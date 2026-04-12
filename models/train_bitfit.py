# In order to run even with unstable internet connection, I used screen to run the script.
#
# GenAI disclosure: Assistive tools (e.g. Cursor/LLM-based coding assistants) were used for
# refactoring, documentation, and boilerplate. All changes were reviewed and tested locally.


# =========================================================
# 0. Cache directories + global setup
# =========================================================
import os
os.environ["HF_HOME"] = "/cs/student/project_msc/2025/ml/jungrlee/huggingface_cache"
os.environ["TORCH_HOME"] = "/cs/student/project_msc/2025/ml/jungrlee/torch_cache"
os.environ["KAGGLEHUB_CACHE"] = "/cs/student/project_msc/2025/ml/jungrlee/kaggle_cache"

import sys
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchaudio
import jiwer

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# =========================================================
# 1. Configuration
# =========================================================
DATA_ROOT = Path("/cs/student/project_msc/2025/ml/jungrlee/ADL")
MODEL_NAME = "openai/whisper-small"
SAMPLE_RATE = 16_000
MAX_DURATION_SEC = 30.0

NUM_TRAIN_SAMPLES = None
NUM_EVAL_SAMPLES = None

BITFIT_TRAIN_BIAS_ONLY = True

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
OUTPUT_DIR = DATA_ROOT / "whisper_bitfit"
SEED = 42

ZERO_SHOT_N = 200
AGE_BUCKETS = ["3-4", "5-7", "8-11"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 2. Logging utilities
# =========================================================
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


log_file = open(OUTPUT_DIR / "run.log", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)


def save_text(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def append_text(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_figure(path: Path, dpi: int = 150):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


# =========================================================
# 3. Reproducibility
# =========================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"torch        : {torch.__version__}")
print(f"CUDA         : {torch.cuda.is_available()}")
import peft
import transformers
print(f"peft         : {peft.__version__}")
print(f"transformers : {transformers.__version__}")
print(f"Device       : {device}")
print(f"Output dir   : {OUTPUT_DIR}")

# =========================================================
# 4. Load and explore dataset
# =========================================================
with open(DATA_ROOT / "train_word_transcripts.jsonl", "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

df = pd.DataFrame(raw_data)

print(f"Total utterances : {len(df):,}")
print(f"Age distribution :\n{df['age_bucket'].value_counts()}")
print(f"\nDuration stats (sec):\n{df['audio_duration_sec'].describe().round(2)}")

df.to_csv(OUTPUT_DIR / "full_dataset_overview.csv", index=False)

dataset_stats = {
    "total_utterances": int(len(df)),
    "age_distribution": {k: int(v) for k, v in df["age_bucket"].value_counts().to_dict().items()},
    "duration_stats_sec": {
        k: float(v) for k, v in df["audio_duration_sec"].describe().round(6).to_dict().items()
    },
}
save_json(dataset_stats, OUTPUT_DIR / "dataset_stats.json")

# =========================================================
# 5. Filter and split
# =========================================================
df_filtered = df[df["audio_duration_sec"] <= MAX_DURATION_SEC].reset_index(drop=True)
data = df_filtered.to_dict("records")

train_data, eval_data = train_test_split(data, test_size=0.1, random_state=SEED)

if NUM_TRAIN_SAMPLES is not None:
    train_data = train_data[:NUM_TRAIN_SAMPLES]
if NUM_EVAL_SAMPLES is not None:
    eval_data = eval_data[:NUM_EVAL_SAMPLES]

print(f"Filtered utterances : {len(df_filtered):,}")
print(f"Train               : {len(train_data):,}")
print(f"Eval                : {len(eval_data):,}")

split_info = {
    "max_duration_sec": MAX_DURATION_SEC,
    "filtered_utterances": int(len(df_filtered)),
    "num_train": int(len(train_data)),
    "num_eval": int(len(eval_data)),
}
save_json(split_info, OUTPUT_DIR / "split_info.json")

# =========================================================
# 6. Audio utility
# =========================================================
def load_audio(audio_path_rel: str) -> np.ndarray:
    path = DATA_ROOT / audio_path_rel
    waveform, sr = torchaudio.load(str(path))   # (C, T) float32
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)
    return waveform.squeeze(0).numpy()

# =========================================================
# 7. Processor
# =========================================================
processor = WhisperProcessor.from_pretrained(
    MODEL_NAME,
    language="English",
    task="transcribe",
)
print("Vocab size:", processor.tokenizer.vocab_size)

# =========================================================
# 8. Dataset and collator
# =========================================================
class WhisperSpeechDataset(Dataset):
    def __init__(self, records, processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        audio = load_audio(item["audio_path"])

        feat = self.processor.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = feat.input_features.squeeze(0)

        labels = self.processor.tokenizer(
            item["orthographic_text"].lower(),
            return_tensors="pt",
        ).input_ids.squeeze(0)

        return {
            "input_features": input_features,
            "labels": labels,
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = torch.stack([f["input_features"] for f in features])

        label_list = [f["labels"] for f in features]
        max_len = max(l.shape[0] for l in label_list)
        labels = torch.full((len(label_list), max_len), -100, dtype=torch.long)

        for i, lbl in enumerate(label_list):
            labels[i, :lbl.shape[0]] = lbl

        return {
            "input_features": input_features,
            "labels": labels,
        }

# =========================================================
# 9. Load base model and apply BitFit
# =========================================================
base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="english",
    task="transcribe",
)

base_model.config.forced_decoder_ids = forced_decoder_ids
base_model.config.suppress_tokens = []
base_model.config.use_cache = False
base_model.generation_config.language = "english"
base_model.generation_config.task = "transcribe"
base_model.generation_config.forced_decoder_ids = forced_decoder_ids

# Freeze everything
for param in base_model.parameters():
    param.requires_grad = False

# Unfreeze only bias terms
if BITFIT_TRAIN_BIAS_ONLY:
    for name, param in base_model.named_parameters():
        if "bias" in name:
            param.requires_grad = True

model = base_model.to(device)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\nTrainable parameter names:")
trainable_lines = []
for name, param in model.named_parameters():
    if param.requires_grad:
        line = f"{name} | shape={tuple(param.shape)} | numel={param.numel():,}"
        trainable_lines.append(line)
        print(" ", line)

save_text("\n".join(trainable_lines), OUTPUT_DIR / "trainable_parameters.txt")

print(f"\nTotal params     : {total:,}")
print(f"Trainable params : {trainable:,} ({100 * trainable / total:.4f}%)")

# =========================================================
# 10. WER metric
# =========================================================
_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ReduceToListOfListOfWords(),
])

def wer_score(refs, hyps):
    return jiwer.process_words(
        refs,
        hyps,
        reference_transform=_transform,
        hypothesis_transform=_transform,
    ).wer

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_strs = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_strs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    return {
        "wer": wer_score(label_strs, pred_strs)
    }

# =========================================================
# 11. Zero-shot WER before fine-tuning
# =========================================================
model.eval()

def transcribe(audio, model, processor, max_new_tokens=225):
    feat = processor.feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    input_features = feat.input_features.to(device)

    with torch.no_grad():
        ids = model.generate(
            input_features=input_features,
            max_new_tokens=max_new_tokens,
        )
    return processor.tokenizer.decode(ids[0], skip_special_tokens=True)

zs_preds, zs_refs = [], []
zs_rows = []

for item in eval_data[:ZERO_SHOT_N]:
    audio = load_audio(item["audio_path"])
    pred = transcribe(audio, model, processor)
    ref = item["orthographic_text"]

    zs_preds.append(pred)
    zs_refs.append(ref)
    zs_rows.append({
        "audio_path": item["audio_path"],
        "age_bucket": item.get("age_bucket", ""),
        "reference": ref,
        "prediction_zero_shot": pred,
    })

wer_zero_shot = wer_score(zs_refs, zs_preds)
print(f"Zero-shot WER (n={ZERO_SHOT_N}): {wer_zero_shot:.4f}")

pd.DataFrame(zs_rows).to_csv(OUTPUT_DIR / "zero_shot_predictions_sample.csv", index=False)

# =========================================================
# 12. Custom Trainer for Whisper + BitFit
# =========================================================
class WhisperBitFitTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_features=inputs["input_features"],
            labels=inputs["labels"],
        )
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)

        if not self.args.predict_with_generate or prediction_loss_only:
            model.eval()
            with torch.no_grad():
                outputs = model(
                    input_features=inputs["input_features"],
                    labels=inputs["labels"],
                )
            return outputs.loss.detach(), None, None

        model.eval()
        max_len = self.args.generation_max_length or 225

        with torch.no_grad():
            generated = model.generate(
                input_features=inputs["input_features"],
                max_new_tokens=max_len,
            )

        labels = inputs.get("labels")

        if generated.shape[-1] < max_len:
            generated = self._pad_tensors_to_max_len(generated, max_len)
        if labels is not None and labels.shape[-1] < max_len:
            labels = self._pad_tensors_to_max_len(labels, max_len)

        return None, generated, labels

# =========================================================
# 13. Training
# =========================================================
train_dataset = WhisperSpeechDataset(train_data, processor)
eval_dataset = WhisperSpeechDataset(eval_data, processor)
collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    warmup_steps=100,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=50,
    report_to="none",
    dataloader_num_workers=4,
    remove_unused_columns=False,
    seed=SEED,
)

trainer = WhisperBitFitTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

train_result = trainer.train()

# Save full BitFit model weights
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"BitFit model saved to {OUTPUT_DIR}")

# Save training result object
save_json(
    {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in train_result.metrics.items()},
    OUTPUT_DIR / "train_result_metrics.json",
)

# Save trainer log history
log_history = pd.DataFrame(trainer.state.log_history)
log_history.to_csv(OUTPUT_DIR / "trainer_log_history.csv", index=False)
save_json(log_history.fillna("").to_dict(orient="records"), OUTPUT_DIR / "trainer_log_history.json")

# =========================================================
# 14. Evaluation after fine-tuning
# =========================================================
eval_results = trainer.evaluate()
eval_results_clean = {}
for k, v in eval_results.items():
    if isinstance(v, (np.floating, float, int, np.integer)):
        eval_results_clean[k] = float(v)
    else:
        eval_results_clean[k] = v

save_json(eval_results_clean, OUTPUT_DIR / "eval_results.json")

wer_bitfit = eval_results["eval_wer"]
print(f"BitFit WER (full eval set): {wer_bitfit:.4f}")

# =========================================================
# 15. Training curves
# =========================================================
if "loss" in log_history.columns:
    train_loss_df = log_history.dropna(subset=["loss"])[["step", "loss"]].copy()
    train_loss_df.to_csv(OUTPUT_DIR / "training_loss_curve.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(train_loss_df["step"], train_loss_df["loss"])
    plt.title("Training Loss - Whisper-small + BitFit")
    plt.xlabel("step")
    plt.ylabel("loss")
    save_figure(OUTPUT_DIR / "whisper_bitfit_train_loss.png")

if "eval_loss" in log_history.columns:
    eval_loss_df = log_history.dropna(subset=["eval_loss"])[["epoch", "eval_loss"]].copy()
    eval_loss_df.to_csv(OUTPUT_DIR / "eval_loss_curve.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(eval_loss_df["epoch"], eval_loss_df["eval_loss"], marker="o")
    plt.title("Eval Loss by Epoch - Whisper-small + BitFit")
    plt.xlabel("epoch")
    plt.ylabel("eval_loss")
    save_figure(OUTPUT_DIR / "whisper_bitfit_eval_loss.png")

if "eval_wer" in log_history.columns:
    eval_wer_df = log_history.dropna(subset=["eval_wer"])[["epoch", "eval_wer"]].copy()
    eval_wer_df.to_csv(OUTPUT_DIR / "eval_wer_curve.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(eval_wer_df["epoch"], eval_wer_df["eval_wer"], marker="o")
    plt.title("Eval WER by Epoch - Whisper-small + BitFit")
    plt.xlabel("epoch")
    plt.ylabel("WER")
    save_figure(OUTPUT_DIR / "whisper_bitfit_eval_wer.png")

# =========================================================
# 16. WER by age bucket for BitFit
# =========================================================
model.eval()
age_wer_bitfit = {}
bitfit_pred_rows = []

for bucket in AGE_BUCKETS:
    bucket_eval = [d for d in eval_data if d["age_bucket"] == bucket]
    if not bucket_eval:
        continue

    preds_bitfit, refs = [], []

    for item in bucket_eval[:200]:
        audio = load_audio(item["audio_path"])
        pred = transcribe(audio, model, processor)
        ref = item["orthographic_text"]

        preds_bitfit.append(pred)
        refs.append(ref)

        bitfit_pred_rows.append({
            "audio_path": item["audio_path"],
            "age_bucket": bucket,
            "reference": ref,
            "prediction_bitfit": pred,
        })

    age_wer_bitfit[bucket] = wer_score(refs, preds_bitfit)
    print(f"  BitFit [{bucket}] WER = {age_wer_bitfit[bucket]:.4f}")

pd.DataFrame(bitfit_pred_rows).to_csv(OUTPUT_DIR / "bitfit_predictions_by_age.csv", index=False)
save_json(age_wer_bitfit, OUTPUT_DIR / "age_wer_bitfit.json")

plt.figure(figsize=(7, 4))
x = np.arange(len(age_wer_bitfit))
plt.bar(x, list(age_wer_bitfit.values()), label="BitFit fine-tuned")
plt.xticks(x, list(age_wer_bitfit.keys()))
plt.xlabel("Age bucket")
plt.ylabel("WER")
plt.title("WER by Age Group - Whisper-small + BitFit")
plt.legend()
save_figure(OUTPUT_DIR / "whisper_bitfit_wer_by_age.png")

# =========================================================
# 17. Per-age zero-shot WER for comparison
# =========================================================
print("Computing per-age zero-shot WER for comparison...")

base_for_eval = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
base_for_eval.config.forced_decoder_ids = forced_decoder_ids
base_for_eval.config.suppress_tokens = []
base_for_eval.generation_config.language = "english"
base_for_eval.generation_config.task = "transcribe"
base_for_eval.generation_config.forced_decoder_ids = forced_decoder_ids
base_for_eval.eval()

age_wer_zero = {}
zero_pred_rows = []

for bucket in AGE_BUCKETS:
    bucket_eval = [d for d in eval_data if d["age_bucket"] == bucket]
    if not bucket_eval:
        continue

    preds, refs = [], []

    for item in bucket_eval[:200]:
        audio = load_audio(item["audio_path"])
        pred = transcribe(audio, base_for_eval, processor)
        ref = item["orthographic_text"]

        preds.append(pred)
        refs.append(ref)

        zero_pred_rows.append({
            "audio_path": item["audio_path"],
            "age_bucket": bucket,
            "reference": ref,
            "prediction_zero_shot": pred,
        })

    age_wer_zero[bucket] = wer_score(refs, preds)
    print(f"  Zero-shot [{bucket}] WER = {age_wer_zero[bucket]:.4f}")

pd.DataFrame(zero_pred_rows).to_csv(OUTPUT_DIR / "zero_shot_predictions_by_age.csv", index=False)
save_json(age_wer_zero, OUTPUT_DIR / "age_wer_zero.json")

del base_for_eval
torch.cuda.empty_cache()

# =========================================================
# 18. Side-by-side comparison plot
# =========================================================
buckets = list(age_wer_zero.keys())
zero_vals = [age_wer_zero[b] for b in buckets]
bitfit_vals = [age_wer_bitfit[b] for b in buckets]

comparison_df = pd.DataFrame({
    "age_bucket": buckets,
    "zero_shot_wer": zero_vals,
    "bitfit_wer": bitfit_vals,
    "absolute_improvement": [age_wer_zero[b] - age_wer_bitfit[b] for b in buckets],
    "relative_improvement_percent": [
        ((age_wer_zero[b] - age_wer_bitfit[b]) / age_wer_zero[b] * 100) if age_wer_zero[b] != 0 else np.nan
        for b in buckets
    ],
})
comparison_df.to_csv(OUTPUT_DIR / "age_bucket_comparison.csv", index=False)

x = np.arange(len(buckets))
w = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - w / 2, zero_vals, w, label="Zero-shot", alpha=0.8)
bars2 = ax.bar(x + w / 2, bitfit_vals, w, label="BitFit fine-tuned", alpha=0.8)

for bar in list(bars1) + list(bars2):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{bar.get_height():.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.set_xticks(x)
ax.set_xticklabels(buckets)
ax.set_xlabel("Age bucket")
ax.set_ylabel("WER")
ax.set_title("Zero-shot vs BitFit Fine-tuned WER by Age Group (Whisper-small)")
ax.legend()

save_figure(OUTPUT_DIR / "whisper_bitfit_comparison.png")

print("\nWER improvement (zero-shot -> BitFit):")
improvement_lines = []
for b in buckets:
    delta = age_wer_zero[b] - age_wer_bitfit[b]
    rel = (delta / age_wer_zero[b] * 100) if age_wer_zero[b] != 0 else float("nan")
    line = (
        f"[{b}] {age_wer_zero[b]:.4f} -> {age_wer_bitfit[b]:.4f} "
        f"(delta = {delta:+.4f}, {rel:+.1f}% relative)"
    )
    improvement_lines.append(line)
    print(" ", line)

save_text("\n".join(improvement_lines), OUTPUT_DIR / "wer_improvement_by_age.txt")

# =========================================================
# 19. Beam-search confidence (uncertainty proxy)
# =========================================================
def beam_confidence(audio, model, processor, num_beams=5, max_new_tokens=225):
    feat = processor.feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    input_features = feat.input_features.to(device)

    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_features,
            num_beams=num_beams,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )

    scores = torch.stack(out.scores, dim=1).squeeze(0)
    probs = torch.softmax(scores, dim=-1)
    token_conf = probs.max(dim=-1).values
    token_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

    transcription = processor.tokenizer.decode(
        out.sequences[0],
        skip_special_tokens=True,
    )

    return {
        "transcription": transcription,
        "mean_conf": float(token_conf.mean().item()),
        "mean_entropy": float(token_entropy.mean().item()),
        "token_conf": token_conf.cpu().numpy(),
        "token_entropy": token_entropy.cpu().numpy(),
    }

# Example uncertainty run on one sample
if len(eval_data) > 0:
    sample_unc = eval_data[0]
    audio_unc = load_audio(sample_unc["audio_path"])
    unc_result = beam_confidence(audio_unc, model, processor)
    unc_result_to_save = {
        "audio_path": sample_unc["audio_path"],
        "reference": sample_unc["orthographic_text"],
        "transcription": unc_result["transcription"],
        "mean_conf": unc_result["mean_conf"],
        "mean_entropy": unc_result["mean_entropy"],
        "token_conf_len": int(len(unc_result["token_conf"])),
        "token_entropy_len": int(len(unc_result["token_entropy"])),
    }
    save_json(unc_result_to_save, OUTPUT_DIR / "beam_confidence_example.json")

    pd.DataFrame({
        "token_index": np.arange(len(unc_result["token_conf"])),
        "token_conf": unc_result["token_conf"],
        "token_entropy": unc_result["token_entropy"],
    }).to_csv(OUTPUT_DIR / "beam_confidence_tokens.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(unc_result["token_conf"])
    plt.xlabel("Generated token index")
    plt.ylabel("Max token probability")
    plt.title("Beam-search Token Confidence")
    save_figure(OUTPUT_DIR / "beam_confidence_plot.png")

    plt.figure(figsize=(8, 4))
    plt.plot(unc_result["token_entropy"])
    plt.xlabel("Generated token index")
    plt.ylabel("Entropy")
    plt.title("Beam-search Token Entropy")
    save_figure(OUTPUT_DIR / "beam_entropy_plot.png")

# =========================================================
# 20. Reload BitFit model for inference sanity-check
# =========================================================
def load_bitfit_model(model_dir):
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    model.config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []
    return model

reloaded = load_bitfit_model(OUTPUT_DIR).to(device).eval()
sample = eval_data[0]
audio = load_audio(sample["audio_path"])
hyp = transcribe(audio, reloaded, processor)

print(f"Reference  : {sample['orthographic_text']}")
print(f"Hypothesis : {hyp}")

sanity_check = {
    "audio_path": sample["audio_path"],
    "reference": sample["orthographic_text"],
    "hypothesis": hyp,
}
save_json(sanity_check, OUTPUT_DIR / "reload_sanity_check.json")

del reloaded
torch.cuda.empty_cache()

# =========================================================
# 21. Parameter efficiency summary table
# =========================================================
full_trainable = sum(
    p.numel()
    for p in WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).parameters()
)

rows = [
    ("Whisper-small (zero-shot)", full_trainable, 0, wer_zero_shot),
    ("Whisper-small (full FT)", full_trainable, full_trainable, float("nan")),
    ("Whisper-small + BitFit", full_trainable, trainable, wer_bitfit),
]

summary = pd.DataFrame(
    rows,
    columns=["Model", "Total params", "Trainable params", "WER"]
)
summary["Trainable %"] = (
    summary["Trainable params"] / summary["Total params"] * 100
).round(4)

print(summary.to_string(index=False))
summary.to_csv(OUTPUT_DIR / "summary_table.csv", index=False)

# =========================================================
# 22. Final summary
# =========================================================
print("=" * 60)
print("SUMMARY - Whisper-small + BitFit")
print("=" * 60)
print(f"Base model         : {MODEL_NAME}")
print(f"Trainable params   : {trainable:,} ({100 * trainable / total:.2f}% of total)")
print(f"Train samples      : {len(train_data):,}")
print(f"Eval samples       : {len(eval_data):,}")
print(f"Epochs             : {NUM_EPOCHS}")
print(f"Zero-shot WER      : {wer_zero_shot:.4f}")
print(f"BitFit WER         : {wer_bitfit:.4f}")

improvement = ((wer_zero_shot - wer_bitfit) / wer_zero_shot * 100) if wer_zero_shot != 0 else float("nan")
print(f"Relative WER delta : {improvement:+.1f}%")
print()
print("WER by age bucket:")
for b in buckets:
    print(f"  [{b}] zero-shot={age_wer_zero[b]:.4f}  BitFit={age_wer_bitfit[b]:.4f}")

summary_metrics = {
    "model_name": MODEL_NAME,
    "trainable_params": int(trainable),
    "total_params": int(total),
    "trainable_percent": float(100 * trainable / total),
    "num_train_samples": int(len(train_data)),
    "num_eval_samples": int(len(eval_data)),
    "num_epochs": int(NUM_EPOCHS),
    "zero_shot_wer": float(wer_zero_shot),
    "bitfit_wer": float(wer_bitfit),
    "relative_wer_improvement_percent": float(improvement),
    "age_wer_zero": {k: float(v) for k, v in age_wer_zero.items()},
    "age_wer_bitfit": {k: float(v) for k, v in age_wer_bitfit.items()},
}
save_json(summary_metrics, OUTPUT_DIR / "summary_metrics.json")

final_summary_text = []
final_summary_text.append("=" * 60)
final_summary_text.append("SUMMARY - Whisper-small + BitFit")
final_summary_text.append("=" * 60)
final_summary_text.append(f"Base model         : {MODEL_NAME}")
final_summary_text.append(f"Trainable params   : {trainable:,} ({100 * trainable / total:.2f}% of total)")
final_summary_text.append(f"Train samples      : {len(train_data):,}")
final_summary_text.append(f"Eval samples       : {len(eval_data):,}")
final_summary_text.append(f"Epochs             : {NUM_EPOCHS}")
final_summary_text.append(f"Zero-shot WER      : {wer_zero_shot:.4f}")
final_summary_text.append(f"BitFit WER         : {wer_bitfit:.4f}")
final_summary_text.append(f"Relative WER delta : {improvement:+.1f}%")
final_summary_text.append("")
final_summary_text.append("WER by age bucket:")
for b in buckets:
    final_summary_text.append(
        f"  [{b}] zero-shot={age_wer_zero[b]:.4f}  BitFit={age_wer_bitfit[b]:.4f}"
    )

save_text("\n".join(final_summary_text), OUTPUT_DIR / "final_summary.txt")

print("\nAll logs, tables, JSON files, predictions, and plots have been saved.")
print(f"Saved under: {OUTPUT_DIR}")

# Optional: flush and close log file at the end
log_file.flush()