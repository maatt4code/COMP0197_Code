"""
train_by_age_groups_lora.py — Age-specific Whisper-small + LoRA adapter training
=================================================================================
Trains one LoRA adapter for a single age bucket (3-4, 5-7, or 8-11).

Usage (via AdapterTrainerFactory in train.py):
    adapter = LoraAdapter(config, age_bucket, base_model, processor)
    adapter.train(train_json="train_by_age_bucket_3_4.json",
                  val_json="val_by_age_bucket_3_4.json")

The best checkpoint (lowest eval WER) is written to
    weights/best/<adapter_name>/

GenAI disclosure: Assistive tools (e.g. GitHub Copilot, Claude Code, Cursor) were used in an
assistive role for scaffolding and debugging. All outputs were manually reviewed and tested.
"""

from __future__ import annotations

import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

PARENT_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PARENT_DIR)

from config import Config, TrainingConfig
from metrics import wer as compute_wer

import numpy as np
import torch
import torchaudio

from torch.utils.data import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftConfig

__all__ = ["LoraAdapter"]


# ── Dataset ───────────────────────────────────────────────────────────────────

class WhisperSpeechDataset(Dataset):
    def __init__(
        self,
        records: list,
        processor: WhisperProcessor,
        audio_root: Path,
        max_label_tokens: int,
    ):
        self.records    = records
        self.processor  = processor
        self.audio_root = audio_root
        self.max_label_tokens = max_label_tokens

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        item  = self.records[idx]
        audio = _load_audio(self.audio_root / item["audio_path"])
        feat  = self.processor.feature_extractor(
            audio, sampling_rate=Config.sample_rate(), return_tensors="pt"
        )
        input_features = feat.input_features.squeeze(0)
        labels = self.processor.tokenizer(
            item["orthographic_text"].lower(),
            return_tensors="pt",
            truncation=True,
            max_length=self.max_label_tokens,
        ).input_ids.squeeze(0)
        return {"input_features": input_features, "labels": labels}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = torch.stack([f["input_features"] for f in features])
        label_list = [f["labels"] for f in features]
        max_len    = max(l.shape[0] for l in label_list)
        labels     = torch.full((len(label_list), max_len), -100, dtype=torch.long)
        for i, lbl in enumerate(label_list):
            labels[i, :lbl.shape[0]] = lbl
        return {"input_features": input_features, "labels": labels}


# ── WER metric ────────────────────────────────────────────────────────────────

def _wer_score(refs: list, hyps: list) -> float:
    return compute_wer(refs, hyps)


# ── Custom Trainer ────────────────────────────────────────────────────────────
# Overrides compute_loss and prediction_step to work around the two
# PeftModelForSeq2SeqLM incompatibilities with Whisper's forward signature.

class WhisperLoRATrainer(Seq2SeqTrainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model.base_model.model(
            input_features=inputs["input_features"],
            labels=inputs["labels"],
        )
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if not self.args.predict_with_generate or prediction_loss_only:
            model.eval()
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                outputs = model.base_model.model(
                    input_features=inputs["input_features"],
                    labels=inputs["labels"],
                )
            return outputs.loss.detach(), None, None

        model.eval()
        inputs  = self._prepare_inputs(inputs)
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_audio(path: Path) -> np.ndarray:
    waveform, sr = torchaudio.load(str(path))   # (C, T) float32
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != Config.sample_rate():
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=Config.sample_rate())
    return waveform.squeeze(0).numpy()


def _load_json(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _max_label_tokens(model_config) -> int:
    max_tokens = getattr(model_config, "max_target_positions", None)
    if max_tokens is None:
        max_tokens = getattr(model_config, "max_length", None)
    return int(max_tokens or 448)


def _filter_records_by_label_length(
    records: list,
    processor: WhisperProcessor,
    max_label_tokens: int,
) -> tuple[list, int]:
    kept: list = []
    dropped = 0
    for record in records:
        label_ids = processor.tokenizer(
            record["orthographic_text"].lower(),
            return_tensors="pt",
        ).input_ids.squeeze(0)
        if int(label_ids.shape[0]) <= max_label_tokens:
            kept.append(record)
        else:
            dropped += 1
    return kept, dropped


# ── Adapter ───────────────────────────────────────────────────────────────────

class LoraAdapter:
    """Trains a LoRA adapter for one age bucket and saves the best checkpoint."""

    def __init__(self, config: Config, age_bucket: str, base_model, processor: WhisperProcessor,
                 mock: bool = False, ta_train: bool = False):
        self.config     = config
        self.device     = config.device()
        self.processor  = processor
        self.age_bucket = age_bucket
        self.mock        = mock
        self.ta_train    = ta_train
        # Snapshot base model weights + config on CPU so _build_lora_model() can
        # construct a fresh unwrapped instance each time without hitting HuggingFace
        # and without ever calling get_peft_model() on an already-wrapped model.
        self._base_cfg   = base_model.config
        self._base_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
        self.max_label_tokens = _max_label_tokens(base_model.config)
        assert self.age_bucket in Config.lora_age_buckets(), \
            f"Age bucket {self.age_bucket!r} not in {Config.lora_age_buckets()}"

    # ── public API ────────────────────────────────────────────────────────────

    def train(self, train_json: str, val_json: str) -> None:
        """Train the adapter and write the best checkpoint to weights/best/<name>/.

        Parameters
        ----------
        train_json:
            Filename of the training manifest (resolved via Config.data_dir()).
        val_json:
            Filename of the validation manifest (resolved via Config.data_dir()).
        """
        if self.mock:
            self._mock_load()
            return

        train_path = Config.data_dir() / train_json
        val_path   = Config.data_dir() / val_json
        assert train_path.exists(), f"Train JSON not found: {train_path}"
        assert val_path.exists(),   f"Val JSON not found: {val_path}"

        audio_root  = Config.audio_dir()
        train_data  = _load_json(train_path)
        val_data    = _load_json(val_path)

        n_epochs = TrainingConfig.NUM_EPOCHS
        if self.ta_train:
            train_data = train_data[:TrainingConfig.TA_TRAIN_SAMPLES]
            val_data   = val_data[:TrainingConfig.TA_TRAIN_SAMPLES]
            n_epochs   = TrainingConfig.TA_TRAIN_EPOCHS

        train_data, train_dropped = _filter_records_by_label_length(
            train_data, self.processor, self.max_label_tokens
        )
        val_data, val_dropped = _filter_records_by_label_length(
            val_data, self.processor, self.max_label_tokens
        )
        if not train_data:
            raise ValueError(
                f"All training records for {self.age_bucket} exceed the decoder label limit "
                f"of {self.max_label_tokens} tokens."
            )
        if not val_data:
            raise ValueError(
                f"All validation records for {self.age_bucket} exceed the decoder label limit "
                f"of {self.max_label_tokens} tokens."
            )

        print(f"\n{'='*60}")
        print(f"  LoRA training — age bucket [{self.age_bucket}]"
              + (" [TA_TRAIN]" if self.ta_train else ""))
        print(f"  Train JSON : {train_path}  ({len(train_data)} records)")
        print(f"  Val JSON   : {val_path}  ({len(val_data)} records)")
        if train_dropped or val_dropped:
            print(
                f"  Dropped {train_dropped} train / {val_dropped} val records "
                f"with labels longer than {self.max_label_tokens} tokens"
            )
        print(f"  Epochs: {n_epochs}  Device: {self.device}")
        print('='*60)

        model = self._build_lora_model()
        collator  = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        train_ds  = WhisperSpeechDataset(
            train_data, self.processor, audio_root, self.max_label_tokens
        )
        val_ds    = WhisperSpeechDataset(
            val_data, self.processor, audio_root, self.max_label_tokens
        )

        best_dir = self.config.adapter_best_weights_path(self.age_bucket)
        best_dir.mkdir(parents=True, exist_ok=True)

        training_args = Seq2SeqTrainingArguments(
            output_dir                  = str(best_dir),
            per_device_train_batch_size = TrainingConfig.BATCH_SIZE,
            per_device_eval_batch_size  = TrainingConfig.BATCH_SIZE,
            gradient_accumulation_steps = 1,
            warmup_steps                = TrainingConfig.WARMUP_STEPS,
            num_train_epochs            = n_epochs,
            learning_rate               = TrainingConfig.LEARNING_RATE,
            fp16                        = (self.device == "cuda"),
            eval_strategy               = "epoch",
            save_strategy               = "epoch",
            load_best_model_at_end      = True,
            metric_for_best_model       = "wer",
            greater_is_better           = False,
            predict_with_generate       = True,
            generation_max_length       = 225,
            logging_steps               = 25,
            report_to                   = "none",
            dataloader_num_workers      = 0,
            remove_unused_columns       = False,
            seed                        = Config.seed(),
        )

        def _compute_metrics(pred):
            pred_ids  = pred.predictions
            label_ids = pred.label_ids.copy()
            label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
            pred_strs  = self.processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
            label_strs = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            return {"wer": _wer_score(label_strs, pred_strs)}

        trainer = WhisperLoRATrainer(
            model           = model,
            args            = training_args,
            train_dataset   = train_ds,
            eval_dataset    = val_ds,
            data_collator   = collator,
            compute_metrics = _compute_metrics,
            processing_class = self.processor.feature_extractor,
        )
        trainer.train()

        # After training with load_best_model_at_end=True the model in memory IS
        # the best checkpoint — save the adapter to weights/best/<name>/.
        model.save_pretrained(str(best_dir))
        self.processor.save_pretrained(str(best_dir))
        print(f"  Best adapter saved → {best_dir}")

        eval_results = trainer.evaluate()
        print(f"  [{self.age_bucket}] best eval WER = {eval_results.get('eval_wer', float('nan')):.4f}")

    # ── private helpers ───────────────────────────────────────────────────────

    def _mock_load(self) -> None:
        """Mock mode: load the adapter config from the load weights directory."""
        load_dir = self.config.adapter_load_weights_path(self.age_bucket)
        config_path = load_dir / "adapter_config.json"
        print(f"\n[mock] [{self.age_bucket}] loading adapter config from {load_dir}")
        assert config_path.exists(), (
            f"[mock] adapter_config.json not found at {config_path}. "
            "Run without --mock first to produce a checkpoint."
        )
        cfg = PeftConfig.from_pretrained(str(load_dir))
        print(f"[mock] [{self.age_bucket}] OK — base_model={cfg.base_model_name_or_path}, "
              f"r={getattr(cfg, 'r', '?')}, lora_alpha={getattr(cfg, 'lora_alpha', '?')}")

    def _build_lora_model(self):
        """Construct a fresh LoRA-wrapped Whisper model on the target device.

        Instantiates from the snapshotted config + state dict so that:
        - no HuggingFace download happens on subsequent buckets, and
        - get_peft_model() is always called on a plain (unwrapped) model.
        """
        base = WhisperForConditionalGeneration(self._base_cfg)
        base.load_state_dict(self._base_state)

        base.config.use_cache = False
        base.generation_config = GenerationConfig.from_pretrained(
            self._base_cfg._name_or_path,
            language="english",
            task="transcribe",
        )
        base.generation_config.max_length = None

        lora_cfg = LoraConfig(
            r              = TrainingConfig.LORA_R,
            lora_alpha     = TrainingConfig.LORA_ALPHA,
            target_modules = TrainingConfig.LORA_TARGET_MODULES,
            lora_dropout   = TrainingConfig.LORA_DROPOUT,
            bias           = "none",
            task_type      = TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(base, lora_cfg).to(self.device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")
        return model
