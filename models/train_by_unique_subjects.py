#!/usr/bin/env python3
"""Strong-model experiment entrypoint (Whisper LoRA + future recipes)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from config import Config
import torch
import os

__all__ = ["UniqueSubjectsAdapter"]

class UniqueSubjectsAdapter:
    def __init__(self, config: Config, age_bucket: str, base_model, processor):
        self.config = config
        self.device = config.device()
        self.processor = processor
        self.age_bucket = age_bucket
        self.base_model = base_model.to(self.device).eval()
        assert self.age_bucket in Config.AGE_BUCKETS, f"Age bucket {self.age_bucket} not found in {Config.AGE_BUCKETS}"

    def train(self, peft_model, adapter_name, train_json, val_json):
        weights_path = self.config.adapter_weights_path(self.age_bucket)
        assert os.path.exists(weights_path), f"Adapter weights file not found: {weights_path}"
        assert os.path.exists(train_json), f"Train JSON file not found: {train_json}"
        assert os.path.exists(val_json), f"Val JSON file not found: {val_json}"
        try:
            adapter_weights = torch.load(weights_path, map_location=self.device)
            print(f"Loaded adapter weights for age bucket {self.age_bucket} from {weights_path}")
        except Exception as e:
            print(f"Error loading adapter weights for age bucket {self.age_bucket}: {e}")
            raise e
        peft_model.load_adapter(
            weights_path,
            adapter_name=adapter_name
        )
        peft_model.set_adapter(adapter_name)
        peft_model.delete_adapter(adapter_name)
    


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=["nemo_noise", "whisper_lora"], required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--age-stratified", action="store_true")

    # Whisper LoRA specific
    parser.add_argument("--whisper-model", default="openai/whisper-small")
    parser.add_argument("--language", default="english")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA config
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,v_proj",
        help="Comma-separated module names to LoRA-wrap",
    )
    parser.add_argument("--merge-and-save", action="store_true")
    parser.add_argument(
        "--noise-roots",
        type=Path,
        nargs="*",
        default=[],
        help="Optional noise directories for waveform-level mixing",
    )
    parser.add_argument("--noise-prob", type=float, default=0.0, help="Probability of noise mix per train sample")
    parser.add_argument("--snr-min", type=float, default=5.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_manifest(path: Path, max_samples: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def _write_snapshot(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = {
        "recipe": args.recipe,
        "train_manifest": str(args.train_manifest),
        "val_manifest": str(args.val_manifest),
        "snr_min": args.snr_min,
        "snr_max": args.snr_max,
        "age_stratified": args.age_stratified,
        "whisper_model": args.whisper_model,
        "language": args.language,
        "task": args.task,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "num_workers": args.num_workers,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "seed": args.seed,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "merge_and_save": args.merge_and_save,
    }
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2)


def _run_whisper_lora(args: argparse.Namespace) -> None:
    import random

    import numpy as np
    import librosa
    import torch
    from data.realclass_mixer import list_noise_files
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.utils.data import Dataset, WeightedRandomSampler
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        set_seed,
    )

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    train_rows = _read_manifest(args.train_manifest, args.max_train_samples)
    val_rows = _read_manifest(args.val_manifest, args.max_val_samples)
    if not train_rows:
        raise ValueError(f"No records found in train manifest: {args.train_manifest}")
    if not val_rows:
        raise ValueError(f"No records found in val manifest: {args.val_manifest}")

    print(
        f"Loaded train={len(train_rows)} rows, val={len(val_rows)} rows from manifests. "
        f"Model={args.whisper_model}"
    )
    processor = WhisperProcessor.from_pretrained(args.whisper_model)
    model = WhisperForConditionalGeneration.from_pretrained(args.whisper_model)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, task=args.task
    )
    model.config.use_cache = False

    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    class WaveformNoiseAugmentor:
        def __init__(self, noise_files: list[Path], snr_min: float, snr_max: float):
            self.noise_files = noise_files
            self.snr_min = snr_min
            self.snr_max = snr_max

        @staticmethod
        def _rms(x: torch.Tensor) -> torch.Tensor:
            return torch.sqrt(torch.mean(x * x) + 1e-12)

        def _sample_noise(self, target_len: int, sr: int) -> torch.Tensor:
            noise_path = random.choice(self.noise_files)
            noise, _ = librosa.load(str(noise_path), sr=sr, mono=True)
            noise = torch.tensor(noise, dtype=torch.float32)
            if noise.numel() >= target_len:
                start = random.randint(0, noise.numel() - target_len)
                return noise[start : start + target_len]
            repeat = (target_len + noise.numel() - 1) // noise.numel()
            return noise.repeat(repeat)[:target_len]

        def __call__(self, clean: torch.Tensor, sr: int) -> torch.Tensor:
            noise = self._sample_noise(clean.numel(), sr)
            snr_db = random.uniform(self.snr_min, self.snr_max)
            clean_rms = self._rms(clean)
            noise_rms = self._rms(noise)
            target_noise_rms = clean_rms / (10 ** (snr_db / 20))
            scaled_noise = noise * (target_noise_rms / (noise_rms + 1e-8))
            mixed = clean + scaled_noise
            peak = torch.max(torch.abs(mixed)) + 1e-8
            if peak > 1.0:
                mixed = mixed / peak
            return mixed

    noise_files = list_noise_files(list(args.noise_roots))
    augmentor = (
        WaveformNoiseAugmentor(noise_files, args.snr_min, args.snr_max)
        if noise_files and args.noise_prob > 0
        else None
    )
    if augmentor:
        print(f"Using waveform noise augmentation with {len(noise_files)} files")

    @dataclass
    class ManifestDataset(Dataset):
        records: list[dict[str, Any]]
        processor: WhisperProcessor
        augmentor: WaveformNoiseAugmentor | None = None
        noise_prob: float = 0.0

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            row = self.records[idx]
            audio_path = row["audio_filepath"]
            text = row["text"]
            waveform_np, _ = librosa.load(audio_path, sr=16000, mono=True)
            waveform = torch.tensor(waveform_np, dtype=torch.float32)
            if self.augmentor is not None and random.random() < self.noise_prob:
                # v3.1 order: waveform noise mix first, then feature extraction.
                waveform = self.augmentor(waveform, 16000)
            input_features = self.processor(
                waveform.numpy(), sampling_rate=16000, return_tensors="pt"
            ).input_features[0]
            labels = self.processor.tokenizer(text, return_tensors="pt").input_ids[0]
            if labels.numel() > 0 and labels[0].item() == self.processor.tokenizer.bos_token_id:
                labels = labels[1:]
            return {"input_features": input_features, "labels": labels}

    @dataclass
    class DataCollatorSpeechSeq2Seq:
        processor: WhisperProcessor

        def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            return batch

    train_ds = ManifestDataset(train_rows, processor, augmentor=augmentor, noise_prob=args.noise_prob)
    val_ds = ManifestDataset(val_rows, processor)
    run_dir = args.output_dir / "whisper_lora"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_sampler = None
    if args.age_stratified:
        age_counts: dict[str, int] = {}
        for row in train_rows:
            age = (row.get("age_bucket") or "unknown").strip() or "unknown"
            age_counts[age] = age_counts.get(age, 0) + 1
        weights = []
        for row in train_rows:
            age = (row.get("age_bucket") or "unknown").strip() or "unknown"
            weights.append(1.0 / age_counts[age])
        train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        print(f"Enabled age-stratified sampling across buckets: {sorted(age_counts.keys())}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.num_workers > 0:
        print(
            "Note: forcing num_workers=0 for trainer dataloaders to avoid "
            "pickle issues with local dataset/collator classes."
        )
        args.num_workers = 0
    use_fp16 = device == "cuda"
    print(f"Training device: {device}")
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        predict_with_generate=False,
        fp16=use_fp16,
        bf16=False,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        report_to=[],
        seed=args.seed,
        load_best_model_at_end=False,
    )

    class WhisperTrainer(Seq2SeqTrainer):
        def __init__(self, *trainer_args, train_sampler=None, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            self._train_sampler = train_sampler

        def get_train_dataloader(self):
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")
            data_collator = self.data_collator
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=self._train_sampler,
                shuffle=self._train_sampler is None,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                drop_last=self.args.dataloader_drop_last,
            )

    trainer = WhisperTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorSpeechSeq2Seq(processor=processor),
        processing_class=processor,
        train_sampler=train_sampler,
    )
    trainer.train()
    trainer.save_model()

    adapter_dir = args.output_dir / "whisper_lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to: {adapter_dir}")

    if args.merge_and_save:
        print("Merging LoRA adapter into base Whisper model...")
        base_model = WhisperForConditionalGeneration.from_pretrained(args.whisper_model)
        peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
        merged = peft_model.merge_and_unload()
        merged_dir = args.output_dir / "whisper_lora_merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(merged_dir)
        processor.save_pretrained(merged_dir)
        print(f"Saved merged model to: {merged_dir}")


def main() -> None:
    args = parse_args()
    if not args.train_manifest.exists():
        raise FileNotFoundError(f"Missing train manifest: {args.train_manifest}")
    if not args.val_manifest.exists():
        raise FileNotFoundError(f"Missing val manifest: {args.val_manifest}")
    _write_snapshot(args)

    if args.dry_run:
        print("Saved run configuration only (--dry-run).")
        return

    if args.recipe == "whisper_lora":
        _run_whisper_lora(args)
        return

    print(
        "Saved run configuration. 'nemo_noise' trainer is still pending; "
        "use --recipe whisper_lora for active training."
    )


if __name__ == "__main__":
    main()
