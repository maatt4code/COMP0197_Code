#!/usr/bin/env python3
"""
Plot adapter training logs to a PDF suitable for LaTeX import.

By default this script discovers ``training_log.json`` files under
``weights/best/<adapter>/`` and creates a multi-panel PDF with one subplot per
model, showing train loss, validation loss, and validation WER when available.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_DIR = HERE / "weights" / "best"
DEFAULT_OUT = HERE / "training_logs.pdf"
PREFERRED_ORDER = ["age_3_4", "age_5_7", "age_8_11", "gate_mlp", "unique_subjects"]
LORA_ADAPTER_TO_VAL_JSON = {
    "age_3_4": "val_by_age_bucket_3_4.json",
    "age_5_7": "val_by_age_bucket_5_7.json",
    "age_8_11": "val_by_age_bucket_8_11.json",
    "unique_subjects": "val_by_child_id.json",
}
POSTHOC_LOSS_SOURCE = "posthoc_checkpoint_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training_log.json files to a PDF.")
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=DEFAULT_WEIGHTS_DIR,
        help="Directory containing per-model subdirectories with training_log.json files.",
    )
    parser.add_argument(
        "--logs",
        type=Path,
        nargs="*",
        default=None,
        help="Explicit training_log.json paths. If provided, --weights-dir is ignored.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output PDF path.",
    )
    parser.add_argument(
        "--recover-lora-val-loss",
        action="store_true",
        help=(
            "Evaluate validation loss for supported LoRA checkpoints and write the "
            "recovered losses back into training_log.json before plotting."
        ),
    )
    parser.add_argument(
        "--eval-device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use when recovering LoRA validation loss.",
    )
    parser.add_argument(
        "--base-data-dir",
        type=Path,
        default=None,
        help="Optional override for the dataset base directory (parent of audio/ and noise/).",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Optional override for the audio directory used during post-hoc evaluation.",
    )
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=None,
        help=(
            "Optional local Whisper base-model directory to use for post-hoc "
            "LoRA evaluation, e.g. a local copy of whisper-small."
        ),
    )
    return parser.parse_args()


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _display_name(log_path: Path, payload: object) -> str:
    if isinstance(payload, dict):
        metadata = payload.get("metadata", {})
        if isinstance(metadata, dict):
            adapter = metadata.get("adapter")
            if isinstance(adapter, str) and adapter:
                return adapter
    return log_path.parent.name


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _discover_logs(weights_dir: Path) -> list[Path]:
    logs = sorted(weights_dir.glob("*/training_log.json"))
    order = {name: idx for idx, name in enumerate(PREFERRED_ORDER)}
    return sorted(logs, key=lambda path: (order.get(path.parent.name, math.inf), path.parent.name))


def _is_recoverable_lora_adapter(adapter_name: str) -> bool:
    return adapter_name in LORA_ADAPTER_TO_VAL_JSON


def _approx_equal(lhs: float | None, rhs: float | None, *, tol: float = 1e-6) -> bool:
    if lhs is None or rhs is None:
        return False
    return math.isclose(lhs, rhs, rel_tol=tol, abs_tol=tol)


def _checkpoint_step(checkpoint_dir: Path) -> int | None:
    try:
        return int(checkpoint_dir.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return None


def _load_trainer_state(checkpoint_dir: Path) -> dict[str, Any]:
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        return {}
    payload = _load_json(trainer_state_path)
    return payload if isinstance(payload, dict) else {}


def _checkpoint_epoch(checkpoint_dir: Path) -> float | None:
    trainer_state = _load_trainer_state(checkpoint_dir)
    epoch = _safe_float(trainer_state.get("epoch"))
    if epoch is not None:
        return epoch
    for item in reversed(trainer_state.get("log_history", [])):
        if not isinstance(item, dict):
            continue
        epoch = _safe_float(item.get("epoch"))
        if epoch is not None:
            return epoch
    return None


def _discover_checkpoint_dirs(adapter_dir: Path) -> list[Path]:
    checkpoints = [path for path in adapter_dir.glob("checkpoint-*") if path.is_dir()]
    return sorted(checkpoints, key=_checkpoint_step)


def _legacy_summary_from_points(
    eval_points: list[tuple[float, float]],
    wer_points: list[tuple[float, float]],
) -> dict[str, float]:
    summary: dict[str, float] = {}
    if eval_points:
        best_epoch, best_val_loss = min(eval_points, key=lambda item: item[1])
        summary["best_val_loss"] = best_val_loss
        summary["best_epoch"] = best_epoch
    if wer_points:
        best_wer_epoch, best_eval_wer = min(wer_points, key=lambda item: item[1])
        summary["best_eval_wer"] = best_eval_wer
        summary.setdefault("best_epoch", best_wer_epoch)
    return summary


def _infer_val_manifest(log_path: Path, payload: object) -> Path | None:
    adapter_name = _display_name(log_path, payload)
    if not _is_recoverable_lora_adapter(adapter_name):
        return None

    if isinstance(payload, dict):
        metadata = payload.get("metadata", {})
        if isinstance(metadata, dict):
            val_json = metadata.get("val_json")
            if isinstance(val_json, str) and val_json:
                return HERE / "data" / val_json

    fallback = LORA_ADAPTER_TO_VAL_JSON.get(adapter_name)
    if fallback is None:
        return None
    return HERE / "data" / fallback


def _resolve_eval_device(device_name: str) -> str:
    if device_name != "auto":
        return device_name

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _configure_whisper_model(model) -> None:
    model.config.suppress_tokens = None
    model.config.begin_suppress_tokens = None
    model.config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = None
    model.generation_config.begin_suppress_tokens = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.max_length = None
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"


def _normalise_manifest_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        flattened: list[dict[str, Any]] = []
        for value in payload.values():
            if isinstance(value, list):
                flattened.extend(item for item in value if isinstance(item, dict))
        return flattened
    raise ValueError(f"Unsupported manifest format: {type(payload).__name__}")


def _evaluate_lora_checkpoint_loss(
    checkpoint_dir: Path,
    *,
    val_manifest: Path,
    audio_dir: Path,
    device: str,
    base_model_source: str,
) -> float:
    import torch
    from peft import PeftModel
    from torch.utils.data import DataLoader
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from config import Config, TrainingConfig
    from models.train_by_age_groups_lora import (
        DataCollatorSpeechSeq2SeqWithPadding,
        WhisperSpeechDataset,
        _filter_records_by_label_length,
        _load_json as load_manifest_json,
        _max_label_tokens,
    )

    try:
        processor = WhisperProcessor.from_pretrained(
            base_model_source,
            local_files_only=True,
        )
    except OSError as exc:
        raise RuntimeError(
            "Unable to load the Whisper processor locally. "
            "Pass --base-model-dir to a local whisper-small directory."
        ) from exc
    try:
        base_model = WhisperForConditionalGeneration.from_pretrained(
            base_model_source,
            local_files_only=True,
        )
    except OSError as exc:
        raise RuntimeError(
            "Unable to load the Whisper base model locally. "
            "Pass --base-model-dir to a local whisper-small directory."
        ) from exc
    _configure_whisper_model(base_model)
    model = PeftModel.from_pretrained(base_model, str(checkpoint_dir)).to(device)
    model.eval()

    max_label_tokens = _max_label_tokens(base_model.config)
    val_records = _normalise_manifest_records(load_manifest_json(val_manifest))
    val_records, _ = _filter_records_by_label_length(val_records, processor, max_label_tokens)
    dataset = WhisperSpeechDataset(
        val_records,
        processor,
        audio_dir,
        max_label_tokens,
    )
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    dataloader = DataLoader(
        dataset,
        batch_size=TrainingConfig.BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in dataloader:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            outputs = model.base_model.model(
                input_features=input_features,
                labels=labels,
            )
            batch_size = labels.shape[0]
            total_loss += float(outputs.loss.item()) * batch_size
            total_examples += int(batch_size)

    del model
    del base_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    if total_examples == 0:
        raise ValueError(f"No validation examples available for {checkpoint_dir}")
    return total_loss / total_examples


def _recover_lora_val_losses_for_log(
    log_path: Path,
    *,
    eval_device: str,
    base_data_dir: Path | None,
    audio_dir: Path | None,
    base_model_dir: Path | None,
) -> str | None:
    payload = _load_json(log_path)
    adapter_name = _display_name(log_path, payload)
    if not _is_recoverable_lora_adapter(adapter_name):
        return None

    checkpoint_dirs = _discover_checkpoint_dirs(log_path.parent)
    if not checkpoint_dirs:
        return f"{adapter_name}: no checkpoint directories found under {log_path.parent}"

    val_manifest = _infer_val_manifest(log_path, payload)
    if val_manifest is None or not val_manifest.exists():
        return f"{adapter_name}: validation manifest not found for {log_path}"

    from config import Config

    if base_data_dir is not None:
        Config.set_base_data_dir(base_data_dir)
    if audio_dir is not None:
        Config.set_audio_dir(audio_dir)

    resolved_audio_dir = audio_dir.resolve() if audio_dir is not None else Config.audio_dir().resolve()
    base_model_source = str(base_model_dir.resolve()) if base_model_dir is not None else Config.model_name()
    results: list[dict[str, Any]] = []
    for checkpoint_dir in checkpoint_dirs:
        step = _checkpoint_step(checkpoint_dir)
        epoch = _checkpoint_epoch(checkpoint_dir)
        val_loss = _evaluate_lora_checkpoint_loss(
            checkpoint_dir,
            val_manifest=val_manifest,
            audio_dir=resolved_audio_dir,
            device=eval_device,
            base_model_source=base_model_source,
        )
        results.append(
            {
                "checkpoint_dir": str(checkpoint_dir.resolve()),
                "epoch": epoch,
                "step": step,
                "val_loss": val_loss,
            }
        )

    if isinstance(payload, dict) and "events" in payload:
        events = payload.get("events", [])
        if not isinstance(events, list):
            raise ValueError(f"Expected events list in {log_path}")

        for result in results:
            matched_event = None
            for event in events:
                if not isinstance(event, dict) or event.get("event") != "eval":
                    continue
                event_step = event.get("step")
                event_epoch = _safe_float(event.get("epoch"))
                if result["step"] is not None and event_step == result["step"]:
                    matched_event = event
                    break
                if _approx_equal(event_epoch, result["epoch"]):
                    matched_event = event
                    break

            if matched_event is None:
                matched_event = {
                    "event": "eval",
                    "epoch": result["epoch"],
                    "step": result["step"],
                    "metrics": {},
                }
                events.append(matched_event)

            metrics = matched_event.setdefault("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
                matched_event["metrics"] = metrics
            metrics["loss"] = result["val_loss"]
            metrics["loss_source"] = POSTHOC_LOSS_SOURCE
            metrics["checkpoint_dir"] = result["checkpoint_dir"]

        payload["events"] = sorted(
            events,
            key=lambda event: (
                _safe_float(event.get("epoch")) if isinstance(event, dict) and _safe_float(event.get("epoch")) is not None else math.inf,
                event.get("step", math.inf) if isinstance(event, dict) and event.get("step") is not None else math.inf,
            ),
        )
        summary = payload.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        best_result = min(results, key=lambda item: item["val_loss"])
        summary["best_val_loss"] = best_result["val_loss"]
        summary["best_val_loss_epoch"] = best_result["epoch"]
        summary["best_val_loss_checkpoint_dir"] = best_result["checkpoint_dir"]
        summary["best_val_loss_source"] = POSTHOC_LOSS_SOURCE
        payload["summary"] = summary
    elif isinstance(payload, list):
        preserved_rows = []
        for row in payload:
            if not isinstance(row, dict):
                preserved_rows.append(row)
                continue
            if row.get("metric_source") == POSTHOC_LOSS_SOURCE:
                continue
            preserved_rows.append(row)

        for result in results:
            preserved_rows.append(
                {
                    "epoch": result["epoch"],
                    "step": result["step"],
                    "val_loss": result["val_loss"],
                    "checkpoint_dir": result["checkpoint_dir"],
                    "metric_source": POSTHOC_LOSS_SOURCE,
                }
            )
        payload = preserved_rows
    else:
        raise ValueError(f"Unsupported training log format: {log_path}")

    _save_json(log_path, payload)
    best_result = min(results, key=lambda item: item["val_loss"])
    return (
        f"{adapter_name}: recovered {len(results)} checkpoint val losses "
        f"(best={best_result['val_loss']:.4f} at epoch {best_result['epoch']})"
    )


def _parse_structured_log(payload: dict) -> dict:
    train_points: list[tuple[float, float]] = []
    eval_points: list[tuple[float, float]] = []
    wer_points: list[tuple[float, float]] = []
    summary = payload.get("summary", {})

    for event in payload.get("events", []):
        if not isinstance(event, dict):
            continue
        metrics = event.get("metrics", {})
        if not isinstance(metrics, dict):
            continue

        x = _safe_float(event.get("epoch"))
        if x is None:
            x = _safe_float(event.get("step"))
        if x is None:
            continue

        event_name = event.get("event")
        if event_name == "train":
            loss = _safe_float(metrics.get("loss"))
            if loss is None:
                continue
            train_points.append((x, loss))
        elif event_name == "eval":
            eval_loss = _safe_float(metrics.get("loss"))
            if eval_loss is not None:
                eval_points.append((x, eval_loss))

            wer = _safe_float(metrics.get("wer"))
            if wer is not None:
                wer_points.append((x, wer))

    return {
        "train_points": train_points,
        "eval_points": eval_points,
        "wer_points": wer_points,
        "summary": summary if isinstance(summary, dict) else {},
    }


def _parse_legacy_log(payload: list[dict]) -> dict:
    train_points: list[tuple[float, float]] = []
    eval_points: list[tuple[float, float]] = []
    wer_points: list[tuple[float, float]] = []

    for idx, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            continue

        x = _safe_float(row.get("epoch"))
        if x is None:
            x = float(idx)

        if "train_loss" in row or "val_loss" in row:
            train_loss = _safe_float(row.get("train_loss"))
            val_loss = _safe_float(row.get("val_loss"))
            if train_loss is not None:
                train_points.append((x, train_loss))
            if val_loss is not None:
                eval_points.append((x, val_loss))
            continue

        loss = _safe_float(row.get("loss"))
        if loss is not None:
            train_points.append((x, loss))

        eval_loss = _safe_float(row.get("eval_loss"))
        if eval_loss is not None:
            eval_points.append((x, eval_loss))

        eval_wer = _safe_float(row.get("eval_wer"))
        if eval_wer is not None:
            wer_points.append((x, eval_wer))

    return {
        "train_points": train_points,
        "eval_points": eval_points,
        "wer_points": wer_points,
        "summary": _legacy_summary_from_points(eval_points, wer_points),
    }


def _parse_log(path: Path) -> dict:
    payload = _load_json(path)
    if isinstance(payload, dict) and "events" in payload:
        parsed = _parse_structured_log(payload)
    elif isinstance(payload, list):
        parsed = _parse_legacy_log(payload)
    else:
        raise ValueError(f"Unsupported training log format: {path}")

    parsed["name"] = _display_name(path, payload)
    parsed["path"] = path
    return parsed


def _format_summary_text(summary: dict) -> str:
    parts: list[str] = []
    best_val_loss = _safe_float(summary.get("best_val_loss"))
    best_eval_wer = _safe_float(summary.get("best_eval_wer"))
    best_epoch = summary.get("best_epoch")

    if best_val_loss is not None:
        parts.append(f"best val loss={best_val_loss:.4f}")
    if best_eval_wer is not None:
        parts.append(f"best eval WER={best_eval_wer:.4f}")
    if best_epoch is not None:
        epoch = _safe_float(best_epoch)
        if epoch is not None:
            parts.append(f"best epoch={epoch:g}")

    return " | ".join(parts)


def _plot_model(ax, parsed: dict) -> None:
    train_points = parsed["train_points"]
    eval_points = parsed["eval_points"]
    wer_points = parsed.get("wer_points", [])
    wer_ax = None

    if train_points:
        xs, ys = zip(*train_points)
        ax.plot(xs, ys, color="#1f77b4", linewidth=1.8, label="train loss")
    if eval_points:
        xs, ys = zip(*eval_points)
        ax.plot(xs, ys, color="#d62728", linewidth=1.8, marker="o", markersize=3, label="val loss")
    if wer_points:
        wer_ax = ax.twinx()
        xs, ys = zip(*wer_points)
        wer_ax.plot(
            xs,
            ys,
            color="#2ca02c",
            linewidth=1.8,
            marker="s",
            markersize=3,
            linestyle="--",
            label="val WER",
        )
        wer_ax.set_ylabel("WER", color="#2ca02c")
        wer_ax.tick_params(axis="y", colors="#2ca02c")
        wer_ax.grid(False)

    if not train_points and not eval_points and not wer_points:
        ax.text(0.5, 0.5, "No training data found", ha="center", va="center", transform=ax.transAxes)

    ax.set_title(parsed["name"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)

    summary_text = _format_summary_text(parsed.get("summary", {}))
    if summary_text:
        ax.text(0.02, 0.98, summary_text, fontsize=8, color="#555555",
                va="top", ha="left", transform=ax.transAxes)

    handles, labels = ax.get_legend_handles_labels()
    if wer_ax is not None:
        wer_handles, wer_labels = wer_ax.get_legend_handles_labels()
        handles.extend(wer_handles)
        labels.extend(wer_labels)
    if handles:
        ax.legend(handles, labels, frameon=False, fontsize=8, loc="best")


def main() -> None:
    args = parse_args()
    log_paths = [path.resolve() for path in args.logs] if args.logs else _discover_logs(args.weights_dir.resolve())
    if not log_paths:
        raise FileNotFoundError("No training_log.json files found.")

    if args.recover_lora_val_loss:
        eval_device = _resolve_eval_device(args.eval_device)
        print(f"Recovering LoRA validation loss on device: {eval_device}")
        for log_path in log_paths:
            status = _recover_lora_val_losses_for_log(
                log_path,
                eval_device=eval_device,
                base_data_dir=args.base_data_dir,
                audio_dir=args.audio_dir,
                base_model_dir=args.base_model_dir,
            )
            if status:
                print(status)

    parsed_logs = [_parse_log(path) for path in log_paths]
    n = len(parsed_logs)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11.0, 3.4 * nrows), squeeze=False)
    fig.suptitle("Training Log Overview", fontsize=14, fontweight="bold")

    for ax, parsed in zip(axes.flat, parsed_logs):
        _plot_model(ax, parsed)

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
