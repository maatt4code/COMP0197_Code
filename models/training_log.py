from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


def _coerce_json_value(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _coerce_json_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_json_value(item) for item in value]
    return value


class StructuredTrainingLogger:
    """Persist structured training events to JSON as metrics are produced."""

    def __init__(
        self,
        log_path: Path,
        metadata: dict[str, Any] | None = None,
    ):
        self.log_path = log_path
        self.payload: dict[str, Any] = {
            "schema_version": 1,
            "metadata": _coerce_json_value(metadata or {}),
            "events": [],
            "summary": {},
        }

    def log_event(
        self,
        event: str,
        *,
        metrics: dict[str, Any] | None = None,
        epoch: float | int | None = None,
        step: int | None = None,
    ) -> None:
        event_payload: dict[str, Any] = {"event": event}
        if epoch is not None:
            event_payload["epoch"] = _coerce_json_value(epoch)
        if step is not None:
            event_payload["step"] = int(step)
        if metrics:
            event_payload["metrics"] = _coerce_json_value(metrics)

        self.payload["events"].append(event_payload)
        self.save()

    def update_summary(self, **summary_fields: Any) -> None:
        self.payload["summary"].update(_coerce_json_value(summary_fields))
        self.save()

    def save(self) -> Path:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", encoding="utf-8") as f:
            json.dump(self.payload, f, indent=2)
            f.write("\n")
        return self.log_path


class TrainerMetricsCallback(TrainerCallback):
    """Feed Hugging Face trainer metrics into the shared structured logger."""

    def __init__(self, logger: StructuredTrainingLogger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        raw_metrics = dict(logs)
        if "eval_loss" in raw_metrics or "eval_wer" in raw_metrics:
            event = "eval"
            metrics = {
                "loss": raw_metrics.get("eval_loss"),
                "wer": raw_metrics.get("eval_wer"),
                "runtime": raw_metrics.get("eval_runtime"),
                "samples_per_second": raw_metrics.get("eval_samples_per_second"),
                "steps_per_second": raw_metrics.get("eval_steps_per_second"),
            }
        elif "train_runtime" in raw_metrics or "train_loss" in raw_metrics:
            event = "summary"
            metrics = {
                "loss": raw_metrics.get("train_loss"),
                "runtime": raw_metrics.get("train_runtime"),
                "samples_per_second": raw_metrics.get("train_samples_per_second"),
                "steps_per_second": raw_metrics.get("train_steps_per_second"),
                "total_flos": raw_metrics.get("total_flos"),
            }
        else:
            event = "train"
            metrics = {
                "loss": raw_metrics.get("loss"),
                "grad_norm": raw_metrics.get("grad_norm"),
                "learning_rate": raw_metrics.get("learning_rate"),
            }

        metrics = {
            key: value
            for key, value in metrics.items()
            if value is not None
        }

        self.logger.log_event(
            event,
            epoch=state.epoch,
            step=state.global_step,
            metrics=metrics,
        )
