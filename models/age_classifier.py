"""
age_classifier.py — Shared gate classifier utilities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import Config
from models.whisper_common import encode_audios, load_audio

__all__ = [
    "AgeClassifierHead",
    "TemperatureScaler",
    "expected_calibration_error",
    "load_gate_checkpoint",
    "predict_gate_probs",
    "run_gate_inference",
    "save_gate_checkpoint",
]


class AgeClassifierHead(nn.Module):
    """MLP classifier over pooled Whisper encoder embeddings."""

    def __init__(self, d_model: int, hidden: int, n_classes: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.n_classes = n_classes
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, pooled_embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_embeddings)


class TemperatureScaler(nn.Module):
    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(float(init_temperature)))
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp_min(1e-6)


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)

    for bin_idx in range(n_bins):
        left = bin_edges[bin_idx]
        right = bin_edges[bin_idx + 1]
        if bin_idx == 0:
            in_bin = (confidences >= left) & (confidences <= right)
        else:
            in_bin = (confidences > left) & (confidences <= right)

        if in_bin.any():
            bin_accuracy = accuracies[in_bin].float().mean()
            bin_confidence = confidences[in_bin].mean()
            ece += in_bin.float().mean() * torch.abs(bin_accuracy - bin_confidence)

    return float(ece.item())


def save_gate_checkpoint(
    checkpoint_path: Path,
    classifier_head: AgeClassifierHead,
    temperature_scaler: TemperatureScaler,
    age_dict: dict[str, int],
    adapter_names: list[str],
    history: list[dict],
    best_epoch: int,
    whisper_model_name: str,
) -> None:
    checkpoint = {
        "classifier_head_state_dict": classifier_head.state_dict(),
        "age_dict": age_dict,
        "temperature": float(temperature_scaler.temperature.detach().cpu().item()),
        "whisper_model_name": whisper_model_name,
        "dropout": classifier_head.dropout,
        "hidden": classifier_head.hidden,
        "d_model": classifier_head.d_model,
        "n_classes": classifier_head.n_classes,
        "history": history,
        "best_epoch": best_epoch,
        "adapter_names": adapter_names,
    }
    torch.save(checkpoint, str(checkpoint_path))


def _infer_head_shape_from_state_dict(state_dict: dict) -> tuple[int, int, int]:
    first_linear = state_dict["net.0.weight"]
    second_linear = state_dict["net.3.weight"]
    d_model = int(first_linear.shape[1])
    hidden = int(first_linear.shape[0])
    n_classes = int(second_linear.shape[0])
    return d_model, hidden, n_classes


def load_gate_checkpoint(checkpoint_path: Path, device: str) -> dict:
    """
    Load a gate checkpoint, supporting both the new classifier schema and the
    older ``{\"state_dict\": ..., \"config\": ...}`` schema.
    """
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=device,
        weights_only=True,
    )

    if "classifier_head_state_dict" in checkpoint:
        state_dict = checkpoint["classifier_head_state_dict"]
        d_model = int(checkpoint.get("d_model", _infer_head_shape_from_state_dict(state_dict)[0]))
        hidden = int(checkpoint.get("hidden", _infer_head_shape_from_state_dict(state_dict)[1]))
        n_classes = int(checkpoint.get("n_classes", _infer_head_shape_from_state_dict(state_dict)[2]))
        dropout = float(checkpoint.get("dropout", 0.3))
        age_dict = checkpoint.get(
            "age_dict",
            {bucket: idx for idx, bucket in enumerate(Config.lora_age_buckets())},
        )
        adapter_names = checkpoint.get(
            "adapter_names",
            [Config.lora_bucket_to_adapter(bucket) for bucket in Config.lora_age_buckets()],
        )
        temperature = float(checkpoint.get("temperature", 1.0))
        history = checkpoint.get("history", [])
        best_epoch = int(checkpoint.get("best_epoch", 0))
        whisper_model_name = checkpoint.get("whisper_model_name", Config.model_name())
    elif "state_dict" in checkpoint and "config" in checkpoint:
        legacy_config = checkpoint["config"]
        state_dict = checkpoint["state_dict"]
        d_model = int(legacy_config.get("d_model", _infer_head_shape_from_state_dict(state_dict)[0]))
        hidden = int(legacy_config.get("hidden", _infer_head_shape_from_state_dict(state_dict)[1]))
        n_classes = int(legacy_config.get("n_classes", _infer_head_shape_from_state_dict(state_dict)[2]))
        dropout = float(legacy_config.get("dropout", 0.1))
        age_buckets = legacy_config.get("age_buckets", Config.lora_age_buckets())
        age_dict = {bucket: idx for idx, bucket in enumerate(age_buckets)}
        adapter_names = legacy_config.get(
            "adapter_names",
            [Config.lora_bucket_to_adapter(bucket) for bucket in age_buckets],
        )
        temperature = 1.0
        history = []
        best_epoch = 0
        whisper_model_name = Config.model_name()
    else:
        raise ValueError(
            f"Unrecognised gate checkpoint schema at {checkpoint_path}"
        )

    classifier_head = AgeClassifierHead(
        d_model=d_model,
        hidden=hidden,
        n_classes=n_classes,
        dropout=dropout,
    ).to(device)
    classifier_head.load_state_dict(state_dict)
    classifier_head.eval()

    scaler = TemperatureScaler(init_temperature=temperature).to(device)
    scaler.eval()

    age_buckets = [bucket for bucket, _ in sorted(age_dict.items(), key=lambda item: item[1])]
    idx_to_age = {idx: bucket for bucket, idx in age_dict.items()}

    return {
        "classifier_head": classifier_head,
        "temperature_scaler": scaler,
        "age_dict": age_dict,
        "age_buckets": age_buckets,
        "idx_to_age": idx_to_age,
        "adapter_names": adapter_names,
        "history": history,
        "best_epoch": best_epoch,
        "whisper_model_name": whisper_model_name,
        "raw_checkpoint": checkpoint,
    }


def _normalise_audio_inputs(
    inputs,
    audio_root: Path | None = None,
) -> list[np.ndarray]:
    if isinstance(inputs, np.ndarray):
        return [inputs]
    if isinstance(inputs, dict):
        assert audio_root is not None, "audio_root is required when passing manifest records."
        return [load_audio(audio_root / inputs["audio_path"])]
    if isinstance(inputs, list):
        if not inputs:
            return []
        first = inputs[0]
        if isinstance(first, np.ndarray):
            return inputs
        if isinstance(first, dict):
            assert audio_root is not None, "audio_root is required when passing manifest records."
            return [load_audio(audio_root / item["audio_path"]) for item in inputs]
    raise TypeError("Unsupported gate input type.")


def run_gate_inference(
    inputs,
    encoder,
    processor,
    classifier_head: AgeClassifierHead,
    temperature_scaler: TemperatureScaler,
    device: str,
    age_buckets: list[str] | None = None,
    audio_root: Path | None = None,
    adapter_model=None,
) -> dict[str, torch.Tensor | list[str]]:
    """Run the gate classifier and return raw + calibrated outputs."""
    age_buckets = age_buckets or Config.lora_age_buckets()
    audios = _normalise_audio_inputs(inputs, audio_root=audio_root)
    classifier_head.eval()
    temperature_scaler.eval()
    encoder.eval()

    with torch.no_grad():
        pooled, _ = encode_audios(
            encoder=encoder,
            processor=processor,
            audios=audios if len(audios) > 1 else audios[0],
            device=device,
            adapter_model=adapter_model,
        )
        logits = classifier_head(pooled)
        calibrated_logits = temperature_scaler(logits)
        probs = torch.softmax(calibrated_logits, dim=-1)

    return {
        "logits": logits.detach().cpu(),
        "calibrated_logits": calibrated_logits.detach().cpu(),
        "probs": probs.detach().cpu(),
        "age_buckets": age_buckets,
    }


def predict_gate_probs(
    inputs,
    encoder,
    processor,
    classifier_head: AgeClassifierHead,
    temperature_scaler: TemperatureScaler,
    device: str,
    age_buckets: list[str] | None = None,
    audio_root: Path | None = None,
    adapter_model=None,
) -> dict[str, float] | list[dict[str, float]]:
    """Return calibrated routing probabilities keyed by age bucket."""
    outputs = run_gate_inference(
        inputs=inputs,
        encoder=encoder,
        processor=processor,
        classifier_head=classifier_head,
        temperature_scaler=temperature_scaler,
        device=device,
        age_buckets=age_buckets,
        audio_root=audio_root,
        adapter_model=adapter_model,
    )

    rows = []
    for probs in outputs["probs"]:
        rows.append(
            {
                bucket: float(probs[idx].item())
                for idx, bucket in enumerate(outputs["age_buckets"])
            }
        )
    return rows[0] if len(rows) == 1 else rows
