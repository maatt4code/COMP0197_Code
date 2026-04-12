"""
whisper_common.py — Shared Whisper audio / encoder / generation helpers.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchaudio

from config import Config

__all__ = [
    "batch_records",
    "encode_audios",
    "extract_pooled_embeddings",
    "get_whisper_encoder",
    "load_audio",
    "load_manifest_records",
    "mean_pool_encoder_outputs",
    "prepare_audio_features",
    "transcribe_audio",
    "transcribe_audio_with_details",
    "transcribe_record",
]


def load_manifest_records(path: Path) -> list[dict]:
    """Load a manifest JSON and flatten child-id keyed manifests to a record list."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        records: list[dict] = []
        for rows in data.values():
            records.extend(rows)
        return records

    return data


def load_audio(path: Path | str) -> np.ndarray:
    """Load mono audio and resample to Whisper's expected sample rate."""
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != Config.sample_rate():
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sr,
            new_freq=Config.sample_rate(),
        )
    return waveform.squeeze(0).numpy()


def batch_records(records: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def prepare_audio_features(
    processor,
    audios: np.ndarray | list[np.ndarray],
    device: str | None = None,
    padding: str = "max_length",
) -> dict[str, torch.Tensor]:
    """Convert raw audio arrays into Whisper input features + attention mask."""
    features = processor(
        audios,
        sampling_rate=Config.sample_rate(),
        return_tensors="pt",
        padding=padding,
        return_attention_mask=True,
    )
    if device is None:
        return features
    return {name: tensor.to(device) for name, tensor in features.items()}


def get_whisper_encoder(model):
    """Return the underlying Whisper encoder for a base or PEFT-wrapped model."""
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model.get_encoder()
    if hasattr(model, "model") and hasattr(model.model, "get_encoder"):
        return model.model.get_encoder()
    if hasattr(model, "get_encoder"):
        return model.get_encoder()
    raise TypeError(f"Unsupported Whisper model type: {type(model)!r}")


def mean_pool_encoder_outputs(
    encoder_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean-pool only the valid encoded timesteps, following the notebook logic.

    Whisper's encoder halves the effective number of timesteps, so we project the
    audio attention mask down before pooling.
    """
    num_audio_frames = attention_mask.sum(dim=1)
    enc_lengths = (num_audio_frames + 1) // 2

    t = torch.arange(
        encoder_hidden_states.size(1),
        device=encoder_hidden_states.device,
    ).unsqueeze(0)
    enc_mask = (t < enc_lengths.unsqueeze(1)).to(encoder_hidden_states.dtype)

    num_valid_steps = enc_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled = (encoder_hidden_states * enc_mask.unsqueeze(-1)).sum(dim=1) / num_valid_steps
    return pooled


def encode_audios(
    encoder,
    processor,
    audios: np.ndarray | list[np.ndarray],
    device: str,
    adapter_model=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run Whisper's frozen encoder and return mean-pooled embeddings."""
    features = prepare_audio_features(processor, audios, device=device)
    encode_context = nullcontext()
    if adapter_model is not None and hasattr(adapter_model, "disable_adapter"):
        encode_context = adapter_model.disable_adapter()

    with encode_context:
        encoder_outputs = encoder(
            input_features=features["input_features"],
            output_hidden_states=False,
            return_dict=True,
        )
    pooled = mean_pool_encoder_outputs(
        encoder_outputs.last_hidden_state,
        features["attention_mask"],
    )
    return pooled, features


def extract_pooled_embeddings(
    records: list[dict],
    processor,
    encoder,
    audio_root: Path,
    device: str,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a full manifest into pooled Whisper embeddings and age labels."""
    all_embeddings: list[torch.Tensor] = []
    all_labels: list[int] = []
    age_dict = {bucket: idx for idx, bucket in enumerate(Config.lora_age_buckets())}

    encoder.eval()
    with torch.no_grad():
        for batch in batch_records(records, batch_size=batch_size):
            audios = [load_audio(audio_root / record["audio_path"]) for record in batch]
            pooled, _ = encode_audios(encoder, processor, audios, device=device)
            all_embeddings.append(pooled.cpu())
            all_labels.extend(age_dict[record["age_bucket"]] for record in batch)

    embeddings = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty((0, Config.model_dim()))
    labels = torch.tensor(all_labels, dtype=torch.long)
    return embeddings, labels


def _generate_context(model, adapter_name: str | None):
    if adapter_name is None and hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    if adapter_name is not None and hasattr(model, "set_adapter"):
        model.set_adapter(adapter_name)
    return nullcontext()


def _summarise_generation_scores(scores: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> dict[str, float | int]:
    if not scores:
        return {
            "mean_token_entropy": float("nan"),
            "mean_max_token_probability": float("nan"),
            "generated_token_count": 0,
        }

    stacked_scores = torch.stack(list(scores), dim=1)
    probs = torch.softmax(stacked_scores, dim=-1)
    token_confidence = probs.max(dim=-1).values.squeeze(0)
    token_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).squeeze(0)
    return {
        "mean_token_entropy": float(token_entropy.mean().item()),
        "mean_max_token_probability": float(token_confidence.mean().item()),
        "generated_token_count": int(token_entropy.numel()),
    }


def transcribe_audio_with_details(
    model,
    processor,
    audio: np.ndarray,
    device: str,
    adapter_name: str | None = None,
    max_new_tokens: int = 225,
) -> dict[str, float | int | str]:
    """Generate a transcription plus token-level uncertainty summary."""
    model.eval()
    inputs = prepare_audio_features(processor, audio, device=device)
    generation_context = _generate_context(model, adapter_name=adapter_name)

    with generation_context:
        with torch.no_grad():
            generation = model.generate(
                input_features=inputs["input_features"],
                attention_mask=inputs["attention_mask"],
                max_length=None,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )

    transcription = processor.batch_decode(
        generation.sequences,
        skip_special_tokens=True,
    )[0]
    return {
        "transcription": transcription,
        **_summarise_generation_scores(generation.scores),
    }


def transcribe_audio(
    model,
    processor,
    audio: np.ndarray,
    device: str,
    adapter_name: str | None = None,
    max_new_tokens: int = 225,
) -> str:
    """Generate a transcription for one audio array."""
    model.eval()
    inputs = prepare_audio_features(processor, audio, device=device)
    generation_context = _generate_context(model, adapter_name=adapter_name)

    with generation_context:
        with torch.no_grad():
            generated_ids = model.generate(
                input_features=inputs["input_features"],
                attention_mask=inputs["attention_mask"],
                max_length=None,
                max_new_tokens=max_new_tokens,
            )

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def transcribe_record(
    model,
    processor,
    record: dict,
    audio_root: Path,
    device: str,
    adapter_name: str | None = None,
    max_new_tokens: int = 225,
) -> str:
    audio = load_audio(audio_root / record["audio_path"])
    return transcribe_audio(
        model=model,
        processor=processor,
        audio=audio,
        device=device,
        adapter_name=adapter_name,
        max_new_tokens=max_new_tokens,
    )
