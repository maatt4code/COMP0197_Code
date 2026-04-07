"""
train_by_age_groups_gatingmlp.py — Gating MLP for MoLE (Mixture of LoRA Experts)
==================================================================================
Trains a lightweight gating MLP (768 → 256 → 3) that routes input audio to a
weighted combination of the three age-specific LoRA adapters (3-4, 5-7, 8-11).

Prerequisites: all three age-specific LoRA adapters must already be trained and
present in weights/best/age_*/. They are loaded by AdapterTrainerFactory.load_prereqs()
and passed in as a PeftModel before this adapter is trained.

Pipeline
--------
1. Extract frozen mean-pooled encoder features from training audio (adapters disabled).
2. Train GatingMLP on those features to predict age-bucket probabilities.
3. Save best checkpoint (lowest val loss) to weights/best/gate_mlp/.

GenAI Disclosure: GitHub Copilot and Claude Code were used in an assistive role
for code scaffolding and debugging. All outputs were manually reviewed.
"""

from __future__ import annotations

import sys
import json
import time
from collections import Counter
from pathlib import Path

PARENT_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PARENT_DIR)

from config import Config, TrainingConfig

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import WhisperProcessor
from peft import PeftModel

__all__ = ["GatingMLPAdapter"]


# ── Gating MLP ────────────────────────────────────────────────────────────────

class GatingMLP(nn.Module):
    def __init__(self, d_model: int, n_classes: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Hyper-parameters ──────────────────────────────────────────────────────────

GATE_HIDDEN  = 256
GATE_DROPOUT = 0.1
GATE_LR      = 1e-3
GATE_EPOCHS  = 10
BATCH_SIZE   = 16


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_audio(path: Path) -> np.ndarray:
    waveform, sr = torchaudio.load(str(path))       # (C, T) float32
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != Config.sample_rate():
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=Config.sample_rate()
        )
    return waveform.squeeze(0).numpy()


def _load_json(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_features(
    data: list,
    peft_model: PeftModel,
    processor: WhisperProcessor,
    device: str,
    age_bucket_map: dict[str, int],
    audio_root: Path,
    batch_size: int = BATCH_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract mean-pooled encoder features with all LoRA adapters disabled."""
    peft_model.eval()
    all_features: list[torch.Tensor] = []
    all_labels:   list[int]          = []

    for i in range(0, len(data), batch_size):
        batch  = data[i : i + batch_size]
        audios = [_load_audio(audio_root / item["audio_path"]) for item in batch]

        inputs = processor.feature_extractor(
            audios,
            sampling_rate=Config.sample_rate(),
            return_tensors="pt",
            padding="max_length",
        )
        input_features = inputs.input_features.to(device)

        with peft_model.disable_adapter(), torch.no_grad():
            encoder  = peft_model.base_model.model.get_encoder()
            enc_out  = encoder(input_features)
            hidden   = enc_out.last_hidden_state      # (B, 1500, 768)

            pooled_list: list[torch.Tensor] = []
            for j, audio in enumerate(audios):
                # Pool only speech frames; Whisper's encoder outputs ~50 frames/sec.
                n_frames = min(
                    int(len(audio) / Config.sample_rate() * 50), hidden.shape[1]
                )
                n_frames = max(n_frames, 1)
                pooled_list.append(hidden[j, :n_frames, :].mean(dim=0))
            pooled = torch.stack(pooled_list)

        all_features.append(pooled.cpu())
        all_labels.extend(age_bucket_map[item["age_bucket"]] for item in batch)
        print(f"  Extracted {min(i + batch_size, len(data)):>4d} / {len(data)}")

    return torch.cat(all_features), torch.tensor(all_labels)


# ── Adapter ───────────────────────────────────────────────────────────────────

class GatingMLPAdapter:
    """Trains the gating MLP on top of pre-loaded age-specific LoRA adapters."""

    def __init__(
        self,
        config: Config,
        peft_model: PeftModel,
        processor: WhisperProcessor,
        mock: bool = False,
        ta_train: bool = False,
    ):
        self.config     = config
        self.device     = config.device()
        self.peft_model = peft_model.to(self.device).eval()
        self.processor  = processor
        self.mock       = mock
        self.ta_train   = ta_train

    # ── public API ────────────────────────────────────────────────────────────

    def train(self, train_json: str, val_json: str) -> None:
        """Train the gating MLP and save the best checkpoint to weights/best/gate_mlp/.

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

        audio_root     = Config.audio_dir()
        train_data     = _load_json(train_path)
        val_data       = _load_json(val_path)
        age_bucket_map = {b: i for i, b in enumerate(Config.lora_age_buckets())}

        n_epochs = GATE_EPOCHS
        if self.ta_train:
            train_data = train_data[:TrainingConfig.TA_TRAIN_SAMPLES]
            val_data   = val_data[:TrainingConfig.TA_TRAIN_SAMPLES]
            n_epochs   = TrainingConfig.TA_TRAIN_EPOCHS

        print(f"\n{'='*60}")
        print(f"  GatingMLP training" + (" [TA_TRAIN]" if self.ta_train else ""))
        print(f"  Train JSON : {train_path}  ({len(train_data)} records)")
        print(f"  Val JSON   : {val_path}  ({len(val_data)} records)")
        print(f"  Epochs: {n_epochs}  Device: {self.device}")
        print(f"  Age bucket map: {age_bucket_map}")
        print(f"  Loaded adapters: {list(self.peft_model.peft_config.keys())}")
        print('='*60)

        # 1. Extract frozen encoder features
        print("\n[1/3] Extracting train features...")
        t0 = time.time()
        train_feats, train_labels = _extract_features(
            train_data, self.peft_model, self.processor,
            self.device, age_bucket_map, audio_root,
        )
        print(f"[1/3] Extracting val features...")
        val_feats, val_labels = _extract_features(
            val_data, self.peft_model, self.processor,
            self.device, age_bucket_map, audio_root,
        )
        print(f"  Feature shape: {train_feats.shape}  ({time.time()-t0:.1f}s)")
        print(f"  Train distribution: {Counter(train_labels.tolist())}")

        # 2. Train gating MLP
        print(f"\n[2/3] Training gating MLP ({n_epochs} epochs)...")
        n_classes = len(Config.lora_age_buckets())
        gate      = GatingMLP(Config.model_dim(), n_classes, GATE_HIDDEN, GATE_DROPOUT)
        gate      = gate.to(self.device)
        optimizer = torch.optim.Adam(gate.parameters(), lr=GATE_LR)
        criterion = nn.CrossEntropyLoss()

        best_val_loss  = float("inf")
        best_state     = None

        for epoch in range(n_epochs):
            gate.train()
            perm                  = torch.randperm(len(train_feats))
            epoch_loss, correct   = 0.0, 0

            for i in range(0, len(train_feats), BATCH_SIZE):
                idx    = perm[i : i + BATCH_SIZE]
                x      = train_feats[idx].to(self.device)
                y      = train_labels[idx].to(self.device)
                logits = gate(x)
                loss   = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
                correct    += (logits.argmax(-1) == y).sum().item()

            train_acc  = correct / len(train_feats)
            train_loss = epoch_loss / len(train_feats)

            # Validation
            gate.eval()
            with torch.no_grad():
                val_logits = gate(val_feats.to(self.device))
                val_loss   = criterion(val_logits, val_labels.to(self.device)).item()
                val_acc    = (val_logits.argmax(-1) == val_labels.to(self.device)).float().mean().item()

            print(f"  Epoch {epoch+1:2d}/{n_epochs}  "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in gate.state_dict().items()}
                print(f"    ^ new best val_loss={best_val_loss:.4f}")

        # 3. Save best checkpoint
        best_dir = self.config.adapter_best_weights_path("gate_mlp")
        best_dir.mkdir(parents=True, exist_ok=True)
        output_path = best_dir / "gate_mlp.pt"

        gate_config = {
            "d_model":       Config.model_dim(),
            "n_classes":     n_classes,
            "hidden":        GATE_HIDDEN,
            "dropout":       GATE_DROPOUT,
            "age_buckets":   Config.lora_age_buckets(),
            "adapter_names": [Config.lora_bucket_to_adapter(b) for b in Config.lora_age_buckets()],
        }
        gate.load_state_dict(best_state)
        torch.save({"state_dict": gate.state_dict(), "config": gate_config}, str(output_path))
        print(f"\n[3/3] Best gate saved → {output_path}  "
              f"(val_loss={best_val_loss:.4f})")

    # ── private helpers ───────────────────────────────────────────────────────

    def _mock_load(self) -> None:
        """Mock mode: verify the saved gate checkpoint is readable."""
        best_dir    = self.config.adapter_best_weights_path("gate_mlp")
        output_path = best_dir / "gate_mlp.pt"
        print(f"\n[mock] GatingMLP — loading checkpoint from {output_path}")
        assert output_path.exists(), (
            f"[mock] gate_mlp.pt not found at {output_path}. "
            "Run without --mock first to produce a checkpoint."
        )
        ckpt = torch.load(str(output_path), map_location="cpu")
        cfg  = ckpt["config"]
        print(f"[mock] OK — d_model={cfg['d_model']}, n_classes={cfg['n_classes']}, "
              f"hidden={cfg['hidden']}")
