"""
train.py — Mixture of LoRA Experts (MoLE) for Children's Speech ASR
====================================================================
Trains a lightweight gating MLP that routes input audio to a weighted
combination of three age-specific LoRA adapters (3-4, 5-7, 8-11).

Pipeline
--------
1. Load the base Whisper-small model from Hugging Face.
2. Load three pre-trained age-specific LoRA adapters.
3. Extract frozen encoder features from a small training set.
4. Train a gating MLP (768 → 256 → 3) to predict age-bucket probabilities.
5. Save the gating MLP weights + config to models/gate_mlp.pt.

This script is designed to run on CPU with a small audio subset (~200 samples,
3–10 seconds each) for demonstration purposes.

GenAI Disclosure: GitHub Copilot and Claude Code were used in an assistive role
for code scaffolding and debugging. All outputs were manually reviewed for
correctness.
"""

import sys
from pathlib import Path

PARENT_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PARENT_DIR)

from config import Config

import os
import json
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

__all__ = ["GatingMLPAdapter"]

class GatingMLPAdapter:
    def __init__(self, config: Config, base_model, processor):
        self.config = config
        self.device = config.device()
        self.processor = processor
        self.base_model = base_model.to(self.device).eval()
        self.gate = None

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


# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
MODEL_NAME   = Config.MODEL_NAME
SAMPLE_RATE  = Config.SAMPLE_RATE

AGE_BUCKETS   = ["3-4", "5-7", "8-11"]
AGE_BUCKET_MAP = {"3-4": 0, "5-7": 1, "8-11": 2}
ADAPTER_NAMES = [name for name in Config.ADAPTER_NAMES if name.startswith("age_")]
ADAPTER_DIRS = {k: v
                for k, v in zip(
                    AGE_BUCKETS,
                    [PARENT_DIR / Config.ADAPTER_WEIGHTS_DIR / f"{Config.ADAPTER_WEIGHTS_FILE_PREFIX}_{name}" for name in ADAPTER_NAMES]
                    )
                }

TRAIN_JSONL  = SCRIPT_DIR / "data" / "train_samples.jsonl"
OUTPUT_PATH  = SCRIPT_DIR / "models" / "gate_mlp.pt"

# Gating MLP hyper-parameters
SEED         = Config.SEED
ENCODER_DIM  = Config.MODEL_DIM
GATE_HIDDEN  = 256
GATE_DROPOUT = 0.1
GATE_LR      = 1e-3
GATE_EPOCHS  = 10
BATCH_SIZE   = 16

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_audio(audio_path_rel):
    path = SCRIPT_DIR / audio_path_rel
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio.astype(np.float32),
                                 orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


# ── Gating MLP ────────────────────────────────────────────────────────────────
class GatingMLP(nn.Module):
    def __init__(self, d_model, n_classes, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(data, model, processor, device, batch_size=BATCH_SIZE):
    """Extract mean-pooled encoder features (speech frames only) with all adapters disabled."""
    model.eval()
    all_features, all_labels = [], []

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        audios = [load_audio(item["audio_path"]) for item in batch]

        inputs = processor.feature_extractor(
            audios, sampling_rate=SAMPLE_RATE, return_tensors="pt",
            padding="max_length",
        )
        input_features = inputs.input_features.to(device)

        with model.disable_adapter(), torch.no_grad():
            encoder = model.base_model.model.get_encoder()
            enc_out = encoder(input_features)
            hidden = enc_out.last_hidden_state          # (B, 1500, 768)

            # Pool only speech frames (not silence/padding) — Whisper encodes
            # at 50 frames/sec, but the encoder halves the time dimension so
            # the effective rate is 1500 frames for 30 s ≈ 50 frames/sec.
            pooled_list = []
            for j, audio in enumerate(audios):
                n_speech_frames = min(
                    int(len(audio) / SAMPLE_RATE * 50), hidden.shape[1]
                )
                n_speech_frames = max(n_speech_frames, 1)   # safety floor
                pooled_list.append(hidden[j, :n_speech_frames, :].mean(dim=0))
            pooled = torch.stack(pooled_list)

        all_features.append(pooled.cpu())
        all_labels.extend(AGE_BUCKET_MAP[item["age_bucket"]] for item in batch)
        print(f"  Extracted {min(i + batch_size, len(data)):>4d} / {len(data)}")

    return torch.cat(all_features), torch.tensor(all_labels)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cpu")
    print("=" * 65)
    print("  MoLE Training — Gating MLP for Age-Adaptive LoRA Routing")
    print("=" * 65)
    print(f"Device: {device}")

    # 1. Load training data
    print("\n[1/5] Loading training data...")
    train_data = load_jsonl(TRAIN_JSONL)
    print(f"  Samples: {len(train_data)}")
    print(f"  Age distribution: {Counter(r['age_bucket'] for r in train_data)}")

    # 2. Load base model + adapters
    print("\n[2/5] Loading Whisper-small + LoRA adapters...")
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language="English", task="transcribe"
    )
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    model = PeftModel.from_pretrained(
        base_model, str(ADAPTER_DIRS["3-4"]), adapter_name="age_3_4"
    )
    model.load_adapter(str(ADAPTER_DIRS["5-7"]),  adapter_name="age_5_7")
    model.load_adapter(str(ADAPTER_DIRS["8-11"]), adapter_name="age_8_11")
    model = model.to(device).eval()
    print(f"  Loaded adapters: {list(model.peft_config.keys())}")

    # 3. Extract encoder features
    print("\n[3/5] Extracting encoder features, this may take a while.")
    t0 = time.time()
    train_feats, train_labels = extract_features(
        train_data, model, processor, device
    )
    print(f"  Feature shape: {train_feats.shape}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # 4. Train gating MLP
    print(f"\n[4/5] Training gating MLP ({GATE_EPOCHS} epochs)...")
    gate = GatingMLP(ENCODER_DIM, len(AGE_BUCKETS), GATE_HIDDEN, GATE_DROPOUT)
    gate = gate.to(device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=GATE_LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(GATE_EPOCHS):
        gate.train()
        perm = torch.randperm(len(train_feats))
        epoch_loss, correct = 0.0, 0

        for i in range(0, len(train_feats), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            x = train_feats[idx].to(device)
            y = train_labels[idx].to(device)

            logits = gate(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
            correct += (logits.argmax(-1) == y).sum().item()

        acc = correct / len(train_feats)
        print(f"  Epoch {epoch+1:2d}/{GATE_EPOCHS}  "
              f"loss={epoch_loss/len(train_feats):.4f}  acc={acc:.4f}")

    # 5. Save gate weights + config
    print(f"\n[5/5] Saving gating MLP to {OUTPUT_PATH}...")
    gate_config = {
        "d_model":       ENCODER_DIM,
        "n_classes":     len(AGE_BUCKETS),
        "hidden":        GATE_HIDDEN,
        "dropout":       GATE_DROPOUT,
        "age_buckets":   AGE_BUCKETS,
        "adapter_names": ADAPTER_NAMES,
    }
    torch.save({"state_dict": gate.state_dict(), "config": gate_config},
               str(OUTPUT_PATH))
    print(f"  Saved ({os.path.getsize(OUTPUT_PATH) / 1024:.0f} KB)")
    print(f"  Config: {gate_config}")

    print("\n" + "=" * 65)
    print("  Training complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
