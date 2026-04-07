# COMP0197 Group 33 — Children's Speech ASR

Whisper-small fine-tuned for children's speech via age-specific LoRA adapters, a
gating MLP router, and a unique-subjects adapter.

---

## Environment setup

```bash
micromamba env create -f env_comp0197_g33_submission.yml
micromamba activate comp0197-pt-g33-submission
```

---

## Repository layout

```
config.py                          — Config and TrainingConfig
train.py                           — Main training entry point
models/
  train_by_age_groups_lora.py      — LoRA adapter per age bucket (3-4, 5-7, 8-11)
  train_by_age_groups_gatingmlp.py — Gating MLP router (requires age adapters)
  train_by_unique_subjects.py      — LoRA adapter across all child IDs
data/
  build_age_bucket_splits.py       — Build train/val/test JSON splits from JSONL
  *.json                           — Pre-built split manifests
weights/
  best/                            — Best checkpoints written by training
  final/                           — Release checkpoints used for inference
```

---

## Data paths

Audio files and noise files live on the cluster under a shared base directory.
JSON manifests in `data/` store paths relative to `<base-data-dir>/audio/`.

| Path | Default |
|------|---------|
| Base data dir | `/cs/student/projects3/COMP0158/grp_1/data` |
| Audio files | `<base-data-dir>/audio/` |
| Noise files | `<base-data-dir>/noise/` |

Override any of these at runtime — see CLI flags below.

---

## Building data splits

Run once from the project root to (re)generate the JSON manifests in `data/`:

```bash
python data/build_age_bucket_splits.py \
    --transcripts /cs/student/projects3/COMP0158/grp_1/data/train_word_transcripts.jsonl \
    --audio-roots audio_part_0 audio_part_1 audio_part_2 \
    --out-dir data/
```

| Flag | Default | Description |
|------|---------|-------------|
| `--transcripts` | `train_word_transcripts.jsonl` | Path to source JSONL |
| `--audio-roots` | `audio_part_0 audio_part_1 audio_part_2` | Audio directories to index |
| `--out-dir` | `data/processed/splits_age` | Output directory |
| `--train-ratio` | `0.8` | Train fraction |
| `--val-ratio` | `0.1` | Validation fraction |
| `--test-ratio` | `0.1` | Test fraction |
| `--seed` | `42` | Random seed |

---

## Training

### Training modes

Exactly one mode must be active (default: `--mock`):

| Flag | Behaviour |
|------|-----------|
| `--mock` *(default)* | Skip training; load and verify `adapter_config.json` from `weights/best/` |
| `--ta-train` | Train on 100 samples for 5 epochs — fast verification without a GPU |
| `--really-train` | Full training run; prompts for confirmation before starting |

`--really-train` and `--train-ensemble` cannot be used together.

### Examples

```bash
# Verify all adapters are loadable (default mock mode)
python train.py

# TA verification — runs quickly, no GPU required
python train.py --ta-train

# Full training — all age adapters + unique_subjects
python train.py --really-train

# Full training — specific adapter only
python train.py --really-train --adapters age_3_4

# Full training — multiple adapters
python train.py --really-train --adapters age_3_4 --adapters age_5_7

# Train gating MLP (age adapters loaded automatically as prerequisites)
python train.py --really-train --adapters gate_mlp

# Train age_3_4 AND gate_mlp together
# (age_5_7 and age_8_11 loaded from weights/best/ as prerequisites)
python train.py --really-train --adapters age_3_4 --adapters gate_mlp

# Override dataset paths
python train.py --ta-train \
    --base-data-dir /path/to/data \
    --audio-dir /path/to/data/audio \
    --noise-dir /path/to/data/noise
```

### Path override flags

| Flag | Default | Description |
|------|---------|-------------|
| `--base-data-dir` | `/cs/student/projects3/COMP0158/grp_1/data` | Parent of `audio/` and `noise/` |
| `--audio-dir` | `<base-data-dir>/audio/` | Root for audio files referenced in JSONs |
| `--noise-dir` | `<base-data-dir>/noise/` | Noise files for augmentation |

### Adapter prerequisites

Some adapters require others to be present in `weights/best/` before training:

| Adapter | Prerequisites |
|---------|---------------|
| `age_3_4`, `age_5_7`, `age_8_11` | none |
| `unique_subjects` | none |
| `gate_mlp` | `age_3_4`, `age_5_7`, `age_8_11` |
| ensemble | all adapters |

Prerequisites are loaded automatically — you do not need to list them in `--adapters`.

### Output

Best checkpoints are saved to `weights/best/<adapter_name>/` after each training run.
