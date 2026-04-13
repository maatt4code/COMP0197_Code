# COMP0197 Group 33 — Children's Speech ASR

Whisper-small fine-tuned for children's speech via age-specific LoRA adapters, a
gating MLP router, and a unique-subjects adapter.

---

## Environment setup

```bash
micromamba env create -f env_comp0197_g33_submission.yml
micromamba activate comp0197-pt-g33-submission
```

## Repository layout

```
config.py                          — Config and TrainingConfig
train.py                           — Main training entry point
test.py                            — Evaluation, metrics, CSV/JSON, and PNG figures
metrics.py                         — WER implementation (no third-party deps)
models/
  __init__.py                      — Public API + re-exports
  age_classifier.py                — Gate MLP classifier + calibration metrics
  whisper_common.py                — Shared audio loading, transcription, encoder utils
  training_log.py                  — Structured JSON logger for train.py
  train_by_age_groups_lora.py      — LoRA adapter per age bucket (3-4, 5-7, 8-11)
  train_by_age_groups_gatingmlp.py — Gating MLP router / classifier training
  train_by_unique_subjects.py      — LoRA adapter across all child IDs
data/
  build_age_bucket_splits.py       — Build train/val/test JSON splits from JSONL
  *.json                           — Pre-built split manifests (train/val/test_by_*)
  test_ta_200.json                 — 200-utterance TA subset (stratified by age, seed=42)
weights/
  best/                            — Default checkpoints directory (read by test.py / mock)
  final/                           — Checkpoints from previous full runs
  README.txt                       — What to include in the Moodle zip
```

---

## Data paths

JSON manifests in `data/` store paths relative to the **audio root** directory.

| Path | Default |
|------|---------|
| Manifests (JSON) | `data/*.json` in the repo |
| Base data dir (audio + noise) | `/cs/student/projects3/COMP0158/grp_1/data` (override with `--base-data-dir`) |
| Audio files | `<base-data-dir>/audio/` |
| Noise files | `<base-data-dir>/noise/` |

On machines without that path, pass `--base-data-dir` (and optionally `--audio-dir` / `--noise-dir`).

Training modes (`--ta-train`, `--really-train`) expect `noise/` under the base data directory for augmentation unless you override `--noise-dir`.

---

## Building data splits (optional)

The pre-built manifests in `data/` are ready to use. To regenerate from a fresh `train_word_transcripts.jsonl`:

```bash
python data/build_age_bucket_splits.py \
    --transcripts /path/to/train_word_transcripts.jsonl \
    --audio-roots audio_part_0 audio_part_1 audio_part_2 \
    --out-dir data/
```

---

## Training

### Training modes

Exactly one mode must be active (default: `--mock`):

| Flag | Behaviour |
|------|-----------|
| `--mock` *(default)* | Skip training; load and verify `adapter_config.json` from `weights/best/` |
| `--ta-train` | Train on 100 samples for 5 epochs — fast verification without a GPU |
| `--really-train` | Full training run; prompts for confirmation before starting |

### Examples

```bash
# Verify all adapters are loadable (default mock mode)
python train.py

# TA verification — runs quickly, no GPU required
python train.py --ta-train --base-data-dir /path/to/data

# Full training — all adapters
python train.py --really-train --base-data-dir /path/to/data

# Full training — specific adapter only
python train.py --really-train --adapters age_3_4 --base-data-dir /path/to/data

# Train gating MLP
python train.py --really-train --adapters gate_mlp --base-data-dir /path/to/data

# Save to a new run directory, load prerequisites from a previous run
python train.py --really-train --adapters gate_mlp \
    --best-dir run_02 --load-dir run_01 --base-data-dir /path/to/data
```

### Path override flags

| Flag | Default | Description |
|------|---------|-------------|
| `--best-dir` | `best` | Subdirectory under `weights/` for new checkpoints |
| `--load-dir` | *(same as `--best-dir`)* | Subdirectory under `weights/` for loading prereqs / mock |
| `--base-data-dir` | UCL lab path in `config.DEFAULT_BASE_DATA_DIR` | Parent of `audio/` and `noise/` |
| `--audio-dir` | `<base-data-dir>/audio/` | Root for audio files referenced in JSONs |
| `--noise-dir` | `<base-data-dir>/noise/` | Noise files for augmentation |
| `--train-ensemble` | — | After normal adapter training, load `weights/final/` for the ensemble hook (incompatible with `--really-train`) |

### Adapter prerequisites

| Adapter | Prerequisites loaded from |
|---------|--------------------------|
| `age_3_4`, `age_5_7`, `age_8_11` | none |
| `unique_subjects` | none |
| `gate_mlp` | none at the PEFT level (frozen encoder + classifier head; see `train_by_age_groups_gatingmlp.py`) |

### Output

Trained checkpoints are saved to `weights/<best-dir>/<adapter_name>/`.

### Ensemble mode (`--train-ensemble`)

Use **mock** (default) or **TA** mode with `--train-ensemble` — not `--really-train`.

Loads all adapter checkpoints from **`weights/final/`** (age LoRAs, `unique_subjects`, `gate_mlp/gate_mlp.pt`) via `load_final_ensemble()`. The CLI still prints a placeholder until the ensemble trainer is wired up; your group can extend that path without changing the flag.

`--really-train` and `--train-ensemble` cannot be combined.

---

## Evaluation (`test.py`)

Writes a timestamped folder under `test_results/` containing `summary.csv`,
`classifier_metrics.csv`, `predictions.jsonl`, `metrics.json`, and figures
`wer_by_bucket.png`, `gate_reliability.png`.

```bash
# Full test manifest (large)
python test.py --load-dir best --base-data-dir /path/to/data \
    --test-json test_by_child_id.json

# TA / quick run — 200 utterances, stratified (data/test_ta_200.json)
python test.py --load-dir best --base-data-dir /path/to/data \
    --test-json test_ta_200.json
```

| Flag | Description |
|------|-------------|
| `--test-json` | Filename under `data/` or absolute path to the test manifest |
| `--max-samples` | Optional cap on the first N records (debug) |
| `--load-dir` | Weights subdirectory (default `best`) |

---

## Quick TA test

```bash
python test.py --load-dir best --base-data-dir /path/to/your/data \
    --test-json test_ta_200.json
```


