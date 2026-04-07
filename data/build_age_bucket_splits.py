#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build train/val/test JSON files bucketed by age from transcripts, "
            "using only audio files present in audio_part_0/1/2 and skipping noise."
        )
    )
    parser.add_argument(
        "--transcripts",
        default="train_word_transcripts.jsonl",
        help="Path to transcript JSONL with age_bucket and utterance_id fields.",
    )
    parser.add_argument(
        "--audio-roots",
        nargs="+",
        default=["audio_part_0", "audio_part_1", "audio_part_2"],
        help="Audio roots to include. Noise roots are intentionally excluded.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/splits_age",
        help="Output directory for age-bucketed train/val/test JSON files.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def is_noise_row(row: dict) -> bool:
    utt_id = str(row.get("utterance_id", ""))
    audio_path = str(row.get("audio_path", ""))
    text = str(row.get("orthographic_text", "")).strip().lower()
    if utt_id.startswith("N_"):
        return True
    if "noise" in audio_path.lower():
        return True
    if text in {"noise", "[noise]", "<noise>", "(noise)"}:
        return True
    return False


def discover_audio_index(audio_roots: list[Path]) -> dict[str, str]:
    index = {}
    for root in audio_roots:
        if not root.exists():
            continue
        for fp in root.glob("*.flac"):
            stem = fp.stem
            index[stem] = str(fp.as_posix())
    return index


def split_bucket_by_child(
    items: list[dict], train_ratio: float, val_ratio: float, test_ratio: float, rng: random.Random
):
    by_child = defaultdict(list)
    for s in items:
        child_id = s.get("child_id") or f"UNKNOWN_{s['utterance_id']}"
        by_child[child_id].append(s)

    child_groups = list(by_child.items())
    rng.shuffle(child_groups)

    total = len(items)
    target_train = total * train_ratio
    target_val = total * val_ratio
    target_test = total * test_ratio

    splits = {"train": [], "val": [], "test": []}
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_targets = {"train": target_train, "val": target_val, "test": target_test}

    # Assign each child as an indivisible group to the most under-target split.
    for _, group_items in child_groups:
        best_split = min(
            ("train", "val", "test"),
            key=lambda name: (split_counts[name] - split_targets[name], split_counts[name]),
        )
        splits[best_split].extend(group_items)
        split_counts[best_split] += len(group_items)

    return splits["train"], splits["val"], splits["test"]


def main() -> None:
    args = parse_args()
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError("train/val/test ratios must sum to 1.0")

    transcript_path = Path(args.transcripts)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_roots = [Path(p) for p in args.audio_roots]
    audio_index = discover_audio_index(audio_roots)
    if not audio_index:
        raise RuntimeError("No audio files found in provided audio roots.")

    by_bucket = defaultdict(list)
    seen_ids = set()
    kept = 0
    skipped_noise = 0
    skipped_missing_audio = 0
    skipped_missing_bucket = 0

    with transcript_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if is_noise_row(row):
                skipped_noise += 1
                continue
            age_bucket = row.get("age_bucket")
            utt_id = row.get("utterance_id")
            if not age_bucket or not utt_id:
                skipped_missing_bucket += 1
                continue
            if utt_id in seen_ids:
                continue
            abs_audio = audio_index.get(utt_id)
            if not abs_audio:
                skipped_missing_audio += 1
                continue

            seen_ids.add(utt_id)
            sample = {
                "utterance_id": utt_id,
                "child_id": row.get("child_id"),
                "session_id": row.get("session_id"),
                "age_bucket": age_bucket,
                "audio_path": abs_audio,
                "audio_duration_sec": row.get("audio_duration_sec"),
                "orthographic_text": row.get("orthographic_text"),
            }
            by_bucket[age_bucket].append(sample)
            kept += 1

    rng = random.Random(args.seed)
    train = {}
    val = {}
    test = {}
    for bucket in sorted(by_bucket.keys()):
        tr, va, te = split_bucket_by_child(
            by_bucket[bucket], args.train_ratio, args.val_ratio, args.test_ratio, rng
        )
        train[bucket] = tr
        val[bucket] = va
        test[bucket] = te

    (out_dir / "train_by_age_bucket.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "val_by_age_bucket.json").write_text(
        json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "test_by_age_bucket.json").write_text(
        json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Also write separate files per age bucket for convenience.
    per_bucket_dir = out_dir / "by_bucket"
    per_bucket_dir.mkdir(parents=True, exist_ok=True)
    for bucket in sorted(by_bucket.keys()):
        bucket_dir = per_bucket_dir / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        (bucket_dir / "train.json").write_text(
            json.dumps(train[bucket], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (bucket_dir / "val.json").write_text(
            json.dumps(val[bucket], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (bucket_dir / "test.json").write_text(
            json.dumps(test[bucket], ensure_ascii=False, indent=2), encoding="utf-8"
        )

    summary = {
        "kept_total": kept,
        "skipped_noise": skipped_noise,
        "skipped_missing_bucket_or_id": skipped_missing_bucket,
        "skipped_missing_audio_in_audio_parts": skipped_missing_audio,
        "age_buckets": sorted(by_bucket.keys()),
        "counts": {
            "train": {k: len(v) for k, v in train.items()},
            "val": {k: len(v) for k, v in val.items()},
            "test": {k: len(v) for k, v in test.items()},
        },
        "unique_children": {
            "train": {k: len({s.get("child_id") for s in v}) for k, v in train.items()},
            "val": {k: len({s.get("child_id") for s in v}) for k, v in val.items()},
            "test": {k: len({s.get("child_id") for s in v}) for k, v in test.items()},
        },
        "per_bucket_output_dir": str(per_bucket_dir.as_posix()),
    }
    (out_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
