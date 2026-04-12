#!/usr/bin/env python3
"""Build a fixed 200-utterance TA test manifest (stratified by age_bucket, seed=42).

GenAI disclosure: Assistive tools (e.g. Cursor/LLM-based coding assistants) were used for
refactoring, documentation, and boilerplate. All changes were reviewed and tested locally.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent / "test_by_child_id.json",
        help="Full test manifest (bucket-keyed dict).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "test_ta_200.json",
        help="Output JSON path (flat list of records).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.source.open("r", encoding="utf-8") as f:
        data = json.load(f)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    if isinstance(data, dict):
        for rows in data.values():
            for row in rows:
                by_bucket[row["age_bucket"]].append(row)
    else:
        for row in data:
            by_bucket[row["age_bucket"]].append(row)

    buckets = sorted(by_bucket.keys())
    n_buckets = len(buckets)
    if n_buckets == 0:
        raise SystemExit("No records found in source manifest.")

    base, rem = divmod(args.total, n_buckets)
    counts = [base + (1 if i < rem else 0) for i in range(n_buckets)]

    rng = random.Random(args.seed)
    selected: list[dict] = []
    for bucket, want in zip(buckets, counts):
        pool = by_bucket[bucket][:]
        if len(pool) < want:
            raise SystemExit(
                f"Bucket {bucket!r} has only {len(pool)} utterances; need {want}."
            )
        rng.shuffle(pool)
        selected.extend(pool[:want])

    rng.shuffle(selected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=True)
    print(f"Wrote {len(selected)} records to {args.out} (seed={args.seed}).")


if __name__ == "__main__":
    main()
