"""
Validate JSON manifests under data/ and external audio (and optionally noise) directories.

Used by ``train.py --prepare-data`` and before TA/full training runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from config import Config


def _flatten_records(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        out: list[dict] = []
        for v in payload.values():
            if isinstance(v, list):
                out.extend(x for x in v if isinstance(x, dict))
        return out
    return []


def validate_data_layout(*, require_noise: bool = False) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    ok = True

    data_dir = Config.data_dir()
    if not data_dir.is_dir():
        return False, [f"Data directory missing: {data_dir}"]

    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        ok = False
        msgs.append(f"No JSON manifests under {data_dir}")
    else:
        msgs.append(f"Found {len(json_files)} manifest(s) under {data_dir}")

    audio_root = Config.audio_dir()
    if not audio_root.is_dir():
        ok = False
        msgs.append(f"Audio root is not a directory: {audio_root}")
    else:
        msgs.append(f"Audio root OK: {audio_root}")

    noise_root = Config.noise_dir()
    if require_noise:
        if not noise_root.is_dir():
            ok = False
            msgs.append(f"Noise directory required for training but missing: {noise_root}")
        else:
            msgs.append(f"Noise directory OK: {noise_root}")

    for manifest in json_files[:5]:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            ok = False
            msgs.append(f"{manifest.name}: invalid JSON ({exc})")
            continue
        records = _flatten_records(payload)
        for rec in records[:8]:
            if not isinstance(rec, dict):
                continue
            ap = rec.get("audio_path")
            if not isinstance(ap, str) or not ap:
                continue
            full = audio_root / ap
            if not full.is_file():
                ok = False
                msgs.append(f"Missing audio file (from {manifest.name}): {full}")
                break
        if not ok:
            break

    return ok, msgs


def print_validation_report(ok: bool, msgs: list[str]) -> None:
    for line in msgs:
        print(line)
    if ok:
        print("[prepare-data] OK — manifests, audio, and (if required) noise paths look valid.")
    else:
        print("[prepare-data] Problems detected:", file=sys.stderr)
