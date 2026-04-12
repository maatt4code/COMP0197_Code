#!/usr/bin/env python3
"""
GenAI disclosure: Assistive tools (e.g. Cursor/LLM-based coding assistants) were used for
refactoring, documentation, and boilerplate. All changes were reviewed and tested locally.

Build instruction.pdf (module submission requirement) using matplotlib only.
Run from repo root:  python build_instruction_pdf.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
OUT = HERE / "instruction.pdf"

# Line width for wrapping (characters, monospace approx)
WRAP = 92

SECTIONS: list[tuple[str, list[str]]] = [
    (
        "COMP0197 Group 33 — Reproduction instructions",
        [
            "This PDF lists extra Python packages beyond the Coursework 1 micromamba env",
            "and the exact steps to reproduce training, evaluation, and figures.",
            "",
            "Additional packages (max 3, pip on top of comp0197-pt):",
            "  1. peft",
            "  2. soundfile",
            "  (No third pip package is required; PyTorch, transformers, torchaudio come from the env file.)",
        ],
    ),
    (
        "1. Environment",
        [
            "From the project root:",
            "  micromamba env create -f env_comp0197_g33_submission.yml -y",
            "  micromamba activate comp0197-pt-g33-submission",
            "",
            "The file is PyTorch-based and aligned with comp0197-pt (single framework: PyTorch).",
        ],
    ),
    (
        "2. Expected submission zip layout",
        [
            "  config.py  train.py  test.py  metrics.py  build_instruction_pdf.py",
            "  env_comp0197_g33_submission.yml  instruction.pdf  README.md",
            "  data/*.json  (manifests including test_ta_200.json)",
            "  models/",
            "  weights/best/<adapter_name>/  (trained checkpoints — required for marking)",
            "  Audio corpus: not in repo; markers use --base-data-dir (see section 3).",
        ],
    ),
    (
        "3. Data layout",
        [
            "JSON manifests live in data/ in the zip. Audio files are not in the repo.",
            "Default base (UCL lab): /cs/student/projects3/COMP0158/grp_1/data with audio/ and noise/.",
            "Elsewhere, pass your corpus root with --base-data-dir (and optionally --audio-dir / --noise-dir).",
            "Obtain the dataset per module rules; this repo does not download the corpus.",
        ],
    ),
    (
        "4. Training (bundle the checkpoints you report)",
        [
            "Smoke test (no GPU required, small run):",
            "  python train.py --ta-train --base-data-dir <your_base>",
            "",
            "Full fine-tune (long-running):",
            "  python train.py --really-train --base-data-dir <your_base>",
            "",
            "Optional ensemble hook (loads weights/final/ after adapter training; not with --really-train):",
            "  python train.py --train-ensemble --base-data-dir <your_base>",
            "",
            "Checkpoints are written to weights/best/<adapter_name>/. Include that tree in the zip.",
        ],
    ),
    (
        "5. Evaluation — full test set",
        [
            "Produces metrics, JSON/CSV, and PNG figures under test_results/<timestamp>/.",
            "  python test.py --load-dir best --base-data-dir <your_base> \\",
            "      --test-json test_by_child_id.json",
        ],
    ),
    (
        "6. Evaluation — TA quick run (200 utterances)",
        [
            "Stratified subset: data/test_ta_200.json (seed 42). Typical runtime is much lower",
            "than the full test_by_child_id.json set.",
            "  python test.py --load-dir best --base-data-dir <your_base> \\",
            "      --test-json test_ta_200.json",
        ],
    ),
    (
        "7. Outputs",
        [
            "Under test_results/<YYYYMMDD_HHMMSS>/:",
            "  summary.csv           — WER by mode and age split",
            "  classifier_metrics.csv — gate accuracy, NLL, ECE",
            "  predictions.jsonl     — per-utterance transcriptions and gate probs",
            "  metrics.json          — machine-readable summary",
            "  wer_by_bucket.png     — WER bar chart (selected modes)",
            "  gate_reliability.png  — calibration-style reliability diagram for the gate",
        ],
    ),
]


def wrap_line(line: str, width: int) -> list[str]:
    if len(line) <= width:
        return [line]
    words = line.split()
    out: list[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur = f"{cur} {w}"
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def main() -> None:
    lines: list[str] = []
    for title, body in SECTIONS:
        lines.append(f"=== {title} ===")
        lines.extend(body)
        lines.append("")

    flat: list[str] = []
    for line in lines:
        if not line.strip():
            flat.append("")
            continue
        flat.extend(wrap_line(line, WRAP))

    page_lines = 42
    pages: list[list[str]] = []
    buf: list[str] = []
    for line in flat:
        if len(buf) >= page_lines:
            pages.append(buf)
            buf = []
        buf.append(line)
    if buf:
        pages.append(buf)

    with PdfPages(OUT) as pdf:
        for chunk in pages:
            fig = plt.figure(figsize=(8.5, 11))
            fig.patch.set_facecolor("white")
            y = 0.97
            for ln in chunk:
                fig.text(
                    0.06,
                    y,
                    ln,
                    fontsize=9,
                    va="top",
                    ha="left",
                    family="monospace",
                    wrap=False,
                )
                y -= 0.022
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
