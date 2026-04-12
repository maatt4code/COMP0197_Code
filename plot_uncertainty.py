"""
plot_uncertainty.py — Probabilistic output figures for the Group 33 report.

Reads predictions.jsonl and produces two figures saved as PDF + PNG:
  1. uncertainty_histograms.pdf  — side-by-side histograms of mean token entropy
                                   and mean max token probability, one subplot per model
  2. uncertainty_by_age.pdf      — grouped bar chart of mean entropy per model x age bucket

Usage:
    python plot_uncertainty.py --results test_results/20260412_140213
    python plot_uncertainty.py --results test_results/20260412_140213 --out Report/pictures
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────

MODELS = [
    ("base",            "Base\n(zero-shot)"),
    ("age_3_4",         "LoRA\n3–4"),
    ("age_5_7",         "LoRA\n5–7"),
    ("age_8_11",        "LoRA\n8–11"),
    ("unique_subjects", "Unique\nSubjects"),
    ("mole_weighted",   "MoLE\nWeighted"),
    ("gated_router",    "Gated\nRouter"),
]

AGE_BUCKETS  = ["3-4", "5-7", "8-11"]
BUCKET_LABEL = {"3-4": "Age 3–4", "5-7": "Age 5–7", "8-11": "Age 8–11"}

PALETTE = {
    "base":            "#5B9BD5",
    "age_3_4":         "#C55A11",
    "age_5_7":         "#548235",
    "age_8_11":        "#7030A0",
    "unique_subjects": "#1F6B75",
    "mole_weighted":   "#C00000",
    "gated_router":    "#2E75B6",
}

FONT = "DejaVu Sans"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_records(results_dir: Path) -> list[dict]:
    path = results_dir / "predictions.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_series(records: list[dict]) -> dict[str, dict]:
    """
    Returns dict keyed by model name, each value a dict with:
        "entropy":  list of mean_token_entropy (all utterances)
        "max_prob": list of mean_max_token_probability (all utterances)
        "entropy_by_age": dict[age_bucket -> list[float]]
    """
    data: dict[str, dict] = {
        m: {"entropy": [], "max_prob": [], "entropy_by_age": {b: [] for b in AGE_BUCKETS}}
        for m, _ in MODELS
    }
    for rec in records:
        age = rec.get("age_bucket", "")
        for m, _ in MODELS:
            unc = rec.get("uncertainty", {}).get(m, {})
            h   = unc.get("mean_token_entropy")
            p   = unc.get("mean_max_token_probability")
            if h is not None:
                data[m]["entropy"].append(h)
                if age in AGE_BUCKETS:
                    data[m]["entropy_by_age"][age].append(h)
            if p is not None:
                data[m]["max_prob"].append(p)
    return data


# ── PNG helper ───────────────────────────────────────────────────────────────

def _save_png(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    """Render matplotlib figure to PNG via Pillow."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi, facecolor="#F9F9F9")
    buf.seek(0)
    img = Image.open(buf)
    img.save(path, format="PNG")
    buf.close()


# ── Figure 1: histograms ──────────────────────────────────────────────────────

def plot_histograms(series: dict, out_dir: Path) -> None:
    n_models = len(MODELS)
    fig, axes = plt.subplots(
        2, n_models,
        figsize=(2.2 * n_models, 5),
        sharey="row",
        facecolor="#F9F9F9",
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.12)

    bins = 40
    for col, (m, label) in enumerate(MODELS):
        color = PALETTE[m]

        # Row 0 — token entropy
        ax_h = axes[0, col]
        vals_h = np.array(series[m]["entropy"])
        ax_h.hist(vals_h, bins=bins, color=color, alpha=0.85, edgecolor="none")
        ax_h.axvline(float(np.mean(vals_h)), color="#111", lw=1.2, ls="--")
        ax_h.set_title(label, fontsize=7.5, fontfamily=FONT, pad=3)
        ax_h.set_xlabel("Entropy $H$", fontsize=6.5, fontfamily=FONT)
        if col == 0:
            ax_h.set_ylabel("Count", fontsize=7, fontfamily=FONT)
        ax_h.tick_params(labelsize=6)
        ax_h.annotate(
            f"μ={np.mean(vals_h):.2f}",
            xy=(0.97, 0.93), xycoords="axes fraction",
            ha="right", va="top", fontsize=6, color="#111", fontfamily=FONT,
        )

        # Row 1 — max token probability
        ax_p = axes[1, col]
        vals_p = np.array(series[m]["max_prob"])
        ax_p.hist(vals_p, bins=bins, color=color, alpha=0.85, edgecolor="none")
        ax_p.axvline(float(np.mean(vals_p)), color="#111", lw=1.2, ls="--")
        ax_p.set_xlabel("Max prob $p^*$", fontsize=6.5, fontfamily=FONT)
        if col == 0:
            ax_p.set_ylabel("Count", fontsize=7, fontfamily=FONT)
        ax_p.tick_params(labelsize=6)
        ax_p.annotate(
            f"μ={np.mean(vals_p):.2f}",
            xy=(0.97, 0.93), xycoords="axes fraction",
            ha="right", va="top", fontsize=6, color="#111", fontfamily=FONT,
        )

    axes[0, 0].set_facecolor("#F0F0F0")
    for ax in axes.flat:
        ax.set_facecolor("#F4F4F4")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    fig.suptitle(
        "Distribution of mean token entropy (top) and mean max token probability (bottom) per model",
        fontsize=8, fontfamily=FONT, y=1.01,
    )

    fig.savefig(out_dir / "uncertainty_histograms.pdf",
                bbox_inches="tight", dpi=200, facecolor="#F9F9F9")
    _save_png(fig, out_dir / "uncertainty_histograms.png", dpi=200)
    plt.close(fig)
    print(f"Saved uncertainty_histograms.pdf/png → {out_dir}")


# ── Figure 2: entropy by age group ───────────────────────────────────────────

def plot_entropy_by_age(series: dict, out_dir: Path) -> None:
    model_keys  = [m for m, _ in MODELS]
    model_labels = [l for _, l in MODELS]
    x = np.arange(len(model_keys))
    width = 0.25
    offsets = np.array([-1, 0, 1]) * width

    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor="#F9F9F9")
    ax.set_facecolor("#F4F4F4")

    age_colors = {"3-4": "#C55A11", "5-7": "#548235", "8-11": "#2E75B6"}

    for i, bucket in enumerate(AGE_BUCKETS):
        means = [float(np.mean(series[m]["entropy_by_age"][bucket]))
                 if series[m]["entropy_by_age"][bucket] else 0.0
                 for m in model_keys]
        bars = ax.bar(x + offsets[i], means, width * 0.9,
                      label=BUCKET_LABEL[bucket],
                      color=age_colors[bucket], alpha=0.88, edgecolor="none")
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=5.5, fontfamily=FONT, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=7.5, fontfamily=FONT)
    ax.set_ylabel("Mean token entropy $H$", fontsize=8, fontfamily=FONT)
    ax.set_title("Mean token entropy by model and age group",
                 fontsize=9, fontfamily=FONT, pad=6)
    ax.legend(fontsize=7.5, framealpha=0.7)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    fig.savefig(out_dir / "uncertainty_by_age.pdf",
                bbox_inches="tight", dpi=200, facecolor="#F9F9F9")
    _save_png(fig, out_dir / "uncertainty_by_age.png", dpi=200)
    plt.close(fig)
    print(f"Saved uncertainty_by_age.pdf/png → {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Path to test_results/<run>/ directory")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: same as --results)")
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_dir     = Path(args.out) if args.out else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from {results_dir / 'predictions.jsonl'} ...")
    records = load_records(results_dir)
    print(f"  {len(records):,} utterances loaded")

    series = extract_series(records)

    plot_histograms(series, out_dir)
    plot_entropy_by_age(series, out_dir)


if __name__ == "__main__":
    main()
