#!/usr/bin/env python3
"""
draw_architecture.py — SVG architecture diagram for MoLE (Mixture of LoRA Experts).

Outputs architecture.svg (vector) and optionally architecture.pdf via cairosvg.

Usage:
    python draw_architecture.py
    python draw_architecture.py --out figs/arch

Dependencies: stdlib only. For PDF: pip install cairosvg

GenAI Disclosure: GitHub Copilot and Claude Code were used in an assistive role.
All outputs were manually reviewed.
"""

import argparse
import base64
import io
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Canvas & palette
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

W, H = 1320, 680

C = dict(
    bg        = "#FFFFFF",
    base      = "#263238",   # encoder / decoder  (blue-grey 900)
    base_hi   = "#37474F",   # lighter for stack layers
    gate      = "#0D47A1",   # gating MLP          (blue 900)
    gate_bg   = "#E3F2FD",   # gate box fill
    l0        = "#B71C1C",   # LoRA age 3-4        (red 900)
    l1        = "#6A1520",   # LoRA age 5-7
    l2        = "#4A148C",   # LoRA age 8-11       (purple 900)
    uniq      = "#E65100",   # unique subjects      (orange 900)
    io        = "#1B5E20",   # input / output       (green 900)
    wt        = "#FFFFFF",   # white text
    dt        = "#1A1A2E",   # dark text
    gray      = "#666666",
    lgray     = "#BBBBBB",
    line      = "#444444",
    sum_bg    = "#FAFAFA",
    ens       = "#2E7D32",   # ensemble box border  (green 800)
    ens_bg    = "#F1F8E9",   # ensemble fill
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layout (px, y-down)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TITLE_H = 0           # no title bar
Y_SP    = 315         # spine y (encoder / decoder)

X_IN   = 66
X_ENC  = 280
X_FORK = 326
X_GATE = X_ENC;  Y_GATE = 104
X_ADPT = 612
ADAPTER_Y_TOP = 162
ADAPTER_Y_BOTTOM = 468
ADAPTER_COUNT = 3
Y_A = [
    ADAPTER_Y_TOP + i * (ADAPTER_Y_BOTTOM - ADAPTER_Y_TOP) / (ADAPTER_COUNT - 1)
    for i in range(ADAPTER_COUNT)
]   # adapter centres
X_BUS  = 500          # routing-weight bus x
# Σ node just touches decoder left face:
#   X_DEC - DEC_HW = X_SIG + SIGMA_R
SIGMA_R = 20
SIGMA_OFFSET = 40
MUL_R = 13
X_MUL = 706
DEC_HW  = 76
X_DEC   = 956
X_SIG   = X_DEC - DEC_HW - SIGMA_R   # = 884
X_OUT   = 1161

# half-extents
ENC_HW, ENC_HH    = 74,  62
GATE_HW, GATE_HH  = 88,  56
ADPT_HW, ADPT_HH  = 66,  32
DEC_HH            = 62
OUT_HW, OUT_HH    = 62,  38

CR      = 9     # corner radius for boxes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SVG document builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SVG:
    def __init__(self, w, h):
        self.w = w; self.h = h
        self._defs = []; self._els = []

    def d(self, s): self._defs.append(s)
    def e(self, s): self._els.append(s)

    def render(self):
        defs = f'<defs>{"".join(self._defs)}</defs>'
        body = "\n  ".join(self._els)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}" '
            f'font-family="DejaVu Sans, Segoe UI, Arial, sans-serif">\n'
            f'  {defs}\n  {body}\n</svg>'
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Primitives
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _marker(svg, mid, color, sz=7):
    svg.d(
        f'<marker id="{mid}" markerWidth="{sz}" markerHeight="{sz}" '
        f'refX="{sz-1}" refY="{sz/2:.1f}" orient="auto" markerUnits="strokeWidth">'
        f'<polygon points="0 0,{sz} {sz/2:.1f},0 {sz}" fill="{color}"/>'
        f'</marker>'
    )


def _rect(svg, x, y, w, h, fill, stroke=None, sw=1.2, rx=CR):
    st = f'stroke="{stroke}" stroke-width="{sw}"' if stroke else 'stroke="none"'
    svg.e(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
          f'rx="{rx}" fill="{fill}" {st}/>')


def _shadow(svg, x, y, w, h, rx=CR, opacity=0.18):
    # soft drop-shadow via feDropShadow filter (defined once in defs)
    svg.e(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
          f'rx="{rx}" fill="#000000" opacity="{opacity}" '
          f'transform="translate(3,4)"/>')


def _text(svg, x, y, s, sz=13, fill=C["dt"], anchor="middle",
          bold=False, italic=False, opacity=1.0):
    fw = "bold" if bold else "normal"
    fs = "italic" if italic else "normal"
    op = f'opacity="{opacity}"' if opacity != 1.0 else ""
    svg.e(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{sz}" fill="{fill}" '
          f'text-anchor="{anchor}" font-weight="{fw}" font-style="{fs}" {op}>{s}</text>')


def _line(svg, x1, y1, x2, y2, color=C["line"], w=1.5,
          dashed=False, marker=None):
    dash = 'stroke-dasharray="7,4"' if dashed else ""
    me   = f'marker-end="url(#{marker})"' if marker else ""
    svg.e(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
          f'stroke="{color}" stroke-width="{w}" {dash} {me}/>')


def _path(svg, d, color=C["line"], w=1.5, dashed=False, marker=None, fill="none"):
    dash = 'stroke-dasharray="7,4"' if dashed else ""
    me   = f'marker-end="url(#{marker})"' if marker else ""
    svg.e(f'<path d="{d}" stroke="{color}" stroke-width="{w}" '
          f'fill="{fill}" {dash} {me}/>')


def _circle(svg, cx, cy, r, fill=C["wt"], stroke=C["line"], sw=1.5):
    svg.e(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" '
          f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def _blend(h1, h2, t):
    """Blend two hex colours."""
    def p(h): h = h.lstrip("#"); return [int(h[i:i+2],16) for i in (0,2,4)]
    r1,g1,b1 = p(h1); r2,g2,b2 = p(h2)
    return "#{:02X}{:02X}{:02X}".format(
        int(r1+(r2-r1)*t), int(g1+(g2-g1)*t), int(b1+(b2-b1)*t))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Composite components
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def stacked_block(svg, cx, cy, hw, hh, title, subtitle, fill):
    """Encoder/decoder: drop-shadow + bold label + italic subtitle."""
    # shadow layers (back to front)
    for i in (3, 2, 1):
        f = _blend(fill, "#FFFFFF", 0.10 * i)
        _rect(svg, cx - hw + i*5, cy - hh - i*4, hw*2, hh*2,
              fill=f, stroke=C["lgray"], sw=0.7, rx=CR)
    # front face
    _rect(svg, cx - hw, cy - hh, hw*2, hh*2,
          fill=fill, stroke="#90A4AE", sw=1.4, rx=CR)
    _text(svg, cx, cy - 10, title,  sz=14, fill=C["wt"], bold=True)
    _text(svg, cx, cy + 10, subtitle, sz=9.5, fill=C["wt"], italic=True)


def adapter_box(svg, cx, cy, hw, hh, name, sub, fill):
    _shadow(svg, cx-hw, cy-hh, hw*2, hh*2, rx=6, opacity=0.14)
    _rect(svg, cx-hw, cy-hh, hw*2, hh*2, fill=fill, stroke="#90A4AE", sw=1.2, rx=6)
    _text(svg, cx, cy - 7,  name, sz=11.5, fill=C["wt"], bold=True)
    _text(svg, cx, cy + 10, sub,  sz=8.5,  fill=C["wt"], italic=True)


def sigma_node(svg, cx, cy, r=SIGMA_R):
    """Σ drawn as circle + embedded path tracing the Sigma letterform."""
    _circle(svg, cx, cy, r, fill=C["wt"], stroke=C["dt"], sw=2.0)
    # Σ as a path: top-right → top-left → centre-right → bottom-left → bottom-right
    s = r * 0.52
    svg.e(
        f'<polyline points="'
        f'{cx+s:.1f},{cy-s:.1f} '
        f'{cx-s:.1f},{cy-s:.1f} '
        f'{cx+s*0.2:.1f},{cy:.1f} '
        f'{cx-s:.1f},{cy+s:.1f} '
        f'{cx+s:.1f},{cy+s:.1f}" '
        f'fill="none" stroke="{C["dt"]}" stroke-width="1.9" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
    )


def mul_node(svg, cx, cy, r=MUL_R, stroke=C["dt"]):
    """× drawn as a circle with diagonal cross-bars."""
    _circle(svg, cx, cy, r, fill=C["wt"], stroke=stroke, sw=1.6)
    inset = r * 0.45
    svg.e(
        f'<line x1="{cx-inset:.1f}" y1="{cy-inset:.1f}" '
        f'x2="{cx+inset:.1f}" y2="{cy+inset:.1f}" '
        f'stroke="{stroke}" stroke-width="1.5"/>'
    )
    svg.e(
        f'<line x1="{cx-inset:.1f}" y1="{cy+inset:.1f}" '
        f'x2="{cx+inset:.1f}" y2="{cy-inset:.1f}" '
        f'stroke="{stroke}" stroke-width="1.5"/>'
    )


def mlp_schematic(svg, cx, cy, layers, hgap=42, vgap=24, nr=7, color=C["gate"]):
    """
    Compact MLP node diagram.  Draws edges first (behind), then nodes on top.
    layers = list of node counts, e.g. [3, 5, 3]
    """
    xs = [cx + (i - (len(layers)-1)/2)*hgap for i in range(len(layers))]
    ys_per_layer = [
        [cy + (j - (n-1)/2)*vgap for j in range(n)]
        for n, _ in zip(layers, xs)
    ]
    # edges
    for li in range(len(layers)-1):
        for y0 in ys_per_layer[li]:
            for y1 in ys_per_layer[li+1]:
                svg.e(
                    f'<line x1="{xs[li]:.1f}" y1="{y0:.1f}" '
                    f'x2="{xs[li+1]:.1f}" y2="{y1:.1f}" '
                    f'stroke="{color}" stroke-width="0.7" opacity="0.4"/>'
                )
    # nodes
    for li, (x, ys) in enumerate(zip(xs, ys_per_layer)):
        for y in ys:
            svg.e(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{nr}" '
                f'fill="{color}" stroke="white" stroke-width="1.2" opacity="0.95"/>'
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main diagram
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def draw(out_stem: Path):
    svg = SVG(W, H)

    # ── Arrowhead markers ─────────────────────────────────────────────────────
    for mid, col in [
        ("a_dk",   C["dt"]),
        ("a_gy",   C["gray"]),
        ("a_gate", C["gate"]),
        ("a_l0",   C["l0"]),
        ("a_l1",   C["l1"]),
        ("a_l2",   C["l2"]),
        ("a_io",   C["io"]),
    ]:
        _marker(svg, mid, col)

    # ── Background ────────────────────────────────────────────────────────────
    _rect(svg, 0, 0, W, H, fill=C["bg"])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # (a) Input — mel spectrogram thumbnail
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    mel_x, mel_y = X_IN - 32, Y_SP - 56
    mel_w, mel_h = 64, 48

    # Render a smooth mel spectrogram via matplotlib, embed as base64 PNG
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.random.seed(7)
    data = np.random.rand(20, 40) * 0.4
    # formant-like horizontal bands
    data[4:7,  :] += 0.55
    data[10:13,:] += 0.35
    data[15:17,:] += 0.20
    # add some temporal variation
    t = np.linspace(0, 2 * np.pi, 40)
    data[4:7,  :] *= (0.7 + 0.3 * np.sin(t))
    data[10:13,:] *= (0.6 + 0.4 * np.sin(2*t + 1))
    data = np.clip(data, 0, 1)

    fig_mel, ax_mel = plt.subplots(figsize=(1.5, 1.1), dpi=120)
    ax_mel.imshow(data, aspect="auto", origin="lower", cmap="magma",
                  interpolation="bilinear", vmin=0, vmax=1)
    ax_mel.axis("off")
    fig_mel.patch.set_facecolor("#111827")
    fig_mel.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig_mel.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    pad_inches=0, facecolor="#111827")
    plt.close(fig_mel)
    buf.seek(0)
    mel_b64 = base64.b64encode(buf.read()).decode()

    svg.e(
        f'<image x="{mel_x}" y="{mel_y}" width="{mel_w}" height="{mel_h}" '
        f'href="data:image/png;base64,{mel_b64}" '
        f'preserveAspectRatio="none"/>'
    )
    # thin border over the image
    _rect(svg, mel_x, mel_y, mel_w, mel_h, fill="none",
          stroke=C["lgray"], sw=0.8, rx=3)

    _text(svg, X_IN, mel_y - 12, "Whisper Processor", sz=10, fill=C["io"], bold=True)
    _text(svg, X_IN + 5, mel_y + mel_h + 20, "waveform → input features", sz=7.5, fill=C["gray"], italic=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # (b) Whisper Encoder
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    stacked_block(svg, X_ENC, Y_SP, ENC_HW, ENC_HH,
                  "Whisper Encoder", "(frozen)", C["base"])

    proc_out_x = X_IN + 32
    proc_fork_x = X_ENC - ENC_HW - 34
    whisper_cx = X_DEC + SIGMA_OFFSET
    whisper_left = whisper_cx - DEC_HW
    whisper_bottom = Y_SP + DEC_HH
    shared_input_route_y = Y_A[-1] + ADPT_HH + 52

    _line(svg, proc_out_x, Y_SP, proc_fork_x, Y_SP, color=C["line"], w=1.8)
    _circle(svg, proc_fork_x, Y_SP, 5, fill=C["line"], stroke=C["line"])
    _path(svg, f"M {proc_fork_x:.0f},{Y_SP} L {X_ENC-ENC_HW:.0f},{Y_SP}",
          color=C["line"], w=1.8, marker="a_dk")
    _text(svg, (proc_fork_x + X_ENC-ENC_HW)/2 - 28, Y_SP - 16,
          "input features", sz=8, fill=C["gray"], italic=True)
    _path(
        svg,
        f"M {proc_fork_x:.0f},{Y_SP} "
        f"L {proc_fork_x:.0f},{shared_input_route_y:.0f} "
        f"L {whisper_cx:.0f},{shared_input_route_y:.0f} "
        f"L {whisper_cx:.0f},{whisper_bottom:.0f}",
        color=C["line"], w=1.8, marker="a_dk"
    )
    _text(svg, (proc_fork_x + whisper_cx)/2, shared_input_route_y + 10,
          "input features", sz=8, fill=C["gray"], italic=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # (c) Gate MLP — title ABOVE box, MLP schematic INSIDE, dims BELOW
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    gate_top    = Y_GATE - GATE_HH
    gate_bottom = Y_GATE + GATE_HH
    gate_left   = X_GATE - GATE_HW
    gate_right  = X_GATE + GATE_HW

    # Title ABOVE
    _text(svg, X_GATE, gate_top - 18, "Age Gating Network",
          sz=12, fill=C["gate"], bold=True)
    _text(svg, X_GATE, gate_top - 5, "(Routing Module)",
          sz=8.5, fill=C["gate"], italic=True)

    # Box
    _shadow(svg, gate_left, gate_top, GATE_HW*2, GATE_HH*2, rx=CR, opacity=0.12)
    _rect(svg, gate_left, gate_top, GATE_HW*2, GATE_HH*2,
          fill=C["gate_bg"], stroke=C["gate"], sw=1.8, rx=CR)

    # MLP schematic inside
    mlp_schematic(svg, X_GATE, Y_GATE, layers=[3, 5, 3],
                  hgap=40, vgap=20, nr=7, color=C["gate"])

    # Encoder → Gate: pooled encoder embeddings only
    _path(svg,
          f"M {X_ENC},{Y_SP-ENC_HH} L {X_ENC},{gate_bottom}",
          color=C["gate"], w=1.6, marker="a_gate")
    _text(svg, X_ENC - 28, (Y_SP + Y_GATE)//2 - 6, "pooled",
          sz=8, fill=C["gate"], italic=True, anchor="middle")
    _text(svg, X_ENC - 28, (Y_SP + Y_GATE)//2 + 8, "embeds",
          sz=8, fill=C["gate"], italic=True, anchor="middle")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # (d) LoRA adapter bank (parameter deltas only; no direct activations)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    adpt_left  = X_ADPT - ADPT_HW
    adpt_right = X_ADPT + ADPT_HW

    adapter_defs = [
        (Y_A[0], "LoRA  (age 3–4)",  "q,v-proj · r=16 · &#x3B1;=32", C["l0"],   "a_l0",   "w&#x2081;"),
        (Y_A[1], "LoRA  (age 5–7)",  "q,v-proj · r=16 · &#x3B1;=32", C["l1"],   "a_l1",   "w&#x2082;"),
        (Y_A[2], "LoRA  (age 8–11)", "q,v-proj · r=16 · &#x3B1;=32", C["l2"],   "a_l2",   "w&#x2083;"),
    ]

    for ya, name, sub, col, arr, _ in adapter_defs:
        adapter_box(svg, X_ADPT, ya, ADPT_HW, ADPT_HH, name, sub, col)

    ens_pad = 18
    ens_y = Y_A[0] - ADPT_HH - ens_pad
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # (e) Weight bus: gate right → horizontal at Y_GATE → vertical over LoRAs
    #     Dashed blue = routing/control weights
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    weight_branch_ys = [ya + ADPT_HH + 18 for ya in Y_A]

    # Horizontal from gate right to X_BUS at Y_GATE (above all adapters)
    _line(svg, gate_right, Y_GATE, X_BUS, Y_GATE,
          color=C["gate"], w=1.5, dashed=True)
    # Vertical bus down to the lowest branch point
    _line(svg, X_BUS, Y_GATE, X_BUS, weight_branch_ys[-1],
          color=C["gate"], w=1.5, dashed=True)

    # "routing weights" label
    _text(svg, gate_right + 12, Y_GATE - 10,
          "weights", sz=8, fill=C["gate"], italic=True, anchor="start")
    _text(svg, gate_right + 12, Y_GATE - 22,
          "routing", sz=8, fill=C["gate"], italic=True, anchor="start")
    _text(svg, gate_right + 12, Y_GATE + 14,
          "(temperature scaled)", sz=7.5, fill=C["gray"], italic=True, anchor="start")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # (g) Adapter bank label (above ensemble box, outside)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    _text(svg, X_ADPT, ens_y - 18, "LoRA Adapter Bank",
          sz=10.5, fill=C["dt"], bold=True)
    _text(svg, X_ADPT, ens_y - 1,
          "(stored parameter deltas)", sz=8, fill=C["gray"], italic=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # (h) Dashed blue = weights into ×, solid coloured = parameter deltas
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    sigma_left = X_SIG + SIGMA_OFFSET - SIGMA_R

    for i in range(len(adapter_defs)):
        ya = Y_A[i]
        weight_branch_y = weight_branch_ys[i]
        _path(
            svg,
            f"M {X_BUS:.0f},{weight_branch_y:.0f} "
            f"L {X_MUL:.0f},{weight_branch_y:.0f} "
            f"L {X_MUL:.0f},{(ya + MUL_R):.0f}",
            color=C["gate"],
            w=1.2,
            dashed=True,
            marker="a_gate",
        )

    for i, (ya, _, _, col, arr, wl) in enumerate(adapter_defs):
        _path(
            svg,
            f"M {adpt_right},{ya:.0f} L {X_MUL - MUL_R:.0f},{ya:.0f}",
            color=col,
            w=1.5,
            marker=arr,
        )
        mul_node(svg, X_MUL, ya, r=MUL_R, stroke=col)
        if ya < Y_SP:
            _path(
                svg,
                f"M {X_MUL + MUL_R:.0f},{ya:.0f} "
                f"L {sigma_left - 18:.0f},{ya:.0f} "
                f"L {sigma_left - 18:.0f},{(Y_SP - SIGMA_R):.0f} "
                f"L {sigma_left:.0f},{(Y_SP - SIGMA_R):.0f}",
                color=col,
                w=1.5,
                marker=arr,
            )
        elif ya > Y_SP:
            _path(
                svg,
                f"M {X_MUL + MUL_R:.0f},{ya:.0f} "
                f"L {sigma_left - 18:.0f},{ya:.0f} "
                f"L {sigma_left - 18:.0f},{(Y_SP + SIGMA_R):.0f} "
                f"L {sigma_left:.0f},{(Y_SP + SIGMA_R):.0f}",
                color=col,
                w=1.5,
                marker=arr,
            )
        else:
            _path(
                svg,
                f"M {X_MUL + MUL_R:.0f},{ya:.0f} L {sigma_left:.0f},{Y_SP:.0f}",
                color=col,
                w=1.5,
                marker=arr,
            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # (i) Equation — above ensemble box, centred on X_SIG
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    eq_y = ens_y - 36
    _text(svg, X_SIG, eq_y,
          "&#x0394;&#x03B8; = \u03A3(i)  w(i) \u00B7 &#x0394;&#x03B8;(i)",
          sz=12, fill=C["dt"])
    _text(svg, X_SIG, eq_y + 14,
          "weighted adapter merge", sz=8, fill=C["gray"], italic=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # (j) Σ node — vertical bus then node touching decoder
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    sigma_node(svg, X_SIG + SIGMA_OFFSET, Y_SP)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # (k) Whisper Decoder — Σ right edge = Decoder left edge (touching)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    stacked_block(svg, X_DEC + SIGMA_OFFSET, Y_SP, DEC_HW, DEC_HH,
                  "Whisper", "(frozen)", C["base"])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # (l) Output
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    _path(svg, f"M {X_DEC + DEC_HW + SIGMA_OFFSET},{Y_SP} L {X_OUT + SIGMA_OFFSET - OUT_HW},{Y_SP}",
          color=C["io"], w=1.8, marker="a_io")

    _shadow(svg, X_OUT-OUT_HW + SIGMA_OFFSET, Y_SP-OUT_HH, OUT_HW*2, OUT_HH*2, rx=6, opacity=0.12)
    _rect(svg, X_OUT-OUT_HW + SIGMA_OFFSET, Y_SP-OUT_HH, OUT_HW*2, OUT_HH*2,
          fill=C["io"], stroke="#A5D6A7", sw=1.4, rx=6)
    _text(svg, X_OUT + SIGMA_OFFSET, Y_SP - 8,  "Transcript",     sz=12, fill=C["wt"], bold=True)
    _text(svg, X_OUT + SIGMA_OFFSET, Y_SP + 9, "token sequence",  sz=8,  fill=C["wt"], italic=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Section labels
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    foot_y   = H - 42
    line_x0  = 0
    line_x1  = W
    _line(svg, line_x0, foot_y - 28, line_x1, foot_y - 28, color=C["lgray"], w=0.6)
    for label, x in [
        ("(a) Audio preprocessing",  X_IN + 42),
        ("(b) Gating Network",     X_ENC),
        ("(c) Adapter bank",         X_ADPT),
        ("(d) LoRA blending",        X_SIG + SIGMA_OFFSET),
        ("(e) Transcription",        X_DEC + SIGMA_OFFSET),
    ]:
        _text(svg, x, foot_y - 12, label, sz=8, fill=C["gray"], italic=True)

    fn_cx = (line_x0 + line_x1) / 2

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Legend
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    legend_items = [
        (C["base"],  "Whisper base (frozen)"),
        (C["gate"],  "Age gating network"),
        (C["l0"],    "LoRA — age 3–4"),
        (C["l1"],    "LoRA — age 5–7"),
        (C["l2"],    "LoRA — age 8–11"),
        (C["io"],    "Input / output"),
    ]

    leg_w  = 230
    leg_rx = X_OUT + OUT_HW          # right edge aligned with output box
    lx     = leg_rx - leg_w
    ly     = 14
    _rect(svg, lx - 8, ly - 12, leg_w + 8, len(legend_items)*19 + 20,
          fill="#FAFAFA", stroke=C["lgray"], sw=0.8, rx=6)
    _text(svg, lx + (leg_w // 2), ly + 2, "Legend", sz=9.5, fill=C["dt"], bold=True)
    for i, (col, label) in enumerate(legend_items):
        yi = ly + 18 + i * 19
        _rect(svg, lx, yi - 8, 12, 11, fill=col, stroke=None, rx=2)
        _text(svg, lx + 18, yi, label, sz=9, fill=C["dt"], anchor="start")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Save
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    svg_path = out_stem.with_suffix(".svg")
    svg_path.write_text(svg.render(), encoding="utf-8")
    print(f"Saved: {svg_path}")

    # Optional PDF via cairosvg
    try:
        import cairosvg
        pdf_path = out_stem.with_suffix(".pdf")
        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
        print(f"Saved: {pdf_path}")
    except ImportError:
        print("cairosvg not found — SVG only.  For PDF: pip install cairosvg")
        print("  or:  inkscape architecture.svg --export-filename=architecture.pdf")


def main():
    parser = argparse.ArgumentParser(description="Draw MoLE architecture diagram (SVG).")
    parser.add_argument("--out", type=Path, default=Path("architecture"),
                        help="Output file stem. Default: architecture  →  architecture.svg")
    args = parser.parse_args()
    draw(args.out.with_suffix(""))


if __name__ == "__main__":
    main()
