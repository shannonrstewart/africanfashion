"""Conceptual figure: community + expert input -> dataset -> benchmark -> (sketched) outputs.

Writes results/report_assets/figures/fig23_feedback_concept.png. The three output panels are
ILLUSTRATIVE SKETCHES (hand-set shapes, no data); the figure says so on its face.

Run:  uv run python scripts/16_feedback_concept_figure.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("ra", ROOT / "scripts" / "06_report_assets.py")
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

SURF, INK, INK2, MUTED, GRID, BASE, GRAY, BLUE = ra.SURF, ra.INK, ra.INK2, ra.MUTED, ra.GRID, ra.BASE, ra.GRAY, ra.BLUE
RUNG, SEQ = ra.RUNG, ra.SEQ
LIGHT_BLUE, MID_BLUE = "#86b6ef", BLUE
W, H = 900, 610

fig = plt.figure(figsize=(W / 100, H / 100), dpi=ra.DPI, facecolor=SURF)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")


def box(x0, y0, x1, y1, fc="white", ec=BASE, lw=1.0, ls="-", r=8, z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def text(x, y, s, size=8, color=INK, weight="normal", ha="left", va="center", rot=0, z=5):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, rotation=rot, zorder=z)


def arrow(p0, p1, ls="-", color=INK2, rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4, shrinkA=0, shrinkB=0))


text(24, 24, "Feedback and expert review feed one dataset, which feeds the benchmark", 12.5, weight="bold")
text(24, 46, "Conceptual flow of the proposed feedback loop. The output panels are illustrative sketches, not results.",
     8.5, INK2)

# ---------------------------------------------------------------- column 1: inputs
box(30, 86, 250, 186, fc="#f1f0ea")
text(44, 106, "Community feedback", 9.5, weight="bold")
text(44, 126, "Form: image upload, proposed", 8, INK2)
text(44, 140, "tradition, cultural notes", 8, INK2)
text(44, 160, "Weight = consistency with", 8, INK2)
text(44, 174, "verified facts × quiz score*", 8, INK2)

box(30, 232, 250, 332, fc="#f1f0ea")
text(44, 252, "Expert review", 9.5, weight="bold")
text(44, 272, "Validates labels and", 8, INK2)
text(44, 286, "textile names in the taxonomy", 8, INK2)
text(44, 306, "Verified label = a vote of 3.0,", 8, INK2)
text(44, 320, "each community vote is ≤ 1.0", 8, INK2)

arrow((140, 188), (140, 230))
text(150, 209, "contested, new data,", 7.5, MUTED)
text(150, 221, "claims → expert queue", 7.5, MUTED)

# ---------------------------------------------------------------- column 2: dataset
box(330, 86, 556, 332, fc="#e9f2fd", ec=SEQ[4], lw=1.2)
text(344, 108, "Dataset", 10.5, weight="bold")
text(344, 128, "Cleaned, deduplicated images", 8, INK2)
text(344, 142, "with grouped train/val/test split", 8, INK2)
text(344, 176, "Per image cluster:", 8, INK2, weight="bold")
text(344, 194, "• weighted vote can relabel", 8, INK2)
text(344, 208, "• new images: candidates only", 8, INK2)
text(344, 222, "   (dedup + rights check first)", 8, INK2)
text(344, 246, "split v1 → split v2", 8.5, weight="bold")
text(344, 261, "every change tagged by source", 8, INK2)
text(344, 276, "(label_source column)", 8, INK2)
text(344, 304, "v1 is kept so scores stay comparable", 7.5, MUTED)

arrow((252, 140), (328, 154))
text(290, 126, "weighted vote", 7, MUTED, ha="center")
arrow((252, 282), (328, 268))
text(290, 296, "verified labels", 7, MUTED, ha="center")

# ---------------------------------------------------------------- column 3: benchmark
box(620, 86, 876, 332, fc="#e9f2fd", ec=SEQ[4], lw=1.2)
text(636, 108, "Benchmark: granularity ladder", 9.5, weight="bold")
text(636, 126, "Same test images, same scorer", 8, INK2)
for i, (lvl, name, shade) in enumerate([("L1", "Africa", RUNG[0]), ("L2", "Region", RUNG[1]),
                                        ("L3", "People", RUNG[2]), ("L4", "Textile / technique", RUNG[3])]):
    y0 = 146 + i * 36
    box(636, y0, 636 + 150 + i * 22, y0 + 28, fc=["#e1e0d9", "#9ec5f4", "#5598e7", "#256abf"][i], ec="none", r=5, z=3)
    text(646, y0 + 14, f"{lvl}  {name}", 8.5, color=[INK, INK, "white", "white"][i], weight="bold")
text(636, 304, "More specific = harder; scored per level", 7.5, MUTED)

arrow((558, 209), (618, 209))
text(588, 198, "gold labels", 7.5, MUTED, ha="center")

# future loop (dashed): outputs -> community
ax.plot([22, 22], [560, 136], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=4)
arrow((22, 136), (29, 136), ls="-", color=MUTED, lw=1.2)
text(11, 340, "future: where models fail guides what to ask the community", 7.5, MUTED, ha="center", rot=90)

# ---------------------------------------------------------------- outputs
arrow((748, 334), (748, 372))
box(30, 378, 876, 586, fc=SURF, ec=BASE, ls=(0, (5, 3)), r=8)
text(44, 396, "Benchmark output", 10.5, weight="bold")
text(212, 396, "ILLUSTRATIVE SKETCHES – hand-set shapes, no data", 7.5, MUTED, weight="bold")

PW, PY0, PY1 = 250, 412, 540
panels = [(46, "Accuracy by ladder level"), (330, "Deepest correct rung"), (614, "Per-class confusion")]
for px, title in panels:
    box(px, PY0, px + PW, PY1, fc="white", ec=GRID, r=5)
    text(px + 10, PY0 + 12, title, 8.5, weight="bold")

# panel 1: grouped bars, collapse vs specific
px = panels[0][0]
for i, (zs, ft) in enumerate([(0.78, 1.0), (0.60, 0.95), (0.14, 0.88), (0.05, 0.86)]):
    y = PY0 + 32 + i * 20
    text(px + 10, y + 7, f"L{i + 1}", 7.5, INK2)
    for j, (v, c) in enumerate([(zs, LIGHT_BLUE), (ft, MID_BLUE)]):
        ax.add_patch(Rectangle((px + 30, y + j * 8), 200 * v, 7, fc=c, ec="none", zorder=3))
text(px + 30, PY1 - 10, "light: off-the-shelf   dark: adapted", 7, MUTED)

# panel 2: stacked rung bars
px = panels[1][0]
for i, (lab, parts) in enumerate([("off-the-shelf", [0.25, 0.4, 0.3, 0.05]), ("adapted", [0.02, 0.05, 0.1, 0.83])]):
    y = PY0 + 44 + i * 34
    text(px + 10, y - 8, lab, 7.5, INK2)
    x = px + 10
    for rung, frac in zip([0, 1, 2, 3], parts):
        ax.add_patch(Rectangle((x, y - 2), 230 * frac, 16, fc=RUNG[rung], ec="white", lw=0.8, zorder=3))
        x += 230 * frac
for k, (rung, lab) in enumerate([(0, "none"), (1, "Africa"), (2, "region"), (3, "people / textile")]):
    xx = px + 10 + [0, 44, 92, 138][k]
    ax.add_patch(Rectangle((xx, PY1 - 22), 8, 8, fc=RUNG[rung], ec="none", zorder=3))
    text(xx + 11, PY1 - 18, lab, 6.5, MUTED)

# panel 3: confusion matrix sketch
px = panels[2][0]
n, cell = 5, 17
x0, y0 = px + 20, PY0 + 34
dark = [(0, 0, 3), (1, 1, 3), (2, 2, 3), (3, 3, 3), (4, 4, 3)]
off = {(0, 1): 0, (1, 0): 0, (2, 3): 1, (3, 2): 1, (1, 4): 0, (4, 1): 0}
for r in range(n):
    for c in range(n):
        fc = "#e9f2fd"
        if r == c:
            fc = SEQ[8]
        elif (r, c) in off and off[(r, c)] == 1:
            fc = SEQ[3]
        ax.add_patch(Rectangle((x0 + c * cell, y0 + r * cell), cell - 1.5, cell - 1.5, fc=fc, ec="none", zorder=3))
text(x0 + n * cell + 12, y0 + 18, "rows: true", 7, MUTED)
text(x0 + n * cell + 12, y0 + 32, "cols: predicted", 7, MUTED)
text(x0 + n * cell + 12, y0 + 56, "dark diagonal =", 7, MUTED)
text(x0 + n * cell + 12, y0 + 68, "correct", 7, MUTED)

# captions
for (px, _), cap in zip(panels, ["Generic models collapse to “Africa”", "How specific is the answer?",
                                 "Which peoples get confused?"]):
    text(px + PW / 2, PY1 + 22, cap, 8, INK2, ha="center")

text(24, 600, "*The quiz block is not on the form yet; weight currently = consistency with expert-verified taxonomy facts. "
     "See benchmark/FEEDBACK.md.", 7, MUTED)

out = ra.FIG / "fig23_feedback_concept.png"
fig.savefig(out, dpi=ra.DPI, facecolor=SURF)
print("wrote", out)
