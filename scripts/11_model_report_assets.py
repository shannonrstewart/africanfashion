"""Report assets for the two Qwen3-VL-4B runs and the three-way model comparison.

Adds to results/report_assets/ (nothing from scripts/06 is modified or overwritten):
    summary_models.json            headline numbers, Wilson CIs, paired comparisons (single source for the text)
    tables/models_*.csv            table twin of each comparison figure; one per-class CSV per model
    figures/fig7..fig15_*.png      comparison figures, per-model figures, and the example-image figures
    tables/examples_selected.csv   exactly which test images appear in the example figures and why

Example images are a SEEDED RANDOM draw within pre-declared outcome groups (not hand-picked); the group
sizes are printed on the figure so a reader can see how typical each example is.

Run (after scripts 05, 07, 10):  uv run python scripts/11_model_report_assets.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))

# reuse the drawing helpers and palette from scripts/06 (its main() is guarded)
_spec = importlib.util.spec_from_file_location("ra", ROOT / "scripts" / "06_report_assets.py")
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.transforms import IdentityTransform  # noqa: E402

from benchmark.taxonomy import Taxonomy  # noqa: E402

RES, OUT, TAB = ra.RES, ra.OUT, ra.TAB
PX, SURF, INK, INK2, MUTED, GRID, BASE, GRAY, BLUE = ra.PX, ra.SURF, ra.INK, ra.INK2, ra.MUTED, ra.GRID, ra.BASE, ra.GRAY, ra.BLUE
SEED = 42

# Categorical slots 1-3 of the reference palette, in fixed order (validated: all checks pass; aqua is below 3:1
# contrast on the surface, so every bar carries a direct value label and a table twin exists).
MODELS = {
    "zs": {"label": "Qwen zero-shot", "long": "Qwen3-VL-4B, zero-shot", "color": "#2a78d6",
           "dir": RES / "qwen3vl_4b_zeroshot" / "structured", "raw": RES / "qwen3vl_4b_zeroshot" / "structured_raw.jsonl"},
    "ft": {"label": "Qwen fine-tuned (LoRA)", "long": "Qwen3-VL-4B, LoRA fine-tuned", "color": "#eb6834",
           "dir": RES / "qwen3vl_4b_lora_eval" / "structured", "raw": RES / "qwen3vl_4b_lora_eval" / "structured_raw.jsonl"},
    "dino": {"label": "DINOv2 linear probe", "long": "DINOv2 linear probe", "color": "#1baf7a",
             "dir": RES / "dinov2_probe" / "ladder", "raw": None},
}
MODELS["gzs"] = {"label": "Gemma zero-shot", "long": "Gemma 4 E4B, zero-shot", "color": "#eda100",
                 "dir": RES / "gemma4_e4b_zeroshot" / "structured", "raw": RES / "gemma4_e4b_zeroshot" / "structured_raw.jsonl"}
MODELS["gft"] = {"label": "Gemma fine-tuned (LoRA)", "long": "Gemma 4 E4B, LoRA fine-tuned", "color": "#e87ba4",
                 "dir": RES / "gemma4_e4b_lora_eval" / "structured", "raw": RES / "gemma4_e4b_lora_eval" / "structured_raw.jsonl"}
# models whose results exist; display order groups zero-shot / fine-tuned / classifier
MODELS = {k: MODELS[k] for k in ("zs", "gzs", "ft", "gft", "dino") if (MODELS[k]["dir"] / "metrics.json").exists()}
KEYS = ["acc_l1_africa", "acc_l2_region", "acc_l3_people", "acc_l4_technique"]
LEVELS = ["L1 Africa", "L2 Region", "L3 People", "L4 Textile"]
RUNG_TXT = {3: "specific", 2: "region only", 1: "Africa only", 0: "no credit"}


# ============================================================================ data
def load_all():
    for m in MODELS.values():
        met = json.loads((m["dir"] / "metrics.json").read_text())
        m["overall"], m["per_class"] = met["overall"], met["per_class"]
        pi = pd.read_csv(m["dir"] / "per_item.csv", dtype={"id": str})
        m["items"] = pi.set_index("id")
    zso = json.loads((RES / "qwen3vl_4b_zeroshot" / "open" / "metrics.json").read_text())
    return zso


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def ok3(m):
    return m["items"].rung.astype(int) == 3


def summarise(zso):
    s = {"n_test": int(MODELS["ft"]["overall"]["n"]), "models": {}, "paired_specific_rung": {}}
    for k, m in MODELS.items():
        o = m["overall"]
        n = o["n"]
        spec = int(round(o["specific_rate"] * n))
        lo, hi = wilson(spec, n)
        s["models"][k] = {
            "label": m["long"], "n": n,
            **{k2: o[k2] for k2 in KEYS},
            "rung_dist": o["rung_dist"], "specific_rate": o["specific_rate"], "specific_rate_ci95": [lo, hi],
            "wrong_specific_rate": o["wrong_specific_rate"], "monolithic_collapse_rate": o["monolithic_collapse_rate"],
            "generic_term_rate": o.get("generic_term_rate"),
            "per_class_specific_rate": {c: v["specific_rate"] for c, v in m["per_class"].items()},
        }
    zo = zso["overall"]
    s["models"]["zs_open_protocol"] = {
        "label": "Qwen3-VL-4B, zero-shot, open-ended prompt", "n": zo["n"],
        **{k2: zo[k2] for k2 in KEYS}, "rung_dist": zo["rung_dist"], "specific_rate": zo["specific_rate"],
        "wrong_specific_rate": zo["wrong_specific_rate"], "monolithic_collapse_rate": zo["monolithic_collapse_rate"],
        "generic_term_rate": zo.get("generic_term_rate")}
    for a, b in [(a, b) for a, b in (("ft", "dino"), ("ft", "zs"), ("dino", "zs"), ("gzs", "zs"), ("gft", "ft"), ("gft", "gzs"))
                  if a in MODELS and b in MODELS]:
        ia = ok3(MODELS[a]).sort_index()
        ib = ok3(MODELS[b]).reindex(ia.index)
        both, only_a, only_b = int((ia & ib).sum()), int((ia & ~ib).sum()), int((~ia & ib).sum())
        s["paired_specific_rung"][f"{a}_vs_{b}"] = {
            "both_right": both, f"only_{a}": only_a, f"only_{b}": only_b, "both_wrong": int((~ia & ~ib).sum()),
            "mcnemar_exact_p": mcnemar_exact(only_a, only_b)}
    return s


def write_tables(zso):
    TAB.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, m in MODELS.items():
        o = m["overall"]
        rows.append({"model": m["long"], "n": o["n"], **{lv: o[key] for lv, key in zip(LEVELS, KEYS)},
                     **{f"rung{r}": o["rung_dist"][str(r)] for r in (3, 2, 1, 0)},
                     "specific_rate": o["specific_rate"], "wrong_specific_rate": o["wrong_specific_rate"],
                     "monolithic_collapse_rate": o["monolithic_collapse_rate"]})
    pd.DataFrame(rows).to_csv(TAB / "models_overall.csv", index=False)
    pc = pd.DataFrame({m["label"]: {c: v["specific_rate"] for c, v in m["per_class"].items()} for m in MODELS.values()})
    pc.index.name = "class"
    pc.to_csv(TAB / "models_specific_rate_by_class.csv")
    files = {"zs": ("qwen_zeroshot_structured_per_class.csv", MODELS["zs"]["per_class"], MODELS["zs"]["overall"]),
             "ft": ("qwen_lora_per_class.csv", MODELS["ft"]["per_class"], MODELS["ft"]["overall"]),
             "zso": ("qwen_zeroshot_open_per_class.csv", zso["per_class"], zso["overall"])}
    if "gzs" in MODELS:
        files["gzs"] = ("gemma_zeroshot_structured_per_class.csv", MODELS["gzs"]["per_class"], MODELS["gzs"]["overall"])
    if "gft" in MODELS:
        files["gft"] = ("gemma_lora_per_class.csv", MODELS["gft"]["per_class"], MODELS["gft"]["overall"])
    for _, (fn, per, ov) in files.items():
        out = []
        for c, m in {**per, "OVERALL": ov}.items():
            out.append({"class": c, "n": m["n"], **{lv: m[key] for lv, key in zip(LEVELS, KEYS)},
                        **{f"rung{r}": m["rung_dist"][str(r)] for r in (3, 2, 1, 0)}, "mean_rung": m["mean_rung"],
                        "wrong_specific_rate": m["wrong_specific_rate"],
                        "monolithic_collapse_rate": m["monolithic_collapse_rate"]})
        pd.DataFrame(out).to_csv(TAB / fn, index=False)


# ============================================================================ chart helpers
def grouped_hbars(fig, ax, rows, series, thick=11, gap=2, fmt=lambda v: f"{v:.0f}%"):
    """rows: [(label, [value per series])] on a 0-100 axis. series: [(color, name)]. Direct value label on every bar."""
    n, S = len(rows), len(series)
    ax.set_xlim(0, 100)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n), [r[0] for r in rows])
    ra.style_axis(ax, "x")
    ax.set_xticks(range(0, 101, 25), [f"{t}%" for t in range(0, 101, 25)])
    for i, (_, vals) in enumerate(rows):
        for k, v in enumerate(vals):
            (X0, Yc), (X1, _) = ax.transData.transform([(0, i), (v, i)])
            Y = Yc - (k - (S - 1) / 2) * (thick + gap) * PX
            if v > 0:
                ra.add_rr(ax, X0, Y - thick * PX / 2, X1, Y + thick * PX / 2, series[k][0], (0, 1, 1, 0), r=3 * PX)
            ax.text(*ax.transData.inverted().transform((X1 + 6 * PX, Y)), fmt(v), fontsize=7.8, color=INK2, va="center")


def rung_rows(m, per_class=True):
    rows = []
    if per_class:
        rows = [(c, [100 * v["rung_dist"][str(r)] for r in (3, 2, 1, 0)]) for c, v in m["per_class"].items()]
        rows.sort(key=lambda r: -r[1][0])
    return rows


# ============================================================================ comparison figures
def title_people():
    v = {k: 100 * m["overall"]["acc_l3_people"] for k, m in MODELS.items()}
    if "gft" in v and "gzs" in v:
        return f"Fine-tuning lifts people accuracy: Qwen {v['zs']:.0f}\u2192{v['ft']:.0f}%, Gemma {v['gzs']:.0f}\u2192{v['gft']:.0f}%"
    if "gzs" in v:
        return f"People accuracy: zero-shot Qwen {v['zs']:.0f}%, Gemma {v['gzs']:.0f}%; tuned Qwen {v['ft']:.0f}%"
    return f"Fine-tuning lifts people-level accuracy from {v['zs']:.0f}% to {v['ft']:.0f}%"


def fig_compare_ladder(S):
    ms = list(MODELS.items())
    rows = [(lv, [100 * m["overall"][key] for _, m in ms]) for lv, key in zip(LEVELS, KEYS)]
    pitch = len(ms) * 13 + 24
    fig, ax = ra.new_fig(800, 150 + 4 * pitch, left=100, right=60, top=104, bottom=40)
    ra.heading(fig, title_people(),
               f"Accuracy at each ladder level on {S['n_test']:,} held-out test images; same images, same scorer for all {len(MODELS)} systems",
               "Sources: results/qwen3vl_4b_*, gemma4_e4b_*, dinov2_probe. "
               "DINOv2 picks from a fixed class list, so Africa is 100% by design.")
    ra.legend_row(fig, [(m["color"], m["label"]) for _, m in ms], 64)
    grouped_hbars(fig, ax, rows, [(m["color"], m["label"]) for _, m in ms])
    ra.save(fig, "fig7_model_ladder_comparison")


def fig_compare_rungs(S):
    order = list(MODELS)
    rows = [(MODELS[k]["label"], [100 * MODELS[k]["overall"]["rung_dist"][str(r)] for r in (3, 2, 1, 0)]) for k in order]
    z0, zr3 = MODELS["zs"]["overall"]["rung_dist"], MODELS["zs"]["overall"]["specific_rate"]
    f = MODELS["ft"]["overall"]
    fig, ax = ra.new_fig(800, 180 + len(order) * 44, left=172, right=30, top=104, bottom=40)
    ra.heading(fig, "Share of images reaching each rung, by system" if "ft" not in MODELS else
               f"Specific answers: {100 * zr3:.0f}% Qwen zero-shot, {100 * f['specific_rate']:.0f}% after fine-tuning",
               "Deepest correct rung per test image. Rung 3 = people or textile named correctly.",
               "Rung 1 (Africa only) includes wrong-specific answers such as the wrong people; "
               "rung 0 = not credited as African or no answer.")
    ra.legend_row(fig, [(ra.RUNG[3], "3  People or textile correct"), (ra.RUNG[2], "2  Region correct only")], 64)
    ra.legend_row(fig, [(ra.RUNG[1], "1  Africa only (vague or wrong specific)"), (ra.RUNG[0], "0  Not credited")], 84)
    segs = [(ra.RUNG[3], "3"), (ra.RUNG[2], "2"), (ra.RUNG[1], "1"), (ra.RUNG[0], "0")]
    ra.stacked_hbars(fig, ax, rows, segs, xmax=100, label_fmt=lambda v: f"{v:.0f}%", pitch=44, thick=22)
    ax.set_xticks(range(0, 101, 25), [f"{t}%" for t in range(0, 101, 25)])
    ra.save(fig, "fig8_model_rung_comparison")


def fig_compare_by_class(S):
    ms = list(MODELS.values())
    classes = list(MODELS["ft"]["per_class"])
    classes.sort(key=lambda c: -MODELS["ft"]["per_class"][c]["specific_rate"])
    rows = [(c, [100 * m["per_class"][c]["specific_rate"] for m in ms]) for c in classes]
    ftv = {c: 100 * MODELS["ft"]["per_class"][c]["specific_rate"] for c in classes}
    zsv = {c: 100 * MODELS["zs"]["per_class"][c]["specific_rate"] for c in classes}
    worst = min(ftv, key=ftv.get)
    best_zs = max(zsv, key=zsv.get)
    improved = all(ftv[c] > zsv[c] for c in classes)
    title = (f"Fine-tuning improves every class; the lowest is {worst} at {ftv[worst]:.0f}%" if improved else
             f"Fine-tuning helps most classes; the lowest is {worst} at {ftv[worst]:.0f}%")
    pitch = len(ms) * 12 + 14
    fig, ax = ra.new_fig(800, 150 + 8 * pitch, left=150, right=60, top=104, bottom=40)
    ra.heading(fig, title,
               f"Share of test images where the people or textile is named correctly (rung 3), per class. "
               f"Zero-shot best: {best_zs} ({zsv[best_zs]:.0f}%).",
               "Source: per-class metrics of each run. Gold for Tswana-Shweshwe, Xhosa and Zulu is still a draft.")
    ra.legend_row(fig, [(m["color"], m["label"]) for m in ms], 64)
    grouped_hbars(fig, ax, rows, [(m["color"], m["label"]) for m in ms], thick=10 if len(ms) < 5 else 8)
    ra.save(fig, "fig9_specific_rate_by_class")


# ============================================================================ per-model figures
def fig_model_ladder(key, fname, title_fmt):
    m = MODELS[key]
    pc = m["per_class"]
    rows = list(pc.items()) + [("All test images", m["overall"])]
    M = np.array([[v[k] for k in KEYS] for _, v in rows], dtype=float)
    n, o = len(rows), m["overall"]
    fig, ax = ra.new_fig(800, 108 + n * 44 + 60, left=165, right=150, top=108, bottom=60)
    ra.heading(fig, title_fmt(o),
               f"Accuracy at each rung of the granularity ladder, per class, {m['long']}, {o['n']:,} held-out test images",
               "Source: " + str(m["dir"].relative_to(ROOT)) + "/metrics.json. Gold for Tswana-Shweshwe, Xhosa and Zulu is a draft.")
    ra.heatmap(fig, ax, M, [r[0] for r in rows], LEVELS, show=lambda a, b: True, fmt=lambda v: f"{100 * v:.0f}")
    ax.xaxis.tick_top()
    ax.set_xticks(range(4), LEVELS, rotation=0, ha="center")
    ra.scale_legend(fig, 660, 130, 120, "0%", "100%", "")
    ra.save(fig, fname)


def fig_model_rungs(key, fname, title_fmt):
    m = MODELS[key]
    rows = rung_rows(m)
    o = m["overall"]
    rows.append(("All test images", [100 * o["rung_dist"][str(r)] for r in (3, 2, 1, 0)]))
    n = len(rows)
    fig, ax = ra.new_fig(800, 168 + n * 34, left=150, right=30, top=104, bottom=40)
    ra.heading(fig, title_fmt(o), f"Deepest correct rung per test image, {m['long']}",
               "Source: " + str(m["dir"].relative_to(ROOT)) + "/metrics.json. Labels shown where they fit.")
    ra.legend_row(fig, [(ra.RUNG[3], "3  People or textile correct"), (ra.RUNG[2], "2  Region correct only")], 64)
    ra.legend_row(fig, [(ra.RUNG[1], "1  Africa only (vague or wrong specific)"), (ra.RUNG[0], "0  Not credited")], 84)
    segs = [(ra.RUNG[3], "3"), (ra.RUNG[2], "2"), (ra.RUNG[1], "1"), (ra.RUNG[0], "0")]
    ra.stacked_hbars(fig, ax, rows, segs, xmax=100, label_fmt=lambda v: f"{v:.0f}%")
    ax.set_xticks(range(0, 101, 25), [f"{t}%" for t in range(0, 101, 25)])
    ax.axhline(n - 1.5, color=BASE, linewidth=0.8)
    ra.save(fig, fname)


def fig_named_peoples(key, fname):
    mdl = MODELS[key]
    it = mdl["items"].copy()
    it["named"] = it.pred_people.where(it.pred_people.notna(), "(no answer)").astype(str)
    it["right"] = it.l3_people.astype(str) == "True"
    g = it.groupby("named").agg(n=("right", "size"), right=("right", "sum"))
    noans = g.loc["(no answer)"] if "(no answer)" in g.index else None
    g = g.drop(index="(no answer)", errors="ignore").sort_values("n", ascending=False)
    top, rest = g.head(9), g.iloc[9:]
    rows = [(k, [int(v.right), int(v.n - v.right)]) for k, v in top.iterrows()]
    if len(rest):
        rows.append((f"{len(rest)} other name" + ("s" if len(rest) != 1 else ""), [int(rest.right.sum()), int((rest.n - rest.right).sum())]))
    if noans is not None:
        rows.append(("(no answer)", [0, int(noans.n)]))
    n = len(rows)
    N = len(it)
    first = top.index[0]
    xmax = int(math.ceil(max(sum(r[1]) for r in rows) / 100) * 100)
    fig, ax = ra.new_fig(800, 168 + n * 30, left=150, right=60, top=104, bottom=40)
    n_none = int(noans.n) if noans is not None else 0
    ra.heading(fig, f"{mdl['label']} names {first} most often ({100 * top.n.iloc[0] / N:.0f}% of images)"
                    + (f"; no people for {100 * n_none / N:.0f}%" if n_none > 0.2 * N else ""),
               f"The people named for {N:,} test images, split by whether the answer was right for that image",
               "Source: " + str(mdl["dir"].relative_to(ROOT)) + "/per_item.csv. A name counts as right only if it matches the class.")
    ra.legend_row(fig, [(BLUE, "Right people for this image"), (GRAY, "Wrong people for this image")], 64)
    ra.stacked_hbars(fig, ax, rows, [(BLUE, "right"), (GRAY, "wrong")], xmax=xmax, label_fmt=lambda v: f"{int(v):,}", pitch=30, thick=16)
    step = 100 if xmax <= 600 else 200
    ax.set_xticks(range(0, xmax + 1, step), [f"{t:,}" for t in range(0, xmax + 1, step)])
    ra.save(fig, fname)
    pd.DataFrame([{"answer": k, "right": v[0], "wrong": v[1]} for k, v in rows]).to_csv(TAB / f"named_peoples_{key}.csv", index=False)


# ============================================================================ example images
def parse_json(text):
    import re
    mt = re.search(r"\{.*\}", text or "", re.S)
    if mt:
        try:
            d = json.loads(mt.group(0))
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            pass
    return {}


def load_raw(path):
    return {json.loads(l)["id"]: parse_json(json.loads(l)["text"]) for l in path.read_text().splitlines() if l.strip()}


def trunc(v, n=30):
    if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip().lower() in ("", "none", "null"):
        return None
    s = str(v)
    return s if len(s) <= n else s[:n - 1] + "…"


class Canvas:
    """Figure addressed in css px (origin top-left) so layout reads like HTML."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w / 100, h / 100), dpi=ra.DPI, facecolor=SURF)

    def text(self, x, y, s, size=8, color=INK, weight="normal", ha="left", va="top"):
        return self.fig.text(x / self.w, 1 - y / self.h, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va)

    def image(self, im, x, y, box):
        ax = self.fig.add_axes([x / self.w, 1 - (y + box) / self.h, box / self.w, box / self.h], facecolor="#efeee9")
        ax.imshow(im, aspect="equal", interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(BASE)
            s.set_linewidth(0.6)

    def chip(self, x, y, letter, ok, w=20, h=15):
        X0, Y0 = x * PX, (self.h - y - h) * PX
        self.fig.add_artist(PathPatch(ra.rr_path(X0, Y0, X0 + w * PX, Y0 + h * PX, 3 * PX, (1, 1, 1, 1)),
                                      facecolor=BLUE if ok else "#e9e8e2", edgecolor="none", transform=IdentityTransform()))
        self.text(x + w / 2, y + h / 2, letter, size=7.5, color="#ffffff" if ok else MUTED, weight="bold", ha="center", va="center")

    def chips(self, x, y, flags):
        for i, (letter, ok) in enumerate(zip("ARPT", flags)):
            self.chip(x + i * 24, y, letter, ok)

    def line(self, x0, y0, x1, y1):
        self.fig.add_artist(plt.Line2D([x0 / self.w, x1 / self.w], [1 - y0 / self.h, 1 - y1 / self.h],
                                       color=GRID, linewidth=0.8, transform=self.fig.transFigure))

    def save(self, name):
        ra.FIG.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(ra.FIG / f"{name}.png", dpi=ra.DPI, facecolor=SURF)
        plt.close(self.fig)
        print(f"  figures/{name}.png")


def flags(row):
    return [str(row[c]) == "True" for c in ("l1_africa", "l2_region", "l3_people", "l4_technique")]


def gold_lines(tax, cls):
    g = tax.class_gold(cls)
    region = {"West": "West Africa", "South": "Southern Africa", "East": "East Africa", "North": "North Africa",
              "Central": "Central Africa"}.get(g["region"], g["region"])
    return region, " / ".join(g["peoples"]), " / ".join(g["techniques"])


def answer_block(cv, x, y, pred, row, compact=False):
    """Region / people / textile as the model answered, plus the four ladder chips and the rung."""
    for i, (lab, key) in enumerate((("Region", "region"), ("People", "people"), ("Textile", "technique"))):
        v = trunc(pred.get(key), 17 if compact else 30)
        cv.text(x, y + i * 15, f"{lab}:", size=7.6, color=MUTED)
        cv.text(x + 44, y + i * 15, v if v else "(no answer)", size=8, color=INK if v else MUTED)
    cv.chips(x, y + 52, flags(row))
    if compact:
        cv.text(x, y + 72, f"rung {int(row.rung)}: {RUNG_TXT[int(row.rung)]}", size=7.2, color=INK2)
    else:
        cv.text(x + 100, y + 55, f"rung {int(row.rung)}: {RUNG_TXT[int(row.rung)]}", size=7.2, color=INK2)


def pick(rng, df, used):
    ids = list(df.index)
    rng.shuffle(ids)
    for i in ids:
        if df.loc[i, "gold_class"] not in used:
            return i
    return ids[0]


def fig_examples_outcomes(ds, tax):
    zs, ft = MODELS["zs"], MODELS["ft"]
    ids = ft["items"].index
    a, b = ok3(zs).reindex(ids), ok3(ft)
    N = len(ids)
    groups = [("Zero-shot missed, fine-tuned got it", ~a & b), ("Both got it", a & b),
              ("Zero-shot got it, fine-tuned missed", a & ~b), ("Both missed", ~a & ~b)]
    raw_zs, raw_ft = load_raw(zs["raw"]), load_raw(ft["raw"])
    rng, used, chosen = random.Random(SEED), set(), []
    for name, mask in groups:
        sub = ft["items"][mask.values]
        if len(sub) == 0:
            continue
        i = pick(rng, sub, used)
        used.add(sub.loc[i, "gold_class"])
        chosen.append((name, int(mask.sum()), i))
    RH = 156
    cv = Canvas(800, 108 + RH * len(chosen) + 22)
    cv.text(16, 14, "The same test image, answered by the model before and after fine-tuning", size=11.5, weight="bold")
    cv.text(16, 34, "One image drawn at random (seed 42) from each outcome group; the share of test images in each group is on the right.",
            size=8.2, color=INK2)
    cv.text(16, 48, "Chips show which ladder levels were right: A Africa, R Region, P People, T Textile.", size=8.2, color=INK2)
    for x, lab in ((144, "Answer key"), (312, "Qwen zero-shot"), (562, "Qwen fine-tuned (LoRA)")):
        cv.text(x, 74, lab, size=8.5, weight="bold", color=INK2)
    rows_out = []
    for r, (name, cnt, i) in enumerate(chosen):
        y0 = 100 + r * RH
        cv.line(16, y0 - 4, 784, y0 - 4)
        cv.text(16, y0 + 2, f"{name}", size=8.6, weight="bold")
        cv.text(784, y0 + 2, f"{cnt:,} of {N:,} test images ({100 * cnt / N:.0f}%)", size=8, color=INK2, ha="right")
        im = ds[int(i)]["image"].convert("RGB")
        cv.image(im, 16, y0 + 22, 118)
        cls = ft["items"].loc[i, "gold_class"]
        reg, peo, tec = gold_lines(tax, cls)
        cv.text(144, y0 + 24, f"Class: {cls}", size=8, weight="bold")
        for k, (lab, val) in enumerate((("Region", reg), ("People", peo), ("Textile", tec))):
            cv.text(144, y0 + 40 + k * 28, f"{lab}:", size=7.6, color=MUTED)
            cv.text(144, y0 + 40 + k * 28 + 11, "\n".join(textwrap.wrap(val, 24)[:2]), size=8)
        answer_block(cv, 312, y0 + 24, raw_zs[i], zs["items"].loc[i])
        answer_block(cv, 562, y0 + 24, raw_ft[i], ft["items"].loc[i])
        rows_out.append({"figure": "fig14_examples_outcomes", "group": name, "group_n": cnt, "dataset_idx": i, "class": cls})
    cv.text(16, cv.h - 14, "Source: qwen3vl_4b_zeroshot and qwen3vl_4b_lora_eval (structured prompt). "
                           "Any value in the draft taxonomy's list counts as right.", size=7.2, color=MUTED, va="center")
    cv.save("fig14_examples_outcomes")
    return rows_out


def fig_examples_systems(ds, tax):
    """The same random test images answered by every Qwen/Gemma system (one image from each of 4 random classes)."""
    keys = [k for k in ("zs", "gzs", "ft", "gft") if k in MODELS]
    raws = {k: load_raw(MODELS[k]["raw"]) for k in keys}
    items = MODELS["zs"]["items"]
    rng = random.Random(SEED + 2)
    classes = sorted(items.gold_class.unique())
    rng.shuffle(classes)
    classes = classes[:4]
    chosen = []
    for c in classes:
        ids = sorted(items.index[items.gold_class == c])
        chosen.append(ids[rng.randrange(len(ids))])
    colw = (784 - 296) / len(keys)
    RH = 150
    cv = Canvas(800, 108 + RH * len(chosen) + 22)
    cv.text(16, 14, "The same test image, answered by each system", size=11.5, weight="bold")
    cv.text(16, 34, "One image drawn at random (seed 42) from each of four randomly chosen classes. No image was picked for its answers.",
            size=8.2, color=INK2)
    cv.text(16, 48, "Chips show which ladder levels were right: A Africa, R Region, P People, T Textile.", size=8.2, color=INK2)
    cv.text(144, 74, "Answer key", size=8.5, weight="bold", color=INK2)
    for j, k in enumerate(keys):
        cv.text(296 + j * colw, 74, MODELS[k]["label"], size=8.5, weight="bold", color=INK2)
    out = []
    for r, i in enumerate(chosen):
        y0 = 100 + r * RH
        cv.line(16, y0 - 4, 784, y0 - 4)
        cls = items.loc[i, "gold_class"]
        cv.image(ds[int(i)]["image"].convert("RGB"), 16, y0 + 6, 118)
        reg, peo, tec = gold_lines(tax, cls)
        cv.text(144, y0 + 8, cls, size=8.4, weight="bold")
        yy = y0 + 26
        for lab, val in (("Region", reg), ("People", peo), ("Textile", tec)):
            cv.text(144, yy, f"{lab}:", size=7.6, color=MUTED)
            for ln in textwrap.wrap(val, 20)[:2]:
                yy += 11
                cv.text(144, yy, ln, size=8)
            yy += 17
        for j, k in enumerate(keys):
            answer_block(cv, 296 + j * colw, y0 + 8, raws[k][i], MODELS[k]["items"].loc[i], compact=True)
        out.append({"figure": "fig22_examples_systems", "group": "random image from random class", "group_n": "",
                    "dataset_idx": i, "class": cls})
    cv.text(16, cv.h - 14, "Sources: results/qwen3vl_4b_*, results/gemma4_e4b_* (structured prompt). "
                           "Any value in the draft taxonomy's list counts as right.", size=7.2, color=MUTED, va="center")
    cv.save("fig22_examples_systems")
    return out


def fig_examples_errors(ds, tax):
    ft = MODELS["ft"]
    raw_ft = load_raw(ft["raw"])
    err = ft["items"][ft["items"].rung.astype(int) != 3]
    rng = random.Random(SEED + 1)
    classes = sorted(err.gold_class.unique())
    rng.shuffle(classes)
    classes = sorted(classes[:6], key=lambda c: list(ft["per_class"]).index(c))
    CW, CH = 256, 158
    cv = Canvas(800, 76 + 2 * CH + 26)
    cv.text(16, 14, "How the fine-tuned model goes wrong", size=11.5, weight="bold")
    cv.text(16, 34, "One error drawn at random (seed 42) from each of six classes.", size=8.2, color=INK2)
    cv.text(16, 48, f"An error means neither people nor textile was right ({len(err)} of {len(ft['items']):,} test images).",
            size=8.2, color=INK2)

    def wrapped(x, y, s, size=7.6, color=INK2, width=27):
        for ln in textwrap.wrap(s, width)[:2]:
            cv.text(x, y, ln, size=size, color=color)
            y += 11
        return y

    out = []
    for n, c in enumerate(classes):
        sub = err[err.gold_class == c]
        i = sub.index[rng.randrange(len(sub))]
        x, y = 16 + (n % 3) * (CW + 8), 76 + (n // 3) * CH
        cv.image(ds[int(i)]["image"].convert("RGB"), x, y + 2, 104)
        reg, peo, tec = gold_lines(tax, c)
        tx = x + 112
        cv.text(tx, y + 2, "Answer key", size=7.4, color=MUTED)
        cv.text(tx, y + 13, f"{c}", size=8.4, weight="bold")
        yy = wrapped(tx, y + 30, f"People: {peo}")
        yy = wrapped(tx, yy, f"Textile: {tec}")
        cv.text(tx, yy + 6, "Model said", size=7.4, color=MUTED)
        p = raw_ft[i]
        cv.text(tx, yy + 17, f"People: {trunc(p.get('people'), 18) or '(none)'}", size=8)
        cv.text(tx, yy + 29, f"Textile: {trunc(p.get('technique'), 18) or '(none)'}", size=8)
        cv.chips(tx, yy + 46, flags(ft["items"].loc[i]))
        out.append({"figure": "fig15_examples_errors", "group": f"fine-tuned error, class {c}", "group_n": len(sub),
                    "dataset_idx": i, "class": c})
    cv.text(16, cv.h - 14, "Source: qwen3vl_4b_lora_eval. Chips: A Africa, R Region, P People, T Textile; filled = right. "
                           "Images resized to at most 768 px, as the model saw them.", size=7.2, color=MUTED, va="center")
    cv.save("fig15_examples_errors")
    return out


# ============================================================================ main
def main() -> None:
    zso = load_all()
    S = summarise(zso)
    (OUT / "summary_models.json").write_text(json.dumps(S, indent=2))
    write_tables(zso)
    print("wrote summary_models.json and tables/models_*.csv, qwen_*_per_class.csv")
    print("figures:")
    fig_compare_ladder(S)
    fig_compare_rungs(S)
    fig_compare_by_class(S)
    fig_model_ladder("zs", "fig10_qwen_zeroshot_ladder",
                     lambda o: f"Zero-shot Qwen: region {100 * o['acc_l2_region']:.0f}%, people {100 * o['acc_l3_people']:.0f}%, "
                               f"textile {100 * o['acc_l4_technique']:.0f}%")
    fig_model_rungs("zs", "fig11_qwen_zeroshot_rungs",
                    lambda o: f"Zero-shot Qwen reaches the top rung on {100 * o['specific_rate']:.0f}% of images; "
                              f"{100 * o['rung_dist']['0']:.0f}% get no credit")
    fig_named_peoples("zs", "fig12_zeroshot_named_peoples")
    fig_model_ladder("ft", "fig13_qwen_lora_ladder",
                     lambda o: f"Fine-tuned Qwen: region {100 * o['acc_l2_region']:.0f}%, people {100 * o['acc_l3_people']:.0f}%, "
                               f"textile {100 * o['acc_l4_technique']:.0f}%")
    fig_model_rungs("ft", "fig13b_qwen_lora_rungs",
                    lambda o: f"Fine-tuned Qwen reaches the top rung on {100 * o['specific_rate']:.0f}% of images; "
                              f"{100 * (1 - o['specific_rate']):.0f}% stop short")
    for k, tag in (("gzs", "16"), ("gft", "19")):
        if k not in MODELS:
            continue
        lab = MODELS[k]["label"]
        fig_model_ladder(k, f"fig{tag}_{k}_ladder",
                         lambda o, lab=lab: f"{lab}: region {100 * o['acc_l2_region']:.0f}%, people {100 * o['acc_l3_people']:.0f}%, "
                                            f"textile {100 * o['acc_l4_technique']:.0f}%")
        fig_model_rungs(k, f"fig{int(tag) + 1}_{k}_rungs",
                        lambda o, lab=lab: f"{lab} reaches the top rung on {100 * o['specific_rate']:.0f}% of images; "
                                           f"{100 * o['monolithic_collapse_rate']:.0f}% collapse to Africa")
        fig_named_peoples(k, f"fig{int(tag) + 2}_{k}_named_peoples")
    from datasets import load_dataset
    ds = load_dataset("inuwamobarak/african-atire", split="train")
    tax = Taxonomy.load()
    sel = fig_examples_outcomes(ds, tax) + fig_examples_errors(ds, tax) + fig_examples_systems(ds, tax)
    pd.DataFrame(sel).to_csv(TAB / "examples_selected.csv", index=False)


if __name__ == "__main__":
    main()
