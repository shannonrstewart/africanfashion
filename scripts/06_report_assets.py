"""Consolidate every metric into one summary + CSV tables, and draw the report figures.

Outputs, all under results/report_assets/:
    summary.json            every headline number, with provenance (single source for the narrative report)
    tables/*.csv            the table twin of each figure
    figures/*.png           figures (light theme, 200 dpi, 800 css-px wide)

Reads the committed result files plus data/clean/manifest.csv and data/splits/split_v1.csv.
Re-run after any upstream step:  uv run python scripts/06_report_assets.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.transforms import IdentityTransform  # noqa: E402

from benchmark.taxonomy import Taxonomy  # noqa: E402

RES = ROOT / "results"
OUT = RES / "report_assets"
FIG, TAB = OUT / "figures", OUT / "tables"

# ---- palette: reference instance of the dataviz skill, light mode, values unchanged
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b9b8b0"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]   # steps 100..700
RUNG = {3: "#104281", 2: "#2a78d6", 1: "#86b6ef", 0: GRAY}                # ordinal: steps 650/450/250 + gray
DPI, PX = 200, 2          # figure dpi; device px per css px (figsize inches = css px / 100)
FONT = "DejaVu Sans"
plt.rcParams.update({"font.family": FONT, "text.color": INK, "axes.edgecolor": BASE, "svg.fonttype": "none"})


# ============================================================================ drawing helpers
def rr_path(x0, y0, x1, y1, r, corners):
    """Rounded rectangle in device px. corners = (bottom-left, bottom-right, top-right, top-left) flags."""
    k = 0.5523
    bl, br, tr, tl = (r if c else 0 for c in corners)
    v = [(x0 + bl, y0), (x1 - br, y0),
         (x1 - br + k * br, y0), (x1, y0 + br - k * br), (x1, y0 + br),
         (x1, y1 - tr),
         (x1, y1 - tr + k * tr), (x1 - tr + k * tr, y1), (x1 - tr, y1),
         (x0 + tl, y1),
         (x0 + tl - k * tl, y1), (x0, y1 - tl + k * tl), (x0, y1 - tl),
         (x0, y0 + bl),
         (x0, y0 + bl - k * bl), (x0 + bl - k * bl, y0), (x0 + bl, y0), (x0 + bl, y0)]
    c = [MPath.MOVETO, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
         MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
         MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    return MPath(v, c)


def add_rr(ax, x0, y0, x1, y1, color, corners, r=4 * PX, z=3):
    r = max(0, min(r, abs(x1 - x0) / 2, abs(y1 - y0) / 2))
    ax.add_patch(PathPatch(rr_path(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), r, corners),
                           facecolor=color, edgecolor="none", transform=IdentityTransform(), zorder=z, clip_on=False))


def new_fig(w, h, left, right, top, bottom):
    """Figure of w x h css px; axes inset by the given css-px margins (fixed, no tight_layout)."""
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=DPI, facecolor=SURF)
    ax = fig.add_axes([left / w, bottom / h, (w - left - right) / w, (h - top - bottom) / h], facecolor=SURF)
    for s in ax.spines.values():
        s.set_visible(False)
    fig._wh = (w, h)
    return fig, ax


def heading(fig, title, sub, note=None):
    w, h = fig._wh
    fig.text(16 / w, 1 - 24 / h, title, fontsize=11.5, fontweight="bold", color=INK, va="center")
    fig.text(16 / w, 1 - 44 / h, sub, fontsize=8.5, color=INK2, va="center")
    if note:
        fig.text(16 / w, 10 / h, note, fontsize=7.2, color=MUTED, va="center")


def legend_row(fig, items, y_css, x_css=16):
    """items: [(color, label)] drawn as 10px swatches + secondary-ink text (identity by mark, not by text colour)."""
    w, h = fig._wh
    rend = fig.canvas.get_renderer()
    x = x_css
    for color, label in items:
        add_swatch(fig, x, y_css, color)
        t = fig.text((x + 15) / w, 1 - y_css / h, label, fontsize=8, color=INK2, va="center")
        x += 15 + t.get_window_extent(rend).width / PX + 16


def add_swatch(fig, x_css, y_css, color):
    w, h = fig._wh
    X0, Y0 = x_css * PX, (h - y_css - 5) * PX
    fig.add_artist(PathPatch(rr_path(X0, Y0, X0 + 10 * PX, Y0 + 10 * PX, 2 * PX, (1, 1, 1, 1)),
                             facecolor=color, edgecolor="none", transform=IdentityTransform()))


def style_axis(ax, grid="x"):
    ax.tick_params(length=0, labelsize=8.5, labelcolor=INK2, pad=6)
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_facecolor(SURF)


def luminance(hex_):
    r, g, b = (int(hex_[i:i + 2], 16) / 255 for i in (1, 3, 5))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def label_ink(fill):
    return "#ffffff" if luminance(fill) < 0.30 else INK


def ramp(v):
    """Sequential blue ramp, 0..1 -> steps 100..700 (linear interpolation between steps)."""
    v = float(np.clip(v, 0, 1)) * (len(SEQ) - 1)
    i = min(int(v), len(SEQ) - 2)
    a = np.array([int(SEQ[i][j:j + 2], 16) for j in (1, 3, 5)])
    b = np.array([int(SEQ[i + 1][j:j + 2], 16) for j in (1, 3, 5)])
    c = (a + (b - a) * (v - i)).round().astype(int)
    return "#%02x%02x%02x" % tuple(c)


def stacked_hbars(fig, ax, rows, segs, xmax, label_fmt, pitch=34, thick=18):
    """Horizontal stacked bars. rows: [(label, [v1, v2, ...])]; segs: [(color, name)].
    2px surface gap between segments; 4px rounded outer end, square at the baseline; in-bar value labels
    only where the text fits with padding (otherwise skipped - the table twin carries it)."""
    n = len(rows)
    ax.set_xlim(0, xmax)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n), [r[0] for r in rows])
    style_axis(ax, "x")
    rend = fig.canvas.get_renderer()
    gap = 2 * PX
    for i, (_, vals) in enumerate(rows):
        nz = [k for k, v in enumerate(vals) if v > 0]
        cum = 0
        for k, v in enumerate(vals):
            if v <= 0:
                continue
            (X0, Y), (X1, _) = ax.transData.transform([(cum, i), (cum + v, i)])
            x0 = X0 + (gap / 2 if k != nz[0] else 0)
            x1 = X1 - (gap / 2 if k != nz[-1] else 0)
            h = thick * PX / 2
            add_rr(ax, x0, Y - h, x1, Y + h, segs[k][0], (0, k == nz[-1], k == nz[-1], 0))
            txt = label_fmt(v)
            t = ax.text(0, 0, txt, fontsize=7.8, color=label_ink(segs[k][0]), alpha=0)
            tw = t.get_window_extent(rend).width
            t.remove()
            if x1 - x0 > tw + 12 * PX:
                ax.text(*ax.transData.inverted().transform((x0 + 7 * PX, Y)), txt, fontsize=7.8, va="center",
                        ha="left", color=label_ink(segs[k][0]), zorder=5)
            cum += v
    return ax


def heatmap(fig, ax, M, rows, cols, show, fmt, cell=44, gap_px=2):
    """Sequential heatmap, 2px surface gaps, direct labels where show(i, j) is True."""
    n, m = M.shape
    ax.set_xlim(-0.5, m - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(m), cols, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n), rows)
    ax.tick_params(length=0, labelsize=8.5, labelcolor=INK2, pad=6)
    for i in range(n):
        for j in range(m):
            (X0, Y0), (X1, Y1) = ax.transData.transform([(j - 0.5, i + 0.5), (j + 0.5, i - 0.5)])
            g = gap_px * PX / 2
            color = ramp(M[i, j])
            add_rr(ax, X0 + g, Y0 + g, X1 - g, Y1 - g, color, (1, 1, 1, 1), r=3 * PX, z=2)
            if show(i, j):
                ax.text(j, i, fmt(M[i, j]), ha="center", va="center", fontsize=7.8, color=label_ink(color), zorder=5)


def scale_legend(fig, x_css, y_css, w_css, label_lo, label_hi, caption):
    W, H = fig._wh
    n = 40
    for k in range(n):
        X0 = (x_css + k * w_css / n) * PX
        Y0 = (H - y_css - 8) * PX
        fig.add_artist(PathPatch(rr_path(X0, Y0, X0 + w_css / n * PX + 0.6, Y0 + 8 * PX, 0, (0, 0, 0, 0)),
                                 facecolor=ramp(k / (n - 1)), edgecolor="none", transform=IdentityTransform()))
    fig.text(x_css / W, 1 - (y_css + 17) / H, label_lo, fontsize=7.5, color=MUTED, va="center", ha="left")
    fig.text((x_css + w_css) / W, 1 - (y_css + 17) / H, label_hi, fontsize=7.5, color=MUTED, va="center", ha="right")
    fig.text((x_css - 8) / W, 1 - (y_css + 4) / H, caption, fontsize=8, color=INK2, va="center", ha="right")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=DPI, facecolor=SURF)
    plt.close(fig)
    print(f"  figures/{name}.png")


# ============================================================================ data
def load():
    tax = Taxonomy.load()
    man = pd.read_csv(ROOT / "data" / "clean" / "manifest.csv")
    spl = pd.read_csv(ROOT / "data" / "splits" / "split_v1.csv")
    split_meta = json.loads((ROOT / "data" / "splits" / "split_v1.json").read_text())
    probe = json.loads((RES / "dinov2_probe" / "metrics.json").read_text())
    ladder = json.loads((RES / "dinov2_probe" / "ladder_by_subset.json").read_text())
    dedup_hdr = (RES / "dedup_report.md").read_text().splitlines()[0].lstrip("# ").strip()
    return tax, man, spl, split_meta, probe, ladder, dedup_hdr


def build_summary(tax, man, spl, split_meta, probe, ladder, dedup_hdr):
    reason = man.reason.map(lambda r: {"duplicate_near": "removed_duplicate", "duplicate_exact": "removed_duplicate",
                                       "label_conflict": "held_conflict", "kept": "kept", "broken": "broken"}[r])
    by_class = (pd.crosstab(man.clean_label, reason)
                .reindex(columns=["kept", "removed_duplicate", "held_conflict", "broken"], fill_value=0))
    by_class.insert(0, "before_cleaning", by_class.sum(axis=1))
    by_class = by_class.sort_values("before_cleaning", ascending=False)
    kept = man[man.keep]
    short = kept[["w", "h"]].min(axis=1)

    conflict = man[man.cluster_n_labels > 1]
    combos = conflict.groupby("cluster").clean_label.agg(lambda s: " vs ".join(sorted(set(s)))).value_counts()
    split_tab = pd.crosstab(spl.clean_label, spl.split).reindex(columns=["train", "val", "test"])
    split_tab["total"] = split_tab.sum(axis=1)
    sizes = spl.groupby("group").size()

    ov = ladder["test (all)"]["overall"]
    pc = ladder["test (all)"]["per_class"]
    summary = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "source": "inuwamobarak/african-atire (Hugging Face), 1 train split",
            "n_images_source": int(len(man)),
            "images_per_source_label": man.source_label.value_counts().to_dict(),
            "label_merges_confirmed_by_expert": tax.source_label_map,
            "classes_after_merge": int(man.clean_label.nunique()),
            "cleaning": {
                "dedup_parameters": dedup_hdr,
                "kept": int(man.keep.sum()),
                "removed_duplicate": int((reason == "removed_duplicate").sum()),
                "held_for_expert_label_conflict": int((reason == "held_conflict").sum()),
                "fraction_removed_or_held": float(1 - man.keep.mean()),
                "label_conflict_clusters": int(conflict.cluster.nunique()),
                "label_conflict_pairs": {k: int(v) for k, v in combos.items()},
                "by_class": by_class.to_dict("index"),
            },
            "resolution_of_kept_images": {
                "median_short_side_px": float(short.median()),
                "min_short_side_px": int(short.min()),
                "under_128px": int((short < 128).sum()),
                "under_128px_pct": float(100 * (short < 128).mean()),
                "decision": "keep all small images (user decision 2026-11-13)",
            },
            "expert_review_sample": {"images": int(man.in_expert_sample.sum()),
                                     "also_kept_after_dedup": int((man.in_expert_sample & man.keep).sum())},
        },
        "split": {
            "name": "split_v1", "seed": split_meta["seed"], "embedding_group_cosine_tau": split_meta["tau"],
            "n_groups": int(len(sizes)), "largest_group": int(sizes.max()),
            "counts": {s: int((spl.split == s).sum()) for s in ("train", "val", "test")},
            "fractions_pct": {s: float(100 * (spl.split == s).mean()) for s in ("train", "val", "test")},
            "by_class": split_tab.to_dict("index"),
            "forced_into_test_expert_images": int(spl.forced_test.sum()),
            "small_images_per_split": {s: int(spl[(spl.split == s) & spl.small].shape[0]) for s in ("train", "val", "test")},
        },
        "dinov2_probe": {
            "model": probe["model"], "features": probe["features"], "C": probe["C"],
            "selection": "feature set and C chosen on validation macro-F1 only; test evaluated once",
            "test": probe["test"], "n_test": probe["n_test"],
            "per_class": probe["per_class"], "by_subset": probe["subsets"],
            "validation_grid": probe["val_grid"],
        },
        "ladder_benchmark_dinov2_test": {
            "note": "closed-set classifier: L1 is 100% by construction and collapse is 0%; gold for "
                    "Tswana-Shweshwe, Xhosa and Zulu is still an unverified draft",
            "taxonomy_status": ladder["test (all)"].get("taxonomy_status"),
            "overall": {k: ov[k] for k in ("n", "acc_l1_africa", "acc_l2_region", "acc_l3_people", "acc_l4_technique",
                                            "rung_dist", "mean_rung", "specific_rate", "wrong_specific_rate",
                                            "monolithic_collapse_rate")},
            "per_class": {c: {k: m[k] for k in ("n", "acc_l1_africa", "acc_l2_region", "acc_l3_people",
                                                 "acc_l4_technique", "rung_dist", "mean_rung")}
                          for c, m in pc.items()},
            "by_subset": {name: {"n": m["overall"]["n"], "specific_rate": m["overall"]["specific_rate"],
                                 "wrong_specific_rate": m["overall"]["wrong_specific_rate"]}
                          for name, m in ladder.items()},
        },
        "files": {
            "probe_report": "results/dinov2_probe/report.md", "probe_metrics": "results/dinov2_probe/metrics.json",
            "ladder_per_item": "results/dinov2_probe/ladder/per_item.csv",
            "test_predictions": "results/dinov2_probe/test_predictions.csv",
            "dedup_report": "results/dedup_report.md", "split_report": "results/split_report.md",
            "benchmark_spec": "benchmark/BENCHMARK.md", "taxonomy": "benchmark/taxonomy.yaml",
        },
    }
    return summary, by_class, kept, split_tab


def write_tables(summary, by_class, split_tab, probe, ladder):
    TAB.mkdir(parents=True, exist_ok=True)
    by_class.to_csv(TAB / "dataset_cleaning_by_class.csv")
    split_tab.to_csv(TAB / "split_by_class.csv")
    pd.DataFrame(probe["per_class"]).T.to_csv(TAB / "probe_per_class.csv", index_label="class")
    pc = ladder["test (all)"]["per_class"]
    rows = []
    for c, m in {**pc, "OVERALL": ladder["test (all)"]["overall"]}.items():
        rows.append({"class": c, "n": m["n"], "L1_africa": m["acc_l1_africa"], "L2_region": m["acc_l2_region"],
                     "L3_people": m["acc_l3_people"], "L4_technique": m["acc_l4_technique"],
                     **{f"rung{k}": m["rung_dist"][str(k)] for k in range(4)}, "mean_rung": m["mean_rung"]})
    pd.DataFrame(rows).to_csv(TAB / "ladder_per_class.csv", index=False)
    pd.DataFrame(probe["subsets"]).T.to_csv(TAB / "probe_by_subset.csv", index_label="subset")
    cm = probe["confusion"]
    pd.DataFrame(cm["counts"], index=cm["classes"], columns=cm["classes"]).to_csv(TAB / "confusion_counts.csv")
    pd.DataFrame(probe["val_grid"]).to_csv(TAB / "validation_grid.csv", index=False)


# ============================================================================ figures
def fig_cleaning(summary, by_class):
    c = summary["dataset"]["cleaning"]
    n = len(by_class)
    fig, ax = new_fig(800, 150 + n * 34, left=140, right=120, top=84, bottom=40)
    heading(fig, f"Cleaning removed or held {100 * c['fraction_removed_or_held']:.0f}% of the images, mostly duplicates",
            f"Images per class after merging Idgo into Igbo and Idama into Idoma; "
            f"{c['kept']:,} of {summary['dataset']['n_images_source']:,} kept",
            "Source: data/clean/manifest.csv. Held = the same photo filed under different classes, pending expert review.")
    legend_row(fig, [(BLUE, "Kept"), (GRAY, "Removed as near-duplicate"), (ORANGE, "Held: conflicting labels")], 64)
    rows = [(k, [v.kept, v.removed_duplicate, v.held_conflict]) for k, v in by_class.iterrows()]
    segs = [(BLUE, "kept"), (GRAY, "dup"), (ORANGE, "held")]
    stacked_hbars(fig, ax, rows, segs, xmax=4000, label_fmt=lambda v: f"{int(v):,}")
    ax.set_xticks(range(0, 4001, 1000), [f"{t:,}" for t in range(0, 4001, 1000)])
    ax.axhline(n - 0.5, color=BASE, linewidth=0.8)
    for i, (_, v) in enumerate(by_class.iterrows()):
        (X, Y), = ax.transData.transform([(v.before_cleaning, i)])
        ax.text(*ax.transData.inverted().transform((X + 10 * PX, Y)), f"{v.kept / v.before_cleaning:.0%} kept",
                fontsize=8, color=INK2, va="center")
    save(fig, "fig1_cleaning_by_class")


def fig_resolution(summary, kept):
    r = summary["dataset"]["resolution_of_kept_images"]
    short = kept[["w", "h"]].min(axis=1).to_numpy()
    edges = 128 * 2.0 ** np.arange(-1, 5.01, 0.25)
    counts, _ = np.histogram(np.clip(short, edges[0] + 1e-6, edges[-1] - 1e-6), bins=edges)
    fig, ax = new_fig(800, 350, left=84, right=24, top=84, bottom=76)
    heading(fig, f"Typical images are small: median short side {r['median_short_side_px']:.0f} px",
            f"Short side of each kept image, log scale. {r['under_128px']} images ({r['under_128px_pct']:.1f}%) are "
            f"under 128 px; they were kept (decision 2026-11-13).",
            "Source: data/clean/manifest.csv, kept images only (n = %s)." % f"{len(short):,}")
    ax.set_xscale("log", base=2)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, counts.max() * 1.12)
    style_axis(ax, "y")
    ax.set_xticks([64, 128, 256, 512, 1024, 2048, 4096], ["64", "128", "256", "512", "1,024", "2,048", "4,096"])
    ax.set_yticks(np.arange(0, counts.max() * 1.1, 500), [f"{t:,}" for t in np.arange(0, counts.max() * 1.1, 500).astype(int)])
    ax.minorticks_off()
    ax.axhline(0, color=BASE, linewidth=0.8, zorder=2)
    gap = 2 * PX
    for k, cnt in enumerate(counts):
        if cnt == 0:
            continue
        (X0, Y0), (X1, Y1) = ax.transData.transform([(edges[k], 0), (edges[k + 1], cnt)])
        add_rr(ax, X0 + gap / 2, Y0, X1 - gap / 2, Y1, BLUE if edges[k + 1] <= 128 else GRAY, (0, 0, 1, 1))
    ax.set_xlabel("Short side of image (px)", fontsize=8.5, color=INK2, labelpad=8)
    ax.set_ylabel("Images", fontsize=8.5, color=INK2, labelpad=8)
    ax.annotate(f"{r['under_128px']} images\nunder 128 px", xy=(98, counts[edges[:-1] < 128].max() + 60),
                xytext=(66, counts.max() * 0.5), fontsize=8.5, color=INK2, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8, shrinkA=4, shrinkB=2))
    save(fig, "fig2_image_resolution")


def fig_f1(summary, probe):
    pc = probe["per_class"]
    order = sorted(pc, key=lambda c: -pc[c]["f1"])
    best, worst = order[0], order[-1]
    t = probe["test"]
    n = len(order)
    fig, ax = new_fig(800, 140 + n * 30, left=215, right=60, top=70, bottom=44)
    heading(fig, f"Per-class F1: {best} highest ({pc[best]['f1']:.2f}), {worst} lowest ({pc[worst]['f1']:.2f})",
            f"DINOv2 linear probe on {probe['n_test']:,} held-out images; accuracy {t['accuracy']:.1%} "
            f"(95% CI {t['accuracy_ci95'][0]:.1%}-{t['accuracy_ci95'][1]:.1%}), macro-F1 {t['macro_f1']:.3f}",
            "Source: results/dinov2_probe/metrics.json. n = test images per class.")
    ax.set_xlim(0, 1)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n), [f"{c}  (n={pc[c]['support']})" for c in order])
    style_axis(ax, "x")
    ax.set_xticks(np.arange(0, 1.01, 0.25), ["0", "0.25", "0.50", "0.75", "1.00"])
    for i, c in enumerate(order):
        (X0, Y), (X1, _) = ax.transData.transform([(0, i), (pc[c]["f1"], i)])
        add_rr(ax, X0, Y - 9 * PX, X1, Y + 9 * PX, BLUE, (0, 1, 1, 0))
        ax.text(*ax.transData.inverted().transform((X1 + 7 * PX, Y)), f"{pc[c]['f1']:.2f}", fontsize=8.5,
                color=INK2, va="center")
    save(fig, "fig3_f1_by_class")


def fig_confusion(probe):
    cm = np.array(probe["confusion"]["counts"], dtype=float)
    cls = probe["confusion"]["classes"]
    M = cm / cm.sum(1, keepdims=True)
    off = M.copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    n = len(cls)
    fig, ax = new_fig(800, 70 + n * 46 + 140, left=185, right=160, top=70, bottom=140)
    heading(fig, f"Largest confusion: {cls[i]} read as {cls[j]} ({off[i, j]:.0%} of {cls[i]} images)",
            "Share of each true class (rows) assigned to each predicted class (columns), held-out test images. "
            "Diagonal = correct.",
            "Source: results/dinov2_probe/metrics.json (counts in tables/confusion_counts.csv). "
            "Cells under 5% off the diagonal are unlabelled.")
    heatmap(fig, ax, M, cls, cls, show=lambda a, b: a == b or M[a, b] >= 0.05, fmt=lambda v: f"{100 * v:.0f}")
    ax.set_xlabel("Predicted class", fontsize=8.5, color=INK2, labelpad=8)
    ax.set_ylabel("True class", fontsize=8.5, color=INK2, labelpad=8)
    scale_legend(fig, 660, 120, 120, "0%", "100%", "")
    save(fig, "fig4_confusion_matrix")


def fig_ladder_columns(summary):
    L = summary["ladder_benchmark_dinov2_test"]
    names = ["L1 Africa", "L2 Region", "L3 People", "L4 Textile"]
    keys = ["acc_l1_africa", "acc_l2_region", "acc_l3_people", "acc_l4_technique"]
    rows = list(L["per_class"].items()) + [("All test images", L["overall"])]
    M = np.array([[m[k] for k in keys] for _, m in rows], dtype=float)
    n = len(rows)
    o = L["overall"]
    fig, ax = new_fig(800, 108 + n * 44 + 60, left=165, right=150, top=108, bottom=60)
    heading(fig, f"Region is read right {100 * o['acc_l2_region']:.0f}% of the time; people {100 * o['acc_l3_people']:.0f}%, "
                 f"textile {100 * o['acc_l4_technique']:.0f}%",
            "Accuracy at each rung of the granularity ladder, per class, DINOv2 probe on held-out test images",
            "Source: ladder_by_subset.json. L3 and L4 mostly match (a classifier answers both at once). "
            "Draft key: Tswana-Shweshwe, Xhosa, Zulu.")
    heatmap(fig, ax, M, [r[0] for r in rows], names, show=lambda a, b: True, fmt=lambda v: f"{100 * v:.0f}")
    ax.xaxis.tick_top()
    ax.set_xticks(range(4), names, rotation=0, ha="center")
    scale_legend(fig, 660, 130, 120, "0%", "100%", "")
    save(fig, "fig5_ladder_accuracy")


def fig_rungs(summary):
    L = summary["ladder_benchmark_dinov2_test"]
    rows = [(c, [100 * m["rung_dist"][str(k)] for k in (3, 2, 1, 0)]) for c, m in L["per_class"].items()]
    rows.sort(key=lambda r: -r[1][0])
    o = L["overall"]
    rows.append(("All test images", [100 * o["rung_dist"][str(k)] for k in (3, 2, 1, 0)]))
    n = len(rows)
    fig, ax = new_fig(800, 168 + n * 34, left=150, right=30, top=104, bottom=40)
    heading(fig, f"{100 * o['specific_rate']:.0f}% of answers reach the top rung; "
                 f"{100 * o['rung_dist']['2']:.0f}% stop at region",
            "Deepest correct rung per test image (DINOv2 probe). Rung 1 here means a wrong people or region, "
            "never \"Africa\" alone.",
            "Source: results/dinov2_probe/ladder_by_subset.json. Rung 0 is empty for a closed-set classifier. "
            "Labels shown where they fit.")
    legend_row(fig, [(RUNG[3], "3  People or textile correct"), (RUNG[2], "2  Region correct only")], 64)
    legend_row(fig, [(RUNG[1], "1  Africa only correct (here: a wrong people/region)"),
                     (RUNG[0], "0  Not recognised as African")], 84)
    segs = [(RUNG[3], "3"), (RUNG[2], "2"), (RUNG[1], "1"), (RUNG[0], "0")]
    stacked_hbars(fig, ax, rows, segs, xmax=100, label_fmt=lambda v: f"{v:.0f}%")
    ax.set_xticks(range(0, 101, 25), [f"{t}%" for t in range(0, 101, 25)])
    ax.axhline(n - 1.5, color=BASE, linewidth=0.8)
    save(fig, "fig6_rung_distribution")


def main() -> None:
    tax, man, spl, split_meta, probe, ladder, dedup_hdr = load()
    OUT.mkdir(parents=True, exist_ok=True)
    summary, by_class, kept, split_tab = build_summary(tax, man, spl, split_meta, probe, ladder, dedup_hdr)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    write_tables(summary, by_class, split_tab, probe, ladder)
    print("wrote results/report_assets/summary.json and tables/")
    print("figures:")
    fig_cleaning(summary, by_class)
    fig_resolution(summary, kept)
    fig_f1(summary, probe)
    fig_confusion(probe)
    fig_ladder_columns(summary)
    fig_rungs(summary)


if __name__ == "__main__":
    main()
