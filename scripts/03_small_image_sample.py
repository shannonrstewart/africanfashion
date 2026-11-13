"""Draw a stratified sample of small images (after dedup) for manual inspection.

Nothing is dropped here. The sample mixes images below 128 px (short side) with a few from
the 128-159 px band, shuffled together so the review is not anchored by size, and is written
to labelled contact sheets for the user to review.

Usage:
    uv run python scripts/03_small_image_sample.py [--per-class-small 6] [--per-class-ref 2] [--seed 42]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from datasets import load_dataset  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

DATASET_ID = "inuwamobarak/african-atire"
CLEAN_DIR = ROOT / "data" / "clean"
OUT_DIR = CLEAN_DIR / "small_review"
RESULTS_DIR = ROOT / "results"
SMALL = 128            # the cutoff under discussion
REF_HI = 160           # comparison band: 128 <= short side < 160
BANDS = [(0, 96), (96, 112), (112, 128), (128, 160)]
TILE, COLS, ROWS = 200, 5, 4


def band_of(m: int) -> str:
    for lo, hi in BANDS:
        if lo <= m < hi:
            return f"{lo}-{hi - 1}"
    return f"{BANDS[-1][1]}+"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class-small", type=int, default=6)
    ap.add_argument("--per-class-ref", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    m = pd.read_csv(CLEAN_DIR / "manifest.csv")
    m = m[m.keep].copy()
    m["short"] = m[["w", "h"]].min(axis=1)
    m["band"] = m.short.map(band_of)

    # counts by class x band, over the kept (deduplicated) images
    counts = pd.crosstab(m.clean_label, m.band).reindex(columns=[band_of(lo) for lo, _ in BANDS] + ["160+"], fill_value=0)
    counts["total_kept"] = counts.sum(axis=1)
    counts["under_128"] = counts[[band_of(lo) for lo, _ in BANDS[:3]]].sum(axis=1)
    counts["pct_under_128"] = (100 * counts.under_128 / counts.total_kept).round(1)
    RESULTS_DIR.mkdir(exist_ok=True)
    counts.to_csv(RESULTS_DIR / "small_image_counts.csv")
    print("Kept images by class and short-side band (px):")
    print(counts.to_string())

    rng = np.random.default_rng(args.seed)
    picks = []
    for cls, g in m.groupby("clean_label"):
        small, ref = g[g.short < SMALL], g[(g.short >= SMALL) & (g.short < REF_HI)]
        # within a class, spread the small picks across size bands rather than sampling by volume
        per_band = max(1, args.per_class_small // 3)
        chosen = []
        for lo, hi in BANDS[:3]:
            pool = small[(small.short >= lo) & (small.short < hi)]
            chosen += pool.sample(min(len(pool), per_band), random_state=int(rng.integers(1 << 30))).idx.tolist()
        rest = small[~small.idx.isin(chosen)]
        extra = args.per_class_small - len(chosen)
        if extra > 0 and len(rest):
            chosen += rest.sample(min(len(rest), extra), random_state=int(rng.integers(1 << 30))).idx.tolist()
        chosen += ref.sample(min(len(ref), args.per_class_ref), random_state=int(rng.integers(1 << 30))).idx.tolist()
        picks += chosen
    sample = m[m.idx.isin(picks)].sample(frac=1, random_state=args.seed).reset_index(drop=True)
    sample["sample_id"] = [f"S-{i + 1:03d}" for i in range(len(sample))]

    ds = load_dataset(DATASET_ID, split="train")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in OUT_DIR.glob("*"):
        f.unlink()
    per_sheet = COLS * ROWS
    n_sheets = (len(sample) + per_sheet - 1) // per_sheet
    for s in range(n_sheets):
        chunk = sample.iloc[s * per_sheet:(s + 1) * per_sheet]
        sheet = Image.new("RGB", (COLS * TILE, ROWS * (TILE + 30)), "#e6e6e6")
        d = ImageDraw.Draw(sheet)
        for n, r in enumerate(chunk.itertuples()):
            x, y = (n % COLS) * TILE, (n // COLS) * (TILE + 30)
            im = ds[int(r.idx)]["image"].convert("RGB")
            scale = (TILE - 8) / max(im.size)          # fit the tile; small images are enlarged
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.BICUBIC)
            sheet.paste(im, (x + (TILE - im.width) // 2, y + (TILE - im.height) // 2))
            d.text((x + 4, y + TILE + 2), f"{r.sample_id}  {r.clean_label}", fill="black")
            d.text((x + 4, y + TILE + 14), f"{r.w}x{r.h} px   (idx {r.idx})", fill="#333333")
        sheet.save(OUT_DIR / f"small_review_{s + 1}.png")
    cols = ["sample_id", "idx", "clean_label", "source_label", "w", "h", "band"]
    sample[cols].assign(decision="", notes="").to_csv(OUT_DIR / "sample.csv", index=False)

    print(f"\nSample: {len(sample)} images on {n_sheets} sheets in {OUT_DIR}")
    print(pd.crosstab(sample.clean_label, sample.band).to_string())


if __name__ == "__main__":
    main()
