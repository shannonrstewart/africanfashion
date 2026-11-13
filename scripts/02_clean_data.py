"""Relabel merged classes and flag near-duplicate images (non-destructive).

1. Apply the expert-confirmed label merges (taxonomy.yaml: source_label_map).
2. Hash every image: SHA-256 of the file bytes, plus a 64-bit perceptual hash (pHash)
   under all 8 rotations/flips, since the dataset contains rotated and mirrored copies.
3. Find near-duplicate pairs (Hamming distance of the best-matching orientation),
   write calibration sheets of sample pairs per distance, and cluster at --threshold.
4. Keep the highest-resolution image of each cluster; flag the rest. Clusters whose
   members have different clean labels are flagged for review and not kept.

Hashes are cached in data/clean/, so re-running with another --threshold is fast.

Usage:
    uv run python scripts/02_clean_data.py [--threshold 6] [--rehash]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `benchmark` is importable when run as a script
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from datasets import Image as HFImage  # noqa: E402
from datasets import load_dataset  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy.fft import dctn  # noqa: E402

from benchmark.taxonomy import Taxonomy  # noqa: E402

DATASET_ID = "inuwamobarak/african-atire"
CLEAN_DIR = ROOT / "data" / "clean"
RESULTS_DIR = ROOT / "results"
HASH_FILE = CLEAN_DIR / "hashes.npy"
BASE_FILE = CLEAN_DIR / "base.csv"
THUMB_FILE = CLEAN_DIR / "thumbs.npy"
MAX_CAND = 12          # record all pairs up to this Hamming distance (of 64 bits)
FLAT_STD = 3.0         # near-constant images give meaningless hashes
BUCKETS = [(0, 0), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]


# ----------------------------------------------------------------------------- hashing
def _transforms(a: np.ndarray):
    """The 8 elements of the dihedral group applied to a square array."""
    for k in range(4):
        r = np.rot90(a, k)
        yield r
        yield np.fliplr(r)


def _phash_bits(a: np.ndarray) -> int:
    d = dctn(a, norm="ortho")[:8, :8]
    bits = (d > np.median(d)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hash_one(args):
    idx, data = args
    sha = hashlib.sha256(data).hexdigest()
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            rgb = im.convert("RGB")
            g = rgb.convert("L").resize((32, 32), Image.LANCZOS)
            thumb = np.asarray(rgb.resize((8, 8), Image.BOX), dtype=np.uint8)
        a = np.asarray(g, dtype=np.float64)
        hashes = [_phash_bits(t) for t in _transforms(a)]
        flat = bool(a.std() < FLAT_STD)
    except Exception:
        w = h = -1
        hashes, flat, thumb = [0] * 8, True, np.zeros((8, 8, 3), dtype=np.uint8)
    return idx, sha, len(data), w, h, flat, hashes, thumb


def compute_base(tax: Taxonomy, rehash: bool):
    if HASH_FILE.exists() and BASE_FILE.exists() and THUMB_FILE.exists() and not rehash:
        return pd.read_csv(BASE_FILE), np.load(HASH_FILE), np.load(THUMB_FILE)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATASET_ID, split="train")
    names = ds.features["label"].names
    raw = ds.cast_column("image", HFImage(decode=False))
    labels = raw["label"]
    items = [(i, ex["bytes"]) for i, ex in enumerate(raw["image"])]
    print(f"Hashing {len(items)} images ...")
    with Pool(max(1, (os.cpu_count() or 4) - 2)) as pool:
        out = pool.map(hash_one, items, chunksize=64)
    out.sort(key=lambda r: r[0])
    base = pd.DataFrame({
        "idx": [r[0] for r in out],
        "source_label": [names[labels[r[0]]] for r in out],
        "sha256": [r[1] for r in out],
        "nbytes": [r[2] for r in out],
        "w": [r[3] for r in out], "h": [r[4] for r in out],
        "flat": [r[5] for r in out],
    })
    base["clean_label"] = base.source_label.map(tax.canonical_class)
    H = np.array([r[6] for r in out], dtype=np.uint64)
    thumbs = np.stack([r[7] for r in out])
    base.to_csv(BASE_FILE, index=False)
    np.save(HASH_FILE, H)
    np.save(THUMB_FILE, thumbs)
    return base, H, thumbs


def add_color_distance(pairs: pd.DataFrame, thumbs: np.ndarray) -> pd.DataFrame:
    """Largest colour difference (0-255) over the cells of the 8x8 RGB thumbnails, after applying
    the same rotation/flip that produced the hash match. pHash is grayscale-only, so this
    separates e.g. the same product layout in different colours. (A mean over cells is too weak:
    white backgrounds dilute it.)"""
    ids = np.unique(np.r_[pairs.i, pairs.k])
    lookup = {int(j): list(_transforms(thumbs[j].astype(np.float64))) for j in ids}
    cd = [float(np.abs(lookup[int(k)][t] - thumbs[i].astype(np.float64)).mean(axis=2).max())
          for i, k, t in zip(pairs.i, pairs.k, pairs["transform"])]
    return pairs.assign(cdist=cd)


# ----------------------------------------------------------------------------- pairs
def find_pairs(H: np.ndarray, flat: np.ndarray):
    """All pairs (i<k) whose best-orientation Hamming distance is <= MAX_CAND."""
    n = len(H)
    base = H[:, 0]
    rows, cols, dists, tids = [], [], [], []
    block = 256
    for s in range(0, n, block):
        e = min(n, s + block)
        best = np.full((e - s, n), 255, dtype=np.uint8)
        best_t = np.zeros((e - s, n), dtype=np.uint8)
        for t in range(8):
            d = np.bitwise_count(base[s:e, None] ^ H[None, :, t])
            m = d < best
            best[m] = d[m]
            best_t[m] = t
        for r in range(e - s):
            i = s + r
            ks = np.nonzero(best[r, i + 1:] <= MAX_CAND)[0] + i + 1
            for k in ks:
                if flat[i] or flat[k]:
                    continue
                rows.append(i); cols.append(int(k)); dists.append(int(best[r, k])); tids.append(int(best_t[r, k]))
    return pd.DataFrame({"i": rows, "k": cols, "dist": dists, "transform": tids})


# ----------------------------------------------------------------------------- clustering
class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def select_pairs(pairs: pd.DataFrame, t0: int, tx: int, cmax: float) -> pd.DataFrame:
    """Which candidate pairs count as duplicates.
    - same orientation: hash distance 0 always; up to t0 only if the colours also agree (cdist <= cmax)
    - needs a rotation/flip: hash distance <= tx (stricter; 8 orientations multiply chance matches)"""
    same = pairs["transform"] == 0
    ok_same = same & ((pairs.dist == 0) | ((pairs.dist <= t0) & (pairs.cdist <= cmax)))
    return pairs[ok_same | (~same & (pairs.dist <= tx))]


def cluster(base: pd.DataFrame, pairs: pd.DataFrame, t0: int, tx: int, cmax: float) -> pd.DataFrame:
    dsu = DSU(len(base))
    # exact byte-identical files always cluster
    for _, g in base.groupby("sha256"):
        ids = g.index.tolist()
        for x in ids[1:]:
            dsu.union(ids[0], x)
    for r in select_pairs(pairs, t0, tx, cmax).itertuples():
        dsu.union(r.i, r.k)
    out = base.copy()
    out["cluster"] = [dsu.find(i) for i in range(len(base))]
    out["pixels"] = out.w.clip(lower=0) * out.h.clip(lower=0)
    out["cluster_size"] = out.groupby("cluster").idx.transform("size")
    out["cluster_n_labels"] = out.groupby("cluster").clean_label.transform("nunique")
    # representative: most pixels, then biggest file, then lowest idx
    order = out.sort_values(["cluster", "pixels", "nbytes", "idx"], ascending=[True, False, False, True])
    rep = order.groupby("cluster").idx.first()
    out["is_rep"] = out.idx.eq(out.cluster.map(rep))
    out["keep"] = out.is_rep & (out.cluster_n_labels == 1)
    out["reason"] = np.where(
        out.keep, "kept",
        np.where(out.cluster_n_labels > 1, "label_conflict",
                 np.where(out.sha256.duplicated(keep=False) & ~out.is_rep, "duplicate_exact", "duplicate_near")))
    out.loc[out.w < 0, ["keep", "reason"]] = [False, "broken"]
    return out


# ----------------------------------------------------------------------------- sheets
def _tile(im: Image.Image, caption: str, size: int = 190) -> Image.Image:
    t = Image.new("RGB", (size, size + 28), "white")
    im = im.convert("RGB"); im.thumbnail((size, size))
    t.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    ImageDraw.Draw(t).text((3, size + 2), caption, fill="black")
    return t


def calibration_sheets(ds, base, pairs, seed=0):
    out_dir = RESULTS_DIR / "dedup_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.jpg"):
        f.unlink()
    rng = np.random.default_rng(seed)
    size, per_row, rows = 190, 3, 4
    groups = [("same", pairs[pairs["transform"] == 0]), ("flip", pairs[pairs["transform"] != 0])]
    for (kind, lo, hi), sub in (((k, lo, hi), g[(g.dist >= lo) & (g.dist <= hi)])
                                for k, g in groups for lo, hi in BUCKETS):
        if sub.empty:
            continue
        pick = sub.iloc[rng.choice(len(sub), size=min(len(sub), per_row * rows), replace=False)]
        sheet = Image.new("RGB", (per_row * (2 * size + 12), rows * (size + 36)), "#dddddd")
        for n, r in enumerate(pick.itertuples()):
            x, y = (n % per_row) * (2 * size + 12), (n // per_row) * (size + 36)
            for j, idx in enumerate((r.i, r.k)):
                row = base.iloc[idx]
                cap = f"{row.clean_label[:9]} {row.w}x{row.h} d={r.dist} t={r.transform}" if j == 0 else \
                      f"{row.clean_label[:9]} {row.w}x{row.h}"
                sheet.paste(_tile(ds[int(idx)]["image"], cap, size), (x + j * size, y))
        name = f"{kind}_dist_{lo:02d}-{hi:02d}.jpg"
        sheet.save(out_dir / name, quality=88)
        print(f"  {name}: {len(sub)} pairs in bucket")


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=10,
                    help="max Hamming distance (of 64) for a match in the original orientation")
    ap.add_argument("--color-max", type=float, default=15.0,
                    help="max colour difference (0-255, worst thumbnail cell) for a same-orientation "
                         "match with hash distance > 0")
    ap.add_argument("--flip-threshold", type=int, default=4,
                    help="max Hamming distance for a match that needs a rotation/flip (stricter)")
    ap.add_argument("--rehash", action="store_true")
    ap.add_argument("--no-sheets", action="store_true")
    args = ap.parse_args()

    tax = Taxonomy.load()
    base, H, thumbs = compute_base(tax, args.rehash)
    print("Source -> clean label counts:")
    print(pd.crosstab(base.source_label, base.clean_label).to_string())

    pairs_file = CLEAN_DIR / "pairs.csv"
    if pairs_file.exists() and not args.rehash:
        pairs = pd.read_csv(pairs_file)
    else:
        print("Searching for near-duplicate pairs ...")
        pairs = find_pairs(H, base.flat.to_numpy())
        pairs.to_csv(pairs_file, index=False)
    pairs = add_color_distance(pairs, thumbs)
    print(f"{len(pairs)} candidate pairs (distance <= {MAX_CAND}); by distance:")
    print(pairs.dist.value_counts().sort_index().to_string())

    if not args.no_sheets:
        ds = load_dataset(DATASET_ID, split="train")
        print("Calibration sheets:")
        calibration_sheets(ds, base, pairs)

    print("Pairs by (needs flip, distance):",
          pairs.assign(flip=pairs["transform"] != 0).groupby(["flip", "dist"]).size().to_dict())
    c = cluster(base, pairs, args.threshold, args.flip_threshold, args.color_max)
    # expert-sample bookkeeping: those images stay out of training later
    manifest = ROOT / "data" / "expert_review" / "manifest.csv"
    c["in_expert_sample"] = False
    if manifest.exists():
        c.loc[c.idx.isin(pd.read_csv(manifest).idx), "in_expert_sample"] = True
    c["cluster_has_expert_sample"] = c.groupby("cluster").in_expert_sample.transform("any")
    c["small"] = c[["w", "h"]].min(axis=1) < 128
    c.to_csv(CLEAN_DIR / "manifest.csv", index=False)

    # ---- report
    before = c.clean_label.value_counts()
    after = c[c.keep].clean_label.value_counts()
    rep = [f"# Dedup report (pHash Hamming threshold {args.threshold}/64 same orientation, "
           f"{args.flip_threshold}/64 rotated/flipped, worst-cell colour diff <= {args.color_max:g})\n",
           "| class | before | kept | removed |", "|---|---:|---:|---:|"]
    for k in before.index:
        rep.append(f"| {k} | {before[k]} | {after.get(k, 0)} | {before[k] - after.get(k, 0)} |")
    rep.append(f"| **total** | {len(c)} | {int(c.keep.sum())} | {len(c) - int(c.keep.sum())} |\n")
    rep.append("Removal reasons: " + ", ".join(f"{k}={v}" for k, v in c.reason.value_counts().items()) + "\n")
    sizes = c[c.is_rep].cluster_size
    rep.append(f"Clusters: {len(sizes)}; singletons {int((sizes == 1).sum())}; "
               f"largest {int(sizes.max())}; clusters of 5+: {int((sizes >= 5).sum())}\n")
    cross = c.groupby("cluster").source_label.nunique()
    merged_dupes = c[c.cluster.map(cross) > 1].groupby("cluster").source_label.agg(lambda s: "+".join(sorted(set(s))))
    rep.append("Clusters spanning more than one *source* label: "
               + ", ".join(f"{k}: {v}" for k, v in Counter(merged_dupes.values).items()) + "\n")
    conflict = c[c.cluster_n_labels > 1]
    rep.append(f"Label-conflict clusters (different clean labels, excluded pending review): "
               f"{conflict.cluster.nunique()} clusters / {len(conflict)} images\n")
    if len(conflict):
        pairs_lbl = conflict.groupby("cluster").clean_label.agg(lambda s: " vs ".join(sorted(set(s))))
        rep.append("  " + ", ".join(f"{k}: {v}" for k, v in Counter(pairs_lbl.values).items()) + "\n")
    rep.append(f"Flat/near-constant images (excluded from hash matching): {int(c.flat.sum())}\n")
    rep.append(f"Small images (<128 px short side) among kept: {int((c.keep & c.small).sum())}\n")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "dedup_report.md").write_text("\n".join(rep))
    print("\n" + "\n".join(rep))
    print(f"Wrote {CLEAN_DIR / 'manifest.csv'} and results/dedup_report.md")


if __name__ == "__main__":
    main()
