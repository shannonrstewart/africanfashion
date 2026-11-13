"""Extract frozen DINOv2 embeddings, group embedding-level near-duplicates, and build the
grouped, stratified train/val/test split.

1. Embeds every kept image (data/clean/manifest.csv, keep == True) with frozen DINOv2:
   feature = [CLS token ; mean of patch tokens]. Images are resized (squashed) to 224x224 so no
   part of the pattern or garment is cropped away. Cached in data/embeddings/.
2. Finds near-duplicate pairs by CLS cosine similarity (catches crops/re-edits that the pHash pass
   cannot). These images are GROUPED so they stay in the same split; nothing is deleted.
3. Splits ~70/15/15 stratified by class, grouped by (pHash cluster + embedding group).
   Images that belong to a cluster containing an expert-review image are forced into the test set.

Usage:
    uv run python scripts/04_embeddings_and_splits.py [--tau 0.93] [--sheets] [--reembed]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from torchvision import transforms as T  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModel  # noqa: E402

DATASET_ID = "inuwamobarak/african-atire"
MODEL_ID = "facebook/dinov2-base"
CLEAN_DIR = ROOT / "data" / "clean"
EMB_DIR = ROOT / "data" / "embeddings"
SPLIT_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results"
MIN_SIM = 0.85          # record candidate pairs down to this cosine for calibration
SEED = 42


# ----------------------------------------------------------------------------- embeddings
class ImgDS(torch.utils.data.Dataset):
    def __init__(self, ds, idxs):
        self.ds, self.idxs = ds, idxs
        self.tf = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        return self.tf(self.ds[int(self.idxs[i])]["image"].convert("RGB"))


@torch.no_grad()
def embed(ds, idxs: np.ndarray, batch: int = 64) -> np.ndarray:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL_ID).to(dev).eval()
    dl = torch.utils.data.DataLoader(ImgDS(ds, idxs), batch_size=batch, num_workers=8)
    out = []
    for x in tqdm(dl, desc="DINOv2 embeddings"):
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            h = model(pixel_values=x.to(dev)).last_hidden_state
        out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).float().cpu())
    return torch.cat(out).numpy()


def load_or_embed(ds, idxs, reembed: bool) -> np.ndarray:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    f_emb, f_idx = EMB_DIR / "dinov2-base_sq224.npy", EMB_DIR / "dinov2-base_sq224_idx.npy"
    if f_emb.exists() and f_idx.exists() and not reembed and np.array_equal(np.load(f_idx), idxs):
        return np.load(f_emb)
    E = embed(ds, idxs)
    np.save(f_emb, E)
    np.save(f_idx, idxs)
    return E


# ----------------------------------------------------------------------------- similar pairs
def similar_pairs(E: np.ndarray, idxs: np.ndarray) -> pd.DataFrame:
    cls = torch.nn.functional.normalize(torch.from_numpy(E[:, :768]).cuda(), dim=1)
    rows = []
    for s in range(0, len(cls), 2048):
        sim = cls[s:s + 2048] @ cls.T
        ii, jj = torch.nonzero(sim >= MIN_SIM, as_tuple=True)
        keep = (ii + s) < jj
        ii, jj = ii[keep], jj[keep]
        rows.append(pd.DataFrame({"a": (ii + s).cpu().numpy(), "b": jj.cpu().numpy(),
                                  "sim": sim[ii, jj].cpu().numpy()}))
    p = pd.concat(rows, ignore_index=True)
    p["idx_a"], p["idx_b"] = idxs[p.a], idxs[p.b]
    return p


def calibration_sheets(ds, pairs: pd.DataFrame, labels: dict[int, str]):
    out = RESULTS_DIR / "embedding_dup_calibration"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()
    rng = np.random.default_rng(0)
    bands = [(0.85, 0.88), (0.88, 0.91), (0.91, 0.94), (0.94, 0.97), (0.97, 1.01)]
    size, per_row, rows = 170, 3, 4
    for lo, hi in bands:
        sub = pairs[(pairs.sim >= lo) & (pairs.sim < hi)]
        print(f"  band {lo:.2f}-{hi:.2f}: {len(sub)} pairs")
        if sub.empty:
            continue
        pick = sub.iloc[rng.choice(len(sub), size=min(len(sub), per_row * rows), replace=False)]
        sheet = Image.new("RGB", (per_row * (2 * size + 10), rows * (size + 16)), "#dddddd")
        d = ImageDraw.Draw(sheet)
        for n, r in enumerate(pick.itertuples()):
            x, y = (n % per_row) * (2 * size + 10), (n // per_row) * (size + 16)
            for j, idx in enumerate((r.idx_a, r.idx_b)):
                im = ds[int(idx)]["image"].convert("RGB")
                im.thumbnail((size, size))
                sheet.paste(im, (x + j * size + (size - im.width) // 2, y))
            d.text((x + 2, y + size + 2), f"sim={r.sim:.3f}  {labels[r.idx_a][:8]} | {labels[r.idx_b][:8]}", fill="black")
        sheet.save(out / f"sim_{lo:.2f}-{hi:.2f}.jpg", quality=88)


# ----------------------------------------------------------------------------- grouping + split
def union_find_groups(n: int, edges) -> np.ndarray:
    p = list(range(n))

    def find(x):
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            p[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


def make_split(m: pd.DataFrame, seed: int) -> pd.Series:
    """m has columns: group, clean_label, forced_test. Returns split name per row (aligned to m.index)."""
    split = pd.Series("train", index=m.index)
    split[m.forced_test] = "test"
    pool = m[~m.forced_test.values & ~m.group.isin(m[m.forced_test].group).values]
    # a group containing a forced image is wholly forced
    split[m.group.isin(m[m.forced_test].group)] = "test"
    y = pool.clean_label.to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    rest_i, test_i = next(sgkf.split(pool, y, pool.group))
    split[pool.index[test_i]] = "test"
    rest = pool.iloc[rest_i]
    sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    _, val_i = next(sgkf.split(rest, rest.clean_label, rest.group))
    split[rest.index[val_i]] = "val"
    return split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.91,
                    help="CLS cosine at/above which images are grouped (0.91: largest group 31 images; "
                         "0.85 chains different fabrics with the same design into a group of 111)")
    ap.add_argument("--sheets", action="store_true", help="write calibration sheets of sample pairs per band")
    ap.add_argument("--reembed", action="store_true")
    args = ap.parse_args()

    m = pd.read_csv(CLEAN_DIR / "manifest.csv")
    m = m[m.keep].sort_values("idx").reset_index(drop=True)
    idxs = m.idx.to_numpy()
    print(f"{len(m)} kept images")

    ds = load_dataset(DATASET_ID, split="train")
    E = load_or_embed(ds, idxs, args.reembed)
    print("embeddings:", E.shape)

    pairs = similar_pairs(E, idxs)
    pairs.to_csv(EMB_DIR / "similar_pairs.csv", index=False)
    print("candidate pairs by cosine band:")
    for lo in (0.85, 0.88, 0.91, 0.93, 0.95, 0.97, 0.99):
        print(f"  >= {lo:.2f}: {int((pairs.sim >= lo).sum())}")
    if args.sheets:
        print("calibration sheets:")
        calibration_sheets(ds, pairs, dict(zip(m.idx, m.clean_label)))

    # --- groups: pHash cluster (already 1 kept image per cluster) + embedding similarity
    edges = pairs[pairs.sim >= args.tau][["a", "b"]].to_numpy()
    m["group"] = union_find_groups(len(m), edges)
    sizes = m.groupby("group").size()
    n_mixed = int((m.groupby("group").clean_label.nunique() > 1).sum())
    print(f"tau={args.tau}: {len(sizes)} groups; largest {int(sizes.max())}; "
          f"groups of 2+: {int((sizes > 1).sum())}; mixed-label groups: {n_mixed}")

    m["forced_test"] = m.cluster_has_expert_sample.astype(bool)
    m["split"] = make_split(m, SEED)
    m["small"] = m[["w", "h"]].min(axis=1) < 128

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["idx", "split", "clean_label", "source_label", "w", "h", "small", "group", "in_expert_sample", "forced_test"]
    m[cols].to_csv(SPLIT_DIR / "split_v1.csv", index=False)
    (SPLIT_DIR / "split_v1.json").write_text(json.dumps({
        "version": "v1", "seed": SEED, "tau": args.tau, "model": MODEL_ID,
        **{s: m[m.split == s].idx.tolist() for s in ("train", "val", "test")}}))

    # --- report
    tab = pd.crosstab(m.clean_label, m.split)[["train", "val", "test"]]
    tab["total"] = tab.sum(axis=1)
    frac = (m.split.value_counts(normalize=True) * 100).round(1)
    small = pd.crosstab(m.small, m.split).reindex(columns=["train", "val", "test"])
    lines = ["# Split v1\n", f"CLS cosine grouping threshold tau={args.tau}; seed {SEED}; "
             f"groups: {len(sizes)} (largest {int(sizes.max())}, mixed-label {n_mixed})\n",
             tab.to_markdown(), "",
             f"Fractions (%): train {frac['train']}, val {frac['val']}, test {frac['test']}\n",
             f"Images forced into test because they (or a duplicate) are in the expert-review sample: "
             f"{int(m.forced_test.sum())}\n", "Small (<128 px short side) images per split:\n", small.to_markdown()]
    (RESULTS_DIR / "split_report.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
