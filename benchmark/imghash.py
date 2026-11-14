"""Perceptual hashing shared by the dedup pipeline and the feedback loop.

A 64-bit DCT pHash of a 32x32 grayscale copy, computed for all 8 rotations/flips so a
rotated or mirrored copy of an image still matches.
"""

from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image
from scipy.fft import dctn

FLAT_STD = 3.0  # near-constant images give meaningless hashes


def transforms(a: np.ndarray):
    """The 8 elements of the dihedral group applied to a square array."""
    for k in range(4):
        r = np.rot90(a, k)
        yield r
        yield np.fliplr(r)


def phash_bits(a: np.ndarray) -> int:
    d = dctn(a, norm="ortho")[:8, :8]
    bits = (d > np.median(d)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hash_bytes(data: bytes) -> dict:
    """Hash raw image bytes. On failure returns w = h = -1, flat = True."""
    sha = hashlib.sha256(data).hexdigest()
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            g = im.convert("RGB").convert("L").resize((32, 32), Image.LANCZOS)
        a = np.asarray(g, dtype=np.float64)
        hashes = [phash_bits(t) for t in transforms(a)]
        flat = bool(a.std() < FLAT_STD)
    except Exception:
        w = h = -1
        hashes, flat = [0] * 8, True
    return {"sha256": sha, "w": w, "h": h, "hashes": hashes, "flat": flat}


def best_match(query_hash0: int, H: np.ndarray, flat: np.ndarray | None = None):
    """Closest dataset image to a query's base hash over the dataset's 8 orientations.
    H is the (n, 8) uint64 array from data/clean/hashes.npy. Returns (row, distance)."""
    q = np.uint64(query_hash0)
    d = np.bitwise_count(q ^ H).min(axis=1).astype(int)
    if flat is not None:
        d = np.where(flat, 255, d)
    i = int(d.argmin())
    return i, int(d[i])
