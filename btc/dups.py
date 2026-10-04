"""Exact + near-duplicate detection (perceptual dHash) and grouping.

The Kaggle set is stitched together from Figshare, SARTAJ and Br35H, so the same scan
can appear several times (re-crops, re-encodes). If copies of one scan land in both
train and test, test accuracy is inflated. We group them and split by group.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(img: Image.Image) -> int:
    """64-bit difference hash (9x8 grayscale, compare horizontal neighbours)."""
    a = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):  # numpy >= 2.0
        return np.bitwise_count(x)
    return _POP8[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1)


def near_pairs(hashes: np.ndarray, thr: int, block: int = 256):
    """Yield (i, j) index arrays, i < j, whose Hamming distance is <= thr."""
    n = len(hashes)
    for s in range(0, n, block):
        d = _popcount(hashes[s : s + block, None] ^ hashes[None, :])
        ii, jj = np.nonzero(d <= thr)
        ii = ii + s
        keep = ii < jj
        yield ii[keep], jj[keep]


def group_images(hashes: np.ndarray, md5s, thr: int) -> np.ndarray:
    """Union-find over exact (md5) and near (dHash <= thr) duplicates -> group id per image."""
    n = len(hashes)
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first: dict[str, int] = {}
    for i, m in enumerate(md5s):
        if m in first:
            union(i, first[m])
        else:
            first[m] = i
    for ii, jj in near_pairs(hashes, thr):
        for a, b in zip(ii.tolist(), jj.tolist()):
            union(a, b)

    roots = np.array([find(i) for i in range(n)])
    _, gid = np.unique(roots, return_inverse=True)
    return gid


def summarize_groups(gid: np.ndarray) -> dict:
    sizes = np.bincount(gid)
    multi = sizes[sizes > 1]
    return {
        "n_groups": int(len(sizes)),
        "n_multi_groups": int(len(multi)),
        "n_images_in_multi": int(multi.sum()),
        "largest_group": int(sizes.max()),
    }
