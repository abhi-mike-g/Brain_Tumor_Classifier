"""Phase 1, step 2: build THE fixed train/val/test split (one file for all four models).

    python -m btc.split

Pools every image (ignores Kaggle's own Training/Testing folders), drops anything matching
data.exclude_name_regex, groups exact/near duplicates, then assigns whole GROUPS to
train/val/test, stratified by class. Writes splits/split.csv and splits/split_report.txt.
Run it once, commit split.csv, and never regenerate it after anyone has trained.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .common import load_config, rel
from .dups import group_images


def assign_groups(df: pd.DataFrame, ratios: dict, seed: int) -> pd.Series:
    """Greedy, class-stratified assignment of whole groups. Returns a split name per row."""
    rng = np.random.default_rng(seed)
    names = list(ratios)
    gl = df.groupby("group").agg(label=("label", lambda s: s.value_counts().index[0]), n=("label", "size"))
    dest: dict[int, str] = {}
    for _, sub in gl.groupby("label"):  # a group counts toward its majority class
        total = sub.n.sum()
        target = {s: ratios[s] * total for s in names}
        have = {s: 0 for s in names}
        # shuffle, then biggest groups first so small ones can fill the gaps
        order = sub.sample(frac=1, random_state=int(rng.integers(1 << 31))).sort_values("n", ascending=False, kind="stable")
        for g, n in order.n.items():
            s = max(names, key=lambda k: target[k] - have[k])
            dest[g] = s
            have[s] += n
    return df.group.map(dest)


def main() -> None:
    cfg = load_config()
    classes = cfg["classes"]
    inv = pd.read_csv(rel(cfg["data"]["inventory_csv"]), keep_default_na=False)
    n0 = len(inv)
    inv = inv[(inv.error == "") & inv.label.isin(classes)]
    rx = cfg["data"]["exclude_name_regex"]
    excluded = pd.DataFrame()
    if rx:
        mask = inv.path.map(lambda p: bool(re.search(rx, p.split("/")[-1], re.I)))
        excluded = inv[mask]
        inv = inv[~mask]
    inv = inv.reset_index(drop=True)

    hashes = np.array([int(h, 16) for h in inv.dhash], dtype=np.uint64)
    inv["group"] = group_images(hashes, inv.md5.tolist(), cfg["split"]["dup_hamming_threshold"])
    conflicts = int((inv.groupby("group").label.nunique() > 1).sum())

    inv["split"] = assign_groups(inv, cfg["split"]["ratios"], cfg["seed"])
    inv["label_idx"] = inv.label.map({c: i for i, c in enumerate(classes)})

    # ---- hard checks -------------------------------------------------------------------
    assert (inv.split != "").all(), "some images were not assigned"
    span = int((inv.groupby("group").split.nunique() > 1).sum())
    assert span == 0, f"{span} groups span more than one split"
    assert not inv.path.duplicated().any()

    out = inv[["path", "label", "label_idx", "split", "group"]]
    split_path = rel(cfg["data"]["split_csv"])
    out.to_csv(split_path, index=False)

    # ---- report ------------------------------------------------------------------------
    order = list(cfg["split"]["ratios"])
    tab = pd.crosstab(inv.label, inv.split, margins=True, margins_name="total")[order + ["total"]]
    tab = tab.reindex(classes + ["total"])
    sizes = inv.groupby("group").size()
    lines = [
        f"Images in inventory: {n0}; dropped (unreadable / unknown class): {n0 - len(inv) - len(excluded)}; "
        f"excluded by regex {rx!r}: {len(excluded)}; final split: {len(inv)} images -> {len(sizes)} groups",
        "",
        tab.to_string(),
        "",
        "Percent of total: " + ", ".join(f"{s} {100 * (inv.split == s).mean():.1f}%" for s in order),
        f"Groups spanning more than one split: {span}",
        f"Groups with conflicting labels: {conflicts}",
        f"Near-duplicate groups (>1 image): {int((sizes > 1).sum())}, containing "
        f"{int(sizes[sizes > 1].sum())} images ({100 * sizes[sizes > 1].sum() / len(inv):.0f}%)",
        f"Largest group: {int(sizes.max())} images",
        f"seed={cfg['seed']}  hamming_threshold={cfg['split']['dup_hamming_threshold']}",
    ]
    if len(excluded):
        lines += ["", "Excluded by regex, per class: " + str(excluded.label.value_counts().to_dict())]
    text = "\n".join(lines)
    print(text)
    (split_path.parent / "split_report.txt").write_text(text + "\n")
    print(f"\nWrote {split_path}. Commit it; do not regenerate after training starts.")


if __name__ == "__main__":
    main()
