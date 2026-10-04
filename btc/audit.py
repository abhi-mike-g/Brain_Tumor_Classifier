"""Phase 1, step 1: inspect the dataset.

    python -m btc.audit

Walks data/raw, records one row per image (label, size, colour mode, md5, perceptual hash,
file-name flags) into splits/inventory.csv, prints a summary and saves reports/samples.png.
Read the printed WARNINGS before you run btc.split.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd
from PIL import Image

from .common import IMG_EXTS, load_config, rel
from .dups import dhash, group_images, summarize_groups


def scan(raw: "Path", classes: list[str]) -> pd.DataFrame:  # noqa: F821
    rows = []
    files = sorted(p for p in raw.rglob("*") if p.suffix.lower() in IMG_EXTS)
    for k, p in enumerate(files):
        parts = p.relative_to(raw).parts
        row = {
            "path": p.relative_to(raw).as_posix(),
            "label": p.parent.name.lower(),
            "orig_split": parts[0].lower() if len(parts) > 2 else "",
            "name_has_aug": bool(re.search(r"aug", p.name, re.I)),
            "bytes": p.stat().st_size,
            "md5": hashlib.md5(p.read_bytes()).hexdigest(),
        }
        try:
            with Image.open(p) as im:
                im.load()
                row.update(width=im.width, height=im.height, mode=im.mode, dhash=f"{dhash(im):016x}", error="")
        except Exception as e:  # corrupt / unreadable
            row.update(width=0, height=0, mode="", dhash="0" * 16, error=str(e)[:80])
        rows.append(row)
        if (k + 1) % 1000 == 0:
            print(f"  scanned {k + 1}/{len(files)}")
    return pd.DataFrame(rows)


def save_samples(df: pd.DataFrame, raw, classes, out, n=6, seed=0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(classes), n, figsize=(2 * n, 2 * len(classes)))
    for r, c in enumerate(classes):
        sub = df[df.label == c]
        pick = rng.choice(len(sub), size=min(n, len(sub)), replace=False) if len(sub) else []
        for k in range(n):
            ax = axes[r, k]
            ax.axis("off")
            if k < len(pick):
                ax.imshow(Image.open(raw / sub.iloc[pick[k]].path).convert("L"), cmap="gray")
            if k == 0:
                ax.set_title(c, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    classes = cfg["classes"]
    raw = rel(cfg["data"]["raw_dir"])
    if not any(raw.rglob("*.jp*g")) and not any(raw.rglob("*.png")):
        raise SystemExit(f"No images under {raw}. Run `python -m btc.download` first.")

    print(f"Scanning {raw} ...")
    df = scan(raw, classes)
    inv_path = rel(cfg["data"]["inventory_csv"])
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(inv_path, index=False)
    warnings: list[str] = []

    print(f"\n=== {len(df)} images found -> {inv_path.relative_to(rel('.'))}")
    print("\nImages per class x original folder:")
    print(pd.crosstab(df.label, df.orig_split.replace("", "(none)"), margins=True).to_string())

    unknown = sorted(set(df.label) - set(classes))
    if unknown:
        warnings.append(f"Folders not in config classes {classes}: {unknown}. Fix `classes` or the folder names.")
    if df.error.ne("").any():
        warnings.append(f"{int(df.error.ne('').sum())} unreadable images (excluded from the split).")

    ok = df[(df.error == "") & df.label.isin(classes)]
    print("\nColour modes:", ok["mode"].value_counts().to_dict())
    print("Image sizes: %d distinct; most common:" % ok.groupby(["width", "height"]).ngroups)
    print(ok.groupby(["width", "height"]).size().sort_values(ascending=False).head(5).to_string())

    n_aug = int(df.name_has_aug.sum())
    if n_aug:
        by = df[df.name_has_aug].groupby("label").size().to_dict()
        warnings.append(
            f"{n_aug} file names contain 'aug' {by}. These look like synthetic augmented copies. "
            "If so, set data.exclude_name_regex: \"aug\" in configs/common.yaml before splitting "
            "(and tell the team: it changes the dataset size and class balance)."
        )

    dup_md5 = int(ok.md5.duplicated(keep=False).sum())
    print(f"\nByte-identical files (md5): {dup_md5} images share content with another file")

    print("\nNear-duplicate groups by Hamming threshold (dHash):")
    hashes = np.array([int(h, 16) for h in ok.dhash], dtype=np.uint64)
    print(f"{'thr':>4} {'groups':>8} {'groups>1':>9} {'imgs in groups>1':>17} {'largest':>8}")
    for t in (0, 2, 4, 6, 8):
        s = summarize_groups(group_images(hashes, ok.md5.tolist(), t))
        print(f"{t:>4} {s['n_groups']:>8} {s['n_multi_groups']:>9} {s['n_images_in_multi']:>17} {s['largest_group']:>8}")
        if t == cfg["split"]["dup_hamming_threshold"] and s["largest_group"] > 0.01 * len(ok):
            warnings.append(
                f"At the configured threshold ({t}) the largest group has {s['largest_group']} images "
                "(>1% of data). Hash collisions on near-blank images? Lower the threshold."
            )

    save_samples(ok, raw, classes, rel("reports") / "samples.png")
    print("\nSaved reports/samples.png (look at it: skull-stripped? odd crops? mislabeled?)")

    print("\n=== WARNINGS ===" if warnings else "\nNo warnings.")
    for w in warnings:
        print(" -", w)
    print("\nnext: python -m btc.split")


if __name__ == "__main__":
    main()
