"""Config loading, seeding, and small helpers shared by every module."""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ROOT = Path(__file__).resolve().parent.parent


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(*names: str) -> dict:
    """load_config('resnet18') merges configs/common.yaml with configs/resnet18.yaml."""
    cfg: dict = {}
    for n in ("common", *names):
        if n == "common" and cfg:
            continue
        path = ROOT / "configs" / f"{n}.yaml"
        with open(path) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def rel(path: str) -> Path:
    """Resolve a repo-relative path from the config."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
