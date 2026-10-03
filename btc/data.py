"""Shared preprocessing + dataset. EVERY model (MLP, ResNet18, CNN-LSTM, ViT) imports this,
so the comparison differs only in architecture. Don't fork it; change it for everyone."""
from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .common import rel


def build_transform(cfg: dict, train: bool) -> T.Compose:
    size, mean, std = cfg["image"]["size"], cfg["image"]["mean"], cfg["image"]["std"]
    if train:
        a = cfg["augment"]
        steps = [
            T.RandomResizedCrop(size, scale=(a["crop_scale_min"], 1.0), ratio=(0.9, 1.1)),
            T.RandomRotation(a["rotation_deg"]),
            T.RandomHorizontalFlip(a["hflip_p"]),
        ]
    else:  # validation / test: deterministic, no augmentation
        steps = [T.Resize((size, size))]
    return T.Compose(steps + [T.ToTensor(), T.Normalize(mean, std)])


class MRIDataset(Dataset):
    """Reads splits/split.csv. Images are loaded as 3-channel RGB (grayscale replicated)."""

    def __init__(self, cfg: dict, split: str, train: bool = False, limit_per_class: int | None = None):
        df = pd.read_csv(rel(cfg["data"]["split_csv"]))
        df = df[df.split == split]
        if limit_per_class:  # smoke tests only
            df = df.groupby("label").head(limit_per_class)
        self.df = df.reset_index(drop=True)
        self.raw = rel(cfg["data"]["raw_dir"])
        self.tf = build_transform(cfg, train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        with Image.open(self.raw / r.path) as im:
            x = self.tf(im.convert("RGB"))
        return x, int(r.label_idx)


def make_loader(cfg: dict, split: str, batch_size: int, num_workers: int = 2,
                limit_per_class: int | None = None, seed: int = 0) -> DataLoader:
    train = split == "train"
    ds = MRIDataset(cfg, split, train=train, limit_per_class=limit_per_class)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), generator=g, drop_last=False)
