"""Train ResNet18 on the fixed split.

    python -m btc.train                       # uses configs/resnet18.yaml
    python -m btc.train --scratch             # from-scratch ablation (pretrained=false)
    python -m btc.train --epochs 1 --limit 20 --device cpu    # quick smoke test

Model selection uses VALIDATION macro-F1 only. The test set is touched by btc.evaluate.
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from .common import load_config, rel, seed_everything
from .data import make_loader
from .resnet18 import build_model


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    tot_loss, ys, ps = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=scaler is not None and scaler.is_enabled()):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            tot_loss += loss.item() * len(y)
            ys += y.tolist()
            ps += out.argmax(1).tolist()
    n = len(ys)
    acc = sum(int(a == b) for a, b in zip(ys, ps)) / n
    return tot_loss / n, acc, f1_score(ys, ps, average="macro", zero_division=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit", type=int, help="max images per class per split (smoke test)")
    ap.add_argument("--device")
    ap.add_argument("--scratch", action="store_true", help="random init instead of ImageNet weights")
    ap.add_argument("--run-name")
    args = ap.parse_args()

    cfg = load_config("resnet18")
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.scratch:
        cfg["pretrained"] = False
        cfg["run_name"] = "resnet18_scratch"
    if args.run_name:
        cfg["run_name"] = args.run_name

    seed_everything(cfg["seed"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = rel(cfg["out_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], limit_per_class=args.limit, seed=cfg["seed"])
    train_dl, val_dl = make_loader(cfg, "train", **kw), make_loader(cfg, "val", **kw)
    print(f"device={device}  train={len(train_dl.dataset)}  val={len(val_dl.dataset)}  pretrained={cfg['pretrained']}")

    model = build_model(len(cfg["classes"]), cfg["pretrained"], cfg["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    hist, best, bad = [], -1.0, 0
    for ep in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tl, ta, tf = run_epoch(model, train_dl, device, criterion, opt, scaler)
        vl, va, vf = run_epoch(model, val_dl, device, criterion)
        sched.step()
        hist.append(dict(epoch=ep, train_loss=tl, train_acc=ta, train_f1=tf, val_loss=vl, val_acc=va, val_f1=vf,
                         lr=opt.param_groups[0]["lr"], sec=time.time() - t0))
        print(f"ep {ep:02d}  train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f} macroF1 {vf:.3f}  ({hist[-1]['sec']:.0f}s)")
        pd.DataFrame(hist).to_csv(run_dir / "history.csv", index=False)
        if vf > best:
            best, bad = vf, 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val_f1": vf}, run_dir / "best.pt")
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"early stop (no val-F1 gain for {bad} epochs)")
                break
    print(f"best val macro-F1 {best:.4f} -> {run_dir / 'best.pt'}\nnext: python -m btc.evaluate --run-name {cfg['run_name']}")


if __name__ == "__main__":
    main()
