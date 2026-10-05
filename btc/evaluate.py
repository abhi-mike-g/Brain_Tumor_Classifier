"""Evaluate the best checkpoint on val + test and write the shared results files.

    python -m btc.evaluate --run-name resnet18_pretrained

Writes into results/resnet18/<run>/ : metrics.json, confusion_matrix.png,
classification_report.txt, predictions.csv (for error analysis).
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import torch
from sklearn.metrics import classification_report

from .common import load_config, rel
from .data import make_loader
from .metrics import compute_metrics, count_params, measure_latency_ms, save_confusion_png
from .models import LABELS, MODEL_CHOICES, build_model, normalize_name


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), 1).cpu()
        ys += y.tolist()
        ps += p.argmax(1).tolist()
        probs += p.tolist()
    return ys, ps, probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=list(MODEL_CHOICES), help="model architecture (inferred from config.json if not set)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device")
    args = ap.parse_args()

    model_arg = args.model
    if model_arg:
        cfg = load_config(model_arg)
        run_dir = rel(cfg["out_dir"]) / (args.run_name or cfg["run_name"])
    elif args.run_name:
        cfg, run_dir = load_config("resnet18"), None
        for name in MODEL_CHOICES:
            candidate_cfg = load_config(name)
            candidate = rel(candidate_cfg["out_dir"]) / args.run_name
            if (candidate / "config.json").exists():
                cfg, run_dir = candidate_cfg, candidate
                break
        if run_dir is None:
            run_dir = rel(cfg["out_dir"]) / args.run_name
    else:
        cfg = load_config("resnet18")
        run_dir = rel(cfg["out_dir"]) / cfg["run_name"]

    saved = json.loads((run_dir / "config.json").read_text())  # the config this run was trained with
    classes = saved["classes"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model_type = normalize_name(saved.get("model_name", args.model or "resnet18"))
    model_label = LABELS[model_type]
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    saved = {**saved, "model_name": model_type}
    model = build_model(saved, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    result = {"model": model_label, "run_name": saved["run_name"], "pretrained": saved.get("pretrained", False),
              "best_epoch": ckpt["epoch"], "seed": saved["seed"], "device": str(device)}
    for split in ("val", "test"):
        dl = make_loader(saved, split, saved["batch_size"], saved["num_workers"], args.limit)
        ys, ps, probs = predict(model, dl, device)
        m = compute_metrics(ys, ps, classes)
        result[split] = m
        print(f"{split}: n={len(ys)}  acc {m['accuracy']:.4f}  macro P {m['macro_precision']:.4f} "
              f"R {m['macro_recall']:.4f} F1 {m['macro_f1']:.4f}")
        if split == "test":
            print(classification_report(ys, ps, target_names=classes, digits=4, zero_division=0))
            (run_dir / "classification_report.txt").write_text(
                classification_report(ys, ps, target_names=classes, digits=4, zero_division=0))
            save_confusion_png(m["confusion_matrix"], classes, run_dir / "confusion_matrix.png", f"{model_label} - test")
            df = dl.dataset.df[["path", "label"]].copy()
            df["pred"] = [classes[i] for i in ps]
            df["correct"] = df.label == df.pred
            for k, c in enumerate(classes):
                df[f"p_{c}"] = [p[k] for p in probs]
            df.to_csv(run_dir / "predictions.csv", index=False)

    result.update(count_params(model))
    result["latency_ms_bs1"] = measure_latency_ms(model, device, saved["image"]["size"])
    print(f"params {result['params_total']:,}  latency {result['latency_ms_bs1']:.1f} ms/img on {device}")
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print("wrote", run_dir / "metrics.json")


if __name__ == "__main__":
    main()
