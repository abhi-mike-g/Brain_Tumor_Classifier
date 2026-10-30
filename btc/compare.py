"""Build a test-set comparison table from the saved evaluations."""
from __future__ import annotations

import argparse
import json

from .common import load_config, rel

MODEL_NAMES = ("mlp", "resnet18", "cnn_lstm", "vit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write a clearly labelled progress table if one or more model results are absent",
    )
    args = parser.parse_args()

    runs = []
    missing = []
    for model_name in MODEL_NAMES:
        cfg = load_config(model_name)
        run_dir = rel(cfg["out_dir"]) / cfg["run_name"]
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            missing.append(model_name)
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "test" not in metrics or "split_sha256" not in metrics:
            raise ValueError(f"{metrics_path} lacks test metrics or split fingerprint; rerun evaluation.")
        runs.append(metrics)

    if missing and not args.allow_incomplete:
        raise SystemExit(
            "Missing evaluated model(s): " + ", ".join(missing)
            + ". Train and evaluate each model, or use --allow-incomplete for a progress table."
        )
    if not runs:
        raise SystemExit("No evaluated model metrics were found.")

    fingerprints = {run["split_sha256"] for run in runs}
    if len(fingerprints) != 1:
        raise ValueError("Model metrics were produced from different split.csv contents.")
    supports = {
        tuple((name, values["support"]) for name, values in run["test"]["per_class"].items())
        for run in runs
    }
    if len(supports) != 1:
        raise ValueError("Model metrics have different test class supports.")

    lines = [
        "# Model comparison",
        "",
        ("**INCOMPLETE progress table.** Missing results: " + ", ".join(missing) + ".")
        if missing else "**All configured models have saved test evaluations.**",
        "",
        "All rows must use the same test split. Metrics are macro-averaged; latency is batch-size 1.",
        "",
        "| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Parameters | Latency (ms/img) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        test = run["test"]
        lines.append(
            f"| {run['model']} | {test['accuracy']:.4f} | {test['macro_precision']:.4f} "
            f"| {test['macro_recall']:.4f} | {test['macro_f1']:.4f} "
            f"| {run['params_total']:,} | {run['latency_ms_bs1']:.2f} |"
        )
    lines.extend(["", f"Split SHA-256: `{next(iter(fingerprints))}`", ""])
    output = rel("reports/model_comparison.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
