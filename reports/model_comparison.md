# Model comparison

**All configured models have saved test evaluations.**

All rows must use the same test split. Metrics are macro-averaged; latency is batch-size 1.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Parameters | Latency (ms/img) |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 0.7071 | 0.7194 | 0.7018 | 0.7052 | 4,967,108 | 0.80 |
| ResNet18 | 0.9838 | 0.9837 | 0.9835 | 0.9836 | 11,178,564 | 88.87 |
| CNN-LSTM | 0.8387 | 0.8476 | 0.8389 | 0.8394 | 230,532 | 2.89 |
| ViT | 0.6861 | 0.6847 | 0.6801 | 0.6806 | 524,164 | 1.68 |

Split SHA-256: `b31e35259ee3106e8d3360adb28e62c5e1716caefb1d15a24ab3947a36f81e1f`
