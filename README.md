# Brain Tumor Classifier

Comparative performance evaluation of four deep learning architectures — MLP, ResNet18, CNN-LSTM, and Vision Transformer — for four-class brain tumour classification from MRI images.

**ICT-4442 Deep Learning Project, Manipal Institute of Technology, MAHE**

## Abstract

Brain tumor classification from Magnetic Resonance Imaging (MRI) images is an important application of deep learning in medical image analysis. This project presents a comparative evaluation of four deep learning architectures: Multilayer Perceptron (MLP), ResNet18, CNN-LSTM, and Vision Transformer (ViT) for multi-class brain tumour classification.

The study uses the publicly available MRI Brain Tumour Image Dataset compiled by Nickparvar, consisting of T1-weighted MRI images categorized into four classes: Glioma, Meningioma, Pituitary Tumour, and No Tumour.

A common preprocessing and experimental framework is adopted across all four architectures to enable a controlled and consistent comparison. Images undergo uniform resizing and normalization prior to model training and evaluation.

The selected architectures represent different approaches to visual representation learning:

- **MLP** provides a fully connected baseline.
- **ResNet18** uses convolutional residual learning.
- **CNN-LSTM** combines spatial feature extraction with sequential modelling.
- **Vision Transformer (ViT)** uses patch-based representations and self-attention.

Model performance is evaluated using accuracy, precision, recall, F1-score, class-wise confusion patterns, parameter efficiency, and computational requirements. The study analyses how architectural differences influence classification performance and computational characteristics under a common experimental setup.

## Team

| Member | Reg. No. | Model |
|---|---|---|
| Varun Naik | 230953388 | MLP |
| Neema Naveen Kini | 230953070 | ResNet18 |
| Abhidutta Mukund Giri | 230953232 | CNN-LSTM |
| Vansh Vakharia | 230911178 | Vision Transformer |

## Dataset and Shared Split

**Source:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — Nickparvar, Kaggle

**Classes:**

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

### Dataset Preparation

The audited dataset contained **7,200 images**, including **203 meningioma images** with `aug` in the filename. These images were excluded to maintain dataset consistency, resulting in a final dataset of **6,997 images**.

All images from the original Kaggle Training and Testing directories are pooled before generating the shared split.

Exact and near-duplicate scans are grouped using perceptual hashing with a Hamming distance threshold of `<= 2`. Entire duplicate groups are assigned to a single split to reduce potential data leakage.

The final dataset is divided into training, validation, and test sets using an approximately **70/15/15 split**, stratified by class.

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| glioma | 1,260 | 271 | 269 | 1,800 |
| meningioma | 1,118 | 240 | 239 | 1,597 |
| notumor | 1,259 | 270 | 271 | 1,800 |
| pituitary | 1,261 | 270 | 269 | 1,800 |
| **Total** | **4,898** | **1,051** | **1,048** | **6,997** |

The fixed dataset split is stored in:

```text
splits/split.csv
```

Additional dataset analysis is available in:

```text
splits/split_report.txt
```

Sample images are available in:

```text
reports/samples.png
```

## Common Preprocessing

All four models use the shared preprocessing pipeline implemented in:

```text
btc/data.py
```

Each input image is:

- Resized to **224 × 224**
- Converted to **3 channels**
- Normalized using **ImageNet mean and standard deviation**

Training augmentation includes:

- Random crop with scale `>= 0.85`
- Random rotation up to `10°`
- Random horizontal flip

Validation and test images are evaluated without augmentation.

## Model Performance

All four architectures are evaluated using the same **1,048-image test split** and random seed `42`.

Precision, recall, and F1-score are reported as macro-averaged metrics across the four classes.

| Model | Accuracy | Precision | Recall | F1 | Parameters | Latency (ms/img) |
|---|---:|---:|---:|---:|---:|---:|
| **ResNet18** (ImageNet-pretrained) | **0.9838** | **0.9837** | **0.9835** | **0.9836** | 11,178,564 | 88.9 |
| CNN-LSTM | 0.8387 | 0.8476 | 0.8389 | 0.8394 | 230,532 | 2.9 |
| MLP | 0.7071 | 0.7194 | 0.7018 | 0.7052 | 4,967,108 | 0.8 |
| Vision Transformer | 0.6861 | 0.6847 | 0.6801 | 0.6806 | 524,164 | 1.7 |

Latency values shown above were measured on CPU. Latency comparisons are most meaningful when models are evaluated under the same hardware and software configuration.

### ResNet18

ResNet18 achieved the highest overall classification performance.

**Per-class F1-score:**

| Class | F1-score |
|---|---:|
| Glioma | 0.9777 |
| Meningioma | 0.9790 |
| No Tumour | 0.9889 |
| Pituitary | 0.9889 |

The model uses ImageNet-pretrained weights and selects its checkpoint based on validation macro-F1.

### CNN-LSTM

CNN-LSTM combines convolutional feature extraction with LSTM-based sequential modelling.

**Per-class F1-score:**

| Class | F1-score |
|---|---:|
| Glioma | 0.8023 |
| Meningioma | 0.7468 |
| No Tumour | 0.9168 |
| Pituitary | 0.8918 |

The most frequent confusion pattern is **glioma classified as meningioma**, occurring for 57 test images.

### MLP

The MLP serves as the fully connected baseline architecture.

**Per-class F1-score:**

| Class | F1-score |
|---|---:|
| Glioma | 0.6610 |
| Meningioma | 0.5145 |
| No Tumour | 0.8000 |
| Pituitary | 0.8452 |

Despite its relatively simple architecture, the MLP provides a useful baseline for understanding the benefit of spatial and attention-based feature extraction methods.

### Vision Transformer

The compact Vision Transformer uses:

- **32 × 32 image patches**
- **96-dimensional embeddings**
- **2 Transformer encoder layers**

**Per-class F1-score:**

| Class | F1-score |
|---|---:|
| Glioma | 0.6427 |
| Meningioma | 0.5055 |
| No Tumour | 0.8177 |
| Pituitary | 0.7563 |

The most frequent confusion pattern is **meningioma classified as glioma**, occurring for 78 test images.

The complete model comparison is available in:

```text
reports/model_comparison.md
```

## Repository Layout

```text
configs/common.yaml        Shared settings used by all models
configs/<model>.yaml       Model-specific configuration

btc/data.py                Shared dataset loading and preprocessing
btc/metrics.py             Accuracy, precision, recall, F1, confusion matrix,
                           parameter count and latency measurement

btc/audit.py               Dataset inspection utilities
btc/split.py               Fixed dataset split generation

btc/mlp.py                 MLP classifier
btc/resnet18.py            ResNet18 classifier
btc/cnn_lstm.py            CNN-LSTM classifier
btc/vit.py                 Vision Transformer classifier

btc/train.py               Shared model training entry point
btc/evaluate.py            Shared model evaluation entry point
btc/compare.py             Cross-model comparison and consistency checks

splits/split.csv           Fixed train/validation/test split

results/<model>/<run>/     Model evaluation outputs
                           metrics.json
                           confusion_matrix.png
                           predictions.csv
                           history.csv
                           checkpoints

reports/                   Comparison reports and visualisations
```

## Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset:

```bash
python -m btc.download
```

If Kaggle authentication is required, configure the Kaggle API credentials locally.

Do not commit `kaggle.json` or other private credentials to the repository.

## Training and Evaluation

Run each model using the shared training and evaluation pipeline.

### MLP

```powershell
python -m btc.train --model mlp
python -m btc.evaluate --model mlp
```

### ResNet18

```powershell
python -m btc.train --model resnet18
python -m btc.evaluate --model resnet18
```

### CNN-LSTM

```powershell
python -m btc.train --model cnn_lstm
python -m btc.evaluate --model cnn_lstm
```

### Vision Transformer

```powershell
python -m btc.train --model vit
python -m btc.evaluate --model vit
```

Generate the final comparison table:

```powershell
python -m btc.compare
```

The comparison script verifies that all model evaluations use the same dataset split fingerprint and class support before generating the comparison.

## Shared Data Pipeline

Models can access the common dataset pipeline using:

```python
from btc.common import load_config
from btc.data import make_loader
from btc.metrics import (
    compute_metrics,
    save_confusion_png,
    count_params,
    measure_latency_ms
)

cfg = load_config("yourmodel")

train_dl = make_loader(
    cfg,
    "train",
    batch_size=32
)

val_dl = make_loader(
    cfg,
    "val",
    batch_size=32
)
```

## Experimental Protocol

To maintain a fair comparison across architectures:

1. Use `splits/split.csv` unchanged for all models.
2. Use the shared loaders implemented in `btc/data.py`.
3. All model inputs use `3 × 224 × 224` image tensors.
4. Hyperparameters and checkpoint selection are determined using the validation set.
5. The test set is used for final model evaluation.
6. Evaluation outputs are stored using the common metrics framework.
7. Model results are saved under:

```text
results/<model>/<run_name>/
```

8. Raw MRI images are excluded from version control.
9. Model comparisons use the same test split and evaluation metrics.

## Evaluation Metrics

Each architecture is evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Per-class F1-score
- Confusion Matrix
- Parameter Count
- Inference Latency

These metrics provide both predictive performance and computational characteristics for each architecture.

## Key Findings

Among the four evaluated architectures, **ResNet18 achieved the strongest overall classification performance**, reaching an accuracy of **98.38%** and a macro F1-score of **98.36%**.

CNN-LSTM achieved the second-highest overall performance while using substantially fewer parameters than ResNet18.

The MLP serves as a useful baseline and demonstrates the limitations of purely fully connected representations for image classification compared with architectures capable of learning spatial features.

The compact Vision Transformer demonstrates attention-based image representation learning while maintaining a relatively small parameter count.

Across the models, the **No Tumour** and **Pituitary Tumour** classes generally achieved stronger classification performance, while **Glioma** and **Meningioma** showed greater overlap and contributed to the largest observed confusion patterns.

## Limitations

- The system performs image-level classification and does not perform tumour segmentation or grading.
- The dataset does not provide patient identifiers; duplicate control therefore relies on image similarity.
- Evaluation is based on a fixed dataset split and random seed.
- The models are designed for research and educational evaluation and are not intended for clinical diagnosis.

## License

This project is licensed under the MIT License.

See:

```text
LICENSE
```
