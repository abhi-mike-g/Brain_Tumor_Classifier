# Brain Tumor Classifier

Comparative performance evaluation of four deep learning architectures (MLP, ResNet18, CNN-LSTM,
Vision Transformer) for four-class brain tumour classification from MRI images.
ICT-4442 Deep Learning Project, Manipal Institute of Technology, MAHE.

## Abstract

Brain tumor classification from Magnetic Resonance Imaging (MRI) images is an important application of deep learning in medical image analysis. This project presents a comparative evaluation of four deep learning architectures Multilayer Perceptron (MLP), ResNet18, CNN-LSTM, and Vision Transformer (ViT) for multi-class brain tumour classification. The study uses the publicly available MRI Brain Tumour Image Dataset compiled by Nickparvar, consisting of T1-weighted MRI images with contrast enhancement categorized into four classes: Glioma, Meningioma, Pituitary Tumour, and No Tumour. A common preprocessing and experimental framework are adopted across all four architectures to facilitate a controlled comparison. The images are subjected to consistent resizing and normalization before model training and evaluation. The selected architectures represent different approaches to visual representation learning: the MLP provides a fully connected baseline, ResNet18 uses convolutional residual learning, CNN-LSTM combines spatial feature extraction with sequential modelling, and ViT uses patch-based representations and self-attention. Model performance is evaluated using accuracy, precision, recall, F1-score, class-wise confusion patterns, parameter efficiency, and computational requirements. The study aims to analyse how architectural differences influence classification performance and computational characteristics under a common experimental setup, providing a systematic comparison of diverse deep learning approaches for automated brain MRI classification.

## Team

| Member | Reg. No. | Model |
| Varun Naik | 230953388 | MLP |
| Neema Naveen Kini | 230953070 | ResNet18 |
| Abhidutta Mukund Giri | 230953232 | CNN-LSTM |
| Vansh Vakharia | 230911178 | Vision Transformer |

## Dataset and the shared split

Source: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Nickparvar, Kaggle).
Classes: `glioma`, `meningioma`, `notumor`, `pituitary`.

**Dataset version note.** 
The synopsis cites 7,023 images. The current Kaggle download has **7,200** images and includes **203 augmented (synthetic) meningioma files** with `aug` in the file name. These are excluded, so the experiments use **6,997 images**.

**One split for everyone: `splits/split.csv`.** All images from Kaggle's Training and Testing
folders are pooled, exact and near-duplicate scans (perceptual hash distance <= 2) are grouped,
and whole groups are assigned to train/val/test (70/15/15, stratified by class). No group appears
in more than one split. 16 groups contain images with conflicting labels; they are kept together
and counted under their majority label.

| Class | Train | Val | Test | Total |
| glioma | 1,260 | 271 | 269 | 1,800 |
| meningioma | 1,118 | 240 | 239 | 1,597 |
| notumor | 1,259 | 270 | 271 | 1,800 |
| pituitary | 1,261 | 270 | 269 | 1,800 |
| **Total** | **4,898** | **1,051** | **1,048** | **6,997** |

Full report: `splits/split_report.txt`. Sample images: `reports/samples.png`.

**Common preprocessing** (`btc/data.py`, used by every model): resize to 224x224, 3 channels,
ImageNet mean/std normalisation. Train-only augmentation: random crop (scale >= 0.85),
rotation up to 10 degrees, horizontal flip. Validation and test images are not augmented.

## Results

Test set (1,048 images), macro-averaged over the four classes. Chosen checkpoint: best validation
macro-F1. Single run, seed 42.

| Model | Accuracy | Precision | Recall | F1 | Parameters | Latency (ms/img) |
| --- | --- | --- | --- | --- | --- | --- |
| **ResNet18** (ImageNet-pretrained) | **0.9838** | **0.9837** | **0.9835** | **0.9836** | 11,178,564 | 4.4 (Colab GPU) |
| CNN-LSTM (from scratch) | 0.9046 | 0.9037 | 0.9025 | 0.9026 | 2,360,068 | 6.9 (CPU) |
| ViT-B/32 (ImageNet-pretrained) | 0.9017 | 0.9037 | 0.9018 | 0.9016 | 87,458,308 | 26.9 (CPU) |

ResNet18 per-class F1: glioma 0.9777, meningioma 0.9790, notumor 0.9889, pituitary 0.9889.
CNN-LSTM per-class F1: glioma 0.9005, meningioma 0.8337, notumor 0.9313, pituitary 0.9449.
ViT-B/32 per-class F1: glioma 0.8757, meningioma 0.8780, notumor 0.9141, pituitary 0.9385.

CNN-LSTM trained for 15 epochs on CPU (best validation epoch 13). ViT-B/32 trained for 12 epochs on CPU (best validation epoch 12).
Latency is batch size 1. ResNet18 was timed on a Colab GPU; CNN-LSTM and ViT were timed on CPU. Compare latency only across runs on the same device.

## Repository layout

```
configs/common.yaml        settings shared by ALL models (seed, image size, split rules, augmentation)
configs/<model>.yaml       settings for one model
btc/data.py                shared dataset + transforms
btc/metrics.py             shared metrics (accuracy, P/R/F1, confusion matrix, params, latency)
btc/audit.py, split.py     Phase 1: dataset inspection and the fixed split
btc/resnet18.py, train.py, evaluate.py     ResNet18 (template for the other models)
splits/split.csv           THE fixed split. Do not regenerate.
results/<model>/<run>/     metrics.json, confusion_matrix.png, predictions.csv, history.csv, ...
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell: first run  Set-ExecutionPolicy -Scope Process Bypass
# source .venv/bin/activate     # Mac / Linux
pip install -r requirements.txt
python -m btc.download          # downloads the Kaggle data into data/raw
```

`splits/split.csv` is already in the repo, so you do **not** need to run `btc.audit` or
`btc.split`. Training needs a GPU; the free Colab T4 works (about 30 s per epoch for ResNet18).

```python
from btc.common import load_config
from btc.data import make_loader
from btc.metrics import compute_metrics, save_confusion_png, count_params, measure_latency_ms

cfg = load_config("yourmodel")                 # reads configs/common.yaml + configs/yourmodel.yaml
train_dl = make_loader(cfg, "train", batch_size=32)
val_dl = make_loader(cfg, "val", batch_size=32)
```

Rules, so the comparison stays fair:

1. Use `splits/split.csv` unchanged. Never regenerate it.
2. Use the loaders from `btc/data.py`. Inputs are `3 x 224 x 224` tensors; the MLP can flatten or downsample inside the model.
3. Pick epochs, learning rate and the checkpoint using the **validation** set only. Evaluate on test once, at the end.
4. Save outputs with `btc/metrics.py` into `results/<model>/<run_name>/metrics.json`, so all four files have the same fields.
5. Do not commit raw images (`data/raw` is git-ignored) or large checkpoints, except one final `best.pt` if the team wants it.

## Limitations

- Image-level classification only. No segmentation, grading or clinical use.
- No patient identifiers in this dataset, so near-duplicate grouping is by image similarity, not by patient. Some leakage between splits may remain.
- Results are single-seed runs on one split; differences of a few tenths of a percent between models are within normal run-to-run variation.

## License

MIT, see `LICENSE`.