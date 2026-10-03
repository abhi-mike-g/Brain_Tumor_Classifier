"""Download the Kaggle dataset into data/raw (run once per machine).

    python -m btc.download

Needs internet. If kagglehub asks for credentials, create a Kaggle API token
(kaggle.com -> Settings -> Create New Token) and put kaggle.json in ~/.kaggle/.
Manual alternative: download the zip from the dataset page and unzip it into data/raw
so that data/raw/Training/glioma/... and data/raw/Testing/glioma/... exist.
"""
import shutil

from .common import load_config, rel

SLUG = "masoudnickparvar/brain-tumor-mri-dataset"


def main() -> None:
    import kagglehub

    cfg = load_config()
    dest = rel(cfg["data"]["raw_dir"])
    src = kagglehub.dataset_download(SLUG)
    print("downloaded to cache:", src)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, dirs_exist_ok=True)
    print("copied to:", dest)
    print("next: python -m btc.audit")


if __name__ == "__main__":
    main()
