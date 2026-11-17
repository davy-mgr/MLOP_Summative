"""Build data/train and data/test from a downloaded UrbanSound8K folder.

Usage:
    python -m src.data_acquisition --source /path/to/UrbanSound8K
    python -m src.data_acquisition --source /path/to/UrbanSound8K --max-per-class 30   # small subset for deployment

Folds 1-9 -> train, fold 10 -> test (the dataset authors warn against reshuffling across
folds because clips from the same recording are in the same fold -> leakage).
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR


def build_dataset(source, test_folds=(10,), max_per_class=None, out_dir=DATA_DIR):
    source, out_dir = Path(source), Path(out_dir)
    meta = pd.read_csv(source / "metadata" / "UrbanSound8K.csv")
    meta["split"] = np.where(meta["fold"].isin(test_folds), "test", "train")
    if max_per_class:
        meta = meta.sample(frac=1, random_state=42).groupby(["class", "split"]).head(max_per_class)

    copied = 0
    for _, r in meta.iterrows():
        src = source / "audio" / f"fold{r['fold']}" / r["slice_file_name"]
        if not src.exists():                       # some copies of the dataset have no 'audio/' level
            src = source / f"fold{r['fold']}" / r["slice_file_name"]
        dst = out_dir / r["split"] / r["class"] / r["slice_file_name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    print(f"Copied {copied} files. Train: {(meta.split == 'train').sum()}  Test: {(meta.split == 'test').sum()}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Path to the UrbanSound8K folder")
    ap.add_argument("--max-per-class", type=int, default=None)
    a = ap.parse_args()
    build_dataset(a.source, max_per_class=a.max_per_class)
