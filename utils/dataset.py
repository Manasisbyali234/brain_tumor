"""
dataset.py
-----------
Step 1: Data Collection (loading) + assembling fused CT-MRI dataset
for training/testing.

Expected folder structure:

    data/
      ct/
        tumor/    *.png|jpg
        no_tumor/ *.png|jpg
      mri/
        tumor/    *.png|jpg   (filenames should correspond to CT counterparts)
        no_tumor/ *.png|jpg

If you only have a single-modality dataset (common for public Kaggle sets),
set `paired=False` and the loader will duplicate the same image as both
CT and MRI input so the fusion pipeline still runs end-to-end.
"""

import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

from utils.preprocessing import preprocess_image, clahe_enhance
from utils.registration import register_images
from utils.fusion import fuse_images

CLASSES = ["no_tumor", "tumor"]  # label 0, 1


def build_mri_dataset(data_root="data", limit_per_class=None):
    """
    Loads MRI images directly (no fusion) for single-modality datasets.
    Returns (X, y) with X shape (N, 224, 224, 1).
    """
    X, y = [], []
    for label, cls in enumerate(CLASSES):
        mri_dir = os.path.join(data_root, "mri", cls)
        files = _list_images(mri_dir)
        if limit_per_class:
            files = files[:limit_per_class]
        for path in files:
            img = clahe_enhance(preprocess_image(path))
            X.append(img)
            y.append(label)
    X = np.expand_dims(np.array(X, dtype=np.float32), axis=-1)
    y = np.array(y, dtype=np.float32)
    return X, y


def _list_images(folder):
    exts = ("*.png", "*.jpg", "*.jpeg")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)


def build_fused_dataset(data_root="data", fusion_method="dwt", paired=True, limit_per_class=None):
    """
    Walks the CT/MRI folders, preprocesses + registers + fuses each pair,
    and returns (X, y) numpy arrays ready for train/test split.
    """
    X, y = [], []

    for label, cls in enumerate(CLASSES):
        mri_dir = os.path.join(data_root, "mri", cls)
        mri_files = _list_images(mri_dir)
        if limit_per_class:
            mri_files = mri_files[:limit_per_class]

        ct_files = _list_images(os.path.join(data_root, "ct", cls)) if not paired else []
        if limit_per_class and ct_files:
            ct_files = ct_files[:limit_per_class]

        for i, mri_path in enumerate(mri_files):
            if paired:
                ct_dir = os.path.join(data_root, "ct", cls)
                fname = os.path.basename(mri_path)
                ct_path = os.path.join(ct_dir, fname)
                if not os.path.exists(ct_path):
                    continue
            else:
                # Cross-pair: use CT image at same index (wrap around) so
                # fused image combines two different images with real signal
                if ct_files:
                    ct_path = ct_files[i % len(ct_files)]
                else:
                    ct_path = mri_path  # fallback if no CT data at all

            ct_img = clahe_enhance(preprocess_image(ct_path))
            mri_img = clahe_enhance(preprocess_image(mri_path))

            mri_registered = register_images(mri_img, ct_img)
            fused = fuse_images(ct_img, mri_registered, method=fusion_method)

            X.append(fused)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    X = np.expand_dims(X, axis=-1)  # add channel dim -> (N, H, W, 1)
    y = np.array(y, dtype=np.float32)

    return X, y


def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Train / validation / test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, stratify=y_train, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
