"""
preprocessing.py
-----------------
Step 2: Image Preprocessing
Handles resizing, denoising, and normalization of CT/MRI brain images.
"""

import cv2
import numpy as np


IMG_SIZE = (224, 224)  # standard size expected by most CNN backbones


def load_image(path, color_mode="grayscale"):
    """Load an image from disk as grayscale or RGB."""
    flag = cv2.IMREAD_GRAYSCALE if color_mode == "grayscale" else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_image(img, size=IMG_SIZE):
    """Resize image to a standard size."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def denoise_image(img, method="gaussian"):
    """Remove noise using a filter."""
    if method == "gaussian":
        return cv2.GaussianBlur(img, (5, 5), 0)
    elif method == "median":
        return cv2.medianBlur(img, 5)
    elif method == "bilateral":
        return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    else:
        raise ValueError(f"Unknown denoise method: {method}")


def normalize_image(img):
    """Normalize pixel values to [0, 1]."""
    img = img.astype(np.float32)
    return img / 255.0


def preprocess_image(path, color_mode="grayscale", denoise_method="gaussian", size=IMG_SIZE):
    """
    Full preprocessing pipeline for a single image:
    load -> resize -> denoise -> normalize
    """
    img = load_image(path, color_mode=color_mode)
    img = resize_image(img, size)
    img = denoise_image(img, denoise_method)
    img = normalize_image(img)
    return img


def clahe_enhance(img):
    """
    Optional contrast enhancement using CLAHE — helps make tumor
    boundaries more distinguishable in CT/MRI scans before fusion.
    """
    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_uint8)
    return normalize_image(enhanced)
