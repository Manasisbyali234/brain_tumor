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


def preprocess_image(path, color_mode="grayscale", denoise_method="bilateral", size=IMG_SIZE):
    """
    Full preprocessing pipeline for a single image:
    load -> resize -> denoise -> normalize
    Uses bilateral filter by default: preserves tumor edges while removing noise.
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


def skull_strip(img):
    """
    Step 2b: Skull Stripping — removes skull/background, keeping only brain tissue.
    Uses Otsu thresholding + largest connected component + morphological fill.
    img: float32 [0,1] grayscale. Returns skull-stripped float32 [0,1].
    """
    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    # Otsu threshold to separate brain from background
    _, thresh = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Keep only the largest connected component (brain)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    if num_labels <= 1:
        brain_mask = thresh
    else:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        brain_mask = np.uint8(labels == largest_label) * 255

    # Fill holes with morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

    stripped = cv2.bitwise_and(img_uint8, img_uint8, mask=brain_mask)
    return normalize_image(stripped)
