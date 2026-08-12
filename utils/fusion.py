"""
fusion.py
----------
Step 4: Image Fusion
Combines registered CT and MRI images into a single fused image carrying
complementary information (CT: bone/dense-tissue detail, MRI: soft-tissue
contrast). Provides three fusion strategies of increasing sophistication.
"""

import cv2
import numpy as np
import pywt


def average_fusion(ct_img, mri_img):
    """Simplest baseline: pixel-wise averaging."""
    return (ct_img.astype(np.float32) + mri_img.astype(np.float32)) / 2.0


def pca_fusion(ct_img, mri_img):
    """
    Principal Component Analysis based fusion.
    Weighs each image by the variance it contributes.
    """
    ct_flat = ct_img.flatten()
    mri_flat = mri_img.flatten()
    stacked = np.vstack([ct_flat, mri_flat])

    cov = np.cov(stacked)
    eigvals, eigvecs = np.linalg.eig(cov)
    principal = np.abs(eigvecs[:, np.argmax(eigvals)])
    weights = principal / (np.sum(principal) + 1e-8)

    fused = weights[0] * ct_img + weights[1] * mri_img
    return fused.astype(np.float32)


def dwt_fusion(ct_img, mri_img, wavelet="db2", level=2):
    """
    Discrete Wavelet Transform based fusion.
    Rule: average the low-frequency (approximation) sub-band,
    take max-magnitude of high-frequency (detail) sub-bands.
    Returns a float32 image in [0, 1] with the same shape as ct_img.
    """
    # Ensure clean float32 inputs with no NaN/Inf
    ct_f = np.nan_to_num(ct_img.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    mri_f = np.nan_to_num(mri_img.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)

    coeffs_ct = pywt.wavedec2(ct_f, wavelet, level=level)
    coeffs_mri = pywt.wavedec2(mri_f, wavelet, level=level)

    fused_coeffs = []

    # Approximation coefficients (lowest frequency): average
    fused_approx = (coeffs_ct[0] + coeffs_mri[0]) / 2.0
    fused_coeffs.append(fused_approx)

    # Detail coefficients at each level: take max-magnitude pixel-wise
    for (cH_ct, cV_ct, cD_ct), (cH_mri, cV_mri, cD_mri) in zip(coeffs_ct[1:], coeffs_mri[1:]):
        cH = np.where(np.abs(cH_ct) >= np.abs(cH_mri), cH_ct, cH_mri)
        cV = np.where(np.abs(cV_ct) >= np.abs(cV_mri), cV_ct, cV_mri)
        cD = np.where(np.abs(cD_ct) >= np.abs(cD_mri), cD_ct, cD_mri)
        fused_coeffs.append((cH, cV, cD))

    fused_img = pywt.waverec2(fused_coeffs, wavelet)

    # waverec2 can produce a shape slightly larger than input; crop back
    fused_img = fused_img[: ct_img.shape[0], : ct_img.shape[1]]

    # Sanitize reconstruction output before normalizing
    fused_img = np.nan_to_num(fused_img, nan=0.0, posinf=1.0, neginf=0.0)

    mn, mx = fused_img.min(), fused_img.max()
    if mx - mn < 1e-8:
        # Flat image — fall back to average fusion
        fused_img = (ct_f + mri_f) / 2.0
    else:
        fused_img = (fused_img - mn) / (mx - mn)

    return fused_img.astype(np.float32)


def fuse_images(ct_img, mri_img, method="dwt"):
    """
    Unified entry point for Step 4.
    method: 'average' | 'pca' | 'dwt'
    """
    if ct_img.shape != mri_img.shape:
        mri_img = cv2.resize(mri_img, (ct_img.shape[1], ct_img.shape[0]))

    if method == "average":
        fused = average_fusion(ct_img, mri_img)
    elif method == "pca":
        fused = pca_fusion(ct_img, mri_img)
    elif method == "dwt":
        fused = dwt_fusion(ct_img, mri_img)
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    fused = np.clip(fused, 0.0, 1.0)
    return fused
