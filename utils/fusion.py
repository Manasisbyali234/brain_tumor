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


def dwt_fusion(ct_img, mri_img, wavelet="db4", level=3):
    """
    Discrete Wavelet Transform based fusion.
    db4 wavelet at level=3 preserves finer tumor-boundary details than db2/level=2.
    Rule: weighted average of approximation sub-bands (MRI weighted higher for
    soft-tissue contrast), max-magnitude of detail sub-bands.
    Returns a float32 image in [0, 1] with the same shape as ct_img.
    """
    # Ensure clean float32 inputs with no NaN/Inf
    ct_f = np.nan_to_num(ct_img.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    mri_f = np.nan_to_num(mri_img.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)

    coeffs_ct = pywt.wavedec2(ct_f, wavelet, level=level)
    coeffs_mri = pywt.wavedec2(mri_f, wavelet, level=level)

    fused_coeffs = []

    # Approximation coefficients: weight MRI higher (0.6) for soft-tissue contrast
    fused_approx = 0.4 * coeffs_ct[0] + 0.6 * coeffs_mri[0]
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


def fusion_quality_metrics(ct_img, mri_img, fused_img):
    """
    Step 4b: Fusion Quality Assessment.
    Computes SSIM and entropy to quantify how well the fused image
    preserves information from both source images.
    Returns a dict with ssim_ct, ssim_mri, entropy_fused.
    """
    from skimage.metrics import structural_similarity as ssim

    ct_f = ct_img.astype(np.float32)
    mri_f = mri_img.astype(np.float32)
    fused_f = fused_img.astype(np.float32)

    ssim_ct = ssim(ct_f, fused_f, data_range=1.0)
    ssim_mri = ssim(mri_f, fused_f, data_range=1.0)

    # Shannon entropy of fused image (higher = more information retained)
    hist, _ = np.histogram(fused_f, bins=256, range=(0, 1), density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)) * (1.0 / 256))

    return {"ssim_ct": round(ssim_ct, 4), "ssim_mri": round(ssim_mri, 4), "entropy": round(entropy, 4)}
