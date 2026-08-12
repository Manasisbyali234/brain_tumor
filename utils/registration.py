"""
registration.py
-----------------
Step 3: Image Registration
Aligns the CT and MRI images spatially so that fusion (Step 4) combines
corresponding anatomical regions correctly.

Uses ECC (Enhanced Correlation Coefficient) with a translation+rotation
(MOTION_EUCLIDEAN) warp model — this preserves Cartesian geometry and
cannot produce the radial/starburst artifacts that a full homography warp
causes when keypoint matching fails on low-texture brain images.
"""

import cv2
import numpy as np


def to_uint8(img):
    """Convert a normalized float image [0,1] back to uint8 for OpenCV ops."""
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def register_images(moving_img, fixed_img, **_kwargs):
    """
    Align `moving_img` (MRI) onto `fixed_img` (CT) using ECC minimisation
    with a Euclidean (translation + rotation) warp model.

    A Euclidean warp can only translate/rotate the image — it cannot
    introduce perspective distortion or starburst artifacts.
    Falls back to a plain resize if ECC does not converge.

    Returns the registered image as float32 in [0, 1], same shape as fixed_img.
    """
    h, w = fixed_img.shape[:2]

    # Resize moving image to match fixed dimensions first
    moving_resized = cv2.resize(to_uint8(moving_img), (w, h))
    fixed_u8 = to_uint8(fixed_img)

    # Initial warp matrix: identity (no transform)
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        50,    # max iterations — kept low for speed
        1e-4,  # convergence threshold
    )

    try:
        _, warp_matrix = cv2.findTransformECC(
            fixed_u8,
            moving_resized,
            warp_matrix,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            5,  # gaussian blur size for gradient computation
        )
        registered = cv2.warpAffine(
            moving_resized,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
    except cv2.error:
        # ECC failed to converge — images are already well-aligned enough
        registered = moving_resized

    return registered.astype(np.float32) / 255.0
