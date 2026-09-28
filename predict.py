"""
predict.py
-----------
Step 8: Prediction + Step 10: Output Display
Runs a new CT+MRI image pair through the full pipeline:
preprocess -> register -> fuse -> predict -> explain (Grad-CAM/SHAP)
and displays/saves the final output.

Usage:
    python predict.py --ct path/to/ct.png --mri path/to/mri.png --model outputs/best_model.h5
"""

import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.preprocessing import preprocess_image, clahe_enhance, skull_strip
from utils.registration import register_images
from utils.fusion import fuse_images, fusion_quality_metrics
from utils.explainability import make_gradcam_heatmap, overlay_gradcam, draw_attention_contour, segment_tumor_mask, predict_regression_score

CLASS_NAMES = ["Healthy Brain", "Tumor"]


def get_args():
    parser = argparse.ArgumentParser(description="Predict brain tumor from CT + MRI images")
    parser.add_argument("--ct", type=str, required=True, help="Path to CT image")
    parser.add_argument("--mri", type=str, required=True, help="Path to MRI image")
    parser.add_argument("--model", type=str, default="outputs/best_model.h5")
    parser.add_argument("--fusion_method", type=str, default="dwt", choices=["average", "pca", "dwt"])
    parser.add_argument("--out_dir", type=str, default="outputs")
    return parser.parse_args()


def _tta_predict(model, input_tensor, n_aug=8):
    """
    Test-Time Augmentation: average predictions over flipped/transposed
    variants of the input for a more robust confidence score.
    """
    variants = [
        input_tensor,
        input_tensor[:, :, ::-1, :],
        input_tensor[:, ::-1, :, :],
        input_tensor[:, :, ::-1, :][:, ::-1, :, :],
        np.transpose(input_tensor, (0, 2, 1, 3)),
        np.transpose(input_tensor, (0, 2, 1, 3))[:, :, ::-1, :],
        np.transpose(input_tensor, (0, 2, 1, 3))[:, ::-1, :, :],
        np.transpose(input_tensor, (0, 2, 1, 3))[:, ::-1, :, :][:, :, ::-1, :],
    ]
    probs = [float(model.predict(v.astype(np.float32), verbose=0)[0][0]) for v in variants[:n_aug]]
    return float(np.mean(probs))


def run_pipeline(ct_path, mri_path, model, fusion_method="dwt"):
    """
    Full pipeline: preprocess -> skull-strip -> register -> fuse ->
    fusion QA -> predict -> Grad-CAM -> segmentation -> severity grade.
    """
    # Step 2: Preprocess + CLAHE
    ct_img  = clahe_enhance(preprocess_image(ct_path))
    mri_img = clahe_enhance(preprocess_image(mri_path))

    # Step 2b: Skull stripping
    ct_stripped  = skull_strip(ct_img)
    mri_stripped = skull_strip(mri_img)

    # Step 3: Registration
    mri_registered = register_images(mri_stripped, ct_stripped)

    # Step 4: Fusion
    fused = fuse_images(ct_stripped, mri_registered, method=fusion_method)

    # Step 4b: Fusion quality metrics
    fusion_metrics = fusion_quality_metrics(ct_stripped, mri_registered, fused)

    # Step 5-8: Predict with TTA
    input_tensor = np.expand_dims(fused, axis=(0, -1)).astype(np.float32)  # (1,H,W,1)
    prob = _tta_predict(model, input_tensor)
    # Threshold 0.35: biased toward recall (catching tumors > missing them)
    pred_class = int(prob >= 0.35)
    confidence = prob if pred_class == 1 else 1 - prob

    # Step 9: Grad-CAM
    heatmap = make_gradcam_heatmap(input_tensor, model, last_conv_layer_name="last_conv")
    overlay = overlay_gradcam(fused, heatmap)
    contour_img = draw_attention_contour(fused, heatmap)

    # Step 9b: Tumor segmentation mask
    tumor_mask, seg_overlay = segment_tumor_mask(fused, heatmap)

    # Step 9c: Severity grading
    severity = predict_regression_score(model, input_tensor)

    return {
        "ct_img": ct_img,
        "mri_img": mri_img,
        "ct_stripped": ct_stripped,
        "mri_registered": mri_registered,
        "fused_img": fused,
        "fusion_metrics": fusion_metrics,
        "prediction": CLASS_NAMES[pred_class],
        "confidence": confidence,
        "gradcam_overlay": overlay,
        "contour_img": contour_img,
        "tumor_mask": tumor_mask,
        "seg_overlay": seg_overlay,
        "severity": severity,
    }


def display_and_save_results(results, out_dir="outputs"):
    """Step 10: Output Display — shows prediction, confidence, and all visuals."""
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    titles = ["CT (skull-stripped)", "MRI (registered)", "Fused Image",
              "Grad-CAM (Red)", "Tumor Marking"]
    imgs   = [results["ct_stripped"], results["mri_registered"], results["fused_img"],
              results["gradcam_overlay"], results["contour_img"]]

    for ax, img, title in zip(axes, imgs, titles):
        cmap = "gray" if img.ndim == 2 else None
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    sev = results["severity"]
    fm  = results["fusion_metrics"]
    fig.suptitle(
        f"Prediction: {results['prediction']}  |  Confidence: {results['confidence']*100:.2f}%  |  "
        f"{sev['grade']}\n"
        f"Fusion SSIM(CT): {fm['ssim_ct']}  SSIM(MRI): {fm['ssim_mri']}  Entropy: {fm['entropy']}",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    save_path = os.path.join(out_dir, "prediction_result.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print("\n===== Prediction Result =====")
    print(f"Prediction : {results['prediction']}")
    print(f"Confidence : {results['confidence']*100:.2f}%")
    print(f"Severity   : {sev['grade']}")
    print(f"Description: {sev['description']}")
    print(f"Fusion QA  : SSIM(CT)={fm['ssim_ct']}  SSIM(MRI)={fm['ssim_mri']}  Entropy={fm['entropy']}")
    print(f"Visualization saved to: {save_path}")


def main():
    args = get_args()
    model = tf.keras.models.load_model(args.model)
    results = run_pipeline(args.ct, args.mri, model, fusion_method=args.fusion_method)
    display_and_save_results(results, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
