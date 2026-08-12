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

from utils.preprocessing import preprocess_image, clahe_enhance
from utils.registration import register_images
from utils.fusion import fuse_images
from utils.explainability import make_gradcam_heatmap, overlay_gradcam

CLASS_NAMES = ["Healthy Brain", "Tumor"]


def get_args():
    parser = argparse.ArgumentParser(description="Predict brain tumor from CT + MRI images")
    parser.add_argument("--ct", type=str, required=True, help="Path to CT image")
    parser.add_argument("--mri", type=str, required=True, help="Path to MRI image")
    parser.add_argument("--model", type=str, default="outputs/best_model.h5")
    parser.add_argument("--fusion_method", type=str, default="dwt", choices=["average", "pca", "dwt"])
    parser.add_argument("--out_dir", type=str, default="outputs")
    return parser.parse_args()


def run_pipeline(ct_path, mri_path, model, fusion_method="dwt"):
    """
    Runs preprocessing -> registration -> fusion -> prediction -> Grad-CAM.
    Returns a dict with all intermediate + final results.
    """
    ct_img = clahe_enhance(preprocess_image(ct_path))
    mri_img = clahe_enhance(preprocess_image(mri_path))

    mri_registered = register_images(mri_img, ct_img)
    fused = fuse_images(ct_img, mri_registered, method=fusion_method)

    input_tensor = np.expand_dims(fused, axis=(0, -1)).astype(np.float32)  # (1,H,W,1)

    prob = float(model.predict(input_tensor, verbose=0)[0][0])
    pred_class = int(prob >= 0.4)
    confidence = prob if pred_class == 1 else 1 - prob

    heatmap = make_gradcam_heatmap(input_tensor, model, last_conv_layer_name="last_conv")
    overlay = overlay_gradcam(fused, heatmap)

    return {
        "ct_img": ct_img,
        "mri_img": mri_img,
        "mri_registered": mri_registered,
        "fused_img": fused,
        "prediction": CLASS_NAMES[pred_class],
        "confidence": confidence,
        "gradcam_overlay": overlay,
    }


def display_and_save_results(results, out_dir="outputs"):
    """Step 10: Output Display — shows prediction, confidence, and visuals."""
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ["CT (preprocessed)", "MRI (registered)", "Fused Image", "Grad-CAM (Explanation)"]
    imgs = [results["ct_img"], results["mri_registered"], results["fused_img"], results["gradcam_overlay"]]

    for ax, img, title in zip(axes, imgs, titles):
        cmap = "gray" if img.ndim == 2 else None
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"Prediction: {results['prediction']}  |  Confidence: {results['confidence']*100:.2f}%",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    save_path = os.path.join(out_dir, "prediction_result.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print("\n===== Prediction Result =====")
    print(f"Prediction : {results['prediction']}")
    print(f"Confidence : {results['confidence']*100:.2f}%")
    print(f"Visualization saved to: {save_path}")


def main():
    args = get_args()
    model = tf.keras.models.load_model(args.model)
    results = run_pipeline(args.ct, args.mri, model, fusion_method=args.fusion_method)
    display_and_save_results(results, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
