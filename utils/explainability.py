"""
explainability.py
-------------------
Step 9: Explainability
- Grad-CAM: highlights the image regions most responsible for the model's
  prediction (the "tumor region" heatmap).
- SHAP: explains the contribution of individual pixel/feature regions
  to the prediction using Shapley values (via GradientExplainer).
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="last_conv", pred_index=None):
    """
    Generates a Grad-CAM heatmap for a single image.
    Works with Sequential and Functional Keras models by running a manual
    forward pass with GradientTape watching the conv layer output.
    img_array: shape (1, H, W, C) — already preprocessed/normalized.
    """
    img_tensor = tf.cast(img_array, tf.float32)

    # Locate the target conv layer index
    layer_names = [l.name for l in model.layers]
    if last_conv_layer_name not in layer_names:
        raise ValueError(f"Layer '{last_conv_layer_name}' not found. Available: {layer_names}")
    conv_idx = layer_names.index(last_conv_layer_name)

    # Forward pass: run up to (and including) the conv layer, then continue
    with tf.GradientTape() as tape:
        x = img_tensor
        conv_outputs = None
        for i, layer in enumerate(model.layers):
            x = layer(x, training=False)
            if i == conv_idx:
                conv_outputs = x
                tape.watch(conv_outputs)
        predictions = x  # final output
        if pred_index is None:
            class_channel = predictions[:, 0]
        else:
            class_channel = predictions[:, pred_index]

    if conv_outputs is None:
        h, w = img_array.shape[1], img_array.shape[2]
        return np.ones((h // 16, w // 16), dtype=np.float32)

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        h, w = img_array.shape[1], img_array.shape[2]
        return np.ones((h // 16, w // 16), dtype=np.float32)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_gradcam(img, heatmap, alpha=0.4):
    """
    Overlays a Grad-CAM heatmap onto the original (grayscale/fused) image.
    img: (H, W) or (H, W, 1) float image in [0,1]
    Returns an RGB uint8 image ready for display/saving.
    """
    import cv2
    img = np.squeeze(img)
    h, w = img.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_resized = np.clip(heatmap_resized, 0, 1)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    jet_colors = plt.colormaps["jet"](np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]

    img_rgb = np.stack([img] * 3, axis=-1) if img.ndim == 2 else img
    superimposed = jet_heatmap * alpha + img_rgb * (1 - alpha)
    superimposed = np.clip(superimposed * 255, 0, 255).astype(np.uint8)
    return superimposed


def draw_attention_contour(img, heatmap, threshold=0.5, label="Model Attention Region"):
    """
    Draws a red contour around the high-activation region from the Grad-CAM
    heatmap on the original image.  Returns an RGB uint8 image.

    img      : (H, W) or (H, W, 1) float [0,1]
    heatmap  : raw output of make_gradcam_heatmap — 2-D float [0,1]
    threshold: fraction of max activation above which a pixel is "active"
    """
    import cv2
    img_sq = np.squeeze(img)
    h, w = img_sq.shape[:2]

    img_uint8 = np.uint8(np.clip(img_sq * 255, 0, 255))
    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB) if img_sq.ndim == 2 else img_uint8.copy()

    hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    hm = np.clip(hm, 0, 1)

    binary = np.uint8(hm >= threshold) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        overlay = img_rgb.copy()
        cv2.drawContours(overlay, [largest], -1, (220, 50, 50), thickness=cv2.FILLED)
        img_rgb = cv2.addWeighted(overlay, 0.25, img_rgb, 0.75, 0)
        cv2.drawContours(img_rgb, [largest], -1, (220, 30, 30), thickness=2)
        x, y, bw, bh = cv2.boundingRect(largest)
        label_y = max(y - 8, 14)
        cv2.putText(img_rgb, label, (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_rgb, label, (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 30, 30), 1, cv2.LINE_AA)

    return img_rgb


def get_shap_explainer(model, background_images):
    """
    Builds a SHAP GradientExplainer.
    background_images: a small representative sample (e.g. 20-50 images)
    from the training set, shape (N, H, W, C).
    """
    import shap
    explainer = shap.GradientExplainer(model, background_images)
    return explainer


def explain_with_shap(explainer, image, nsamples=50):
    """
    Computes SHAP values for a single image (shape (1, H, W, C)).
    Returns shap_values (list/array) usable with shap.image_plot().
    """
    shap_values = explainer.shap_values(image, nsamples=nsamples)
    return shap_values


def plot_shap(shap_values, image, out_path="outputs/shap_explanation.png"):
    """Saves a SHAP image-attribution plot to disk."""
    import shap
    shap.image_plot(shap_values, image, show=False)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _extract_brain_mask(img_uint8):
    """
    Returns a binary mask (uint8, 0/255) covering only the brain region.
    Strategy: threshold out dark background -> largest connected component.
    """
    import cv2
    # Otsu threshold to separate brain from black background
    _, thresh = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Keep only the largest connected component (= brain)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    if num_labels <= 1:
        return thresh  # fallback: whole image
    # label 0 is background; find largest non-background component
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    brain_mask = np.uint8(labels == largest_label) * 255

    # Fill internal holes with morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
    return brain_mask


def segment_tumor_mask(img, heatmap):
    """
    Produces a pixel-level tumor segmentation mask constrained inside the
    brain region so it never fires on background.

    Steps:
      1. Extract brain mask (largest connected component after Otsu)
      2. Resize Grad-CAM heatmap to full image size
      3. Zero-out heatmap outside the brain mask
      4. Otsu threshold on the brain-only heatmap values (adaptive cutoff)
      5. Morphological closing to fill small holes inside the mask
      6. Keep only the largest contour (removes stray specks)

    img      : (H, W) or (H, W, 1) float [0,1]
    heatmap  : 2-D float [0,1] from make_gradcam_heatmap
    Returns  : (mask_binary uint8, overlay_rgb uint8)
    """
    import cv2
    img_sq = np.squeeze(img)
    h, w = img_sq.shape[:2]

    img_uint8 = np.uint8(np.clip(img_sq * 255, 0, 255))

    # Step 1 — brain mask
    brain_mask = _extract_brain_mask(img_uint8)

    # Step 2 — resize heatmap
    hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    hm = np.clip(hm, 0, 1)

    # Step 3 — zero out background
    hm_brain = hm.copy()
    hm_brain[brain_mask == 0] = 0.0

    # Step 4 — Otsu threshold on brain-only heatmap
    hm_u8 = np.uint8(hm_brain * 255)
    otsu_val, mask = cv2.threshold(hm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If Otsu finds nothing meaningful, fall back to top-30% within brain
    if np.sum(mask > 0) < 50:
        brain_vals = hm_brain[brain_mask > 0]
        if len(brain_vals) > 0:
            fallback_thresh = np.percentile(brain_vals, 70)
            mask = np.uint8(hm_brain >= fallback_thresh) * 255
            mask[brain_mask == 0] = 0

    # Step 5 — morphological closing to fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Step 6 — keep only the largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)
    mask = clean_mask

    # Build overlay
    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB) if img_sq.ndim == 2 else img_uint8.copy()
    overlay = img_rgb.copy()
    overlay[mask > 0] = [220, 30, 30]
    blended = cv2.addWeighted(overlay, 0.40, img_rgb, 0.60, 0)
    if contours:
        cv2.drawContours(blended, [max(contours, key=cv2.contourArea)], -1, (255, 60, 60), 2)

    return mask, blended


def predict_regression_score(model, input_tensor):
    """
    Interprets the sigmoid output as a continuous tumor severity score.
    Returns a dict with score, grade label, and description.
    """
    raw = float(model.predict(input_tensor, verbose=0)[0][0])
    if raw < 0.2:
        grade, desc = "Grade 0 — No Tumor", "No detectable tumor tissue."
    elif raw < 0.4:
        grade, desc = "Grade 1 — Minimal", "Very low probability; likely benign or artifact."
    elif raw < 0.6:
        grade, desc = "Grade 2 — Moderate", "Borderline — further clinical review recommended."
    elif raw < 0.8:
        grade, desc = "Grade 3 — High", "High likelihood of malignant tissue present."
    else:
        grade, desc = "Grade 4 — Critical", "Strong indicator of aggressive tumor."
    return {"score": raw, "grade": grade, "description": desc}
