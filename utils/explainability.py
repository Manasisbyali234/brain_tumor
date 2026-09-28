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
    Grad-CAM that works with Keras 3 loaded models.
    Avoids layer.output / model.output entirely — uses a manual forward pass
    with GradientTape watching a tf.Variable holding the conv activations.
    img_array: shape (1, H, W, C) — already preprocessed/normalized.
    """
    h, w = int(img_array.shape[1]), int(img_array.shape[2])
    img_tensor = tf.cast(img_array, tf.float32)

    # Find target conv layer index; fall back to last Conv2D
    target_idx = None
    for i, layer in enumerate(model.layers):
        if layer.name == last_conv_layer_name:
            target_idx = i
            break
    if target_idx is None:
        for i, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                target_idx = i
    if target_idx is None:
        return np.ones((h // 16, w // 16), dtype=np.float32)

    # First pass: run up to (and including) target conv layer to get its output
    x = img_tensor
    for i, layer in enumerate(model.layers):
        x = layer(x, training=False)
        if i == target_idx:
            conv_out_value = x.numpy()  # capture as numpy
            break

    # Second pass inside GradientTape: use a tf.Variable for the conv output
    # so gradients can flow back through the remaining layers
    conv_var = tf.Variable(conv_out_value, trainable=True, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(conv_var)
        x2 = conv_var
        for i, layer in enumerate(model.layers):
            if i > target_idx:
                x2 = layer(x2, training=False)
        class_channel = x2[:, 0] if pred_index is None else x2[:, pred_index]

    grads = tape.gradient(class_channel, conv_var)
    if grads is None:
        return np.ones((h // 16, w // 16), dtype=np.float32)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))                    # (filters,)
    heatmap = tf.reduce_sum(conv_var[0] * pooled_grads, axis=-1)            # (H, W)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_gradcam(img, heatmap, alpha=0.5):
    """
    Overlays a red Grad-CAM heatmap onto the original image.
    High-activation regions appear bright red; low-activation regions stay transparent.
    img: (H, W) or (H, W, 1) float image in [0,1]
    Returns an RGB uint8 image.
    """
    import cv2
    img = np.squeeze(img)
    h, w = img.shape[:2]

    hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    hm = np.clip(hm, 0, 1)

    # Adaptive threshold: only show top activations
    nonzero = hm[hm > 0.05]
    thresh = float(np.percentile(nonzero, 50)) if len(nonzero) > 0 else 0.4
    hm_thresh = np.where(hm > thresh, hm, 0.0)

    img_uint8 = np.uint8(np.clip(img * 255, 0, 255))
    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img_uint8.copy()

    red_overlay = np.zeros_like(img_rgb, dtype=np.float32)
    red_overlay[:, :, 0] = hm_thresh * 255  # R only

    mask_3ch = np.stack([hm_thresh] * 3, axis=-1)
    blended = img_rgb.astype(np.float32) * (1 - mask_3ch * alpha) + red_overlay * (mask_3ch * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_attention_contour(img, heatmap, threshold=None, label="Tumor Region"):
    """
    Draws a precise red contour + bounding box around the high-activation
    region from the Grad-CAM heatmap.
    threshold=None → adaptive (top-40% of heatmap values).
    img      : (H, W) or (H, W, 1) float [0,1]
    heatmap  : 2-D float [0,1] from make_gradcam_heatmap
    """
    import cv2
    img_sq = np.squeeze(img)
    h, w = img_sq.shape[:2]

    img_uint8 = np.uint8(np.clip(img_sq * 255, 0, 255))
    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB) if img_sq.ndim == 2 else img_uint8.copy()

    # Bicubic upscale → sharper heatmap at full resolution
    hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    hm = np.clip(hm, 0, 1)

    # Adaptive threshold: top 40% of non-trivial activations
    if threshold is None:
        nonzero_vals = hm[hm > 0.05]
        threshold = float(np.percentile(nonzero_vals, 60)) if len(nonzero_vals) > 0 else 0.4

    hm_smooth = cv2.GaussianBlur(hm, (11, 11), 0)
    binary = np.uint8(hm_smooth >= threshold) * 255

    # Morphological cleanup for a clean, filled contour
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)

        # Subtle red fill
        fill_overlay = img_rgb.copy()
        cv2.drawContours(fill_overlay, [largest], -1, (220, 0, 0), thickness=cv2.FILLED)
        img_rgb = cv2.addWeighted(fill_overlay, 0.22, img_rgb, 0.78, 0)

        # Bold red contour (3px)
        cv2.drawContours(img_rgb, [largest], -1, (255, 0, 0), thickness=3)

        # Red bounding box
        x, y, bw, bh = cv2.boundingRect(largest)
        cv2.rectangle(img_rgb, (x, y), (x + bw, y + bh), (255, 0, 0), thickness=2)

        # Label badge
        label_y = max(y - 6, 16)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_rgb, (x, label_y - th - 4), (x + tw + 4, label_y + 2), (255, 0, 0), cv2.FILLED)
        cv2.putText(img_rgb, label, (x + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

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
    Strategy: Otsu threshold -> largest connected component.
    """
    import cv2
    _, thresh = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    if num_labels <= 1:
        return thresh
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    brain_mask = np.uint8(labels == largest_label) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
    return brain_mask


def segment_tumor_mask(img, heatmap):
    """
    Pixel-level tumor segmentation mask constrained inside the brain region.
    img      : (H, W) or (H, W, 1) float [0,1]
    heatmap  : 2-D float [0,1] from make_gradcam_heatmap
    Returns  : (mask_binary uint8, overlay_rgb uint8)
    """
    import cv2
    img_sq = np.squeeze(img)
    h, w = img_sq.shape[:2]
    img_uint8 = np.uint8(np.clip(img_sq * 255, 0, 255))

    brain_mask = _extract_brain_mask(img_uint8)

    # Bicubic upscale for sharper mask
    hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    hm = np.clip(hm, 0, 1)

    hm_brain = hm.copy()
    hm_brain[brain_mask == 0] = 0.0

    hm_u8 = np.uint8(hm_brain * 255)
    _, mask = cv2.threshold(hm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.sum(mask > 0) < 50:
        brain_vals = hm_brain[brain_mask > 0]
        if len(brain_vals) > 0:
            fallback_thresh = np.percentile(brain_vals, 70)
            mask = np.uint8(hm_brain >= fallback_thresh) * 255
            mask[brain_mask == 0] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)
    mask = clean_mask

    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB) if img_sq.ndim == 2 else img_uint8.copy()
    overlay = img_rgb.copy()
    overlay[mask > 0] = [255, 0, 0]
    blended = cv2.addWeighted(overlay, 0.35, img_rgb, 0.65, 0)
    if contours:
        largest_c = max(contours, key=cv2.contourArea)
        cv2.drawContours(blended, [largest_c], -1, (255, 0, 0), thickness=3)
        x, y, bw, bh = cv2.boundingRect(largest_c)
        cv2.rectangle(blended, (x, y), (x + bw, y + bh), (255, 0, 0), thickness=2)

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
