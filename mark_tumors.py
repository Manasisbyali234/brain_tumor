"""
mark_tumors.py
--------------
Loads CT images, runs them through the trained model, and saves
a neatly annotated output for each image detected as tumor.
"""

import os
import numpy as np
import cv2
import tensorflow as tf

from utils.preprocessing import preprocess_image, clahe_enhance, skull_strip
from models.cnn_model import focal_loss

IMAGES = [
    "ct_tumor (1).jpg",
    "ct_tumor (1).png",
    "ct_tumor (2).jpg",
    "ct_tumor (2).png",
]
OUT_DIR = "outputs/marked"
THRESHOLD = 0.35


def load_model():
    # .keras needs focal_loss registered; rebuild + load weights avoids all issues
    from models.cnn_model import build_custom_cnn, compile_model
    model = build_custom_cnn(input_shape=(224, 224, 1))
    compile_model(model)
    # Load weights only — skips optimizer/loss deserialization entirely
    model.load_weights("outputs/best_model.keras")
    return model


def gradcam_heatmap(inp, model):
    """Simple Grad-CAM using a sub-model up to last_conv."""
    last_conv = model.get_layer("last_conv")
    grad_model = tf.keras.Model(inputs=model.input,
                                outputs=[last_conv.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inp, training=False)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    mx = tf.reduce_max(heatmap)
    if mx > 0:
        heatmap = heatmap / mx
    return heatmap.numpy()


def mark_image(img, heatmap, confidence, grade):
    """Draw contour + bounding box + labels on the image."""
    img_sq = np.squeeze(img)
    h, w = img_sq.shape[:2]
    img_u8 = np.uint8(np.clip(img_sq * 255, 0, 255))
    canvas = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)

    hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    hm = np.clip(hm, 0, 1)
    hm_blur = cv2.GaussianBlur(hm, (11, 11), 0)

    nonzero = hm_blur[hm_blur > 0.05]
    thresh = float(np.percentile(nonzero, 60)) if len(nonzero) > 0 else 0.4
    binary = np.uint8(hm_blur >= thresh) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)

        # Subtle red fill
        fill = canvas.copy()
        cv2.drawContours(fill, [largest], -1, (220, 0, 0), cv2.FILLED)
        canvas = cv2.addWeighted(fill, 0.22, canvas, 0.78, 0)

        # Bold red contour
        cv2.drawContours(canvas, [largest], -1, (255, 0, 0), 3)

        # Bounding box
        x, y, bw, bh = cv2.boundingRect(largest)
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (255, 0, 0), 2)

        # Label badge
        label = f"Tumor ({confidence*100:.1f}%)"
        label_y = max(y - 6, 16)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x, label_y - th - 4), (x + tw + 4, label_y + 2),
                      (255, 0, 0), cv2.FILLED)
        cv2.putText(canvas, label, (x + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Severity banner at bottom
    banner = np.zeros((36, w, 3), dtype=np.uint8)
    cv2.putText(banner, grade, (6, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 1, cv2.LINE_AA)
    return np.vstack([canvas, banner])


def process(img_path, model):
    img = clahe_enhance(preprocess_image(img_path))
    img = skull_strip(img)
    inp = np.expand_dims(img, axis=(0, -1)).astype(np.float32)

    prob = float(model.predict(inp, verbose=0)[0][0])
    is_tumor = prob >= THRESHOLD
    confidence = prob if is_tumor else 1 - prob

    if not is_tumor:
        return None, prob, confidence

    heatmap = gradcam_heatmap(inp, model)

    raw = prob
    if raw < 0.4:   grade = "Grade 1 — Minimal"
    elif raw < 0.6: grade = "Grade 2 — Moderate"
    elif raw < 0.8: grade = "Grade 3 — High"
    else:           grade = "Grade 4 — Critical"

    marked = mark_image(img, heatmap, confidence, f"{grade}  |  Score: {raw:.3f}")
    return marked, prob, confidence


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_model()

    for fname in IMAGES:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
        if not os.path.exists(path):
            print(f"[SKIP] Not found: {fname}")
            continue

        marked, prob, conf = process(path, model)
        label = "TUMOR" if prob >= THRESHOLD else "NO TUMOR"
        print(f"{fname:30s}  ->  {label}  (conf: {conf*100:.1f}%)")

        if marked is not None:
            stem = os.path.splitext(fname)[0].replace(" ", "_")
            out_path = os.path.join(OUT_DIR, f"{stem}_marked.png")
            cv2.imwrite(out_path, cv2.cvtColor(marked, cv2.COLOR_RGB2BGR))
            print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
