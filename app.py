"""
app.py  —  Streamlit Web Interface
Brain Tumor Detection using CT–MRI Image Fusion (Deep Learning)
Run with:  streamlit run app.py
"""

import os
import time
import tempfile
import traceback

import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from utils.preprocessing import preprocess_image, clahe_enhance
from utils.registration import register_images
from utils.fusion import fuse_images
from utils.explainability import make_gradcam_heatmap, overlay_gradcam, draw_attention_contour, segment_tumor_mask, predict_regression_score

MODEL_PATH = "outputs/best_model.h5"
CLASS_NAMES = ["Healthy Brain", "Tumor"]

STEPS = [
    "Preprocessing",
    "Registration",
    "Fusion",
    "CNN Prediction",
    "Grad-CAM Explanation",
    "Regression Analysis",
    "Tumor Segmentation",
]

PREREQS = {
    "Preprocessing": None,
    "Registration": "Preprocessing",
    "Fusion": "Registration",
    "CNN Prediction": "Fusion",
    "Grad-CAM Explanation": "CNN Prediction",
    "Regression Analysis": "CNN Prediction",
    "Tumor Segmentation": "Grad-CAM Explanation",
}

STEP_NUMS = {s: i + 1 for i, s in enumerate(STEPS)}

st.set_page_config(
    page_title="Brain Tumor Detection - CT/MRI Fusion",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_model_performance():
    """Show training graphs + saved evaluation plots in an expander."""
    history_path = "outputs/history.npy"
    cm_path = "outputs/confusion_matrix.png"
    roc_path = "outputs/roc_curve.png"

    has_history = os.path.exists(history_path)
    has_cm = os.path.exists(cm_path)
    has_roc = os.path.exists(roc_path)

    if not (has_history or has_cm or has_roc):
        return

    with st.expander("📊 Model Performance & Training Graphs", expanded=False):
        st.markdown(
            "These graphs are generated from the last training run. "
            "Run `python train.py` followed by `python evaluate.py` to refresh them."
        )

        if has_history:
            hist = np.load(history_path, allow_pickle=True).item()
            epochs = range(1, len(hist.get("accuracy", hist.get("acc", []))) + 1)
            acc     = hist.get("accuracy",     hist.get("acc",      []))
            val_acc = hist.get("val_accuracy",  hist.get("val_acc", []))
            loss    = hist.get("loss",    [])
            val_loss= hist.get("val_loss", [])

            col_a, col_b = st.columns(2)

            with col_a:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.plot(epochs, acc,     "o-", color="#2563eb", label="Train Accuracy",      linewidth=2, markersize=4)
                ax.plot(epochs, val_acc, "s--",color="#16a34a", label="Validation Accuracy", linewidth=2, markersize=4)
                ax.set_title("Training vs Validation Accuracy", fontsize=11, fontweight="bold")
                ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
                ax.set_ylim(0, 1.05)
                ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    "The accuracy curves show how well the model learns over epochs. "
                    "A small gap between train and validation accuracy indicates good generalisation."
                )

            with col_b:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.plot(epochs, loss,     "o-", color="#dc2626", label="Train Loss",      linewidth=2, markersize=4)
                ax.plot(epochs, val_loss, "s--",color="#ea580c", label="Validation Loss", linewidth=2, markersize=4)
                ax.set_title("Training vs Validation Loss", fontsize=11, fontweight="bold")
                ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (Binary Cross-Entropy)")
                ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    "Decreasing loss confirms the model is learning. "
                    "If validation loss rises while training loss falls, the model is overfitting."
                )

        if has_cm or has_roc:
            col_c, col_d = st.columns(2)
            if has_cm:
                with col_c:
                    st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
                    st.caption(
                        "The confusion matrix shows true positives, true negatives, "
                        "false positives, and false negatives on the held-out test set."
                    )
            if has_roc:
                with col_d:
                    st.image(roc_path, caption="ROC Curve", use_container_width=True)
                    st.caption(
                        "The ROC curve plots sensitivity vs. 1-specificity. "
                        "An AUC close to 1.0 indicates excellent discriminative ability."
                    )


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def save_upload_to_temp(uploaded_file):
    uploaded_file.seek(0)
    suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def init_state():
    defaults = {
        "ct_img": None, "mri_img": None,
        "mri_registered": None, "fused": None,
        "tumor_prob": None,          # raw sigmoid output [0,1]
        "pred_class": None,          # 0 or 1
        "heatmap": None, "overlay": None, "contour_img": None,
        "input_tensor": None,
        "step_status": {},
        "step_error": {},
        "step_meta": {},             # step -> dict of extra info to display
        "active_step": None,         # last successfully-run step
        "pending_step": None,
        "ct_path": None, "mri_path": None,
        "ct_name": None,
        "fusion_method": "dwt",
        "regression_result": None,
        "seg_mask": None,
        "seg_overlay": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_pipeline():
    for k in ["ct_img", "mri_img", "mri_registered", "fused",
              "tumor_prob", "pred_class",
              "heatmap", "overlay", "contour_img", "input_tensor",
              "regression_result", "seg_mask", "seg_overlay"]:
        st.session_state[k] = None
    st.session_state.step_status = {}
    st.session_state.step_error = {}
    st.session_state.step_meta = {}
    st.session_state.active_step = None
    st.session_state.pending_step = None


def step_done(step):
    return st.session_state.step_status.get(step) == "completed"


def can_run(step, ct_ready):
    if not ct_ready:
        return False
    prereq = PREREQS[step]
    if prereq is None:
        return True
    return step_done(prereq)


def execute_step(step, model):
    s = st.session_state
    t0 = time.time()

    try:
        if step == "Preprocessing":
            ct = clahe_enhance(preprocess_image(s.ct_path))
            mri = clahe_enhance(preprocess_image(s.mri_path))
            s.ct_img = ct
            s.mri_img = mri
            s.step_meta[step] = {
                "CT shape": f"{ct.shape[1]}×{ct.shape[0]} px",
                "MRI shape": f"{mri.shape[1]}×{mri.shape[0]} px",
                "CT pixel range": f"[{ct.min():.4f}, {ct.max():.4f}]",
                "MRI pixel range": f"[{mri.min():.4f}, {mri.max():.4f}]",
                "CT mean intensity": f"{ct.mean():.4f}",
                "MRI mean intensity": f"{mri.mean():.4f}",
                "CT std deviation": f"{ct.std():.4f}",
                "MRI std deviation": f"{mri.std():.4f}",
                "Total pixels (each)": f"{ct.shape[0] * ct.shape[1]:,}",
                "Non-zero CT pixels": f"{int(np.count_nonzero(ct)):,} ({100*np.count_nonzero(ct)/ct.size:.1f}%)",
                "Non-zero MRI pixels": f"{int(np.count_nonzero(mri)):,} ({100*np.count_nonzero(mri)/mri.size:.1f}%)",
            }

        elif step == "Registration":
            registered = register_images(s.mri_img, s.ct_img)
            s.mri_registered = registered
            # Compute pixel-level alignment quality metrics
            ct_u8 = (s.ct_img * 255).clip(0, 255).astype(np.uint8)
            reg_u8 = (registered * 255).clip(0, 255).astype(np.uint8)
            mri_u8 = (s.mri_img * 255).clip(0, 255).astype(np.uint8)
            diff_before = float(np.mean(np.abs(ct_u8.astype(np.float32) - mri_u8.astype(np.float32))))
            diff_after = float(np.mean(np.abs(ct_u8.astype(np.float32) - reg_u8.astype(np.float32))))
            improvement = ((diff_before - diff_after) / (diff_before + 1e-8)) * 100
            s.mri_registered = registered
            s.step_meta[step] = {
                "Fixed image": "CT",
                "Moving image": "MRI",
                "Output shape": f"{registered.shape[1]}×{registered.shape[0]} px",
                "Registered pixel range": f"[{registered.min():.4f}, {registered.max():.4f}]",
                "Registered mean intensity": f"{registered.mean():.4f}",
                "Mean pixel diff (before reg)": f"{diff_before:.2f}",
                "Mean pixel diff (after reg)": f"{diff_after:.2f}",
                "Alignment improvement": f"{improvement:.1f}%",
            }

        elif step == "Fusion":
            method = s.fusion_method
            fused = fuse_images(s.ct_img, s.mri_registered, method=method)
            s.fused = fused
            # Real computed fusion quality metrics
            ct_f = s.ct_img.astype(np.float32)
            mri_f = s.mri_registered.astype(np.float32)
            # Entropy of fused image (information content)
            hist, _ = np.histogram(fused, bins=256, range=(0, 1))
            hist_prob = hist / (hist.sum() + 1e-8)
            entropy = float(-np.sum(hist_prob[hist_prob > 0] * np.log2(hist_prob[hist_prob > 0] + 1e-8)))
            # Contrast (std of fused vs inputs)
            ct_contrast = float(ct_f.std())
            mri_contrast = float(mri_f.std())
            fused_contrast = float(fused.std())
            # Mean structural similarity proxy: correlation
            corr_ct = float(np.corrcoef(ct_f.flatten(), fused.flatten())[0, 1])
            corr_mri = float(np.corrcoef(mri_f.flatten(), fused.flatten())[0, 1])
            s.step_meta[step] = {
                "Method": method.upper(),
                "Output shape": f"{fused.shape[1]}×{fused.shape[0]} px",
                "Fused pixel range": f"[{fused.min():.4f}, {fused.max():.4f}]",
                "Fused mean intensity": f"{fused.mean():.4f}",
                "Fused std (contrast)": f"{fused_contrast:.4f}",
                "CT input std": f"{ct_contrast:.4f}",
                "MRI input std": f"{mri_contrast:.4f}",
                "Fused image entropy (bits)": f"{entropy:.3f}",
                "Correlation with CT": f"{corr_ct:.4f}",
                "Correlation with MRI": f"{corr_mri:.4f}",
            }

        elif step == "CNN Prediction":
            if model is None:
                raise RuntimeError(
                    f"No trained model found at '{MODEL_PATH}'. "
                    "Run `python train.py --no_fusion` first."
                )
            # Use MRI image directly (model trained on MRI, not fused)
            src = s.mri_img if s.fused is None else s.fused
            tensor = src[np.newaxis, :, :, np.newaxis].astype(np.float32)
            s.input_tensor = tensor
            raw_prob = float(model.predict(tensor, verbose=0)[0][0])
            # raw_prob is the sigmoid output = P(tumor)
            s.tumor_prob = raw_prob
            s.pred_class = int(raw_prob >= 0.4)
            confidence = raw_prob if s.pred_class == 1 else 1.0 - raw_prob
            s.step_meta[step] = {
                "Raw sigmoid output": f"{raw_prob:.6f}",
                "Tumor probability": f"{raw_prob * 100:.2f}%",
                "Normal probability": f"{(1 - raw_prob) * 100:.2f}%",
                "Prediction": CLASS_NAMES[s.pred_class],
                "Confidence": f"{confidence * 100:.2f}%",
                "Confidence level": (
                    "HIGH" if confidence >= 0.85
                    else "MEDIUM" if confidence >= 0.65
                    else "LOW"
                ),
            }

        elif step == "Grad-CAM Explanation":
            if model is None:
                raise RuntimeError(f"No trained model found at '{MODEL_PATH}'.")
            heatmap = make_gradcam_heatmap(
                s.input_tensor, model, last_conv_layer_name="last_conv"
            )
            s.heatmap = heatmap
            src = s.mri_img if s.fused is None else s.fused
            s.overlay = overlay_gradcam(src, heatmap)
            s.contour_img = draw_attention_contour(src, heatmap)
            tumor_pct = s.tumor_prob * 100
            hm = heatmap.astype(np.float32)
            peak_y, peak_x = np.unravel_index(np.argmax(hm), hm.shape)
            high_attention_pct = float(np.mean(hm > 0.5) * 100)
            heatmap_entropy_hist, _ = np.histogram(hm, bins=64, range=(0, 1))
            hm_prob = heatmap_entropy_hist / (heatmap_entropy_hist.sum() + 1e-8)
            hm_entropy = float(-np.sum(hm_prob[hm_prob > 0] * np.log2(hm_prob[hm_prob > 0] + 1e-8)))
            s.step_meta[step] = {
                "Prediction": CLASS_NAMES[s.pred_class],
                "Tumor probability": f"{tumor_pct:.2f}%",
                "Normal probability": f"{(100 - tumor_pct):.2f}%",
                "Heatmap min activation": f"{hm.min():.4f}",
                "Heatmap max activation": f"{hm.max():.4f}",
                "Heatmap mean activation": f"{hm.mean():.4f}",
                "Peak attention location": f"({peak_x}, {peak_y}) px",
                "High-attention area (>0.5)": f"{high_attention_pct:.1f}% of image",
                "Heatmap entropy (bits)": f"{hm_entropy:.3f}",
            }

        elif step == "Regression Analysis":
            if model is None:
                raise RuntimeError(f"No trained model found at '{MODEL_PATH}'.")
            reg = predict_regression_score(model, s.input_tensor)
            s.regression_result = reg
            score = reg["score"]
            # Severity bar breakdown (5 bands)
            s.step_meta[step] = {
                "Severity Score (0–1)": f"{score:.4f}",
                "Grade": reg["grade"],
                "Interpretation": reg["description"],
                "Score as %": f"{score * 100:.2f}%",
                "Band": (
                    "0.00–0.20 (No Tumor)" if score < 0.2 else
                    "0.20–0.40 (Minimal)" if score < 0.4 else
                    "0.40–0.60 (Moderate)" if score < 0.6 else
                    "0.60–0.80 (High)" if score < 0.8 else
                    "0.80–1.00 (Critical)"
                ),
            }

        elif step == "Tumor Segmentation":
            if s.heatmap is None:
                raise RuntimeError("Run 'Grad-CAM Explanation' first.")
            src = s.mri_img if s.fused is None else s.fused
            mask, seg_overlay = segment_tumor_mask(src, s.heatmap)
            s.seg_mask = mask
            s.seg_overlay = seg_overlay
            tumor_pixels = int(np.sum(mask > 0))
            total_pixels = mask.size
            coverage_pct = tumor_pixels / total_pixels * 100
            s.step_meta[step] = {
                "Segmentation method": "Brain-masked Grad-CAM + Otsu threshold + morphological closing",
                "Threshold": "Otsu adaptive (brain region only)",
                "Tumor pixels": f"{tumor_pixels:,}",
                "Total pixels": f"{total_pixels:,}",
                "Tumor area coverage": f"{coverage_pct:.2f}%",
                "Mask shape": f"{mask.shape[1]}×{mask.shape[0]} px",
            }

        elapsed = time.time() - t0
        s.step_meta[step]["Processing time"] = f"{elapsed:.2f}s"
        s.step_status[step] = "completed"
        s.active_step = step
        s.step_error.pop(step, None)

    except Exception as exc:
        s.step_status[step] = "failed"
        s.step_error[step] = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(ct_ready):
    with st.sidebar:
        st.title("⚙️ Pipeline Steps")
        st.markdown(
            "Run each step **in order** after uploading both images. "
            "Each step feeds its output into the next."
        )
        st.markdown("---")

        s = st.session_state
        for step in STEPS:
            status = s.step_status.get(step, "pending")
            runnable = can_run(step, ct_ready)
            is_active = s.active_step == step

            if status == "completed":
                icon = "✅"
            elif status == "running":
                icon = "⏳"
            elif status == "failed":
                icon = "❌"
            else:
                icon = "▶️" if runnable else "🔒"

            label = f"{icon} Step {STEP_NUMS[step]}: {step}"
            if is_active:
                label = f"**{label}**"

            col_btn, col_retry = st.columns([4, 1])
            with col_btn:
                clicked = st.button(
                    label,
                    key=f"btn_{step}",
                    disabled=(not runnable) or (status == "running"),
                    use_container_width=True,
                )
            with col_retry:
                retry = False
                if status == "failed":
                    retry = st.button("↺", key=f"retry_{step}",
                                      help="Retry this step", use_container_width=True)

            if status == "failed":
                st.caption(f"❗ {s.step_error.get(step, '')[:80]}…")

            if (clicked or retry) and runnable:
                s.pending_step = step
                s.step_status[step] = "running"
                st.rerun()

        st.markdown("---")
        st.selectbox(
            "Fusion method", ["dwt", "pca", "average"],
            index=["dwt", "pca", "average"].index(s.fusion_method),
            key="fusion_method",
        )

        if ct_ready and st.button("🔄 Reset Pipeline", use_container_width=True):
            reset_pipeline()
            st.rerun()

        # Overall progress
        completed = sum(1 for step in STEPS if step_done(step))
        st.markdown("---")
        st.caption(f"Progress: {completed}/{len(STEPS)} steps completed")
        st.progress(completed / len(STEPS))

        st.markdown("---")
        st.markdown(
            "**About**\n\n"
            "CNN trained on fused CT+MRI images. "
            "Fusion methods: DWT (wavelet), PCA, Average. "
            "Explainability via Grad-CAM on the last convolutional layer."
        )


# ── Result renderers ──────────────────────────────────────────────────────────

def _meta_table(meta: dict):
    """Render a clean key-value info table."""
    rows = "".join(
        f"<tr>"
        f"<td style='color:#1a1a1a;padding:6px 16px 6px 10px;white-space:nowrap;"
        f"font-weight:600;border-bottom:1px solid #d0d0d0;'>{k}</td>"
        f"<td style='color:#111111;padding:6px 10px;font-weight:500;"
        f"border-bottom:1px solid #d0d0d0;'>{v}</td>"
        f"</tr>"
        for k, v in meta.items()
    )
    st.markdown(
        f"<table style='border-collapse:collapse;font-size:.88rem;line-height:1.6;"
        f"border:1px solid #c0c0c0;border-radius:6px;overflow:hidden;"
        f"background:#ffffff;width:auto;'>{rows}</table>",
        unsafe_allow_html=True,
    )


def _status_badge(text, color):
    st.markdown(
        f"<span style='background:{color};color:#fff;padding:3px 10px;"
        f"border-radius:12px;font-size:.8rem;font-weight:700;'>{text}</span>",
        unsafe_allow_html=True,
    )


def render_preprocessing(s):
    st.markdown("### 🔬 Step 1 — Preprocessing")
    _status_badge("✓ Completed", "#43a047")
    st.markdown("")
    c1, c2 = st.columns(2)
    c1.image(s.ct_img, caption="CT — preprocessed", clamp=True, use_container_width=True)
    c2.image(s.mri_img, caption="MRI — preprocessed", clamp=True, use_container_width=True)
    st.markdown("**Preprocessing Details**")
    _meta_table(s.step_meta.get("Preprocessing", {}))
    st.success(
        "✓ CT image preprocessed  ·  ✓ MRI image preprocessed  ·  "
        "✓ Normalized to [0,1]  ·  ✓ Resized to 224×224  ·  ✓ Gaussian denoised"
    )


def render_registration(s):
    st.markdown("### 📐 Step 2 — Registration")
    _status_badge("✓ Completed", "#43a047")
    st.markdown("")
    c1, c2, c3 = st.columns(3)
    c1.image(s.ct_img, caption="CT (fixed reference)", clamp=True, use_container_width=True)
    c2.image(s.mri_img, caption="MRI (original)", clamp=True, use_container_width=True)
    c3.image(s.mri_registered, caption="MRI registered to CT", clamp=True, use_container_width=True)
    st.markdown("**Registration Details**")
    _meta_table(s.step_meta.get("Registration", {}))
    st.success("✓ CT and MRI successfully aligned using ORB + RANSAC homography")


def render_fusion(s):
    method = s.fusion_method.upper()
    st.markdown(f"### 🔀 Step 3 — Image Fusion ({method})")
    _status_badge("✓ Completed", "#43a047")
    st.markdown("")
    c1, c2, c3 = st.columns(3)
    c1.image(s.ct_img, caption="CT input", clamp=True, use_container_width=True)
    c2.image(s.mri_registered, caption="Registered MRI input", clamp=True, use_container_width=True)
    c3.image(s.fused, caption=f"Fused image ({method})", clamp=True, use_container_width=True)
    st.markdown("**Fusion Details**")
    _meta_table(s.step_meta.get("Fusion", {}))
    st.success(f"✓ CT-MRI fusion completed using {method} method — fused image ready for CNN")


def render_prediction(s):
    st.markdown("### 🧠 Step 4 — CNN Prediction")
    _status_badge("✓ Completed", "#43a047")
    st.markdown("")

    tumor_prob = s.tumor_prob
    normal_prob = 1.0 - tumor_prob
    tumor_pct = tumor_prob * 100
    normal_pct = normal_prob * 100
    is_tumor = s.pred_class == 1
    label = CLASS_NAMES[s.pred_class]
    confidence = tumor_pct if is_tumor else normal_pct
    conf_level = (
        "HIGH" if confidence >= 85
        else "MEDIUM" if confidence >= 65
        else "LOW"
    )
    is_low_conf = abs(tumor_pct - normal_pct) < 10

    pred_color  = "#c62828" if is_tumor else "#2e7d32"
    pred_bg     = "#fff5f5" if is_tumor else "#f1f8f1"
    pred_icon   = "🧠" if is_tumor else "✅"
    conf_color  = "#2e7d32" if conf_level == "HIGH" else "#e65100" if conf_level == "MEDIUM" else "#b71c1c"
    conf_bg     = "#e8f5e9" if conf_level == "HIGH" else "#fff3e0" if conf_level == "MEDIUM" else "#fce4ec"
    low_conf_block = (
        f"<div style='margin-top:14px;padding:10px 14px;background:#fff8e1;"
        f"border-left:4px solid #f9a825;border-radius:6px;font-size:.85rem;color:#5d4037;'>"
        f"⚠️ <strong>Low confidence prediction</strong> — "
        f"Model prediction is low confidence because the class probabilities are closely balanced."
        f"</div>"
    ) if is_low_conf else ""

    col_card, col_img = st.columns([3, 2])

    with col_card:
        st.markdown(
            f"""
            <style>
              @media (max-width: 768px) {{
                .pred-layout {{ flex-direction: column !important; }}
              }}
            </style>
            <div style="border:1px solid #e0e0e0;border-radius:14px;padding:24px 28px;
                        background:#ffffff;box-shadow:0 2px 12px rgba(0,0,0,.08);
                        margin-bottom:16px;">

              <div style="font-size:.78rem;font-weight:700;color:#757575;
                          letter-spacing:.12em;margin-bottom:14px;">CNN DETECTION RESULT</div>

              <div style="background:{pred_bg};border-radius:10px;padding:16px 20px;
                          margin-bottom:18px;">
                <div style="font-size:2rem;font-weight:800;color:{pred_color};
                            line-height:1.2;">{pred_icon} {label}</div>
              </div>

              <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;
                          flex-wrap:wrap;">
                <div style="background:{conf_bg};border-radius:8px;padding:8px 16px;
                            text-align:center;">
                  <div style="font-size:.72rem;font-weight:700;color:#757575;
                              letter-spacing:.08em;margin-bottom:2px;">CONFIDENCE</div>
                  <div style="font-size:1.4rem;font-weight:800;color:{conf_color};
                              line-height:1;">{confidence:.2f}%</div>
                </div>
                <div style="background:{conf_bg};border-radius:8px;padding:8px 16px;
                            text-align:center;">
                  <div style="font-size:.72rem;font-weight:700;color:#757575;
                              letter-spacing:.08em;margin-bottom:2px;">LEVEL</div>
                  <div style="font-size:1rem;font-weight:800;color:{conf_color};
                              line-height:1.4;">{conf_level}</div>
                </div>
              </div>

              <div style="margin-bottom:6px;">
                <div style="display:flex;justify-content:space-between;
                            font-size:.85rem;font-weight:600;color:#424242;
                            margin-bottom:5px;">
                  <span>Tumor</span>
                  <span style="color:#c62828;">{tumor_pct:.2f}%</span>
                </div>
                <div style="background:#f5f5f5;border-radius:6px;height:18px;
                            overflow:hidden;">
                  <div style="width:{min(tumor_pct,100):.2f}%;height:100%;
                              background:#e53935;border-radius:6px;
                              transition:width .4s ease;"></div>
                </div>
              </div>

              <div style="margin-top:12px;margin-bottom:4px;">
                <div style="display:flex;justify-content:space-between;
                            font-size:.85rem;font-weight:600;color:#424242;
                            margin-bottom:5px;">
                  <span>Normal</span>
                  <span style="color:#2e7d32;">{normal_pct:.2f}%</span>
                </div>
                <div style="background:#f5f5f5;border-radius:6px;height:18px;
                            overflow:hidden;">
                  <div style="width:{min(normal_pct,100):.2f}%;height:100%;
                              background:#43a047;border-radius:6px;
                              transition:width .4s ease;"></div>
                </div>
              </div>

              {low_conf_block}

              <div style="margin-top:18px;padding:8px 12px;background:#f5f5f5;
                          border-radius:6px;font-size:.8rem;color:#616161;">
                ⓘ Model probabilities are generated from the trained CNN output.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("📋 Model Output Details", expanded=False):
            _meta_table(s.step_meta.get("CNN Prediction", {}))

    with col_img:
        display_img = s.fused if s.fused is not None else s.mri_img
        st.image(
            display_img,
            caption="Image Used for CNN Prediction",
            clamp=True,
            use_container_width=True,
        )


def render_gradcam(s):
    st.markdown("### 🔥 Step 5 — Grad-CAM Explanation")
    _status_badge("✓ Completed", "#43a047")
    st.markdown("")

    tumor_prob = s.tumor_prob
    tumor_pct = tumor_prob * 100
    normal_pct = 100 - tumor_pct
    is_tumor = s.pred_class == 1
    label = CLASS_NAMES[s.pred_class]
    border_color = "#e53935" if is_tumor else "#43a047"
    icon = "🔴" if is_tumor else "🟢"

    src = s.fused if s.fused is not None else s.mri_img
    c1, c2, c3, c4 = st.columns(4)
    c1.image(s.ct_img, caption="Original CT", clamp=True, use_container_width=True)
    c2.image(src, caption="CNN input", clamp=True, use_container_width=True)
    c3.image(s.contour_img, caption="Model Attention Region", use_container_width=True)
    c4.image(s.overlay, caption="Grad-CAM Heatmap", use_container_width=True)

    st.markdown(
        """
        <div style="display:flex;gap:18px;align-items:center;
                    margin:10px 0 14px;flex-wrap:wrap;">
          <span style="font-size:.78rem;color:#9e9e9e;font-weight:700;
                       letter-spacing:.08em;">HEATMAP LEGEND</span>
          <span style="display:flex;align-items:center;gap:5px;font-size:.82rem;color:#e0e0e0;">
            <span style="display:inline-block;width:14px;height:14px;
                         background:#1a237e;border-radius:3px;"></span>Low attention
          </span>
          <span style="display:flex;align-items:center;gap:5px;font-size:.82rem;color:#e0e0e0;">
            <span style="display:inline-block;width:14px;height:14px;
                         background:#00bcd4;border-radius:3px;"></span>Medium attention
          </span>
          <span style="display:flex;align-items:center;gap:5px;font-size:.82rem;color:#e0e0e0;">
            <span style="display:inline-block;width:14px;height:14px;
                         background:#ff9800;border-radius:3px;"></span>High attention
          </span>
          <span style="display:flex;align-items:center;gap:5px;font-size:.82rem;color:#e0e0e0;">
            <span style="display:inline-block;width:14px;height:14px;
                         background:#e53935;border-radius:3px;"></span>Peak attention
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_info, col_note = st.columns([2, 3])
    with col_info:
        st.markdown(
            f"""
            <div style="border:2px solid {border_color};border-radius:10px;
                        padding:16px 20px;background:#1a1a2e;">
              <div style="font-size:.82rem;font-weight:700;color:#9e9e9e;
                          letter-spacing:.08em;margin-bottom:8px;">GRAD-CAM SUMMARY</div>
              <div style="font-size:1.4rem;font-weight:800;color:{border_color};
                          margin-bottom:12px;">{icon} {label}</div>
              <div style="font-size:.88rem;color:#bdbdbd;line-height:1.7;">
                Tumor probability: <strong style="color:#e0e0e0;">{tumor_pct:.2f}%</strong><br>
                Normal probability: <strong style="color:#e0e0e0;">{normal_pct:.2f}%</strong>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_note:
        st.info(
            "**Interpretation:** The highlighted region shows where the CNN focused "
            "most when making its prediction. This is a **Model Attention Region** "
            "generated by Grad-CAM — it is NOT an exact tumor boundary or segmentation mask. "
            "Use it as an explainability aid alongside clinical judgment."
        )
    st.markdown("**Grad-CAM Details**")
    _meta_table(s.step_meta.get("Grad-CAM Explanation", {}))


def render_regression(s):
    st.markdown("### 📈 Step 6 — Regression Analysis (Severity Score)")
    _status_badge("✓ Completed", "#43a047")
    st.markdown("")

    reg = s.regression_result
    score = reg["score"]
    score_pct = score * 100

    # Color by severity
    if score < 0.2:
        bar_color, card_color = "#43a047", "#e8f5e9"
    elif score < 0.4:
        bar_color, card_color = "#8bc34a", "#f1f8e9"
    elif score < 0.6:
        bar_color, card_color = "#ff9800", "#fff3e0"
    elif score < 0.8:
        bar_color, card_color = "#f44336", "#fff5f5"
    else:
        bar_color, card_color = "#b71c1c", "#fce4ec"

    col_card, col_gauge = st.columns([3, 2])
    with col_card:
        st.markdown(
            f"""
            <div style="border:1px solid #e0e0e0;border-radius:14px;padding:24px 28px;
                        background:#ffffff;box-shadow:0 2px 12px rgba(0,0,0,.08);">
              <div style="font-size:.78rem;font-weight:700;color:#757575;
                          letter-spacing:.12em;margin-bottom:14px;">REGRESSION — TUMOR SEVERITY SCORE</div>
              <div style="background:{card_color};border-radius:10px;padding:16px 20px;margin-bottom:18px;">
                <div style="font-size:2.4rem;font-weight:900;color:{bar_color};line-height:1;">{score:.4f}</div>
                <div style="font-size:.9rem;color:#555;margin-top:6px;">{reg['grade']}</div>
              </div>
              <div style="margin-bottom:8px;font-size:.85rem;color:#424242;">{reg['description']}</div>
              <div style="background:#f5f5f5;border-radius:8px;height:22px;overflow:hidden;margin-top:14px;">
                <div style="width:{min(score_pct,100):.2f}%;height:100%;background:{bar_color};
                            border-radius:8px;transition:width .4s ease;"></div>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:.75rem;color:#9e9e9e;margin-top:4px;">
                <span>0.0 — No Tumor</span><span>0.5 — Moderate</span><span>1.0 — Critical</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_gauge:
        # Simple matplotlib gauge bar
        fig, ax = plt.subplots(figsize=(4, 2.5))
        bands = [(0.2, "#43a047"), (0.2, "#8bc34a"), (0.2, "#ff9800"), (0.2, "#f44336"), (0.2, "#b71c1c")]
        left = 0
        for width, color in bands:
            ax.barh(0, width, left=left, color=color, height=0.5)
            left += width
        ax.axvline(score, color="black", linewidth=3, label=f"Score: {score:.3f}")
        ax.set_xlim(0, 1); ax.set_yticks([])
        ax.set_xlabel("Severity Score")
        ax.set_title("Severity Gauge", fontsize=10, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.markdown("**Regression Details**")
    _meta_table(s.step_meta.get("Regression Analysis", {}))
    st.info(
        "**Note:** Regression re-uses the CNN sigmoid output as a continuous severity score. "
        "It is NOT trained as a separate regression head — it maps the tumor probability "
        "to a 5-grade clinical scale for interpretability."
    )


def render_segmentation(s):
    st.markdown("### 🗺️ Step 7 — Tumor Segmentation (Pixel-level Mask)")
    _status_badge("✓ Completed", "#43a047")
    st.markdown("")

    src = s.fused if s.fused is not None else s.mri_img
    c1, c2, c3 = st.columns(3)
    c1.image(src, caption="CNN Input Image", clamp=True, use_container_width=True)
    c2.image(s.seg_mask, caption="Binary Tumor Mask", clamp=True, use_container_width=True)
    c3.image(s.seg_overlay, caption="Segmentation Overlay", use_container_width=True)

    st.markdown(
        """
        <div style="display:flex;gap:18px;align-items:center;margin:10px 0 14px;flex-wrap:wrap;">
          <span style="font-size:.78rem;color:#9e9e9e;font-weight:700;letter-spacing:.08em;">LEGEND</span>
          <span style="display:flex;align-items:center;gap:5px;font-size:.82rem;">
            <span style="display:inline-block;width:14px;height:14px;background:#ffffff;
                         border:1px solid #ccc;border-radius:3px;"></span>Background (no tumor)
          </span>
          <span style="display:flex;align-items:center;gap:5px;font-size:.82rem;">
            <span style="display:inline-block;width:14px;height:14px;background:#dc1e1e;
                         border-radius:3px;"></span>Tumor region (predicted)
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Segmentation Details**")
    _meta_table(s.step_meta.get("Tumor Segmentation", {}))
    st.info(
        "**Note:** The brain region is first isolated using Otsu thresholding + largest connected component. "
        "The Grad-CAM heatmap is then zeroed outside the brain, and an adaptive Otsu threshold is applied "
        "on the brain-only heatmap values to find the tumor boundary. "
        "This prevents false positives on the background."
    )


RENDERERS = {
    "Preprocessing": render_preprocessing,
    "Registration": render_registration,
    "Fusion": render_fusion,
    "CNN Prediction": render_prediction,
    "Grad-CAM Explanation": render_gradcam,
    "Regression Analysis": render_regression,
    "Tumor Segmentation": render_segmentation,
}


def render_results(s):
    """Show the active step result prominently, then collapsed history below."""
    active = s.active_step
    if active is None:
        return

    st.markdown("---")

    # Active step — always expanded, highlighted
    status = s.step_status.get(active)
    if status == "failed":
        st.error(f"❌ **{active}** failed:\n\n{s.step_error.get(active, 'Unknown error')}")
    elif status == "completed":
        RENDERERS[active](s)

    # Previous completed steps — collapsed
    prev_steps = [step for step in STEPS if step != active and s.step_status.get(step) == "completed"]
    if prev_steps:
        st.markdown("---")
        st.markdown("#### Previous Steps")
        for step in prev_steps:
            with st.expander(f"✅ Step {STEP_NUMS[step]}: {step}", expanded=False):
                RENDERERS[step](s)

    # Failed steps (other than active)
    failed_steps = [step for step in STEPS if step != active and s.step_status.get(step) == "failed"]
    for step in failed_steps:
        with st.expander(f"❌ Step {STEP_NUMS[step]}: {step} — Failed", expanded=True):
            st.error(s.step_error.get(step, "Unknown error"))
            st.caption("Click ↺ in the sidebar to retry.")


def main():
    init_state()
    model = load_model()
    s = st.session_state

    st.title("🧠 Brain Tumor Detection — CT/MRI Image Fusion Pipeline")
    st.markdown(
        "This tool detects brain tumors by fusing CT and MRI scans using deep learning. "
        "The pipeline aligns and combines both modalities to produce a richer input for the CNN, "
        "then uses **Grad-CAM** to highlight the regions that influenced the prediction."
    )
    st.caption("Preprocessing → Registration → Fusion → CNN → Grad-CAM")

    render_model_performance()

    if model is None:
        st.warning(
            f"⚠️ No trained model found at `{MODEL_PATH}`. "
            "Run `python train.py` first, then refresh."
        )

    col1, col2 = st.columns(2)
    with col1:
        ct_file = st.file_uploader("📂 Upload CT Image", type=["png", "jpg", "jpeg"])
    with col2:
        mri_file = st.file_uploader("📂 Upload MRI Image", type=["png", "jpg", "jpeg"])

    ct_ready = ct_file is not None and mri_file is not None

    if ct_ready:
        if s.ct_name != ct_file.name:
            s.ct_path = save_upload_to_temp(ct_file)
            s.mri_path = save_upload_to_temp(mri_file)
            s.ct_name = ct_file.name
            reset_pipeline()

        ct_file.seek(0)
        mri_file.seek(0)
        st.image(
            [Image.open(ct_file), Image.open(mri_file)],
            caption=["CT (original upload)", "MRI (original upload)"],
            width=260,
        )
        st.caption("👈 Use the sidebar to run each pipeline step in order.")
    else:
        st.info("👆 Upload both a CT and an MRI image, then click a pipeline step on the left.")

    render_sidebar(ct_ready)

    # Execute pending step (triggered by sidebar button click)
    pending = s.pending_step
    if pending is not None:
        s.pending_step = None
        prereq = PREREQS[pending]
        if prereq and not step_done(prereq):
            s.step_status[pending] = "failed"
            s.step_error[pending] = (
                f"Please complete '{prereq}' before running '{pending}'."
            )
            s.active_step = pending
        else:
            with st.spinner(f"⏳ Running: {pending}…"):
                execute_step(pending, model)
        st.rerun()

    render_results(s)


if __name__ == "__main__":
    main()
