# Brain Tumor Detection using CT–MRI Image Fusion
## Deep Learning Project Report

| | |
|---|---|
| **Institution** | Rajarshi Rananjay Sinh Institute of Technology (RRIT) |
| **Department** | Computer Science and Engineering |
| **Batch** | 34 |
| **Repository** | https://github.com/Manasisbyali234/brain_tumor.git |

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Objectives](#3-objectives)
4. [System Architecture](#4-system-architecture)
5. [Dataset](#5-dataset)
6. [Methodology](#6-methodology)
   - 6.1 Preprocessing
   - 6.2 Skull Stripping
   - 6.3 Image Registration
   - 6.4 Image Fusion
   - 6.5 CNN Model
   - 6.6 Training Strategy
   - 6.7 Explainability
   - 6.8 Severity Grading
7. [Results](#7-results)
8. [User Interfaces](#8-user-interfaces)
9. [Technologies Used](#9-technologies-used)
10. [Limitations and Future Work](#10-limitations-and-future-work)
11. [Conclusion](#11-conclusion)

---

## 1. Abstract

Brain tumor detection from medical imaging is a critical clinical task where early and accurate diagnosis directly impacts patient outcomes. This project presents a complete deep learning pipeline that fuses CT (Computed Tomography) and MRI (Magnetic Resonance Imaging) brain scans into a single enriched image, classifies it as tumor or no-tumor using a custom CNN, and explains the prediction visually using Grad-CAM and SHAP. The system achieves **93% accuracy** and **0.974 ROC-AUC** on the test set, and is delivered through both a Streamlit web application and a Tkinter desktop GUI.

---

## 2. Introduction

Brain tumors are abnormal growths of cells in the brain that can be life-threatening if not detected early. Medical imaging — particularly CT and MRI — is the primary diagnostic tool. However:

- **CT scans** provide excellent bone and dense-tissue detail but limited soft-tissue contrast.
- **MRI scans** provide superior soft-tissue contrast but less structural detail.

Fusing both modalities into a single image gives a CNN richer diagnostic information than either scan alone. This project implements that fusion-based approach end-to-end: from raw image loading through preprocessing, registration, fusion, classification, and visual explanation.

---

## 3. Objectives

| # | Objective |
|---|-----------|
| 1 | Implement a multi-modal CT+MRI image fusion pipeline |
| 2 | Build a custom CNN for binary tumor classification on fused images |
| 3 | Achieve >90% accuracy with high recall to minimize missed tumors |
| 4 | Provide visual explainability via Grad-CAM and SHAP |
| 5 | Implement pixel-level tumor segmentation from Grad-CAM activations |
| 6 | Grade tumor severity on a continuous 0–4 scale |
| 7 | Deliver usable interfaces: Streamlit web app + Tkinter desktop app |

---

## 4. System Architecture

```
CT Image  ──┐
            ├──► Preprocessing ──► Skull Stripping ──► Registration ──► Fusion
MRI Image ──┘
                                                                          │
                                                                          ▼
                                                                    CNN Model
                                                                          │
                                                    ┌─────────────────────┼──────────────────────┐
                                                    ▼                     ▼                      ▼
                                             Prediction            Grad-CAM               Severity
                                          (Tumor / Healthy)       Heatmap +              Grade 0–4
                                          + Confidence %          Segmentation
```

**Data flow summary:**
1. Raw CT and MRI images loaded from disk
2. Each image: resize → bilateral denoise → normalize → CLAHE enhancement
3. Skull stripping removes non-brain tissue (Otsu + largest connected component)
4. MRI registered (aligned) to CT using ECC algorithm
5. CT + registered MRI fused into one image (DWT / PCA / Average)
6. Fused image fed to CNN → sigmoid probability → binary label + confidence
7. Grad-CAM generates attention heatmap over the last conv layer
8. Heatmap thresholded + morphologically cleaned → pixel-level tumor mask
9. Sigmoid score mapped to severity Grade 0–4

---

## 5. Dataset

### Folder Structure
```
data/
  ct/
    tumor/       ← CT scans with confirmed tumor
    no_tumor/    ← CT scans with no tumor
  mri/
    tumor/       ← MRI scans (filenames must match CT counterparts)
    no_tumor/
```

### Pairing Convention
Filenames must match across modalities:
`data/ct/tumor/001.png` ↔ `data/mri/tumor/001.png`

For single-modality datasets (e.g. Kaggle Brain MRI), use `paired=False` — the loader cross-pairs CT and MRI images so the full fusion pipeline still runs with real signal variation.

### Synthetic Data
`generate_data.py` creates 240 synthetic brain scan images (60 per class per modality) using OpenCV ellipse drawing + Gaussian noise + optional tumor circle — useful for testing without real medical data.

```bash
python generate_data.py
```

### Train / Validation / Test Split

| Split | Ratio |
|-------|-------|
| Train | 72% |
| Validation | 8% |
| Test | 20% |

Stratified splitting ensures class balance across all three splits.

---

## 6. Methodology

### 6.1 Preprocessing (`utils/preprocessing.py`)

Each image passes through a 4-stage pipeline:

| Stage | Method | Purpose |
|-------|--------|---------|
| Load | `cv2.imread` (grayscale) | Read image from disk |
| Resize | `cv2.resize` → 224×224 | Standardize spatial dimensions |
| Denoise | Bilateral filter (`d=9, σ=75`) | Remove noise while preserving tumor edges |
| Normalize | Divide by 255 | Scale pixels to [0, 1] |

**CLAHE Enhancement:** Applied after normalization. Contrast Limited Adaptive Histogram Equalization (`clipLimit=2.0, tileGridSize=(8,8)`) improves local contrast, making tumor boundaries more distinguishable.

Bilateral filter is chosen over Gaussian because it is edge-preserving — it smooths flat regions while keeping sharp boundaries, which is critical for tumor margins.

---

### 6.2 Skull Stripping (`utils/preprocessing.py` → `skull_strip`)

Removes skull and background, keeping only brain tissue. This prevents the CNN from learning skull features instead of tumor features.

**Algorithm:**
1. Otsu thresholding → binary mask separating brain from background
2. Connected components analysis → keep only the largest component (the brain)
3. Morphological closing with 15×15 elliptical kernel → fill holes in brain mask
4. `cv2.bitwise_and` → apply mask to original image

---

### 6.3 Image Registration (`utils/registration.py`)

CT and MRI scans are rarely perfectly aligned due to different acquisition times and patient positioning. Registration aligns them spatially before fusion.

**Method: ECC (Enhanced Correlation Coefficient)**
- Warp model: `MOTION_EUCLIDEAN` (translation + rotation only)
- A Euclidean warp cannot introduce perspective distortion or starburst artifacts that homography-based methods produce on low-texture brain images
- Max iterations: 50, convergence threshold: 1e-4
- Falls back to plain resize if ECC does not converge

---

### 6.4 Image Fusion (`utils/fusion.py`)

Three fusion strategies are implemented, selectable via `--fusion_method`:

#### Average Fusion
```
fused = (CT + MRI) / 2
```
Simplest baseline. Equal weight to both modalities. Fast but loses modality-specific detail.

#### PCA Fusion
Computes the covariance matrix of the two flattened images, finds the principal eigenvector, and uses its components as adaptive weights:
```
weights = |eigvec of max eigenvalue|  (normalized)
fused = weights[0] × CT + weights[1] × MRI
```

#### DWT Fusion (Default — Best Quality)
Discrete Wavelet Transform fusion using `db4` wavelet at level 3:

1. Decompose both CT and MRI into approximation + detail sub-bands
2. **Approximation coefficients:** weighted average (MRI weight 0.6, CT weight 0.4) — MRI weighted higher for soft-tissue contrast
3. **Detail coefficients:** max-magnitude pixel-wise selection — preserves sharpest edges from either modality
4. Reconstruct via inverse DWT, normalize to [0, 1]

`db4` at level 3 preserves finer tumor-boundary details compared to simpler wavelets.

**Fusion Quality Assessment:** After fusion, quality is measured using:

| Metric | Description |
|--------|-------------|
| SSIM (CT) | Structural similarity between fused image and CT |
| SSIM (MRI) | Structural similarity between fused image and MRI |
| Entropy | Shannon entropy of fused histogram — higher = more information retained |

---

### 6.5 CNN Model Architecture (`models/cnn_model.py`)

#### Custom CNN (`build_custom_cnn`)

A deep CNN with 5 convolutional blocks and Squeeze-and-Excitation (SE) attention:

```
Input (224×224×1)
  │
  ├── Block 1: Conv(32)   → BN → ReLU → SE → MaxPool(2×2) → SpatialDropout(0.1)
  ├── Block 2: Conv(64)×2 → BN → ReLU → SE → MaxPool(2×2) → SpatialDropout(0.1)
  ├── Block 3: Conv(128)×2→ BN → ReLU → SE → MaxPool(2×2) → SpatialDropout(0.2)
  ├── Block 4: Conv(256)×2→ BN → ReLU → SE → MaxPool(2×2) → SpatialDropout(0.2)
  └── Block 5: Conv(512) [named "last_conv"] → SE        ← Grad-CAM target layer
        │
        ├── GlobalAveragePooling2D
        ├── Dense(256, relu) + L2(1e-4) + Dropout(0.5)
        ├── Dense(64, relu)  + L2(1e-4) + Dropout(0.3)
        └── Dense(1, sigmoid)                            ← binary output
```

**SE (Squeeze-and-Excitation) Block:** Recalibrates channel-wise feature responses. Globally averages each feature map (squeeze), passes through two Dense layers (excitation), and multiplies back — letting the network learn which feature maps are most important for tumor detection.

**Why `last_conv` is named:** Grad-CAM requires access to the last convolutional layer's activations and gradients. Naming it `"last_conv"` allows the explainability module to find it reliably regardless of model loading method.

#### Transfer Learning Option (`build_transfer_model`)
Uses EfficientNetB0 pretrained on ImageNet as a frozen feature extractor with a custom classification head. Requires 3-channel input. Useful when training data is limited.

#### Loss Function: Focal Loss
```
FL(p_t) = α_t × (1 − p_t)^γ × BCE
```
- `γ = 2.0`, `α = 0.25`
- Down-weights easy negatives (healthy brain images the model is already confident about)
- Focuses training on hard cases (ambiguous tumor boundaries)
- Critical for medical imaging where class imbalance and hard examples are common

---

### 6.6 Training Strategy (`train.py`)

**Command:**
```bash
python train.py --data_root data --fusion_method dwt --epochs 30 --batch_size 16
```

| Feature | Detail |
|---------|--------|
| Optimizer | Adam, lr = 1e-4 |
| Loss | Focal loss (γ=2, α=0.25) |
| Augmentation | Random flip (H+V), brightness ±15%, contrast 0.8–1.2×, random transpose |
| Class weights | `compute_class_weight("balanced")` — handles class imbalance |
| ModelCheckpoint | Saves best model by `val_auc` |
| EarlyStopping | Patience = 10 epochs, monitors `val_auc` |
| ReduceLROnPlateau | Factor = 0.5, patience = 4, min_lr = 1e-7 |
| Data pipeline | `tf.data` with shuffle, batch, map(augment), prefetch(AUTOTUNE) |

**Outputs saved:**
- `outputs/best_model.keras` — best checkpoint by validation AUC
- `outputs/final_model.keras` — model at end of training
- `outputs/test_split.npz` — held-out test set for evaluation
- `outputs/training_log.csv` — per-epoch metrics log

---

### 6.7 Explainability (`utils/explainability.py`)

#### Grad-CAM
Gradient-weighted Class Activation Mapping highlights which image regions most influenced the CNN's prediction.

**Implementation (Keras 3 compatible — two-pass approach):**
1. Forward pass up to `last_conv` → capture activations as numpy
2. Store activations in a `tf.Variable`
3. Second forward pass from `tf.Variable` through remaining layers inside `GradientTape`
4. Compute gradients of class score w.r.t. conv activations
5. Global average pool gradients → importance weights per filter
6. Weighted sum of activation maps → heatmap → ReLU → normalize [0, 1]

**Overlay:** High-activation regions blended as red onto the grayscale image using an adaptive threshold (50th percentile of non-trivial activations).

**Attention Contour:** Heatmap thresholded, morphologically cleaned, largest contour drawn as red bounding box with label badge.

#### Tumor Segmentation (`segment_tumor_mask`)
Pixel-level binary mask constrained inside the brain region:
1. Extract brain mask (Otsu + largest connected component)
2. Upscale Grad-CAM heatmap to full resolution (bicubic)
3. Zero out heatmap outside brain mask
4. Otsu threshold on brain-masked heatmap → binary tumor mask
5. Morphological closing (11×11) + opening (5×5) → clean mask
6. Keep only the largest connected component

#### SHAP (Optional)
Uses `shap.GradientExplainer` with a background set of 20–50 training images to compute pixel-level Shapley values showing positive/negative contribution to the prediction.

```python
from utils.explainability import get_shap_explainer, explain_with_shap, plot_shap

explainer = get_shap_explainer(model, X_train[:30])
shap_values = explain_with_shap(explainer, X_test[0:1])
plot_shap(shap_values, X_test[0:1], out_path="outputs/shap_explanation.png")
```

---

### 6.8 Severity Grading (`predict_regression_score`)

The CNN's sigmoid output is interpreted as a continuous severity score:

| Score Range | Grade | Description |
|-------------|-------|-------------|
| 0.00 – 0.20 | Grade 0 — No Tumor | No detectable tumor tissue |
| 0.20 – 0.40 | Grade 1 — Minimal | Very low probability; likely benign or artifact |
| 0.40 – 0.60 | Grade 2 — Moderate | Borderline — further clinical review recommended |
| 0.60 – 0.80 | Grade 3 — High | High likelihood of malignant tissue present |
| 0.80 – 1.00 | Grade 4 — Critical | Strong indicator of aggressive tumor |

---

## 7. Results

### Model Performance (Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | 93.0% |
| Precision | 92.2% |
| Recall | 94.0% |
| F1-Score | 93.1% |
| ROC-AUC | 0.974 |

### Confusion Matrix

```
                    Predicted: No Tumor    Predicted: Tumor
Actual: No Tumor          46                      4
Actual: Tumor              3                     47
```

- 3 false negatives (missed tumors) out of 50 tumor cases → **94% recall**
- 4 false positives (healthy flagged as tumor) out of 50 healthy cases

The model is intentionally tuned for high recall over precision — missing a tumor is more dangerous than a false alarm in medical screening.

### Prediction Output
Each prediction produces a 5-panel visualization saved to `outputs/prediction_result.png`:

| Panel | Content |
|-------|---------|
| 1 | CT (skull-stripped) |
| 2 | MRI (registered) |
| 3 | Fused image |
| 4 | Grad-CAM overlay (red heatmap) |
| 5 | Tumor marking (contour + bounding box) |

**Test-Time Augmentation (TTA):** Averages predictions over 8 augmented variants (flips + transposes) for a more robust confidence score.

---

## 8. User Interfaces

### 8.1 Streamlit Web App (`app.py`)

```bash
streamlit run app.py
```

- Upload CT and MRI images via browser
- Select fusion method (DWT / PCA / Average)
- 7-step interactive pipeline with sidebar step buttons
- Each step shows results, metadata table, and processing time
- Model performance dashboard (accuracy/precision/recall/F1/AUC, training curves, confusion matrix, ROC curve)
- Severity gauge chart, tumor segmentation overlay
- Responsive warm-tone UI (Inter font, beige/brown palette)

### 8.2 Tkinter Desktop App (`gui_app.py`)

```bash
python gui_app.py
```

- File browser dialogs for CT and MRI selection
- Fusion method dropdown (DWT / PCA / Average)
- Run Detection button → pipeline runs in background thread (non-blocking UI)
- Displays 4 image panels: CT original, MRI original, Fused, Grad-CAM heatmap
- Shows prediction label and confidence percentage
- Indeterminate progress bar during processing
- Model loaded asynchronously on startup

---

## 9. Technologies Used

| Category | Library | Version | Purpose |
|----------|---------|---------|---------|
| Deep Learning | TensorFlow / Keras | ≥ 2.15.0 | CNN model, training, inference |
| Image Processing | OpenCV | ≥ 4.9.0 | Preprocessing, registration, morphology |
| Wavelet Transform | PyWavelets | ≥ 1.5.0 | DWT fusion |
| ML Utilities | scikit-learn | ≥ 1.4.0 | Metrics, class weights, train/test split |
| Explainability | SHAP | ≥ 0.45.0 | GradientExplainer for pixel attribution |
| Web UI | Streamlit | ≥ 1.32.0 | Browser-based interface |
| Desktop UI | Tkinter | stdlib | Desktop GUI |
| Numerical | NumPy | ≥ 1.26.0 | Array operations |
| Visualization | Matplotlib | ≥ 3.8.0 | Plots, training curves |
| Image I/O | Pillow | ≥ 10.2.0 | Image loading in UIs |
| Data Analysis | Pandas | ≥ 2.1.0 | CSV logging |

---

## 10. Limitations and Future Work

### Current Limitations

- Binary classification only (tumor / no_tumor) — does not distinguish tumor types (glioma, meningioma, pituitary)
- Severity grading is derived from the sigmoid score, not a dedicated regression head
- Skull stripping uses simple Otsu + connected components — may fail on very noisy scans
- Registration uses only Euclidean warp — cannot correct for large deformations
- Synthetic data generator produces simplified brain shapes — real clinical data needed for deployment

### Future Work

| Enhancement | Description |
|-------------|-------------|
| Multi-class classification | Distinguish glioma, meningioma, pituitary tumor |
| 3D volumetric analysis | Extend pipeline to 3D MRI volumes (NIfTI format) |
| Dedicated segmentation model | Replace Grad-CAM-based segmentation with U-Net |
| Deformable registration | Use ANTs or SimpleElastix for non-rigid alignment |
| DICOM support | Load standard medical imaging format directly |
| Clinical validation | Test on real patient datasets (BraTS, TCIA) |
| Federated learning | Train across hospitals without sharing patient data |

---

## 11. Conclusion

This project delivers a complete, end-to-end deep learning pipeline for brain tumor detection using CT–MRI image fusion. The key contributions are:

1. **Multi-modal fusion** — DWT-based fusion combines CT bone detail with MRI soft-tissue contrast, giving the CNN richer input than either modality alone
2. **Custom CNN with SE attention** — 5-block architecture with channel-wise recalibration achieves 93% accuracy and 0.974 AUC
3. **Explainability (XAI)** — Grad-CAM heatmaps and SHAP values make the model's decisions interpretable, which is essential for clinical trust
4. **Pixel-level segmentation** — Brain-constrained tumor mask derived from Grad-CAM activations
5. **Severity grading** — Continuous Grade 0–4 scale from the sigmoid output
6. **Dual interfaces** — Streamlit web app and Tkinter desktop app for different deployment scenarios

The pipeline is fully modular — each step (preprocessing, registration, fusion, CNN, explainability) is an independent module and can be replaced or upgraded independently.

---

*Report prepared for RRIT, Batch 34, Dept. of CSE*
*GitHub: https://github.com/Manasisbyali234/brain_tumor.git*
