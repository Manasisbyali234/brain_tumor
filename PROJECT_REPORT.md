# Brain Tumor Detection using CT–MRI Image Fusion
## Project Report

**Institution:** RRIT (Rajarshi Rananjay Sinh Institute of Technology)
**Department:** Computer Science and Engineering
**Batch:** 34
**Repository:** https://github.com/Manasisbyali234/brain_tumor.git

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objectives](#2-objectives)
3. [System Architecture](#3-system-architecture)
4. [Dataset](#4-dataset)
5. [Pipeline — Step by Step](#5-pipeline--step-by-step)
   - 5.1 Data Collection
   - 5.2 Preprocessing
   - 5.3 Skull Stripping
   - 5.4 Image Registration
   - 5.5 Image Fusion
   - 5.6 Fusion Quality Assessment
   - 5.7 CNN Model Architecture
   - 5.8 Model Training
   - 5.9 Model Evaluation
   - 5.10 Prediction
   - 5.11 Grad-CAM Explainability
   - 5.12 Tumor Segmentation
   - 5.13 Severity Grading
6. [Model Performance](#6-model-performance)
7. [Explainability (XAI)](#7-explainability-xai)
8. [User Interfaces](#8-user-interfaces)
9. [Folder Structure](#9-folder-structure)
10. [Technologies Used](#10-technologies-used)
11. [How to Run](#11-how-to-run)
12. [Key Design Decisions](#12-key-design-decisions)
13. [Limitations and Future Work](#13-limitations-and-future-work)
14. [Conclusion](#14-conclusion)

---

## 1. Project Overview

Brain tumor detection is a critical task in medical imaging. Early and accurate detection directly impacts patient survival rates. This project implements a complete deep learning pipeline that:

- Takes paired **CT** (Computed Tomography) and **MRI** (Magnetic Resonance Imaging) brain scans as input
- **Fuses** them into a single enriched image using signal processing techniques
- **Classifies** the fused image as `tumor` or `no_tumor` using a custom Convolutional Neural Network (CNN)
- **Explains** the prediction visually using Grad-CAM heatmaps and SHAP values
- **Segments** the tumor region at pixel level
- **Grades** tumor severity on a 0–4 scale
- Provides both a **desktop GUI** (Tkinter) and a **web application** (Streamlit)

The fusion of CT and MRI is the core innovation — CT provides bone and dense-tissue detail while MRI provides superior soft-tissue contrast. Combining both modalities gives the CNN richer information than either scan alone.

---

## 2. Objectives

| # | Objective |
|---|-----------|
| 1 | Implement a multi-modal medical image fusion pipeline (CT + MRI) |
| 2 | Build a custom CNN for binary tumor classification on fused images |
| 3 | Achieve high accuracy (>90%) with strong recall to minimize missed tumors |
| 4 | Provide visual explainability via Grad-CAM and SHAP |
| 5 | Implement pixel-level tumor segmentation from Grad-CAM activations |
| 6 | Grade tumor severity continuously (Grade 0–4) |
| 7 | Deliver usable interfaces: Streamlit web app + Tkinter desktop app |

---

## 3. System Architecture

```
CT Image ──┐
           ├──► Preprocessing ──► Skull Stripping ──► Registration ──► Fusion ──► CNN ──► Prediction
MRI Image ─┘                                                                         │
                                                                                     ├──► Grad-CAM ──► Segmentation
                                                                                     └──► Severity Grade
```

**Data flow:**
1. Raw CT and MRI images are loaded from disk
2. Each image is preprocessed (resize → denoise → normalize → CLAHE)
3. Skull stripping removes non-brain tissue
4. MRI is registered (aligned) to CT using ECC
5. CT and registered MRI are fused into one image (DWT / PCA / Average)
6. The fused image is fed to the CNN
7. CNN outputs a sigmoid probability → binary prediction + confidence
8. Grad-CAM generates a heatmap of the CNN's attention
9. The heatmap is used to segment the tumor region
10. The sigmoid score is mapped to a severity grade

---

## 4. Dataset

### Structure
```
data/
  ct/
    tumor/      ← CT scans with confirmed tumor
    no_tumor/   ← CT scans with no tumor
  mri/
    tumor/      ← MRI scans with confirmed tumor (paired filenames with CT)
    no_tumor/   ← MRI scans with no tumor
```

### Pairing Convention
- Filenames must match across modalities: `data/ct/tumor/001.png` ↔ `data/mri/tumor/001.png`
- If only single-modality data is available (e.g. Kaggle Brain MRI dataset), set `paired=False` — the loader cross-pairs CT and MRI images so the fusion pipeline still runs end-to-end with real signal variation

### Synthetic Data Generator
For testing without real medical data, `generate_data.py` creates 240 synthetic brain scan images (60 per class per modality) using OpenCV ellipse drawing + Gaussian noise + optional tumor circle.

```bash
python generate_data.py
```

### Train / Validation / Test Split
| Split | Ratio |
|-------|-------|
| Train | 72%   |
| Validation | 8% |
| Test  | 20%   |

Stratified splitting ensures class balance across all three splits.

---

## 5. Pipeline — Step by Step

### 5.1 Data Collection (`utils/dataset.py`)

Two dataset builders are provided:

- `build_fused_dataset()` — loads CT+MRI pairs, runs the full preprocess→register→fuse pipeline, returns `(X, y)` numpy arrays with shape `(N, 224, 224, 1)`
- `build_mri_dataset()` — loads MRI images directly without fusion, for single-modality baselines

Both support a `limit_per_class` argument to cap the number of images per class (useful for quick experiments).

---

### 5.2 Preprocessing (`utils/preprocessing.py`)

Each image goes through a 4-stage pipeline:

| Stage | Method | Purpose |
|-------|--------|---------|
| Load | `cv2.imread` (grayscale) | Read image from disk |
| Resize | `cv2.resize` → 224×224 | Standardize spatial dimensions |
| Denoise | Bilateral filter (`d=9, σ=75`) | Remove noise while preserving tumor edges |
| Normalize | Divide by 255 | Scale pixels to [0, 1] |

**CLAHE Enhancement** (`clahe_enhance`): Applied after normalization. Contrast Limited Adaptive Histogram Equalization improves local contrast, making tumor boundaries more distinguishable. Uses `clipLimit=2.0, tileGridSize=(8,8)`.

Bilateral filter is chosen over Gaussian because it is edge-preserving — it smooths flat regions while keeping sharp boundaries (important for tumor margins).

---

### 5.3 Skull Stripping (`utils/preprocessing.py` → `skull_strip`)

Skull stripping removes the skull and background, keeping only brain tissue. This prevents the CNN from learning skull features instead of tumor features.

**Algorithm:**
1. Otsu thresholding → binary mask separating brain from background
2. Connected components analysis → keep only the largest component (the brain)
3. Morphological closing with a 15×15 elliptical kernel → fill holes in the brain mask
4. `cv2.bitwise_and` → apply mask to original image

---

### 5.4 Image Registration (`utils/registration.py`)

CT and MRI scans of the same patient are rarely perfectly aligned due to different acquisition times and patient positioning. Registration aligns them spatially before fusion.

**Method: ECC (Enhanced Correlation Coefficient)**
- Warp model: `MOTION_EUCLIDEAN` (translation + rotation only)
- A Euclidean warp cannot introduce perspective distortion or starburst artifacts that homography-based methods produce on low-texture brain images
- Max iterations: 50, convergence threshold: 1e-4
- Falls back to plain resize if ECC does not converge

```python
cv2.findTransformECC(fixed_u8, moving_resized, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
cv2.warpAffine(moving_resized, warp_matrix, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
```

---

### 5.5 Image Fusion (`utils/fusion.py`)

Three fusion strategies are implemented, selectable via `--fusion_method`:

#### Average Fusion
```
fused = (CT + MRI) / 2
```
Simplest baseline. Equal weight to both modalities. Fast but loses modality-specific detail.

#### PCA Fusion
Computes the covariance matrix of the two flattened images, finds the principal eigenvector, and uses its components as weights:
```
weights = |eigvec of max eigenvalue|  (normalized)
fused = weights[0] * CT + weights[1] * MRI
```
Adapts weights based on variance contribution of each modality.

#### DWT Fusion (Default — Best Quality)
Discrete Wavelet Transform fusion using `db4` wavelet at level 3:

1. Decompose both CT and MRI into approximation + detail sub-bands
2. **Approximation coefficients:** weighted average (MRI weight 0.6, CT weight 0.4) — MRI weighted higher for soft-tissue contrast
3. **Detail coefficients:** max-magnitude pixel-wise selection — preserves the sharpest edges from either modality
4. Reconstruct via inverse DWT
5. Normalize output to [0, 1]

`db4` at level 3 preserves finer tumor-boundary details compared to simpler wavelets.

---

### 5.6 Fusion Quality Assessment (`utils/fusion.py` → `fusion_quality_metrics`)

After fusion, quality is measured using:

| Metric | Description |
|--------|-------------|
| SSIM (CT) | Structural Similarity between fused image and CT — measures how much CT structure is preserved |
| SSIM (MRI) | Structural Similarity between fused image and MRI — measures how much MRI structure is preserved |
| Entropy | Shannon entropy of the fused image histogram — higher entropy = more information retained |

---

### 5.7 CNN Model Architecture (`models/cnn_model.py`)

#### Custom CNN (`build_custom_cnn`)

A deep CNN with 5 convolutional blocks and Squeeze-and-Excitation (SE) attention:

```
Input (224×224×1)
  │
  ├── Block 1: Conv(32) → BN → ReLU → SE → MaxPool(2×2) → SpatialDropout(0.1)
  ├── Block 2: Conv(64)×2 → BN → ReLU → SE → MaxPool(2×2) → SpatialDropout(0.1)
  ├── Block 3: Conv(128)×2 → BN → ReLU → SE → MaxPool(2×2) → SpatialDropout(0.2)
  ├── Block 4: Conv(256)×2 → BN → ReLU → SE → MaxPool(2×2) → SpatialDropout(0.2)
  └── Block 5: Conv(512) [named "last_conv"] → SE  ← Grad-CAM target layer
        │
        ├── GlobalAveragePooling2D
        ├── Dense(256, relu) + L2(1e-4) + Dropout(0.5)
        ├── Dense(64, relu) + L2(1e-4) + Dropout(0.3)
        └── Dense(1, sigmoid)  ← binary output
```

**SE (Squeeze-and-Excitation) Block:** Recalibrates channel-wise feature responses. Globally averages each feature map (squeeze), passes through two Dense layers (excitation), and multiplies back — letting the network learn which feature maps are most important for tumor detection.

**Why `last_conv` is named:** Grad-CAM requires access to the last convolutional layer's activations and gradients. Naming it `"last_conv"` allows the explainability module to find it reliably.

#### Transfer Learning Option (`build_transfer_model`)
Uses EfficientNetB0 pretrained on ImageNet as a frozen feature extractor, with a custom classification head. Requires 3-channel input. Useful when training data is limited.

#### Loss Function: Focal Loss
```
FL(p_t) = α_t × (1 - p_t)^γ × BCE
```
- `γ = 2.0`, `α = 0.25`
- Down-weights easy negatives (healthy brain images the model is already confident about)
- Focuses training on hard cases (ambiguous tumor boundaries)
- Critical for medical imaging where class imbalance and hard examples are common

---

### 5.8 Model Training (`train.py`)

**Command:**
```bash
python train.py --data_root data --fusion_method dwt --epochs 30 --batch_size 16
```

**Training features:**

| Feature | Detail |
|---------|--------|
| Optimizer | Adam, lr=1e-4 |
| Loss | Focal loss (γ=2, α=0.25) |
| Augmentation | Random flip (H+V), brightness ±15%, contrast 0.8–1.2×, random transpose |
| Class weights | `compute_class_weight("balanced")` — handles class imbalance |
| Callbacks | ModelCheckpoint (best val_auc), EarlyStopping (patience=10), ReduceLROnPlateau (factor=0.5, patience=4) |
| Data pipeline | `tf.data` with shuffle, batch, map(augment), prefetch(AUTOTUNE) |

**Outputs saved:**
- `outputs/best_model.h5` — best checkpoint by validation AUC
- `outputs/final_model.h5` — model at end of training
- `outputs/test_split.npz` — held-out test set for evaluation
- `outputs/history.npy` — training history dictionary
- `outputs/training_log.csv` — per-epoch metrics log

---

### 5.9 Model Evaluation (`evaluate.py`)

**Command:**
```bash
python evaluate.py --model_path outputs/best_model.h5 --test_split outputs/test_split.npz
```

**Metrics computed:**

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions / total |
| Precision | TP / (TP + FP) — of predicted tumors, how many are real |
| Recall | TP / (TP + FN) — of real tumors, how many are caught |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | Area under the ROC curve — threshold-independent performance |
| Confusion Matrix | Visual breakdown of TP, TN, FP, FN |

**Outputs saved:**
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`

---

### 5.10 Prediction (`predict.py`)

**Command:**
```bash
python predict.py --ct path/to/ct.png --mri path/to/mri.png --model outputs/best_model.h5
```

Runs the full pipeline on a new CT+MRI pair and saves a 5-panel visualization:

| Panel | Content |
|-------|---------|
| 1 | CT (skull-stripped) |
| 2 | MRI (registered) |
| 3 | Fused image |
| 4 | Grad-CAM overlay (red heatmap) |
| 5 | Tumor marking (contour + bounding box) |

**Test-Time Augmentation (TTA):** Averages predictions over 8 augmented variants (flips + transposes) for a more robust confidence score.

**Threshold:** 0.35 (instead of 0.5) — biased toward recall to minimize missed tumors (false negatives are more dangerous than false positives in medical diagnosis).

---

### 5.11 Grad-CAM Explainability (`utils/explainability.py`)

Gradient-weighted Class Activation Mapping (Grad-CAM) highlights which regions of the input image most influenced the CNN's prediction.

**Implementation (Keras 3 compatible):**
1. Forward pass up to `last_conv` layer → capture activations as numpy
2. Store activations in a `tf.Variable`
3. Second forward pass from `tf.Variable` through remaining layers inside `GradientTape`
4. Compute gradients of the class score w.r.t. the conv activations
5. Global average pool the gradients → importance weights per filter
6. Weighted sum of activation maps → heatmap
7. ReLU + normalize to [0, 1]

This two-pass approach avoids `layer.output` API issues with loaded Keras 3 models.

**Overlay:** High-activation regions are blended as red onto the grayscale image. An adaptive threshold (50th percentile of non-trivial activations) ensures only the most relevant regions are highlighted.

**Attention Contour:** The heatmap is thresholded, morphologically cleaned, and the largest contour is drawn as a red bounding box with label badge.

---

### 5.12 Tumor Segmentation (`utils/explainability.py` → `segment_tumor_mask`)

Pixel-level binary mask of the tumor region, constrained inside the brain:

1. Extract brain mask via Otsu + largest connected component
2. Upscale Grad-CAM heatmap to full image resolution (bicubic)
3. Zero out heatmap outside brain mask
4. Otsu threshold on brain-masked heatmap → binary tumor mask
5. Fallback: if mask is too small (<50 px), use 70th percentile threshold
6. Morphological closing (11×11) + opening (5×5) → clean mask
7. Keep only the largest connected component

Output: binary mask + RGB overlay with red tumor region and contour.

---

### 5.13 Severity Grading (`utils/explainability.py` → `predict_regression_score`)

The CNN's sigmoid output is interpreted as a continuous severity score:

| Score Range | Grade | Description |
|-------------|-------|-------------|
| 0.00 – 0.20 | Grade 0 — No Tumor | No detectable tumor tissue |
| 0.20 – 0.40 | Grade 1 — Minimal | Very low probability; likely benign or artifact |
| 0.40 – 0.60 | Grade 2 — Moderate | Borderline — further clinical review recommended |
| 0.60 – 0.80 | Grade 3 — High | High likelihood of malignant tissue present |
| 0.80 – 1.00 | Grade 4 — Critical | Strong indicator of aggressive tumor |

---

## 6. Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 93.0% |
| Precision | 92.2% |
| Recall | 94.0% |
| F1-Score | 93.1% |
| ROC-AUC | 0.974 |

**Confusion Matrix (Test Set):**
```
                Predicted: No Tumor    Predicted: Tumor
Actual: No Tumor       46                    4
Actual: Tumor           3                   47
```

- 3 false negatives (missed tumors) out of 50 tumor cases → 94% recall
- 4 false positives (healthy flagged as tumor) out of 50 healthy cases

The model is intentionally tuned for high recall (catching tumors) over precision, which is the correct priority in medical screening.

---

## 7. Explainability (XAI)

### Grad-CAM
- Produces a spatial heatmap showing which pixels drove the prediction
- Overlaid in red on the fused image
- Attention contour draws a bounding box around the highest-activation region
- Implemented without relying on `model.output` — compatible with Keras 3 saved models

### SHAP (SHapley Additive exPlanations)
- Uses `shap.GradientExplainer` with a background set of 20–50 training images
- Computes pixel-level Shapley values showing positive/negative contribution to the prediction
- Optional — not required for the main pipeline

```python
from utils.explainability import get_shap_explainer, explain_with_shap, plot_shap

explainer = get_shap_explainer(model, X_train[:30])
shap_values = explain_with_shap(explainer, X_test[0:1])
plot_shap(shap_values, X_test[0:1], out_path="outputs/shap_explanation.png")
```

---

## 8. User Interfaces

### 8.1 Streamlit Web App (`app.py`)

Run with: `streamlit run app.py`

**Features:**
- Upload CT and MRI images via browser
- Select fusion method (DWT / PCA / Average)
- 7-step interactive pipeline with sidebar step buttons
- Each step shows results, metadata table, and processing time
- Model performance dashboard (accuracy/precision/recall/F1/AUC pills, training curves, confusion matrix, ROC curve)
- Severity gauge chart
- Tumor segmentation overlay
- Responsive warm-tone UI (Inter font, beige/brown palette)

**Pipeline steps in the UI:**
1. Preprocessing — shows CT and MRI after CLAHE
2. Registration — shows CT, original MRI, registered MRI side by side
3. Fusion — shows CT, registered MRI, fused image
4. CNN Prediction — prediction card with confidence bar charts
5. Grad-CAM Explanation — 4-panel view with heatmap legend
6. Regression Analysis — severity score with color-coded gauge
7. Tumor Segmentation — binary mask + red overlay

### 8.2 Tkinter Desktop App (`gui_app.py`)

Run with: `python gui_app.py`

**Features:**
- File browser dialogs for CT and MRI selection
- Fusion method dropdown (DWT / PCA / Average)
- Run Detection button → runs pipeline in a background thread (non-blocking UI)
- Displays 4 image panels: CT original, MRI original, Fused, Grad-CAM heatmap
- Shows prediction label and confidence percentage
- Indeterminate progress bar during processing
- Model loaded asynchronously on startup

---

## 9. Folder Structure

```
brain_tumor_project/
├── app.py                  # Streamlit web UI
├── gui_app.py              # Tkinter desktop UI
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── predict.py              # Inference + output display
├── generate_data.py        # Synthetic data generator
├── requirements.txt        # Python dependencies
├── README.md               # Quick-start guide
├── PROJECT_REPORT.md       # This document
├── models/
│   ├── __init__.py
│   └── cnn_model.py        # CNN architecture + focal loss
├── utils/
│   ├── __init__.py
│   ├── dataset.py          # Data loading + fused dataset builder
│   ├── preprocessing.py    # Resize, denoise, normalize, CLAHE, skull strip
│   ├── registration.py     # ECC-based image registration
│   ├── fusion.py           # Average / PCA / DWT fusion + quality metrics
│   └── explainability.py   # Grad-CAM, SHAP, segmentation, severity grading
├── data/
│   ├── ct/
│   │   ├── tumor/
│   │   └── no_tumor/
│   └── mri/
│       ├── tumor/
│       └── no_tumor/
└── outputs/
    ├── best_model.h5
    ├── final_model.h5
    ├── test_split.npz
    ├── history.npy
    ├── training_log.csv
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── prediction_result.png
```

---

## 10. Technologies Used

| Category | Library / Tool | Version | Purpose |
|----------|---------------|---------|---------|
| Deep Learning | TensorFlow / Keras | ≥2.15.0 | CNN model, training, inference |
| Image Processing | OpenCV | ≥4.9.0 | Preprocessing, registration, morphology |
| Wavelet Transform | PyWavelets | ≥1.5.0 | DWT fusion |
| ML Utilities | scikit-learn | ≥1.4.0 | Metrics, class weights, train/test split |
| Explainability | SHAP | ≥0.45.0 | GradientExplainer for pixel attribution |
| Web UI | Streamlit | ≥1.32.0 | Browser-based interface |
| Desktop UI | Tkinter | stdlib | Desktop GUI |
| Numerical | NumPy | ≥1.26.0 | Array operations |
| Visualization | Matplotlib | ≥3.8.0 | Plots, training curves |
| Image I/O | Pillow | ≥10.2.0 | Image loading in UIs |
| Data Analysis | Pandas | ≥2.1.0 | CSV logging |
| Version Control | Git / GitHub | — | Source control |

---

## 11. How to Run

### Setup
```bash
git clone https://github.com/Manasisbyali234/brain_tumor.git
cd brain_tumor
pip install -r requirements.txt
```

### Generate Synthetic Data (for testing)
```bash
python generate_data.py
```

### Train
```bash
python train.py --data_root data --fusion_method dwt --epochs 30 --batch_size 16
```

### Evaluate
```bash
python evaluate.py --model_path outputs/best_model.h5 --test_split outputs/test_split.npz
```

### Predict on a New Image Pair
```bash
python predict.py --ct data/ct/tumor/001.png --mri data/mri/tumor/001.png --model outputs/best_model.h5
```

### Web App
```bash
streamlit run app.py
```

### Desktop App
```bash
python gui_app.py
```

---

## 12. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| DWT fusion as default | Preserves both low-frequency (anatomy) and high-frequency (tumor edges) information better than average or PCA |
| ECC registration over ORB+RANSAC | Euclidean warp cannot produce starburst/perspective artifacts on low-texture brain images |
| Focal loss over BCE | Down-weights easy negatives; focuses training on hard tumor boundary cases |
| SE attention blocks | Channel-wise recalibration helps the CNN focus on tumor-relevant feature maps |
| Threshold 0.35 for prediction | Biases toward recall — missing a tumor is more dangerous than a false alarm |
| TTA (8 augmented variants) | More robust confidence estimate at inference time |
| Named `last_conv` layer | Ensures Grad-CAM always targets the correct layer regardless of model loading method |
| Two-pass Grad-CAM | Avoids `layer.output` API incompatibilities with Keras 3 saved models |
| Brain-constrained segmentation | Prevents tumor mask from extending outside the brain region |

---

## 13. Limitations and Future Work

### Current Limitations
- Binary classification only (tumor / no_tumor) — does not distinguish tumor types
- Severity grading is derived from the sigmoid score, not a dedicated regression head
- Skull stripping uses simple Otsu + connected components — may fail on very noisy scans
- Registration uses only Euclidean warp — cannot correct for large deformations
- Synthetic data generator produces simplified brain shapes — real data needed for clinical use

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

## 14. Conclusion

This project delivers a complete, end-to-end deep learning pipeline for brain tumor detection using CT–MRI image fusion. The key contributions are:

1. **Multi-modal fusion** — DWT-based fusion combines CT bone detail with MRI soft-tissue contrast, giving the CNN richer input than either modality alone
2. **Custom CNN with SE attention** — 5-block architecture with channel-wise recalibration achieves 93% accuracy and 0.974 AUC
3. **Explainability** — Grad-CAM heatmaps and SHAP values make the model's decisions interpretable, which is essential for clinical trust
4. **Pixel-level segmentation** — Brain-constrained tumor mask derived from Grad-CAM activations
5. **Severity grading** — Continuous Grade 0–4 scale from the sigmoid output
6. **Dual interfaces** — Both a Streamlit web app and a Tkinter desktop app for different deployment scenarios

The pipeline is modular — each step (preprocessing, registration, fusion, CNN, explainability) is implemented as an independent module and can be replaced or upgraded independently.

---

*Report generated for RRIT, Batch 34, Dept. of CSE*
*GitHub: https://github.com/Manasisbyali234/brain_tumor.git*
