# Brain Tumor Detection using CT–MRI Image Fusion (Deep Learning)

> RRIT, Batch 34, Dept. of CSE — Deep Learning pipeline for brain tumor detection via CT+MRI fusion.

---

## What This Project Does

- Fuses CT and MRI brain scans using Average / PCA / DWT (wavelet) methods
- Classifies fused images as **tumor** or **no_tumor** using a CNN
- Explains predictions visually using **Grad-CAM** and **SHAP**
- Provides both a desktop GUI (Tkinter) and a web UI (Streamlit)

---

## Pipeline Steps

| Step | Description | File |
|------|-------------|------|
| 1 | Data Collection | `utils/dataset.py` |
| 2 | Preprocessing (resize, denoise, normalize) | `utils/preprocessing.py` |
| 2b | Skull Stripping (remove skull/background, keep brain only) | `utils/preprocessing.py` |
| 3 | Image Registration (align CT ↔ MRI via ECC) | `utils/registration.py` |
| 4 | Image Fusion (average / PCA / DWT) | `utils/fusion.py` |
| 4b | Fusion Quality Assessment (SSIM + entropy metrics) | `utils/fusion.py` |
| 5 | CNN Model Architecture | `models/cnn_model.py` |
| 6 | Model Training (augmentation, class weights, callbacks) | `train.py` |
| 7 | Model Evaluation (accuracy, F1, ROC-AUC, confusion matrix) | `evaluate.py` |
| 8 | Prediction | `predict.py` |
| 9 | Grad-CAM Explainability (heatmap + attention contour) | `utils/explainability.py` |
| 9b | Tumor Segmentation Mask (brain-constrained pixel-level mask) | `utils/explainability.py` |
| 9c | Severity Grading (Grade 0–4 from sigmoid score) | `utils/explainability.py` |
| 10 | Output Display (5-panel figure + metrics printout) | `predict.py`, `app.py` |

---

## Folder Structure

```
brain_tumor_project/
├── app.py                  # Streamlit web UI
├── gui_app.py              # Tkinter desktop UI
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── predict.py              # Inference on CT+MRI pair
├── generate_data.py        # Synthetic data generator
├── requirements.txt
├── models/
│   └── cnn_model.py        # CNN architecture
├── utils/
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── registration.py
│   ├── fusion.py
│   └── explainability.py
├── data/
│   ├── ct/
│   │   ├── tumor/
│   │   └── no_tumor/
│   └── mri/
│       ├── tumor/
│       └── no_tumor/
└── outputs/                # Saved models, plots, metrics
```

---

## Setup

- **Step 1** — Clone the repo and navigate into it:
  ```bash
  git clone https://github.com/Manasisbyali234/brain_tumor.git
  cd brain_tumor
  ```

- **Step 2** — Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

---

## Dataset

- Place paired CT and MRI images under `data/ct/<class>/` and `data/mri/<class>/` with matching filenames.
  - Example: `data/ct/tumor/001.png` ↔ `data/mri/tumor/001.png`
  - Classes: `tumor`, `no_tumor`

- Using a single-modality dataset (e.g. Kaggle Brain MRI)? Pass `paired=False` in `build_fused_dataset()` — the same image will be used as a stand-in for both modalities.

- To generate synthetic dummy data for testing:
  ```bash
  python generate_data.py
  ```

---

## Commands to Run

- **Train the model:**
  ```bash
  python train.py --data_root data --fusion_method dwt --epochs 30 --batch_size 16
  ```
  > Saves: `outputs/best_model.h5`, `outputs/final_model.h5`, `outputs/test_split.npz`

- **Evaluate the model:**
  ```bash
  python evaluate.py --model_path outputs/best_model.h5 --test_split outputs/test_split.npz
  ```
  > Prints accuracy, precision, recall, F1 — saves confusion matrix + ROC curve to `outputs/`

- **Predict on a new CT + MRI pair:**
  ```bash
  python predict.py --ct path/to/ct.png --mri path/to/mri.png --model outputs/best_model.h5
  ```
  > Saves `outputs/prediction_result.png` with CT, MRI, fused image, and Grad-CAM overlay

- **Run the desktop app (Tkinter):**
  ```bash
  python gui_app.py
  ```
  > Browse CT + MRI images, pick fusion method, click Run Detection — all results shown in one window

- **Run the web app (Streamlit):**
  ```bash
  streamlit run app.py
  ```
  > Upload CT + MRI in browser, get prediction, confidence score, fused image, and Grad-CAM

---

## SHAP Explanations (optional)

```python
from utils.explainability import get_shap_explainer, explain_with_shap, plot_shap
import tensorflow as tf, numpy as np

model = tf.keras.models.load_model("outputs/best_model.h5")
explainer = get_shap_explainer(model, X_train[:30])
shap_values = explain_with_shap(explainer, X_test[0:1])
plot_shap(shap_values, X_test[0:1], out_path="outputs/shap_explanation.png")
```

---

## Key Notes

- **Fusion method:** `dwt` (default) gives best detail preservation; `average` and `pca` are simpler baselines
- **CNN:** `build_custom_cnn()` is from-scratch (good for viva); `build_transfer_model()` uses EfficientNetB0 for higher accuracy (needs 3-channel input)
- **Grad-CAM:** Uses last conv layer named `"last_conv"` (already tagged in `cnn_model.py`)
- **SHAP:** Uses `GradientExplainer` — works efficiently with CNNs
