"""
gui_app.py
-----------
Tkinter Desktop UI for Brain Tumor Detection using CT-MRI Image Fusion.

Lets the user browse/select a CT image and an MRI image, runs them
through the full pipeline (preprocess -> register -> fuse -> predict
-> Grad-CAM), and displays the prediction, confidence score, and
visual explanation in a desktop window.

Run with:
    python gui_app.py
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

import tensorflow as tf

from utils.preprocessing import preprocess_image
from utils.registration import register_images
from utils.fusion import fuse_images
from utils.explainability import make_gradcam_heatmap, overlay_gradcam

MODEL_PATH = "outputs/best_model.h5"
CLASS_NAMES = ["No Tumor", "Tumor"]
DISPLAY_SIZE = (220, 220)


class BrainTumorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Detection - CT/MRI Fusion")
        self.root.geometry("980x640")
        self.root.resizable(False, False)

        self.ct_path = None
        self.mri_path = None
        self.model = None
        self.fusion_method = tk.StringVar(value="dwt")

        self._build_layout()
        self._load_model_async()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_layout(self):
        # ---- Top: title + status ----
        header = tk.Frame(self.root, pady=10)
        header.pack(fill="x")

        tk.Label(
            header, text="🧠 Brain Tumor Detection using CT–MRI Image Fusion",
            font=("Segoe UI", 16, "bold")
        ).pack()

        self.status_label = tk.Label(
            header, text="Loading model...", font=("Segoe UI", 10), fg="gray"
        )
        self.status_label.pack()

        # ---- Controls row ----
        controls = tk.Frame(self.root, pady=8)
        controls.pack(fill="x", padx=20)

        tk.Button(controls, text="Select CT Image", width=18,
                  command=self.select_ct).grid(row=0, column=0, padx=5)
        self.ct_path_label = tk.Label(controls, text="No file selected", fg="gray", width=40, anchor="w")
        self.ct_path_label.grid(row=0, column=1, padx=5, sticky="w")

        tk.Button(controls, text="Select MRI Image", width=18,
                  command=self.select_mri).grid(row=1, column=0, padx=5, pady=5)
        self.mri_path_label = tk.Label(controls, text="No file selected", fg="gray", width=40, anchor="w")
        self.mri_path_label.grid(row=1, column=1, padx=5, sticky="w")

        tk.Label(controls, text="Fusion method:").grid(row=0, column=2, padx=(30, 5))
        fusion_menu = ttk.Combobox(
            controls, textvariable=self.fusion_method,
            values=["dwt", "pca", "average"], width=10, state="readonly"
        )
        fusion_menu.grid(row=0, column=3)

        self.run_button = tk.Button(
            controls, text="Run Detection", bg="#2563eb", fg="white",
            font=("Segoe UI", 10, "bold"), width=18, command=self.run_detection_async
        )
        self.run_button.grid(row=1, column=3, padx=(30, 5), pady=5)

        # ---- Image display row ----
        images_frame = tk.Frame(self.root, pady=10)
        images_frame.pack(fill="x", padx=20)

        self.image_panels = {}
        titles = ["CT (original)", "MRI (original)", "Fused Image", "Grad-CAM Heatmap"]
        for i, title in enumerate(titles):
            col = tk.Frame(images_frame, bd=1, relief="solid")
            col.grid(row=0, column=i, padx=8)
            tk.Label(col, text=title, font=("Segoe UI", 9, "bold")).pack(pady=(4, 2))
            panel = tk.Label(col, bg="#f0f0f0", width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1])
            panel.pack(padx=4, pady=4)
            self.image_panels[title] = panel

        # ---- Result row ----
        result_frame = tk.Frame(self.root, pady=15)
        result_frame.pack(fill="x", padx=20)

        self.prediction_label = tk.Label(
            result_frame, text="Prediction: -", font=("Segoe UI", 14, "bold")
        )
        self.prediction_label.pack()

        self.confidence_label = tk.Label(
            result_frame, text="Confidence: -", font=("Segoe UI", 11)
        )
        self.confidence_label.pack()

        # ---- Progress bar ----
        self.progress = ttk.Progressbar(self.root, mode="indeterminate", length=400)

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def _load_model_async(self):
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.status_label.config(text=f"Model loaded: {MODEL_PATH}", fg="green")
        else:
            self.status_label.config(
                text=f"No trained model found at '{MODEL_PATH}'. Run train.py first.",
                fg="red"
            )

    # ------------------------------------------------------------------ #
    # File selection
    # ------------------------------------------------------------------ #
    def select_ct(self):
        path = filedialog.askopenfilename(
            title="Select CT Image", filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        if path:
            self.ct_path = path
            self.ct_path_label.config(text=os.path.basename(path), fg="black")
            self._show_thumbnail(path, "CT (original)")

    def select_mri(self):
        path = filedialog.askopenfilename(
            title="Select MRI Image", filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        if path:
            self.mri_path = path
            self.mri_path_label.config(text=os.path.basename(path), fg="black")
            self._show_thumbnail(path, "MRI (original)")

    def _show_thumbnail(self, path, panel_key):
        img = Image.open(path).convert("L").resize(DISPLAY_SIZE)
        tk_img = ImageTk.PhotoImage(img)
        panel = self.image_panels[panel_key]
        panel.configure(image=tk_img)
        panel.image = tk_img  # keep a reference

    def _show_array(self, arr, panel_key):
        """arr: numpy float image in [0,1] (grayscale) or uint8 RGB."""
        arr = np.squeeze(arr)
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        mode = "L" if arr.ndim == 2 else "RGB"
        img = Image.fromarray(arr, mode=mode).resize(DISPLAY_SIZE)
        tk_img = ImageTk.PhotoImage(img)
        panel = self.image_panels[panel_key]
        panel.configure(image=tk_img)
        panel.image = tk_img

    # ------------------------------------------------------------------ #
    # Detection pipeline
    # ------------------------------------------------------------------ #
    def run_detection_async(self):
        if not self.ct_path or not self.mri_path:
            messagebox.showwarning("Missing input", "Please select both a CT and an MRI image.")
            return
        if self.model is None:
            messagebox.showerror("No model", f"No trained model found at '{MODEL_PATH}'.")
            return

        self.run_button.config(state="disabled")
        self.progress.pack(pady=(0, 10))
        self.progress.start(10)
        threading.Thread(target=self.run_detection, daemon=True).start()

    def run_detection(self):
        try:
            ct_img = preprocess_image(self.ct_path)
            mri_img = preprocess_image(self.mri_path)
            mri_registered = register_images(mri_img, ct_img)
            fused = fuse_images(ct_img, mri_registered, method=self.fusion_method.get())

            input_tensor = np.expand_dims(fused, axis=(0, -1)).astype(np.float32)
            prob = float(self.model.predict(input_tensor, verbose=0)[0][0])
            pred_class = int(prob >= 0.5)
            confidence = prob if pred_class == 1 else 1 - prob

            heatmap = make_gradcam_heatmap(input_tensor, self.model, last_conv_layer_name="last_conv")
            overlay = overlay_gradcam(fused, heatmap)

            self.root.after(0, self._display_results, fused, overlay, pred_class, confidence)
        except Exception as e:
            self.root.after(0, self._on_error, str(e))

    def _display_results(self, fused, overlay, pred_class, confidence):
        self._show_array(fused, "Fused Image")
        self._show_array(overlay, "Grad-CAM Heatmap")

        label = CLASS_NAMES[pred_class]
        color = "#dc2626" if pred_class == 1 else "#16a34a"
        self.prediction_label.config(text=f"Prediction: {label}", fg=color)
        self.confidence_label.config(text=f"Confidence: {confidence*100:.2f}%")

        self.progress.stop()
        self.progress.pack_forget()
        self.run_button.config(state="normal")

    def _on_error(self, message):
        self.progress.stop()
        self.progress.pack_forget()
        self.run_button.config(state="normal")
        messagebox.showerror("Error", f"Detection failed:\n{message}")


def main():
    root = tk.Tk()
    BrainTumorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
