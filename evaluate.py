"""
evaluate.py
------------
Step 7: Model Evaluation
Loads the trained model and held-out test split, computes accuracy,
precision, recall, F1-score, confusion matrix, and ROC-AUC.

Usage:
    python evaluate.py --model_path outputs/best_model.h5 --test_split outputs/test_split.npz
"""

import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Brain Tumor Detection CNN")
    parser.add_argument("--model_path", type=str, default="outputs/best_model.h5")
    parser.add_argument("--test_split", type=str, default="outputs/test_split.npz")
    parser.add_argument("--out_dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading model and test data ...")
    model = tf.keras.models.load_model(args.model_path)
    data = np.load(args.test_split)
    X_test, y_test = data["X_test"], data["y_test"]

    print("Running inference on test set ...")
    y_probs = model.predict(X_test).flatten()
    y_pred = (y_probs >= 0.5).astype(int)
    y_true = y_test.astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n===== Evaluation Metrics =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["no_tumor", "tumor"]))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["no_tumor", "tumor"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["no_tumor", "tumor"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve"); ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.out_dir, "roc_curve.png"), dpi=150)
    plt.close(fig2)

    print(f"\nConfusion matrix and ROC curve saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
