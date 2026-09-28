"""
train.py
---------
Step 6: Model Training
Loads the fused dataset, builds the CNN, trains it with backpropagation
+ Adam optimizer, and saves the best model checkpoint.

Usage:
    python train.py --data_root data --epochs 30 --batch_size 16
"""

import argparse
import os
import numpy as np
import tensorflow as tf

from utils.dataset import build_fused_dataset, build_mri_dataset, split_dataset
from models.cnn_model import build_custom_cnn, compile_model
from sklearn.utils.class_weight import compute_class_weight


def get_args():
    parser = argparse.ArgumentParser(description="Train Brain Tumor Detection CNN")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--fusion_method", type=str, default="dwt", choices=["average", "pca", "dwt"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--paired", action="store_true", default=False)
    parser.add_argument("--no_fusion", action="store_true", default=False,
                        help="Train on MRI images directly (no fusion). Use when CT/MRI are not truly paired.")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/4] Building dataset ...")
    if args.no_fusion:
        print("   Mode: MRI-only (no fusion)")
        X, y = build_mri_dataset(args.data_root, limit_per_class=args.limit)
    else:
        print("   Mode: CT+MRI fusion")
        X, y = build_fused_dataset(args.data_root, fusion_method=args.fusion_method, paired=args.paired, limit_per_class=args.limit)
    print(f"   Dataset shape: X={X.shape}, y={y.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    print(f"   Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    # Save test split for later evaluation
    np.savez(os.path.join(args.out_dir, "test_split.npz"), X_test=X_test, y_test=y_test)

    print("[2/4] Building CNN model ...")
    model = build_custom_cnn(input_shape=X_train.shape[1:], num_classes=2)
    model = compile_model(model, num_classes=2, learning_rate=args.lr)
    model.summary()

    # Class weights to handle any imbalance
    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {int(c): w for c, w in zip(classes, cw)}
    print(f"   Class weights: {class_weight}")

    def augment(x, y):
        # Spatial
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        # Intensity
        x = tf.image.random_brightness(x, 0.15)
        x = tf.image.random_contrast(x, 0.8, 1.2)
        # Simulate rotation via two flips + transpose
        x = tf.cond(tf.random.uniform(()) > 0.5,
                    lambda: tf.image.transpose(x), lambda: x)
        x = tf.clip_by_value(x, 0.0, 1.0)
        return x, y

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(len(X_train)).batch(args.batch_size)
    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(args.batch_size)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.out_dir, "best_model.h5"),
            monitor="val_auc", save_best_only=True, mode="max", verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=10, restore_best_weights=True, mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(args.out_dir, "training_log.csv")),
    ]

    print("[3/4] Training ...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    print("[4/4] Saving final model + training history ...")
    model.save(os.path.join(args.out_dir, "final_model.h5"))
    np.save(os.path.join(args.out_dir, "history.npy"), history.history)
    print("Done. Artifacts saved to:", args.out_dir)


if __name__ == "__main__":
    main()
