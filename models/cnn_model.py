"""
cnn_model.py
-------------
Step 5: Model Development (CNN)
Defines the Convolutional Neural Network that takes the fused CT-MRI
image as input and classifies it as Tumor / No Tumor.

Two options are provided:
1. build_custom_cnn()   - a CNN built from scratch (good for demonstrating
                           convolution/pooling concepts in a synopsis/report)
2. build_transfer_model() - a transfer-learning model (EfficientNetB0)
                           for higher accuracy with limited data
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications


def build_custom_cnn(input_shape=(224, 224, 1), num_classes=2):
    """A CNN built from scratch: Conv -> Pool blocks -> Dense classifier."""
    model = models.Sequential(name="BrainTumorCNN")

    model.add(layers.Input(shape=input_shape))

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 4
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="last_conv"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Classifier head
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.4))

    if num_classes == 2:
        model.add(layers.Dense(1, activation="sigmoid"))
    else:
        model.add(layers.Dense(num_classes, activation="softmax"))

    return model


def build_transfer_model(input_shape=(224, 224, 3), num_classes=2, trainable_base=False):
    """
    Transfer learning model using EfficientNetB0 pretrained on ImageNet.
    Note: fused images must be converted to 3-channel (RGB-like) for this.
    """
    base_model = applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base_model.trainable = trainable_base

    inputs = layers.Input(shape=input_shape)
    x = applications.efficientnet.preprocess_input(inputs * 255.0)
    x = base_model(x, training=trainable_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="BrainTumor_EfficientNet")
    return model


def compile_model(model, num_classes=2, learning_rate=1e-4):
    """Step 6 (part): configure loss/optimizer before training."""
    loss = "binary_crossentropy" if num_classes == 2 else "categorical_crossentropy"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model
