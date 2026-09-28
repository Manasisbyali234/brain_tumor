"""
cnn_model.py
-------------
Step 5: Model Development (CNN)
Defines the Convolutional Neural Network that takes the fused CT-MRI
image as input and classifies it as Tumor / No Tumor.

Two options are provided:
1. build_custom_cnn()   - deep CNN with SE (Squeeze-and-Excitation) attention
                          blocks for precise tumor feature focus
2. build_transfer_model() - EfficientNetB0 transfer learning for high accuracy
                            with limited data
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications


def _se_block(x, ratio=8):
    """Squeeze-and-Excitation block: recalibrates channel-wise feature responses."""
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(max(1, filters // ratio), activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])


def _conv_block(x, filters, kernel=3, stride=1, name=None):
    """Conv -> BN -> ReLU with optional layer name on the Conv."""
    kwargs = {"padding": "same", "use_bias": False}
    if name:
        kwargs["name"] = name
    x = layers.Conv2D(filters, kernel, strides=stride, **kwargs)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def build_custom_cnn(input_shape=(224, 224, 1), num_classes=2):
    """
    Deep CNN with 5 conv blocks + SE attention for precise tumor detection.
    Each block: Conv -> BN -> ReLU -> SE -> MaxPool -> Dropout.
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1 — 32 filters
    x = _conv_block(inputs, 32)
    x = _se_block(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.1)(x)

    # Block 2 — 64 filters
    x = _conv_block(x, 64)
    x = _conv_block(x, 64)
    x = _se_block(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.1)(x)

    # Block 3 — 128 filters
    x = _conv_block(x, 128)
    x = _conv_block(x, 128)
    x = _se_block(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.2)(x)

    # Block 4 — 256 filters
    x = _conv_block(x, 256)
    x = _conv_block(x, 256)
    x = _se_block(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.2)(x)

    # Block 5 — 512 filters (deepest; named for Grad-CAM)
    x = _conv_block(x, 512, name="last_conv")
    x = _se_block(x)
    # No pooling — preserve spatial resolution for Grad-CAM

    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="BrainTumorCNN")


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
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="BrainTumor_EfficientNet")
    return model


def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss — down-weights easy negatives, focuses on hard tumor cases."""
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1 - p_t, gamma) * bce)
    loss_fn.__name__ = "focal_loss"
    return loss_fn


def compile_model(model, num_classes=2, learning_rate=1e-4):
    """Step 6 (part): configure loss/optimizer before training."""
    if num_classes == 2:
        loss = focal_loss(gamma=2.0, alpha=0.25)
    else:
        loss = "categorical_crossentropy"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model
