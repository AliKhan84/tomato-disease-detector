"""MobileNetV2 transfer-learning classifier as a single tf.keras.Sequential stack."""

from __future__ import annotations

import tensorflow as tf

IMAGE_SIZE = (224, 224)


def build_model(num_classes: int, image_size: tuple[int, int] = IMAGE_SIZE):
    """Frozen-backbone MobileNetV2 classifier.

    Layer order is load-bearing:

    1. Augmentation runs FIRST, on raw 0-255 pixels. RandomContrast rescales around the
       mean assuming that range; running it after normalisation to [-1, 1] distorts the
       images instead of augmenting them. Do not reorder these two blocks.
    2. Rescaling to [-1, 1] is what MobileNetV2 expects, and it is equivalent to
       tf.keras.applications.mobilenet_v2.preprocess_input. Keeping it inside the model
       means the saved .keras file is self-contained -- infer.py hands it a plain
       0-255 array and does not need to know the normalisation.

    Augmentation layers are inference-noop, so they cost nothing at predict time.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    # Frozen backbone: only the classification head trains. This also keeps the base's
    # BatchNorm layers in inference mode, which is what you want when the head is new.
    base_model.trainable = False

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(*image_size, 3)),
            # -- augmentation, on 0-255 --
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomContrast(0.1),
            # -- normalise to [-1, 1] --
            tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0),
            # -- frozen pretrained backbone --
            base_model,
            # -- classification head --
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="tomato_disease_mobilenetv2",
    )


def compile_model(model, learning_rate: float = 1e-3):
    """Compile for integer labels (label_mode='int' in dataset.py)."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
