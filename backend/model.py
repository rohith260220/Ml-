"""FER-2013 CNN architecture and trained-model loading."""

from pathlib import Path
from typing import Any

EMOTION_LABELS = (
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
)
IMAGE_SHAPE = (48, 48, 1)


def build_model() -> Any:
    """Build the refined three-block CNN described in the PBL report."""
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        BatchNormalization,
        Conv2D,
        Dense,
        Dropout,
        Flatten,
        MaxPooling2D,
    )

    model = Sequential(
        [
            Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=IMAGE_SHAPE),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.5),
            Dense(len(EMOTION_LABELS), activation="softmax"),
        ]
    )
    return model


def load_trained_model(model_path: Path) -> Any:
    """Load a trained HDF5 model and reject incompatible output dimensions."""
    from tensorflow.keras.models import load_model

    model = load_model(model_path, compile=False)
    output_shape = model.output_shape
    if not output_shape or output_shape[-1] != len(EMOTION_LABELS):
        raise ValueError(
            f"Expected {len(EMOTION_LABELS)} output classes, got {output_shape!r}."
        )
    return model