"""Train and evaluate the FER-2013 emotion CNN from train/test image folders."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from backend.model import EMOTION_LABELS, build_model

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "backend" / "emotion_cnn.h5"
METRICS_PATH = PROJECT_DIR / "backend" / "evaluation_metrics.json"
CONFUSION_MATRIX_PATH = PROJECT_DIR / "training" / "confusion_matrix.png"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
logger = logging.getLogger("emotion-recognition.training")


def load_image_directory(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one split whose immediate subfolders are the seven emotion labels."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"FER-2013 image directory not found: {directory}. "
            "Expected a train/ or test/ directory with one subfolder per emotion."
        )

    class_directories = {
        child.name.strip().casefold(): child
        for child in directory.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    }
    expected_labels = {label.casefold() for label in EMOTION_LABELS}
    missing_labels = sorted(expected_labels - set(class_directories))
    unexpected_labels = sorted(set(class_directories) - expected_labels)
    if missing_labels or unexpected_labels:
        problems = []
        if missing_labels:
            problems.append(f"missing: {', '.join(missing_labels)}")
        if unexpected_labels:
            problems.append(f"unexpected: {', '.join(unexpected_labels)}")
        raise ValueError(
            f"{directory} must contain exactly these seven emotion folders "
            f"({', '.join(EMOTION_LABELS)}); {'; '.join(problems)}."
        )

    images: list[np.ndarray] = []
    labels: list[int] = []
    for label_index, label in enumerate(EMOTION_LABELS):
        class_directory = class_directories[label.casefold()]
        image_paths = sorted(
            path
            for path in class_directory.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        )
        if not image_paths:
            raise ValueError(
                f"No supported images found for {label} in {class_directory}."
            )

        readable_count = 0
        for image_path in image_paths:
            try:
                grayscale = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            except cv2.error:
                grayscale = None
            if grayscale is None or grayscale.size == 0:
                logger.warning("Skipping empty or unreadable FER-2013 image: %s", image_path)
                continue
            resized = cv2.resize(grayscale, (48, 48))
            images.append(resized)
            labels.append(label_index)
            readable_count += 1
        if readable_count == 0:
            raise ValueError(
                f"No readable images found for {label} in {class_directory}."
            )

    image_array = np.expand_dims(np.asarray(images, dtype=np.uint8), axis=-1)
    label_array = np.asarray(labels, dtype=np.int64)
    return image_array, label_array


def load_dataset(
    train_directory: Path,
    test_directory: Path,
    seed: int,
    validation_fraction: float = 0.2,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load folder splits and reserve a stratified portion of train for validation."""
    all_train_images, all_train_labels = load_image_directory(train_directory)
    test_images, test_labels = load_image_directory(test_directory)
    class_counts = np.bincount(all_train_labels, minlength=len(EMOTION_LABELS))
    if np.any(class_counts < 2):
        raise ValueError(
            "Each train emotion folder must contain at least two images so a "
            "stratified validation split can be created."
        )

    training_indices, validation_indices = train_test_split(
        np.arange(all_train_labels.size),
        test_size=validation_fraction,
        random_state=seed,
        stratify=all_train_labels,
    )
    dataset = {
        "train": (all_train_images[training_indices], all_train_labels[training_indices]),
        "validation": (
            all_train_images[validation_indices],
            all_train_labels[validation_indices],
        ),
        "test": (test_images, test_labels),
    }
    return {
        split: (images.astype(np.float32) / 255.0, labels)
        for split, (images, labels) in dataset.items()
    }


def save_confusion_matrix(matrix: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(EMOTION_LABELS)),
        yticks=np.arange(len(EMOTION_LABELS)),
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        ylabel="True label",
        xlabel="Predicted label",
        title="FER-2013 test-set confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close(figure)


def train(
    train_directory: Path,
    test_directory: Path,
    epochs: int,
    batch_size: int,
    seed: int,
    validation_fraction: float,
) -> dict[str, Any]:
    tf.keras.utils.set_random_seed(seed)
    dataset = load_dataset(
        train_directory,
        test_directory,
        seed=seed,
        validation_fraction=validation_fraction,
    )
    train_images, train_labels = dataset["train"]
    validation_images, validation_labels = dataset["validation"]
    test_images, test_labels = dataset["test"]

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(EMOTION_LABELS)),
        y=train_labels,
    )
    class_weights = {
        int(label): float(weight)
        for label, weight in enumerate(class_weights_array)
    }

    augmentation = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]

    model.fit(
        augmentation.flow(train_images, train_labels, batch_size=batch_size, seed=seed),
        validation_data=(validation_images, validation_labels),
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # The checkpoint is selected using only the train-derived validation split.
    best_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    probabilities = best_model.predict(test_images, batch_size=batch_size, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(
        test_labels,
        predicted_labels,
        labels=np.arange(len(EMOTION_LABELS)),
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "status": "available",
        "evaluation_split": "test",
        "test_samples": int(test_labels.size),
        "accuracy": float(accuracy_score(test_labels, predicted_labels)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "class_labels": list(EMOTION_LABELS),
        "confusion_matrix": matrix.tolist(),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(matrix)

    print(json.dumps(metrics, indent=2))
    print(f"Best model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_DIR / "data",
        help="Directory containing train/ and test/ emotion-folder splits.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        help="Optional explicit path to the train/ folder.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        help="Optional explicit path to the test/ folder.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    arguments = parser.parse_args()

    if arguments.epochs < 1 or arguments.batch_size < 1:
        parser.error("--epochs and --batch-size must both be positive.")
    if not 0 < arguments.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1.")

    train_directory = arguments.train_dir or arguments.data_dir / "train"
    test_directory = arguments.test_dir or arguments.data_dir / "test"

    train(
        train_directory,
        test_directory,
        arguments.epochs,
        arguments.batch_size,
        arguments.seed,
        arguments.validation_fraction,
    )


if __name__ == "__main__":
    main()