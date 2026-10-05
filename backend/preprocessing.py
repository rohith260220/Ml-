"""OpenCV Haar-cascade face detection and FER-2013 image preprocessing."""

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .model import IMAGE_SHAPE


class FaceDetector(Protocol):
    """Interface for detectors that can replace Haar cascades in the future."""

    def detect(self, grayscale_image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return face rectangles as x, y, width, height."""


class HaarCascadeDetector:
    """Frontal-face detector using OpenCV's haarcascade_frontalface_default.xml."""

    def __init__(self, cascade_path: Path) -> None:
        if not cascade_path.is_file():
            raise FileNotFoundError(f"Haar cascade file not found: {cascade_path}")

        self.classifier = cv2.CascadeClassifier(str(cascade_path))
        if self.classifier.empty():
            raise RuntimeError(f"OpenCV could not load Haar cascade: {cascade_path}")

    def detect(self, grayscale_image: np.ndarray) -> list[tuple[int, int, int, int]]:
        rectangles = self.classifier.detectMultiScale(
            grayscale_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        return [
            (int(x), int(y), int(width), int(height))
            for x, y, width, height in rectangles
        ]


def decode_image(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode an uploaded image and return its BGR and grayscale forms."""
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    color_image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if color_image is None or color_image.size == 0:
        raise ValueError("The file is not a readable image.")

    grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    return color_image, grayscale_image


def prepare_face(grayscale_face: np.ndarray) -> np.ndarray:
    """Resize one face to 48x48, normalize to [0, 1], and add CNN dimensions."""
    resized = cv2.resize(grayscale_face, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=(0, -1))