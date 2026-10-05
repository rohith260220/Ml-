"""FastAPI inference service for the FER-2013 facial-expression CNN."""

from __future__ import annotations

import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .model import EMOTION_LABELS, load_trained_model
from .preprocessing import HaarCascadeDetector, decode_image, prepare_face

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emotion-recognition")

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "backend" / "emotion_cnn.h5"
CASCADE_PATH = PROJECT_DIR / "models" / "haarcascade_frontalface_default.xml"
METRICS_PATH = PROJECT_DIR / "backend" / "evaluation_metrics.json"
MAX_IMAGE_BYTES = 8 * 1024 * 1024

model: Any | None = None
model_error: str | None = None
face_detector: HaarCascadeDetector | None = None
inference_lock = threading.Lock()


def _load_metrics() -> dict[str, Any]:
    unavailable = {
        "accuracy": None,
        "precision_macro": None,
        "recall_macro": None,
        "f1_macro": None,
        "confusion_matrix": None,
        "status": "unavailable",
    }
    if not METRICS_PATH.is_file():
        return unavailable

    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        return {
            "accuracy": metrics.get("accuracy"),
            "precision_macro": metrics.get("precision_macro"),
            "recall_macro": metrics.get("recall_macro"),
            "f1_macro": metrics.get("f1_macro"),
            "confusion_matrix": metrics.get("confusion_matrix"),
            "status": "available",
        }
    except (OSError, json.JSONDecodeError, TypeError) as error:
        logger.warning("Could not read evaluation metrics: %s", error)
        return unavailable


@asynccontextmanager
async def lifespan(_: FastAPI):
    global face_detector, model, model_error

    try:
        face_detector = HaarCascadeDetector(CASCADE_PATH)
        logger.info("Loaded Haar cascade from %s", CASCADE_PATH)
    except (FileNotFoundError, RuntimeError) as error:
        face_detector = None
        logger.error("Face detector unavailable: %s", error)

    if MODEL_PATH.is_file():
        try:
            model = load_trained_model(MODEL_PATH)
            model_error = None
            logger.info("Loaded trained emotion CNN from %s", MODEL_PATH)
        except Exception as error:  # Startup remains healthy so /health can explain the issue.
            model = None
            model_error = str(error)
            logger.exception("Could not load the trained emotion CNN")
    else:
        model = None
        model_error = f"Model weights are not present at {MODEL_PATH}."
        logger.warning("%s", model_error)

    yield


app = FastAPI(
    title="AI Emotion Recognition API",
    description="FER-2013 facial-expression prediction using TensorFlow/Keras and OpenCV.",
    version="1.0.0",
    openapi_url="/api/emotion/openapi.json",
    docs_url="/api/emotion/docs",
    redoc_url=None,
    lifespan=lifespan,
)

# Same-origin requests are used in the Replit preview. These local origins allow
# the README's separately started Vite and FastAPI development servers to work.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api/emotion", tags=["emotion"])


@router.get("/health")
def health() -> dict[str, Any]:
    model_loaded = model is not None
    detector_loaded = face_detector is not None
    if model_loaded and detector_loaded:
        message = "Emotion recognition model and face detector are ready."
    elif not model_loaded:
        message = (
            f"Trained model unavailable. {model_error or 'Model weights have not been loaded.'} "
            "Train the model with the FER-2013 dataset to enable predictions."
        )
    else:
        message = "The OpenCV Haar cascade is unavailable; face detection cannot run."

    return {
        "status": "ok" if detector_loaded else "degraded",
        "model_loaded": model_loaded,
        "face_detector_loaded": detector_loaded,
        "message": message,
        "evaluation": _load_metrics(),
    }


async def _predict_file(upload: UploadFile) -> dict[str, Any]:
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a JPEG, PNG, or WebP image.")

    image_bytes = await upload.read(MAX_IMAGE_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image must be 8 MB or smaller.")

    try:
        _, grayscale_image = decode_image(image_bytes)
    except (ValueError, cv2.error) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        await upload.close()

    if face_detector is None:
        raise HTTPException(
            status_code=503,
            detail="OpenCV face detector is unavailable. Check the Haar cascade file.",
        )
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Trained model unavailable. {model_error or ''} "
                "Run the documented FER-2013 training command first."
            ).strip(),
        )

    detected_faces = face_detector.detect(grayscale_image)
    if not detected_faces:
        return {
            "emotion": None,
            "confidence": None,
            "probabilities": None,
            "faces_detected": 0,
            "faces": [],
            "message": "No face detected. Try a well-lit, front-facing image.",
        }

    predictions: list[dict[str, Any]] = []
    for x, y, width, height in detected_faces:
        face_crop = grayscale_image[y : y + height, x : x + width]
        input_tensor = prepare_face(face_crop)
        with inference_lock:
            raw_probabilities = model.predict(input_tensor, verbose=0)[0]
        probabilities = {
            label: float(raw_probabilities[index])
            for index, label in enumerate(EMOTION_LABELS)
        }
        dominant_index = int(np.argmax(raw_probabilities))
        predictions.append(
            {
                "bounding_box": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                },
                "emotion": EMOTION_LABELS[dominant_index],
                "confidence": float(raw_probabilities[dominant_index]),
                "probabilities": probabilities,
            }
        )

    # The largest detected face is the summary prediction; every face is also
    # returned separately so the webcam can label multiple people.
    primary = max(
        predictions,
        key=lambda result: (
            result["bounding_box"]["width"] * result["bounding_box"]["height"]
        ),
    )
    return {
        "emotion": primary["emotion"],
        "confidence": primary["confidence"],
        "probabilities": primary["probabilities"],
        "faces_detected": len(predictions),
        "faces": predictions,
        "message": (
            "Prediction ready."
            if len(predictions) == 1
            else f"Detected {len(predictions)} faces; each face has an individual prediction."
        ),
    }

@router.post("/predict")
async def predict_frame(file: UploadFile = File(...)) -> dict[str, Any]:
    return await _predict_file(file)


@router.post("/predict-image")
async def predict_image(file: UploadFile = File(...)) -> dict[str, Any]:
    return await _predict_file(file)


app.include_router(router)