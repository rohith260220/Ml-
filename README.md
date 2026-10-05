# AI Emotion Recognition

A browser-based facial-expression recognition demonstration based on the attached PBL report. It detects frontal faces with OpenCV's Haar cascade, preprocesses each face to normalized 48×48 grayscale, and classifies it into the report's seven FER-2013 labels with a TensorFlow/Keras CNN.

The system describes visible facial-expression patterns. It does not determine a person's internal emotional state, and it is not a psychological assessment.

## Current model and metrics status

The report contains placeholder final and baseline accuracy values (`XX.X%`) and does not include trained weights. No model scores or trained `emotion_cnn.h5` file are fabricated or bundled. The website reports the model and evaluation status from the FastAPI service. Before training, the face detector and health endpoint are available, but prediction endpoints return HTTP 503 with a clear model-not-trained message. Train the model with FER-2013 to create `backend/emotion_cnn.h5` and actual evaluation metrics.

## Architecture

```text
Browser (React, HTML, CSS, TypeScript/JavaScript)
  ├─ Webcam preview → periodically captured JPEG frame ─┐
  └─ User-selected image ───────────────────────────────┤
                                                        ▼
FastAPI: /api/emotion/predict and /predict-image
  ├─ Validate and decode image in memory (OpenCV)
  ├─ Detect every frontal face (Haar cascade)
  ├─ Crop → grayscale → 48×48 → float32 / 255
  ├─ TensorFlow/Keras CNN → seven-class probabilities
  └─ Return face boxes, labels, confidence and probabilities
```

The same-origin API prefix is `/api/emotion` in the Replit preview. The FastAPI routes themselves are `/health`, `/predict`, and `/predict-image` under that prefix. The detector is isolated behind a small interface so it can be replaced without changing the inference API.

## FER-2013

The training script reads the downloaded train/test folder layout directly; no CSV is needed. Place or point `--data-dir` at this structure:

```text
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
```

Folder names are matched case-insensitively to the seven labels below. Supported image files are JPEG, PNG, BMP, and WebP. Empty or unreadable image files are skipped with a warning; each class must still have at least two readable training images and one readable test image. The trainer creates a stratified 20% validation split from `train/` for checkpoint selection; `test/` remains held out for final metrics.

Expected labels, in the model output order:

| FER-2013 ID | Label |
| ---: | --- |
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

The dataset is not redistributed with this project. Keep the downloaded `train/` and `test/` folders together under `artifacts/emotion-recognition/data/`, or point the trainer at their parent directory.

## CNN architecture

Input: 48×48, single-channel grayscale values in `[0, 1]`.

1. Conv2D 64, 3×3, ReLU → batch normalization
2. Conv2D 64, 3×3, ReLU → batch normalization → max pooling → dropout 0.25
3. Conv2D 128, 3×3, ReLU → batch normalization
4. Conv2D 128, 3×3, ReLU → batch normalization → max pooling → dropout 0.25
5. Conv2D 256, 3×3, ReLU → batch normalization
6. Conv2D 256, 3×3, ReLU → batch normalization → max pooling → dropout 0.25
7. Flatten → Dense 256, ReLU → batch normalization → dropout 0.5
8. Dense 7, softmax

## Requirements

- Python 3.10 (as specified in the report)
- Node.js and pnpm for the Vite frontend
- A webcam for live camera use
- FER-2013 train/test image folders for training the CNN

From the repository root, install Python packages:

```bash
python3.10 -m pip install -r artifacts/emotion-recognition/backend/requirements.txt
pnpm install
```

If your machine has a different Python 3.10 executable name, replace `python3.10` with that executable.

## Run the backend

From the repository root:

```bash
cd artifacts/emotion-recognition
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The health endpoint is `http://localhost:8000/api/emotion/health`. FastAPI documentation is available at `http://localhost:8000/api/emotion/docs`.

## Run the frontend

In a second terminal, from the repository root:

```bash
VITE_API_BASE_URL=http://localhost:8000 PORT=5173 BASE_PATH=/ pnpm --filter @workspace/emotion-recognition run dev
```

Open the local Vite URL printed in the terminal. The `VITE_API_BASE_URL` variable is for running the two services directly on a local machine; the Replit preview uses its built-in same-origin routing and does not need it.

## Train and evaluate

Place the extracted folders under `artifacts/emotion-recognition/data/train/` and `artifacts/emotion-recognition/data/test/`. Each must contain the seven emotion subfolders above. From the repository root:

```bash
cd artifacts/emotion-recognition
python -m training.train --data-dir data --epochs 60 --batch-size 64
```

Alternatively, pass explicit paths with `--train-dir /path/to/train --test-dir /path/to/test`. The script uses the existing refined CNN and image preprocessing, Adam at learning rate 0.001, sparse categorical cross entropy, class weights, and training-only augmentation (rotation up to 15°, horizontal/vertical shifts and zoom up to 10%, and horizontal flips). Early stopping monitors the stratified validation subset from `train/`, `ReduceLROnPlateau` halves the learning rate when validation loss stalls, and best-model checkpointing saves the lowest-validation-loss model.

Outputs:

- `backend/emotion_cnn.h5` — best model checkpoint
- `backend/evaluation_metrics.json` — actual held-out test accuracy, macro precision, macro recall, macro F1, and 7×7 confusion matrix
- `training/confusion_matrix.png` — rendered test-set confusion matrix

Metrics are calculated from predictions on the supplied `test/` folder. They are only shown on the website when this file contains actual evaluation results.

## Test the webcam

1. Complete the training step and restart the backend so it loads `backend/emotion_cnn.h5`.
2. Open the frontend at `http://localhost:5173` or use the Replit app preview.
3. Select **Start Camera** and allow camera access in the browser. Use **Stop Camera** to release the camera.
4. For the Replit preview, camera access requires the browser's secure context and permission. If permission is unavailable, use **Upload Image**.

Without a trained checkpoint, the camera preview can still start, but prediction is intentionally paused; the interface reports why and never substitutes mock predictions.

## API

All inference requests use multipart form data with an image field named `file`. Images are processed in memory and are not written to disk.

### `GET /api/emotion/health`

Returns service, detector, model, and actual evaluation-metric availability:

```json
{
  "status": "ok",
  "model_loaded": false,
  "face_detector_loaded": true,
  "message": "Trained model unavailable. ...",
  "evaluation": {
    "accuracy": null,
    "precision_macro": null,
    "recall_macro": null,
    "f1_macro": null,
    "confusion_matrix": null,
    "status": "unavailable"
  }
}
```

### `POST /api/emotion/predict` and `POST /api/emotion/predict-image`

Both routes accept an image as multipart field `file`. When real trained weights are available, the response contains the CNN's seven probabilities and dominant class for each detected face, including bounding boxes. The summary prediction is for the largest face. No example scores or labels are included here because they would not be results from this project's trained model.

With no detected face, the service returns an empty `faces` list, `faces_detected: 0`, and nullable summary fields. Invalid images return HTTP 400. Missing or unloadable model weights return HTTP 503.

For a local API smoke check:

```bash
curl http://localhost:8000/api/emotion/health
```

## Limitations and privacy

- FER-2013 images are low-resolution, grayscale, noisy, and class-imbalanced; class weighting does not remove label noise.
- Haar cascades detect frontal faces and can miss profiles, occluded faces, or faces in difficult lighting.
- Predictions are limited to one of seven classes and one dominant expression per face.
- The model only classifies visible facial patterns. It does not measure a person's actual feelings and should not be used for high-stakes decisions.
- Camera frames and uploaded images are held in memory for inference only. The service does not persist them or store prediction history.
- Metrics remain explicitly unavailable until a real FER-2013 training and evaluation run creates them.