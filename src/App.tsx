import {
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  Activity,
  AlertCircle,
  Aperture,
  Camera,
  CloudUpload,
  RefreshCw,
  ScanFace,
  ShieldCheck,
  Sparkles,
  VideoOff,
} from "lucide-react";
import {
  useGetEmotionHealth,
  getGetEmotionHealthQueryKey,
  setBaseUrl,
  usePredictEmotionFrame,
  usePredictEmotionImage,
} from "@workspace/api-client-react";
import type { EmotionPredictionResponse } from "@workspace/api-client-react";
import { ErrorBoundary } from "@/components/error-boundary";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/not-found";
import { Route, Switch, useLocation, Router as WouterRouter } from "wouter";
import "./index.css";

const queryClient = new QueryClient();
setBaseUrl(import.meta.env.VITE_API_BASE_URL || null);
const labels = [
  "Angry",
  "Disgust",
  "Fear",
  "Happy",
  "Sad",
  "Surprise",
  "Neutral",
] as const;
const classNotes: Record<(typeof labels)[number], string> = {
  Angry: "Brow and mouth patterns",
  Disgust: "Nose and upper-lip patterns",
  Fear: "Eye and brow patterns",
  Happy: "Cheek and mouth patterns",
  Sad: "Brow and mouth patterns",
  Surprise: "Eye and mouth patterns",
  Neutral: "Low-intensity pattern",
};
type Prediction = EmotionPredictionResponse;
type Smoothed = {
  emotion: string | null;
  confidence: number | null;
  probabilities: Prediction["probabilities"];
};
type Dimensions = { width: number; height: number };

function percent(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(1)}%`
    : "—";
}

function responseError(error: unknown) {
  if (error instanceof Error && error.message) return error.message;
  return "The analysis could not be completed. Check that the service is available and try again.";
}

function Home() {
  const healthQuery = useGetEmotionHealth({
    query: {
      queryKey: getGetEmotionHealthQueryKey(),
      refetchInterval: 30000,
      retry: 1,
    },
  });
  const frameMutation = usePredictEmotionFrame();
  const imageMutation = usePredictEmotionImage();
  const health = healthQuery.data;
  const healthRef = useRef(health);
  healthRef.current = health;
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const frameBusyRef = useRef(false);
  const historyRef = useRef<Prediction[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [actionError, setActionError] = useState("");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [smooth, setSmooth] = useState<Smoothed | null>(null);
  const [imageUrl, setImageUrl] = useState("");
  const [imageRatio, setImageRatio] = useState<number | null>(null);
  const [sourceDimensions, setSourceDimensions] = useState<Dimensions | null>(
    null,
  );
  const [source, setSource] = useState<"camera" | "image" | null>(null);

  const ready = !!health?.model_loaded && !!health?.face_detector_loaded;
  const stopCamera = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
    frameBusyRef.current = false;
  }, []);

  const smoothPrediction = useCallback((next: Prediction) => {
    historyRef.current = [...historyRef.current, next].slice(-5);
    const valid = historyRef.current.filter(
      (item) =>
        item.faces_detected > 0 &&
        (item.probabilities || item.faces[0]?.probabilities),
    );
    if (!valid.length || !next.probabilities) {
      setSmooth(null);
      return;
    }
    const probabilities = labels.reduce<
      NonNullable<Prediction["probabilities"]>
    >(
      (scores, label) => {
        scores[label] =
          valid.reduce(
            (sum, item) =>
              sum +
              (item.probabilities?.[label] ??
                item.faces[0]?.probabilities[label] ??
                0),
            0,
          ) / valid.length;
        return scores;
      },
      {} as NonNullable<Prediction["probabilities"]>,
    );
    const emotion = labels.reduce(
      (best, label) =>
        probabilities[label] > probabilities[best] ? label : best,
      labels[0],
    );
    setSmooth({
      emotion,
      confidence: probabilities[emotion],
      probabilities,
    });
  }, []);

  const analyzeFrame = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2 || frameBusyRef.current)
      return;
    if (
      !healthRef.current?.model_loaded ||
      !healthRef.current?.face_detector_loaded
    )
      return;
    frameBusyRef.current = true;
    try {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      setSourceDimensions((current) =>
        current?.width === canvas.width && current.height === canvas.height
          ? current
          : { width: canvas.width, height: canvas.height },
      );
      const context = canvas.getContext("2d");
      if (!context) return;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", 0.82),
      );
      if (!blob) return;
      const file = new File([blob], "camera-frame.jpg", { type: "image/jpeg" });
      const result = await frameMutation.mutateAsync({ data: { file } });
      setPrediction(result);
      smoothPrediction(result);
      setSource("camera");
      setActionError("");
    } catch (error) {
      setActionError(responseError(error));
    } finally {
      frameBusyRef.current = false;
    }
  }, [frameMutation, smoothPrediction]);

  const startCamera = async () => {
    setCameraError("");
    setActionError("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError(
        "Camera access is not available in this browser. Try uploading an image instead.",
      );
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      setCameraOn(true);
      setImageUrl("");
      setSourceDimensions(null);
      setPrediction(null);
      setSmooth(null);
      historyRef.current = [];
      setSource("camera");
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        void videoRef.current
          .play()
          .catch(() =>
            setCameraError(
              "The camera preview could not start. Check browser permissions and try again.",
            ),
          );
      }

      if (timerRef.current !== null) {
        window.clearInterval(timerRef.current);
      }

      timerRef.current = window.setInterval(() => void analyzeFrame(), 1350);
    } catch (error) {
      setCameraError(
        error instanceof Error && error.name === "NotAllowedError"
          ? "Camera permission was denied. Allow camera access in your browser settings, or upload an image."
          : "The camera could not be opened. Check that it is connected and not in use by another app.",
      );
    }
  };

  const onUpload = async (file?: File) => {
    if (!file) return;
    setActionError("");
    setCameraError("");
    if (!file.type.startsWith("image/")) {
      setActionError("Choose a JPEG, PNG, or WebP image to analyze.");
      return;
    }
    stopCamera();
    setPrediction(null);
    setSmooth(null);
    setSource("image");
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    const preview = URL.createObjectURL(file);
    setImageUrl(preview);
    if (!ready) {
      setActionError(
        healthQuery.isError
          ? "The analysis service is unavailable. The selected image has not been sent."
          : "The model is not ready yet. The selected image has not been sent.",
      );
      return;
    }
    try {
      const result = await imageMutation.mutateAsync({ data: { file } });
      setPrediction(result);
      setSmooth({
        emotion: result.faces[0]?.emotion ?? result.emotion,
        confidence: result.faces[0]?.confidence ?? result.confidence,
        probabilities: result.faces[0]?.probabilities ?? result.probabilities,
      });
    } catch (error) {
      setActionError(responseError(error));
    }
  };

  useEffect(
    () => () => {
      if (timerRef.current !== null) window.clearInterval(timerRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
    },
    [],
  );

  useEffect(
    () => () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
    },
    [imageUrl],
  );

  const metrics = health?.evaluation;
  const activeProbabilities = smooth?.probabilities;
  const resultEmotion = smooth?.emotion ?? prediction?.emotion;
  const resultConfidence = smooth?.confidence ?? prediction?.confidence;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Aperture size={17} strokeWidth={1.8} />
          </div>
          <span>Expression Lab</span>
        </div>
        <div className="top-meta">PBL · COMPUTER VISION</div>
      </header>
      <main className="main-wrap">
        <section className="intro">
          <div>
            <div className="eyebrow">
              A learning tool for expression classification
            </div>
            <h1>AI Emotion Recognition</h1>
            <p>
              Real-time facial expression analysis powered by deep learning.
            </p>
          </div>
          <div className="trust-pill">
            <span
              className={`pulse-dot ${healthQuery.isError ? "offline" : ""}`}
            />
            {healthQuery.isLoading
              ? "CHECKING READINESS"
              : healthQuery.isError
                ? "SERVICE UNAVAILABLE"
                : ready
                  ? "MODEL READY"
                  : "MODEL NOT READY"}
          </div>
        </section>

        <section
          className="workbench"
          aria-label="Expression analysis workspace"
        >
          <article className="panel">
            <div className="panel-head">
              <div className="panel-title">
                <ScanFace size={16} /> Live analysis
              </div>
              <span className="panel-kicker">
                {cameraOn
                  ? "CAMERA INPUT"
                  : imageUrl
                    ? "IMAGE INPUT"
                    : "AWAITING INPUT"}
              </span>
            </div>
            <div
              className="stage"
              style={
                sourceDimensions
                  ? {
                      aspectRatio: `${sourceDimensions.width} / ${sourceDimensions.height}`,
                      minHeight: 0,
                    }
                  : imageUrl && imageRatio
                    ? { aspectRatio: `${imageRatio}`, minHeight: 0 }
                    : undefined
              }
              data-testid="region-analysis-preview"
            >
              {cameraOn && (
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                  aria-label="Live camera preview"
                  onLoadedMetadata={(event) => {
                    const { videoWidth, videoHeight } = event.currentTarget;
                    if (videoWidth && videoHeight)
                      setSourceDimensions({
                        width: videoWidth,
                        height: videoHeight,
                      });
                  }}
                />
              )}
              {!cameraOn && imageUrl && (
                <img
                  src={imageUrl}
                  alt="Selected image for expression analysis"
                  onLoad={(event) => {
                    const { naturalWidth, naturalHeight } = event.currentTarget;
                    setImageRatio(naturalWidth / naturalHeight);
                    setSourceDimensions({
                      width: naturalWidth,
                      height: naturalHeight,
                    });
                  }}
                />
              )}
              {!cameraOn && !imageUrl && (
                <div className="stage-placeholder">
                  <div className="camera-orbit">
                    <Camera size={28} strokeWidth={1.25} />
                  </div>
                  <h2>Your analysis starts here</h2>
                  <p>
                    Start a camera preview or choose an image. No analysis runs
                    until the model is ready.
                  </p>
                </div>
              )}
              {cameraOn && <div className="viewfinder" />}
              {cameraOn &&
                prediction?.faces.map((face, index) => (
                  <div
                    key={`camera-face-${index}`}
                    className="face-box"
                    style={{
                      left: `${face.bounding_box.x <= 1 ? face.bounding_box.x * 100 : (face.bounding_box.x / (sourceDimensions?.width || 1)) * 100}%`,
                      top: `${face.bounding_box.y <= 1 ? face.bounding_box.y * 100 : (face.bounding_box.y / (sourceDimensions?.height || 1)) * 100}%`,
                      width: `${face.bounding_box.width <= 1 ? face.bounding_box.width * 100 : (face.bounding_box.width / (sourceDimensions?.width || 1)) * 100}%`,
                      height: `${face.bounding_box.height <= 1 ? face.bounding_box.height * 100 : (face.bounding_box.height / (sourceDimensions?.height || 1)) * 100}%`,
                    }}
                  >
                    <span className="face-tag">
                      {face.emotion} · {percent(face.confidence)}
                    </span>
                  </div>
                ))}
              {!cameraOn &&
                imageUrl &&
                prediction?.faces.map((face, index) => (
                  <div
                    key={`image-face-${index}`}
                    className="face-box"
                    style={{
                      left: `${face.bounding_box.x <= 1 ? face.bounding_box.x * 100 : (face.bounding_box.x / (sourceDimensions?.width || 1)) * 100}%`,
                      top: `${face.bounding_box.y <= 1 ? face.bounding_box.y * 100 : (face.bounding_box.y / (sourceDimensions?.height || 1)) * 100}%`,
                      width: `${face.bounding_box.width <= 1 ? face.bounding_box.width * 100 : (face.bounding_box.width / (sourceDimensions?.width || 1)) * 100}%`,
                      height: `${face.bounding_box.height <= 1 ? face.bounding_box.height * 100 : (face.bounding_box.height / (sourceDimensions?.height || 1)) * 100}%`,
                    }}
                  >
                    <span className="face-tag">
                      {face.emotion} · {percent(face.confidence)}
                    </span>
                  </div>
                ))}
              {cameraOn && (
                <div className="stage-state">
                  <span className="pulse-dot" />
                  {ready ? "LIVE · INFERENCE" : "PREVIEW · MODEL UNAVAILABLE"}
                </div>
              )}
              {cameraOn &&
                (!health?.model_loaded || !health?.face_detector_loaded) && (
                  <div className="preview-warning">
                    Preview only — inference is paused until model and detector
                    are ready.
                  </div>
                )}
              {(frameMutation.isPending || imageMutation.isPending) && (
                <div className="processing">
                  <span className="loader-bars">
                    <i />
                    <i />
                    <i />
                  </span>
                  ANALYZING FRAME
                </div>
              )}
              <canvas ref={canvasRef} className="hidden" aria-hidden="true" />
            </div>
            <div className="controls">
              <div className="button-row">
                {!cameraOn ? (
                  <button
                    type="button"
                    className="btn btn-primary"
                    onClick={() => void startCamera()}
                    data-testid="button-start-camera"
                  >
                    <Camera size={15} />
                    Start Camera
                  </button>
                ) : (
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={stopCamera}
                    data-testid="button-stop-camera"
                  >
                    <VideoOff size={15} />
                    Stop Camera
                  </button>
                )}
                <button
                  type="button"
                  className="btn btn-quiet"
                  onClick={() => fileRef.current?.click()}
                  disabled={imageMutation.isPending}
                  data-testid="button-upload-image"
                >
                  <CloudUpload size={15} />
                  Upload Image
                </button>
                <input
                  ref={fileRef}
                  className="file-input"
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={(event) => {
                    void onUpload(event.currentTarget.files?.[0]);
                    event.currentTarget.value = "";
                  }}
                  data-testid="input-upload-image"
                />
              </div>
              <span className="control-note">
                {cameraOn
                  ? "Camera frames analyzed periodically"
                  : "JPEG · PNG · WEBP"}
              </span>
            </div>
            {(cameraError || actionError) && (
              <div
                className="inline-error"
                role="alert"
                data-testid="status-analysis-error"
              >
                <AlertCircle size={14} />
                {cameraError || actionError}
              </div>
            )}
          </article>

          <article className="panel results-panel">
            <div className="panel-head">
              <div className="panel-title">
                <Activity size={16} /> Analysis result
              </div>
              <span className="panel-kicker">
                {prediction
                  ? `${prediction.faces_detected} FACE${prediction.faces_detected === 1 ? "" : "S"}`
                  : "NO RESULT"}
              </span>
            </div>
            <div
              className="result-content"
              data-testid="region-analysis-result"
            >
              {prediction?.message && (
                <p
                  className={`response-message ${prediction.faces_detected === 0 ? "error" : ""}`}
                  data-testid="text-prediction-message"
                >
                  {prediction.message}
                </p>
              )}
              {actionError && (
                <p className="response-message error">{actionError}</p>
              )}
              {!prediction &&
                !imageMutation.isPending &&
                !frameMutation.isPending && (
                  <div className="result-empty">
                    <div>
                      <strong>Waiting for a face</strong>Prediction results and
                      class scores will appear here after analysis.
                    </div>
                  </div>
                )}
              {(imageMutation.isPending || frameMutation.isPending) &&
                !prediction && (
                  <div className="result-empty">
                    <div>
                      <div className="loader-bars">
                        <i />
                        <i />
                        <i />
                      </div>
                      <strong>Reading the frame</strong>Checking for faces and
                      expression patterns.
                    </div>
                  </div>
                )}
              {prediction && prediction.faces_detected === 0 && (
                <div className="result-empty">
                  <div>
                    <strong>No face detected</strong>Try a clear, front-facing
                    image with the face well lit and unobstructed.
                  </div>
                </div>
              )}
              {prediction && prediction.faces_detected > 0 && (
                <>
                  <div className="result-emotion">
                    <div>
                      <div className="result-caption">Expression class</div>
                      <div
                        className="result-label"
                        data-testid="text-predicted-emotion"
                      >
                        {resultEmotion ?? "Unclassified"}
                      </div>
                    </div>
                    <div
                      className="confidence"
                      data-testid="text-prediction-confidence"
                    >
                      {percent(resultConfidence)}
                      <small>CONFIDENCE</small>
                    </div>
                  </div>
                  {prediction.faces_detected > 1 && (
                    <p className="response-message">
                      Multiple faces detected. The summary shows the largest
                      detected face; boxes label each detected face.
                    </p>
                  )}
                  {activeProbabilities && (
                    <>
                      <div className="section-label">
                        <span>Class probabilities</span>
                        <span>
                          {source === "camera" ? "SMOOTHED" : "MODEL OUTPUT"}
                        </span>
                      </div>
                      <div className="prob-list">
                        {labels.map((label) => (
                          <div
                            className="prob-row"
                            key={label}
                            data-testid={`probability-${label.toLowerCase()}`}
                          >
                            <span>{label}</span>
                            <div className="prob-track">
                              <div
                                className="prob-fill"
                                style={{
                                  width: `${Math.max(0, Math.min(100, activeProbabilities[label] * 100))}%`,
                                }}
                              />
                            </div>
                            <span className="prob-value">
                              {percent(activeProbabilities[label])}
                            </span>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                  <div className="faces-note">
                    <ScanFace size={14} />
                    {prediction.faces_detected}{" "}
                    {prediction.faces_detected === 1 ? "face" : "faces"}{" "}
                    detected in this input
                  </div>
                </>
              )}
            </div>
          </article>
        </section>

        <section className="below-grid">
          <article className="panel info-panel">
            <h2 className="info-heading">
              <Sparkles size={15} /> Seven expression classes
            </h2>
            <p className="info-sub">
              The model maps visible facial patterns to one of seven trained
              labels. A label is a classification output, not a reading of a
              person.
            </p>
            <div className="emotion-grid">
              {labels.map((label) => (
                <div
                  className="emotion-item"
                  key={label}
                  data-testid={`class-${label.toLowerCase()}`}
                >
                  <span className="emotion-swatch" />
                  <strong>{label}</strong>
                  <span>{classNotes[label]}</span>
                </div>
              ))}
            </div>
          </article>
          <article className="panel info-panel">
            <div className="readiness-heading">
              <div>
                <h2 className="info-heading">
                  <Activity size={15} /> Model readiness
                </h2>
                <p className="info-sub">
                  Live status from the analysis service
                </p>
              </div>
              <button
                className="icon-button"
                type="button"
                aria-label="Refresh model readiness"
                onClick={() => void healthQuery.refetch()}
                data-testid="button-refresh-readiness"
              >
                <RefreshCw size={14} />
              </button>
            </div>
            {healthQuery.isLoading && !health && (
              <div
                className="skeleton-readiness"
                aria-label="Loading model readiness"
              >
                <i />
                <i />
                <i />
              </div>
            )}
            {healthQuery.isError && (
              <div
                className="service-unavailable"
                role="status"
                data-testid="status-backend-unavailable"
              >
                <AlertCircle size={16} />
                <div>
                  <strong>Backend unavailable</strong>
                  <span>
                    Could not retrieve model or evaluation status. Predictions
                    are disabled.
                  </span>
                </div>
              </div>
            )}
            {health && (
              <div className="readiness">
                <div className="readiness-row">
                  <span className="readiness-name">
                    <span className="mini-state" />
                    API service
                  </span>
                  <span
                    className={`status-badge ${health.status.toLowerCase() === "ok" || health.status.toLowerCase() === "healthy" ? "ready" : "wait"}`}
                    data-testid="status-api"
                  >
                    {health.status || "Unknown"}
                  </span>
                </div>
                <div className="readiness-row">
                  <span className="readiness-name">
                    <span className="mini-state" />
                    Expression model
                  </span>
                  <span
                    className={`status-badge ${health.model_loaded ? "ready" : "down"}`}
                    data-testid="status-model"
                  >
                    {health.model_loaded ? "Loaded" : "Not loaded"}
                  </span>
                </div>
                <div className="readiness-row">
                  <span className="readiness-name">
                    <span className="mini-state" />
                    Face detector
                  </span>
                  <span
                    className={`status-badge ${health.face_detector_loaded ? "ready" : "down"}`}
                    data-testid="status-detector"
                  >
                    {health.face_detector_loaded ? "Loaded" : "Not loaded"}
                  </span>
                </div>
                {!health.model_loaded && (
                  <div
                    className="health-message"
                    data-testid="text-model-message"
                  >
                    {health.message ||
                      "The model is not available. Inference is paused until it is loaded."}
                  </div>
                )}
                {health.model_loaded && health.message && (
                  <div
                    className="health-message"
                    data-testid="text-model-message"
                  >
                    {health.message}
                  </div>
                )}
                <div className="section-label">
                  <span>Evaluation metrics</span>
                  <span
                    className={`status-badge ${metrics?.status === "available" ? "ready" : "wait"}`}
                  >
                    {metrics?.status === "available"
                      ? "REPORTED"
                      : "UNAVAILABLE"}
                  </span>
                </div>
                {metrics?.status === "available" ? (
                  <div
                    className="metrics"
                    data-testid="region-evaluation-metrics"
                  >
                    <div className="metric">
                      <b>{percent(metrics.accuracy)}</b>
                      <span>Accuracy</span>
                    </div>
                    <div className="metric">
                      <b>{percent(metrics.precision_macro)}</b>
                      <span>Macro precision</span>
                    </div>
                    <div className="metric">
                      <b>{percent(metrics.recall_macro)}</b>
                      <span>Macro recall</span>
                    </div>
                    <div className="metric">
                      <b>{percent(metrics.f1_macro)}</b>
                      <span>Macro F1</span>
                    </div>
                  </div>
                ) : (
                  <div className="health-message">
                    Evaluation metrics have not been reported by the service.
                  </div>
                )}
                {metrics?.confusion_matrix && (
                  <details className="matrix-details">
                    <summary data-testid="button-toggle-confusion-matrix">
                      View confusion matrix
                    </summary>
                    <div className="matrix-scroll">
                      <table aria-label="Confusion matrix">
                        <tbody>
                          {metrics.confusion_matrix.map((row, rowIndex) => (
                            <tr key={`matrix-row-${rowIndex}`}>
                              {row.map((value, colIndex) => (
                                <td key={`matrix-${rowIndex}-${colIndex}`}>
                                  {value}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </details>
                )}
              </div>
            )}
          </article>
        </section>

        <aside className="privacy-strip">
          <ShieldCheck size={18} />
          <div>
            <strong>Private by design. Careful by definition.</strong>
            <p>
              Camera frames are only used for inference and are not stored by
              this interface. Facial-expression labels describe visual patterns
              detected by a model; they do not establish a person’s internal
              emotional state. Results can be uncertain and vary with lighting,
              pose, and training data.
            </p>
          </div>
        </aside>
        <footer className="footer">
          <span>EXPRESSION LAB · PBL DEMONSTRATION</span>
          <span>Classification output is not a psychological assessment.</span>
        </footer>
      </main>
    </div>
  );
}

function Router() {
  return (
    <RoutedErrorBoundary>
      <Switch>
        <Route path="/" component={Home} />
        <Route component={NotFound} />
      </Switch>
    </RoutedErrorBoundary>
  );
}

function RoutedErrorBoundary({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}>{children}</ErrorBoundary>;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, "")}>
          <Router />
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
