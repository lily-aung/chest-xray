from __future__ import annotations

import hashlib, io, json, logging, os, tempfile, threading, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2, numpy as np, torch

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.inference.gradcam_core import build_cam_engine, decode_upload_to_gray, get_module_by_path, overlay_png_bytes, predict_probs
from src.inference.infer_deployment import build_infer_transform, infer_one, load_model_from_bundle, maybe_apply_custom_clahe, to_model_channels

APP_NAME = "chestxray-infer-api"

app = FastAPI(title=APP_NAME, version="1.0.0")

# DEV CORS -=>run on two server: allow any origin (no cookies)
'''
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
'''
# -----------------------------
# UI <<- run front end and backend on the same server
UI_DIR = Path(os.getenv("UI_DIR", "/app/ui"))
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
    @app.get("/", include_in_schema=False)
    def root():
        return FileResponse(str(UI_DIR / "index.html"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(message)s")
logger = logging.getLogger("chestxray_api")

def jlog(event: str, **fields: Any) -> None:
    payload = {"event": event, "ts": time.time(), **fields}
    logger.info(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))

def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

REQ_COUNT = Counter("cxr_requests_total", "Total requests", ["endpoint", "status"])
REQ_LAT = Histogram("cxr_request_latency_seconds", "Request latency", ["endpoint"])
PRED_COUNT = Counter("cxr_predictions_total", "Predictions made", ["pred_name", "pred_policy_name"])
FAIL_COUNT = Counter("cxr_failures_total", "Failures", ["endpoint", "reason"])


def env_bool(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def get_device() -> torch.device:
    # Keep CPU default for reliability; allow override.
    dev = os.getenv("DEVICE", "cpu").strip()
    return torch.device(dev)


def get_bundle_dir() -> Path:
    return Path(os.getenv("BUNDLE_DIR", "/bundle"))


MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10MB default
API_TOKEN = os.getenv("API_TOKEN", "").strip()  # if empty => auth disabled
AUTH_ENABLED = bool(API_TOKEN)


# -----------------------------
# Response schemas
class Prediction(BaseModel):
    filename: str = Field(..., description="Original filename from upload")
    pred_id: int
    pred_name: str
    confidence: float
    probs: List[float]
    pred_policy_id: Optional[int] = None
    pred_policy_name: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    results: List[Prediction]
    errors: List[Dict[str, str]] = Field(default_factory=list)

class InfoResponse(BaseModel):
    status: str
    bundle_dir: str
    device: str
    img_size: int
    to_rgb: bool
    use_imagenet_norm: bool
    use_custom_clahe: bool
    semver: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    git_commit: Optional[str] = None

# Auth dependency
# -----------------------------
def require_auth(request: Request) -> None:
    if not AUTH_ENABLED:
        return
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth.split(" ", 1)[1].strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

# Upload-size guard middleware
# -----------------------------
class UploadTooLarge(Exception):
    pass
STATE: Dict[str, Any] = {}
CAM_LOCK = threading.Lock()

@app.middleware("http")
async def size_limit_and_metrics(request: Request, call_next):
    path = request.url.path
    endpoint = path

    # Enforce upload size based on Content-Length if present
    # (Not perfect; still helps)
    if request.method in ("POST", "PUT", "PATCH"):
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > MAX_UPLOAD_BYTES:
                    FAIL_COUNT.labels(endpoint=endpoint, reason="content_length_exceeded").inc()
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"Upload too large. Max bytes={MAX_UPLOAD_BYTES}"},
                    )
            except ValueError:
                pass

    start = time.time()
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        dur = time.time() - start
        REQ_LAT.labels(endpoint=endpoint).observe(dur)

@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    endpoint = request.url.path
    REQ_COUNT.labels(endpoint=endpoint, status=str(exc.status_code)).inc()
    jlog(
        "http_exception",
        endpoint=endpoint,
        status=exc.status_code,
        detail=str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    endpoint = request.url.path
    REQ_COUNT.labels(endpoint=endpoint, status="500").inc()
    FAIL_COUNT.labels(endpoint=endpoint, reason=exc.__class__.__name__).inc()
    jlog(
        "unhandled_exception",
        endpoint=endpoint,
        error=exc.__class__.__name__,
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# -----------------------------
# Startup: load model + bundle defaults
@app.on_event("startup")
def startup() -> None:
    bundle_dir = get_bundle_dir()
    manifest_path = bundle_dir / "manifest.json"
    model_dir = bundle_dir / "model"
    class_names_path = bundle_dir / "class_names.json"
    policy_path = bundle_dir / "thresholds_policy.json"

    if not manifest_path.exists():
        raise RuntimeError(f"Missing manifest.json in bundle: {manifest_path}")
    if not model_dir.exists():
        raise RuntimeError(f"Missing model/ in bundle: {model_dir}")
    if not class_names_path.exists():
        raise RuntimeError(f"Missing class_names.json in bundle: {class_names_path}")

    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    class_names = json.loads((bundle_dir / "class_names.json").read_text())["class_names"]
    if not class_names:
        raise RuntimeError("class_names.json missing class_names list")

    thresholds = None
    if policy_path.exists():
        thresholds = json.loads(policy_path.read_text())

    device = get_device()

    # Preprocess defaults controlled by ENV for maintainability
    img_size = int(os.getenv("IMG_SIZE", "224"))
    to_rgb = env_bool("TO_RGB", "0")
    use_imagenet_norm = env_bool("USE_IMAGENET_NORM", "0")

    use_custom_clahe = env_bool("USE_CUSTOM_CLAHE", "0")
    clahe_clip_limit = float(os.getenv("CLAHE_CLIP_LIMIT", "2.0"))
    clahe_tile_grid = tuple(int(x) for x in os.getenv("CLAHE_TILE_GRID", "8,8").split(","))

    model = load_model_from_bundle(bundle_dir=bundle_dir, device=device)
    tfm = build_infer_transform(img_size=img_size, use_imagenet_norm=use_imagenet_norm, is_rgb=to_rgb)

 
    STATE.update(bundle_dir=str(bundle_dir), manifest=manifest, class_names=class_names, 
                 thresholds=thresholds, device=str(device), model=model, tfm=tfm, img_size=img_size, 
                 to_rgb=to_rgb, use_imagenet_norm=use_imagenet_norm,
                   use_custom_clahe=use_custom_clahe, clahe_clip_limit=clahe_clip_limit, clahe_tile_grid=clahe_tile_grid)
    jlog("startup_ok", bundle_dir=str(bundle_dir), device=str(device), img_size=img_size,
          auth_enabled=AUTH_ENABLED, semver=manifest.get("semver"), run_id=manifest.get("mlflow_run_id"))

    # ---- Grad-CAM config from manifest
    target_layer_path = os.getenv("GRADCAM_TARGET_LAYER") or manifest.get("gradcam_target_layer")
    cam_method_raw = os.getenv("GRADCAM_METHOD") or manifest.get("cam-method")

    if not target_layer_path:
        raise RuntimeError("Grad-CAM target layer missing. Set manifest['gradcam_target_layer'] or GRADCAM_TARGET_LAYER")
    # ---- Grad-CAM init ----
    cam_method = (cam_method_raw or "smoothgradcampp").strip().lower()
    smooth_n = int(os.getenv("GRADCAM_SMOOTH_N", "25"))
    smooth_std = float(os.getenv("GRADCAM_SMOOTH_STD", "0.10"))
    cam_alpha = float(os.getenv("GRADCAM_ALPHA", "0.35"))

    target_layer = get_module_by_path(model, str(target_layer_path))
    cam_engine = build_cam_engine(model=model, target_layer=target_layer, method=cam_method, smooth_n=smooth_n, smooth_std=smooth_std)

    STATE.update( cam_engine=cam_engine, cam_alpha=cam_alpha,
        cam_method=cam_method, cam_target_layer=str(target_layer_path),
        cam_smooth_n=smooth_n, cam_smooth_std=smooth_std )

    jlog( "gradcam_init_ok", target_layer=str(target_layer_path),
        method=cam_method, alpha=cam_alpha,smooth_n=smooth_n,smooth_std=smooth_std)


# >>>Helpers
def save_upload_to_temp(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "upload.jpg").suffix or ".jpg"

    # Read bytes (enforce MAX_UPLOAD_BYTES even if no content-length)
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Upload too large. Max bytes={MAX_UPLOAD_BYTES}")

    fd, tmp_path = tempfile.mkstemp(prefix="cxr_", suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(data)
    return tmp_path


def predict_file(upload: UploadFile) -> Prediction:
    filename = upload.filename or "upload"
    filename_hash = sha1_str(filename)
    tmp_path = None
    try:
        tmp_path = save_upload_to_temp(upload)

        res = infer_one(model=STATE["model"], device=torch.device(STATE["device"]), image_path=tmp_path, 
                        tfm=STATE["tfm"], class_names=STATE["class_names"], thresholds=STATE["thresholds"],
                          use_custom_clahe=STATE["use_custom_clahe"], clahe_clip_limit=STATE["clahe_clip_limit"], 
                          clahe_tile_grid=STATE["clahe_tile_grid"], to_rgb=STATE["to_rgb"])


        pred = Prediction(filename=filename, pred_id=res["pred_id"], pred_name=res["pred_name"], 
                          confidence=res["confidence"], probs=res["probs"], pred_policy_id=res.get("pred_policy_id"), 
                          pred_policy_name=res.get("pred_policy_name"))


        PRED_COUNT.labels(pred_name=pred.pred_name, pred_policy_name=str(pred.pred_policy_name)).inc()
        jlog( "predict_ok", file=filename_hash, pred=pred.pred_name,
            policy=str(pred.pred_policy_name), conf=round(pred.confidence, 6))
        return pred

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# -----------------------------
# Endpoints

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": "model" in STATE}

@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    m = STATE.get("manifest", {}) or {}
    return InfoResponse(
        status="ok",
        bundle_dir=str(STATE.get("bundle_dir")),
        device=str(STATE.get("device")),
        img_size=int(STATE.get("img_size")),
        to_rgb=bool(STATE.get("to_rgb")),
        use_imagenet_norm=bool(STATE.get("use_imagenet_norm")),
        use_custom_clahe=bool(STATE.get("use_custom_clahe")),
        semver=m.get("semver"),
        mlflow_run_id=m.get("mlflow_run_id"),
        git_commit=m.get("git_commit"),
    )

@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=Prediction)
def predict( request: Request, file: UploadFile = File(...), _: Any = Depends(require_auth)) -> Prediction:
    endpoint = "/predict"
    start = time.time()
    try:
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"Unsupported content_type: {file.content_type}")

        pred = predict_file(file)
        REQ_COUNT.labels(endpoint=endpoint, status="200").inc()
        return pred
    finally:
        REQ_LAT.labels(endpoint=endpoint).observe(time.time() - start)


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    _: Any = Depends(require_auth)) -> BatchPredictionResponse:
    endpoint = "/predict_batch"
    start = time.time()
    results: List[Prediction] = []
    errors: List[Dict[str, str]] = []

    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        # Safety cap on count
        max_files = int(os.getenv("MAX_FILES_PER_BATCH", "32"))
        if len(files) > max_files:
            raise HTTPException(status_code=413, detail=f"Too many files. Max files={max_files}")

        for f in files:
            try:
                if f.content_type and not f.content_type.startswith("image/"):
                    raise ValueError(f"Unsupported content_type: {f.content_type}")
                results.append(predict_file(f))
            except Exception as e:
                FAIL_COUNT.labels(endpoint=endpoint, reason=e.__class__.__name__).inc()
                errors.append({"filename": f.filename or "upload", "error": str(e)})

        REQ_COUNT.labels(endpoint=endpoint, status="200").inc()
        return BatchPredictionResponse(results=results, errors=errors)

    finally:
        REQ_LAT.labels(endpoint=endpoint).observe(time.time() - start)

@app.post("/grad_cam")
def grad_cam(request: Request, file: UploadFile = File(...), _: Any = Depends(require_auth)):
    endpoint = "/grad_cam"
    start = time.time()
    try:
        if STATE.get("cam_engine") is None:
            raise HTTPException(status_code=503, detail="Grad-CAM engine not initialized")

        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"Unsupported content_type: {file.content_type}")

        # Read upload bytes (cap size)
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if not data:
            raise HTTPException(status_code=400, detail="Empty upload")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Upload too large. Max bytes={MAX_UPLOAD_BYTES}")

        # Decode + preprocess (same logic as deployment)
        try:
            img_gray = decode_upload_to_gray(data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        img_gray = maybe_apply_custom_clahe(
            img_gray, use_custom_clahe=STATE["use_custom_clahe"],
            clahe_clip_limit=STATE["clahe_clip_limit"],
            clahe_tile_grid=STATE["clahe_tile_grid"])
        img = to_model_channels(img_gray, to_rgb=STATE["to_rgb"])  # HWC

        out = STATE["tfm"](image=img)
        x_chw: torch.Tensor = out["image"]
        x1 = x_chw.unsqueeze(0).to(torch.device(STATE["device"]))

        # Predict class (no_grad inside predict_probs)
        probs_vec, pred_class, pred_prob = predict_probs(STATE["model"], x1)
        class_names = STATE["class_names"]
        pred_name = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)

        # CAM needs gradients + lock for concurrency safety
        with CAM_LOCK:
            with torch.enable_grad():
                cam_hw_01, _, _ = STATE["cam_engine"](x1, class_id=int(pred_class))

        title = f"Pred: {pred_name} ({pred_prob:.3f})"
        png_bytes = overlay_png_bytes(
            img_chw=x_chw,
            cam_hw=cam_hw_01,
            alpha=float(STATE["cam_alpha"]),
            title=title,
        )

        REQ_COUNT.labels(endpoint=endpoint, status="200").inc()
        return Response(content=png_bytes, media_type="image/png")

    finally:
        REQ_LAT.labels(endpoint=endpoint).observe(time.time() - start)
