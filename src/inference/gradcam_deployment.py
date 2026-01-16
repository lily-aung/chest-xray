from __future__ import annotations

import io, json, logging, os, threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2, numpy as np, torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Responseimport albumentations as A
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.inference.gradcam_core import build_cam_engine, get_module_by_path, overlay_png_bytes, predict_probs
from src.utils.image_utils import check_exposure
from src.utils.gradcam import GradCAM
from src.utils.gradcam_pp import GradCAMPlusPlus
from src.utils.smooth_gradcam import SmoothGradCAM


# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("gradcam_endpoint")


# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_BUNDLE_DIR = "/bundle"

# FastAPI app + state
app = FastAPI()
STATE: Dict[str, Any] = {}
CAM_LOCK = threading.Lock()


# Helpers: env + bundle
def env_bool(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def get_bundle_dir() -> Path:
    return Path(os.getenv("BUNDLE_DIR", DEFAULT_BUNDLE_DIR))

def get_device() -> torch.device:
    dev = os.getenv("DEVICE", "").strip()
    if dev:
        return torch.device(dev)
    # prefer mps if available (mac), then cuda, else cpu
    return torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu")


def jlog(event: str, **kwargs: Any) -> None:
    payload = {"event": event, **kwargs}
    log.info(json.dumps(payload, ensure_ascii=False))

def validate_bundle(bundle_dir: Path) -> None:
    required = ["manifest.json", "class_names.json", "model"]
    missing = [p for p in required if not (bundle_dir / p).exists()]
    if missing:
        raise FileNotFoundError(f"Bundle validation failed. Missing {missing} in {bundle_dir}")

def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {path} ({e})")


def load_model_from_bundle(bundle_dir: Path, device: torch.device) -> torch.nn.Module:
    model_dir = bundle_dir / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Bundle missing model directory: {model_dir}")
    model = mlflow.pytorch.load_model(str(model_dir), map_location="cpu")
    model.eval().to(device)
    return model


# =============================================================================
# Preprocess (mirrors infer_deployment)
def build_infer_transform(img_size: int, use_imagenet_norm: bool, is_rgb: bool) -> A.Compose:
    if use_imagenet_norm and is_rgb:
        norm = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0)
    else:
        norm = A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0)
    return A.Compose([A.Resize(img_size, img_size), norm, ToTensorV2()])

def maybe_apply_custom_clahe(img_gray: np.ndarray, use_custom_clahe: bool, 
                             clahe_clip_limit: float, clahe_tile_grid: Tuple[int, int]) -> np.ndarray:

    if not use_custom_clahe:
        return img_gray
    if check_exposure(img_gray):
        clahe = A.CLAHE( clip_limit=float(clahe_clip_limit),
            tile_grid_size=tuple(clahe_tile_grid), p=1.0)
        img_gray = clahe(image=img_gray)["image"]
    return img_gray

def to_model_channels(img_gray: np.ndarray, to_rgb: bool) -> np.ndarray:
    if to_rgb:
        return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)  # HWC, C=3
    return img_gray[..., None]  # HWC, C=1

def decode_upload_to_gray(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img

# Grad-CAM engine selection
def get_module_by_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    cur: torch.nn.Module = model
    for part in path.split("."):
        if part.isdigit():
            cur = cur[int(part)] 
        else:
            cur = getattr(cur, part)
    return cur

@torch.no_grad()
def predict_probs(model: torch.nn.Module, x1: torch.Tensor) -> tuple[np.ndarray, int, float]:
    logits = model(x1)
    probs = F.softmax(logits, dim=1)[0]
    pred_class = int(torch.argmax(probs).item())
    pred_prob = float(probs[pred_class].item())
    return probs.detach().cpu().numpy(), pred_class, pred_prob

def normalize_cam_method(s: Optional[str]) -> str:
    s = (s or "").strip().lower()
    mapping = {
        "gradcam": "gradcam",
        "gradcampp": "gradcampp",
        "gradcam++": "gradcampp",
        "gradcam-pp": "gradcampp",
        "smoothgradcam": "smoothgradcam",
        "smoothgradcampp": "smoothgradcampp",
        "smooth-gradcam": "smoothgradcam",
        "smooth-gradcampp": "smoothgradcampp",
        "gradcam-pp-smooth": "smoothgradcampp",
        "gradcampp-smooth": "smoothgradcampp" }
    return mapping.get(s, "smoothgradcampp")


def build_cam_engine( model: torch.nn.Module,
    target_layer: torch.nn.Module,
    method: str, smooth_n: int, smooth_std: float):
    if method == "gradcam":
        return GradCAM(model, target_layer)
    if method == "gradcampp":
        return GradCAMPlusPlus(model, target_layer)
    if method == "smoothgradcam":
        base = GradCAM(model, target_layer)
        return SmoothGradCAM(base, n_samples=smooth_n, noise_std=smooth_std)
    if method == "smoothgradcampp":
        base = GradCAMPlusPlus(model, target_layer)
        return SmoothGradCAM(base, n_samples=smooth_n, noise_std=smooth_std)
    raise ValueError(f"Unknown CAM method: {method}")


# =============================================================================
# Rendering: PNG in memory
def overlay_png_bytes(img_chw: torch.Tensor, cam_hw: np.ndarray, alpha: float, title: str) -> bytes:
    import matplotlib.pyplot as plt
    img = img_chw.detach().cpu().float()
    img2d = img.mean(dim=0).numpy() if img.shape[0] == 3 else img[0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img2d, cmap="gray")
    axes[0].axis("off")
    axes[0].set_title("Original")

    axes[1].imshow(img2d, cmap="gray")
    axes[1].imshow(cam_hw, alpha=float(alpha))
    axes[1].axis("off")
    axes[1].set_title("Grad-CAM")

    fig.suptitle(title)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# =============================================================================
# Startup: load bundle + model + cam engine once
def startup() -> None:
    bundle_dir = get_bundle_dir()
    manifest_path = bundle_dir / "manifest.json"
    model_dir = bundle_dir / "model"
    class_names_path = bundle_dir / "class_names.json"

    if not manifest_path.exists():
        raise RuntimeError(f"Missing manifest.json in bundle: {manifest_path}")
    if not model_dir.exists():
        raise RuntimeError(f"Missing model/ in bundle: {model_dir}")
    if not class_names_path.exists():
        raise RuntimeError(f"Missing class_names.json in bundle: {class_names_path}")

    manifest = json.loads(manifest_path.read_text())

    class_names_obj = json.loads(class_names_path.read_text())
    class_names = class_names_obj.get("class_names")
    if not class_names:
        raise RuntimeError("class_names.json missing class_names list")

    device = get_device()
    # Preprocess defaults controlled by ENV for maintainability
    img_size = int(os.getenv("IMG_SIZE", "224"))
    to_rgb = env_bool("TO_RGB", "0")
    use_imagenet_norm = env_bool("USE_IMAGENET_NORM", "0")

    use_custom_clahe = env_bool("USE_CUSTOM_CLAHE", "0")
    clahe_clip_limit = float(os.getenv("CLAHE_CLIP_LIMIT", "2.0"))
    clahe_tile_grid = tuple(int(x) for x in os.getenv("CLAHE_TILE_GRID", "8,8").split(","))

    tfm = build_infer_transform(img_size=img_size, use_imagenet_norm=use_imagenet_norm, is_rgb=to_rgb)

    model = load_model_from_bundle(bundle_dir=bundle_dir, device=device)

    # Grad-CAM config from ENV override or manifest fields
    target_layer_path = os.getenv("GRADCAM_TARGET_LAYER") or manifest.get("gradcam_target_layer")
    method_raw = os.getenv("GRADCAM_METHOD") or manifest.get("cam-method")
    cam_method = normalize_cam_method(method_raw)

    smooth_n = int(os.getenv("GRADCAM_SMOOTH_N", "25"))
    smooth_std = float(os.getenv("GRADCAM_SMOOTH_STD", "0.10"))
    cam_alpha = float(os.getenv("GRADCAM_ALPHA", "0.35"))

    cam_engine = None
    if target_layer_path:
        target_layer = get_module_by_path(model, str(target_layer_path))
        cam_engine = build_cam_engine(model, target_layer, cam_method, smooth_n=smooth_n, smooth_std=smooth_std)

    STATE.clear()

    STATE.update(
        bundle_dir=str(bundle_dir), manifest=manifest, class_names=class_names, device=device,
        model=model, tfm=tfm,
        # preprocess flags
        img_size=img_size, to_rgb=to_rgb, use_imagenet_norm=use_imagenet_norm,
        use_custom_clahe=use_custom_clahe,clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid=clahe_tile_grid,
        # cam
        cam_engine=cam_engine, cam_method=cam_method, cam_alpha=cam_alpha,
        cam_target_layer=str(target_layer_path) if target_layer_path else None,
        cam_smooth_n=smooth_n, cam_smooth_std=smooth_std)

    jlog("startup_ok", bundle_dir=str(bundle_dir), device=str(device), img_size=img_size,
          semver=manifest.get("semver"), run_id=manifest.get("mlflow_run_id"), 
          gradcam_enabled=bool(cam_engine), gradcam_target_layer=str(target_layer_path), gradcam_method=cam_method)



@app.on_event("startup")
def _on_startup() -> None:
    startup()

# =============================================================================
# Endpoint: /gradcam -> PNG

@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)) -> Response:
    if STATE.get("cam_engine") is None:
        raise HTTPException(
            status_code=503,
            detail="Grad-CAM not enabled (missing target layer / CAM engine). Set GRADCAM_TARGET_LAYER or manifest.gradcam_target_layer.",)

    img_bytes = await file.read()

    try:
        img_gray = decode_upload_to_gray(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # preprocess identical to infer_deployment
    img_gray = maybe_apply_custom_clahe(
        img_gray,
        use_custom_clahe=STATE["use_custom_clahe"],
        clahe_clip_limit=STATE["clahe_clip_limit"],
        clahe_tile_grid=STATE["clahe_tile_grid"])
    img = to_model_channels(img_gray, to_rgb=STATE["to_rgb"])

    out = STATE["tfm"](image=img)
    x_chw: torch.Tensor = out["image"]
    x1 = x_chw.unsqueeze(0).to(STATE["device"])

    probs_vec, pred_class, pred_prob = predict_probs(STATE["model"], x1)
    class_names = STATE["class_names"]
    pred_name = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)

    with CAM_LOCK:
        with torch.enable_grad():
            cam_hw_01, _, _ = STATE["cam_engine"](x1, class_id=int(pred_class))

    title = (
        f"Pred: {pred_name} ({pred_prob:.3f}) "
        f"| Layer: {STATE.get('cam_target_layer')} "
        f"| Method: {STATE.get('cam_method')}"
    )

    png_bytes = overlay_png_bytes(x_chw, cam_hw_01, alpha=float(STATE["cam_alpha"]), title=title)
    return Response(content=png_bytes, media_type="image/png")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())

def load_manifest(bundle_dir: Path) -> Dict[str, Any]:
    return load_json(bundle_dir / "manifest.json")

def load_class_names(bundle_dir: Path) -> List[str]:
    obj = load_json(bundle_dir / "class_names.json")
    names = obj.get("class_names")
    if not isinstance(names, list) or not names:
        raise ValueError("class_names.json missing class_names")
    return names
