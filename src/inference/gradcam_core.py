import threading
import torch.nn.functional as F
import torch 
import cv2, numpy as np
from typing import Any, Dict, List, Optional, Tuple
from src.utils.gradcam import GradCAM
from src.utils.gradcam_pp import GradCAMPlusPlus
from src.utils.smooth_gradcam import SmoothGradCAM
import io 
CAM_LOCK = threading.Lock()

def get_module_by_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    cur: torch.nn.Module = model
    for part in path.split("."):
        if part.isdigit():
            cur = cur[int(part)]  # type: ignore[index]
        else:
            cur = getattr(cur, part)
    return cur

@torch.no_grad()
def predict_probs(model: torch.nn.Module, x1: torch.Tensor):
    logits = model(x1)
    probs = F.softmax(logits, dim=1)[0]
    pred = int(torch.argmax(probs).item())
    conf = float(probs[pred].item())
    return probs.detach().cpu().numpy(), pred, conf

def decode_upload_to_gray(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img

def overlay_png_bytes(img_chw: torch.Tensor, cam_hw: np.ndarray, alpha: float, title: str) -> bytes:
    import matplotlib.pyplot as plt
    img = img_chw.detach().cpu().float()
    img2d = img.mean(dim=0).numpy() if img.shape[0] == 3 else img[0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img2d, cmap="gray"); axes[0].axis("off"); axes[0].set_title("Original")
    axes[1].imshow(img2d, cmap="gray"); axes[1].imshow(cam_hw, alpha=float(alpha))
    axes[1].axis("off"); axes[1].set_title("Grad-CAM")
    fig.suptitle(title); fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def normalize_cam_method(s: Optional[str]) -> str:
    s = (s or "").strip().lower()
    m = {
        "gradcam": "gradcam", "gradcampp": "gradcampp",
        "gradcam++": "gradcampp", "gradcam-pp": "gradcampp",
        "smoothgradcam": "smoothgradcam",
        "smoothgradcampp": "smoothgradcampp",
        "gradcam-pp-smooth": "smoothgradcampp"}
    return m.get(s, "smoothgradcampp")

def build_cam_engine(model, target_layer, method: str, smooth_n: int = 25, smooth_std: float = 0.10):
    method = normalize_cam_method(method)
    if method == "gradcam":
        return GradCAM(model, target_layer)
    if method == "gradcampp":
        return GradCAMPlusPlus(model, target_layer)
    if method == "smoothgradcam":
        return SmoothGradCAM(GradCAM(model, target_layer), n_samples=smooth_n, noise_std=smooth_std)
    if method == "smoothgradcampp":
        return SmoothGradCAM(GradCAMPlusPlus(model, target_layer), n_samples=smooth_n, noise_std=smooth_std)
    raise ValueError(f"Unknown CAM method: {method}")
