# src/run_gradcam.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data.dataset import build_test_dataset
from src.utils.config import load_config
from src.utils.gradcam import GradCAM
from src.utils.gradcam_pp import GradCAMPlusPlus
from src.utils.mlflow_utils import load_run_model_pytorch
from src.utils.smooth_gradcam import SmoothGradCAM


def overlay_and_save(img_tensor: torch.Tensor, cam: np.ndarray, out_path: Path, title: str = "") -> None:
    """Save a side-by-side visualization: left=original grayscale, right=original+CAM overlay."""
    img = img_tensor.detach().cpu().float()
    if img.ndim != 3:
        raise ValueError(f"Expected img_tensor (C,H,W), got shape {tuple(img.shape)}")
    img2d = img.mean(dim=0).numpy() if img.shape[0] == 3 else img[0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img2d, cmap="gray"); axes[0].axis("off"); axes[0].set_title("Original")
    axes[1].imshow(img2d, cmap="gray"); axes[1].imshow(cam, alpha=0.35); axes[1].axis("off"); axes[1].set_title("Grad-CAM")
    fig.suptitle(title); fig.tight_layout(); 
    fig.savefig(out_path, dpi=200); plt.close(fig)


def find_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Tailored for CNNBaseline: features[8] is the last conv layer."""
    try:
        layer = model.features[12]#[8]
    except Exception as e:
        raise ValueError("Expected model.features[8] for CNNBaseline, but could not access it.") from e
    if not isinstance(layer, torch.nn.Conv2d):
        raise ValueError(f"model.features[8] is not Conv2d. Got: {type(layer)}")
    return layer


def get_item_by_path(dataset, wanted_path: str):
    """Dataset returns (image, label, path). Matches by exact / suffix / filename."""
    wanted = str(wanted_path)
    wanted_name = Path(wanted).name
    wanted_norm = wanted.replace("\\", "/")

    for i in range(len(dataset)):
        item = dataset[i]
        if not (isinstance(item, (tuple, list)) and len(item) == 3):
            continue
        x, y, p = item
        p_str = str(p)
        p_norm = p_str.replace("\\", "/")
        p_name = Path(p_str).name
        if p_str == wanted or wanted_norm.endswith(p_norm) or p_norm.endswith(wanted_norm) or p_name == wanted_name:
            return x, y, p
    return None


def format_prediction(y_true: int, y_pred: int, prob: float, class_names: list[str]) -> str:
    true_name = class_names[y_true] if 0 <= y_true < len(class_names) else str(y_true)
    pred_name = class_names[y_pred] if 0 <= y_pred < len(class_names) else str(y_pred)
    return f"True: {true_name} | Pred: {pred_name} | Confidence: {prob:.3f}"


@torch.no_grad()
def predict_probs(model: torch.nn.Module, x1: torch.Tensor) -> tuple[np.ndarray, int, float]:
    """Returns probs_vec (K,), pred_class, pred_prob."""
    logits = model(x1)
    probs = F.softmax(logits, dim=1)[0]
    pred_class = int(torch.argmax(probs).item())
    pred_prob = float(probs[pred_class].item())
    return probs.detach().cpu().numpy(), pred_class, pred_prob


def build_cam_engine(method: str, model: torch.nn.Module, target_layer: torch.nn.Module, smooth_n: int, smooth_std: float):
    base_engine = None
    if method == "gradcam":
        base_engine = GradCAM(model, target_layer); cam_engine = base_engine
    elif method == "gradcampp":
        base_engine = GradCAMPlusPlus(model, target_layer); cam_engine = base_engine
    elif method == "smoothgradcam":
        base_engine = GradCAM(model, target_layer); cam_engine = SmoothGradCAM(base_engine, n_samples=smooth_n, noise_std=smooth_std)
    elif method == "smoothgradcampp":
        base_engine = GradCAMPlusPlus(model, target_layer); cam_engine = SmoothGradCAM(base_engine, n_samples=smooth_n, noise_std=smooth_std)
    else:
        raise ValueError(f"Unknown method: {method}")
    return base_engine, cam_engine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", type=str, required=True, help="MLflow run_id for the CNN model")
    p.add_argument("--cnn_cfg", type=str, default="configs/cnn.yaml")
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True, help="Where to write Grad-CAM images")
    p.add_argument("--case_list_json", type=str, required=True, help="JSON containing {'paths':[...]} e.g. reports/.../analysis/sample_cases.json")
    p.add_argument("--max_cases", type=int, default=50)
    p.add_argument("--cam_mode", type=str, default="pred", choices=["pred", "true", "fixed", "topk"], help="Which class to explain: pred / true / fixed / topk")
    p.add_argument("--fixed_class", type=str, default=None, help="Class name when cam_mode='fixed' (e.g., Tuberculosis)")
    p.add_argument("--topk", type=int, default=2, help="K for cam_mode='topk' (explain top-K classes).")
    p.add_argument("--names", type=str, default="Normal,Pneumonia,Tuberculosis", help="Comma-separated class names in index order.")
    p.add_argument("--method", type=str, default="gradcam", choices=["gradcam", "gradcampp", "smoothgradcam", "smoothgradcampp"], help="CAM method to use.")
    p.add_argument("--only_correct", action="store_true", help="Generate Grad-CAM only for correctly predicted samples.")
    p.add_argument("--only_incorrect", action="store_true", help="Generate Grad-CAM only for misclassified samples.")
    p.add_argument("--smooth_n", type=int, default=25, help="Samples for SmoothGrad* methods.")
    p.add_argument("--smooth_std", type=float, default=0.10, help="Noise std for SmoothGrad* methods.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    class_names = [s.strip() for s in args.names.split(",") if s.strip()]
    if not class_names:
        raise ValueError("--names is empty; provide class names like Normal,Pneumonia,Tuberculosis")
    name_to_id = {n: i for i, n in enumerate(class_names)}

    model = load_run_model_pytorch(args.run_id); model.eval().to(device)
    cfg = load_config(args.cnn_cfg)
    ds = build_test_dataset(cfg, args.test_csv)

    case_path = Path(args.case_list_json)
    if not case_path.exists():
        raise FileNotFoundError(f"Case list JSON not found: {case_path}")
    obj = json.loads(case_path.read_text())
    paths = obj.get("paths", [])
    if not paths:
        raise ValueError(f"No 'paths' found in {case_path}. Expected JSON like: {{\"paths\": [ ... ]}}")
    paths = paths[: args.max_cases]

    target_layer = find_target_layer(model)
    base_engine, cam_engine = build_cam_engine(args.method, model, target_layer, args.smooth_n, args.smooth_std)

    saved = 0
    print("GradCAM - Processing in progress >>")
    for img_path in paths:
        got = get_item_by_path(ds, img_path)
        if got is None:
            print(f"[WARN] Could not match path in dataset: {img_path}")
            continue

        x, y, p = got
        x1 = x.unsqueeze(0).to(device)
        probs_vec, pred_class, pred_prob = predict_probs(model, x1)

        is_correct = (int(pred_class) == int(y))
        if (args.only_correct and not is_correct) or (args.only_incorrect and is_correct):
            continue

        if args.cam_mode == "pred":
            explain_ids = [int(pred_class)]
        elif args.cam_mode == "true":
            explain_ids = [int(y)]
        elif args.cam_mode == "fixed":
            if args.fixed_class is None:
                raise ValueError("--fixed_class is required when --cam_mode fixed")
            if args.fixed_class not in name_to_id:
                raise ValueError(f"--fixed_class '{args.fixed_class}' not in {class_names}")
            explain_ids = [int(name_to_id[args.fixed_class])]
        elif args.cam_mode == "topk":
            k = max(1, int(args.topk))
            explain_ids = [int(i) for i in np.argsort(-probs_vec)[:k]]
        else:
            raise ValueError(f"Unknown cam_mode: {args.cam_mode}")

        title_main = format_prediction(int(y), int(pred_class), float(pred_prob), class_names)

        for cid in explain_ids:
            with torch.enable_grad():
                cam, _, _ = cam_engine(x1, class_id=int(cid))
            cid_name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
            cid_prob = float(probs_vec[cid]) if 0 <= cid < len(probs_vec) else float("nan")
            title = f"{title_main} | Explaining: {cid_name} ({cid_prob:.3f})"

            out_path = (out_dir / f"cam_{saved:03d}__{args.method}__explain_{cid_name}__{Path(str(p)).name}").with_suffix(".png")
            overlay_and_save(x, cam, out_path, title=title)
            saved += 1

    if base_engine is not None and hasattr(base_engine, "close"):
        base_engine.close()
    print(f"[Success] Saved {saved} Grad-CAM images to {out_dir}")


if __name__ == "__main__":
    main()
