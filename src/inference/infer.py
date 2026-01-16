from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2, numpy as np, pandas as pd, torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mlflow.pytorch

from src.utils.mlflow_utils import load_run_model_pytorch
from src.utils.image_utils import check_exposure
from src.utils.thresholds import predict_with_policy_idx


# Preprocessing
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_infer_transform(img_size: int, use_imagenet_norm: bool, is_rgb: bool) -> A.Compose:
    if use_imagenet_norm and is_rgb:
        norm = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0)
    else:
        norm = A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0)
    return A.Compose([
         A.Resize(img_size, img_size), norm,
        ToTensorV2()])

def load_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img

def maybe_apply_custom_clahe( img_gray: np.ndarray, use_custom_clahe: bool, 
                             clahe_clip_limit: float,  clahe_tile_grid: Tuple[int, int],) -> np.ndarray:
    if not use_custom_clahe:
        return img_gray
    if check_exposure(img_gray):
        clahe = A.CLAHE(clip_limit=float(clahe_clip_limit), tile_grid_size=tuple(clahe_tile_grid), p=1.0)
        img_gray = clahe(image=img_gray)["image"]
    return img_gray

def to_model_channels(img_gray: np.ndarray, to_rgb: bool) -> np.ndarray:
    # Albumentations expects HWC
    if to_rgb:
        img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)   # HWC, C=3
    else:
        img = img_gray[..., None]                          # HWC, C=1
    return img

def load_model_from_bundle(bundle_dir: Path, device: torch.device) -> torch.nn.Module:
    import mlflow.pytorch

    model_dir = bundle_dir / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Bundle missing model directory: {model_dir}")

    model = mlflow.pytorch.load_model(str(model_dir), map_location="cpu")
    model.eval().to(device)
    return model

def validate_bundle(bundle_dir: Path):
    required = ["manifest.json", "class_names.json", "thresholds_policy.json", "model"]
    missing = [p for p in required if not (bundle_dir / p).exists()]
    if missing:
        raise FileNotFoundError(f"Bundle missing: {missing} in {bundle_dir}")


# -------------------------
# Inference core
# -------------------------
@torch.no_grad()
def infer_one( model: torch.nn.Module, device: torch.device, image_path: str,
    tfm: A.Compose, class_names: List[str], thresholds: Optional[Dict[str, float]] = None,
    use_custom_clahe: bool = False, clahe_clip_limit: float = 2.0,
    clahe_tile_grid: Tuple[int, int] = (8, 8), to_rgb: bool = False) -> Dict:
    img_gray = load_grayscale(image_path)
    img_gray = maybe_apply_custom_clahe(img_gray, use_custom_clahe, clahe_clip_limit, clahe_tile_grid)
    img = to_model_channels(img_gray, to_rgb=to_rgb)  # HWC

    out = tfm(image=img)
    x = out["image"].unsqueeze(0).to(device)          # (1,C,H,W)

    logits = model(x)
    probs_t = F.softmax(logits, dim=1)[0]             # (K,)
    probs = probs_t.detach().cpu().numpy()

    pred_id = int(np.argmax(probs))
    conf = float(np.max(probs))

    pred_policy_id = None
    if thresholds:
        pred_policy_id = predict_with_policy_idx( probs=probs, class_names=class_names,
            thresholds=thresholds, priority=("Tuberculosis", "Pneumonia"), fallback="Normal")

    return { "path": str(image_path), "pred_id": pred_id, "pred_name": class_names[pred_id] if pred_id < len(class_names) else str(pred_id),
        "confidence": conf, "probs": probs.tolist(), "pred_policy_id": pred_policy_id,
        "pred_policy_name": (
            class_names[pred_policy_id] if pred_policy_id is not None and pred_policy_id < len(class_names) else None )}

def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--run_id", type=str, help="MLflow run_id (DEV ONLY)")
    src.add_argument("--bundle_dir", type=str, default=None, help="Path to model bundle (PROD)")

    parser.add_argument("--device", type=str, default=None, help="cuda|mps|cpu (optional)")

    # inputs
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--csv", type=str, default=None, help="CSV with column image_path")
    parser.add_argument("--img_col", type=str, default="image_path")

    # outputs
    parser.add_argument("--out", type=str, required=True, help="Output file or directory")

    # preprocessing config
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--names", type=str, default=None,
                    help="Comma-separated class names. Load from bundle.")
    parser.add_argument("--to_rgb", action="store_true", help="Use 3-channel input (for pretrained backbones)")
    parser.add_argument("--use_imagenet_norm", action="store_true")

    # custom CLAHE option
    parser.add_argument("--use_custom_clahe", action="store_true")
    parser.add_argument("--clahe_clip_limit", type=float, default=2.0)
    parser.add_argument("--clahe_tile_grid", type=str, default="8,8", help="e.g. 8,8")

    # policy thresholds
    parser.add_argument("--policy_json", type=str, default=None,
                        help="JSON file path or JSON string. e.g. '{\"Tuberculosis\":0.18,\"Pneumonia\":0.51}'")

    args = parser.parse_args()

    if (args.image is None) == (args.csv is None):
        raise ValueError("Provide exactly one of --image or --csv")

    # device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


    if args.bundle_dir is None and args.run_id is None:
        if Path("/bundle/manifest.json").exists():
            args.bundle_dir = "/bundle"
        else:
            raise ValueError("Provide --bundle_dir or --run_id (no baked bundle found at /bundle)")
        
    if args.bundle_dir:
        bundle_dir = Path(args.bundle_dir)
        print(bundle_dir)

        model = load_model_from_bundle(bundle_dir, device)
        # default class names from bundle
 
        # default policy from bundle
        if args.policy_json is None:
            thresholds = json.loads((bundle_dir / "thresholds_policy.json").read_text())

    else:
        model = load_run_model_pytorch(args.run_id)
        model.eval().to(device)


   # class names
    if args.names:
        class_names = [s.strip() for s in args.names.split(",") if s.strip()]
    else:
        class_names = json.loads((bundle_dir / "class_names.json").read_text())["class_names"]
    # thresholds
    thresholds = None
    if args.policy_json:
        p = Path(args.policy_json)
        thresholds = json.loads(p.read_text()) if p.exists() else json.loads(args.policy_json)

    # clahe grid
    g = tuple(int(x.strip()) for x in args.clahe_tile_grid.split(","))
    if len(g) != 2:
        raise ValueError("--clahe_tile_grid must be like '8,8'")

    # transforms
    tfm = build_infer_transform(args.img_size, use_imagenet_norm=args.use_imagenet_norm, is_rgb=args.to_rgb)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.image and out_path.suffix.lower() != ".json":
        out_path = out_path.with_suffix(".json")

    if args.image:
        res = infer_one( model=model, device=device, image_path=args.image,
            tfm=tfm, class_names=class_names, thresholds=thresholds,
            use_custom_clahe=args.use_custom_clahe, clahe_clip_limit=args.clahe_clip_limit, clahe_tile_grid=g,
            to_rgb=args.to_rgb)
        print(json.dumps(res, indent=2))

        if out_path.suffix.lower() == ".json":
            out_path.write_text(json.dumps(res, indent=2))
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            (out_path / "inference.json").write_text(json.dumps(res, indent=2))
        return

    df = pd.read_csv(args.csv)
    if args.img_col not in df.columns:
        raise ValueError(f"CSV missing column: {args.img_col}")

    rows = []
    for pth in df[args.img_col].astype(str).tolist():
        try:
            rows.append(infer_one( model=model,device=device, image_path=pth,
                tfm=tfm, class_names=class_names, thresholds=thresholds,
                use_custom_clahe=args.use_custom_clahe, clahe_clip_limit=args.clahe_clip_limit,
                clahe_tile_grid=g, to_rgb=args.to_rgb))
        except Exception as e:
            rows.append({"path": str(pth), "error": str(e)})

    out_df = pd.DataFrame(rows)
    if out_path.suffix.lower() != ".csv":
        out_path.mkdir(parents=True, exist_ok=True)
        out_csv = out_path / "inference.csv"
    else:
        out_csv = out_path
    out_df.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")


if __name__ == "__main__":
    main()
