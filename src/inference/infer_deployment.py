from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mlflow.pytorch 
from src.utils.image_utils import check_exposure
from src.utils.thresholds import predict_with_policy_idx


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("infer_deployment")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def validate_bundle(bundle_dir: Path) -> None:
    required = [ "manifest.json", "class_names.json","thresholds_policy.json", "model" ]
    missing = [p for p in required if not (bundle_dir / p).exists()]
    if missing:
        raise FileNotFoundError(
            f"Bundle validation failed. Missing {missing} in {bundle_dir}\n"
            f"Expected bundle layout:\n"
            f"  {bundle_dir}/manifest.json\n"
            f"  {bundle_dir}/class_names.json\n"
            f"  {bundle_dir}/thresholds_policy.json\n"
            f"  {bundle_dir}/model/ (MLflow model directory)\n" )
def load_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text())
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {path} ({e})")
    
def load_class_names(bundle_dir: Path) -> List[str]:
    obj = load_json(bundle_dir / "class_names.json")
    names = obj.get("class_names")
    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        raise ValueError(f"Invalid class_names.json format: {bundle_dir / 'class_names.json'}")
    names = [x.strip() for x in names if x.strip()]
    if not names:
        raise ValueError("class_names.json produced empty class list")
    return names

def load_thresholds(bundle_dir: Path) -> Dict[str, float]:
    thresholds = load_json(bundle_dir / "thresholds_policy.json")
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds_policy.json must be a JSON object mapping class_name -> threshold")
    out: Dict[str, float] = {}
    for k, v in thresholds.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            raise ValueError(f"Invalid threshold value for {k}: {v}")
    return out

def load_manifest(bundle_dir: Path) -> Dict:
    return load_json(bundle_dir / "manifest.json")

def resolve_preprocess(manifest: Dict, overrides: Dict) -> Dict:
    mp = manifest.get("preprocess", {}) if isinstance(manifest, dict) else {}
    if not isinstance(mp, dict):
        mp = {}
    def pick(key: str, default):
        return overrides[key] if overrides.get(key) is not None else mp.get(key, default)
    cfg = {
        "img_size": int(pick("img_size", 224)),
        "to_rgb": bool(pick("to_rgb", False)),
        "use_imagenet_norm": bool(pick("use_imagenet_norm", False)),
        "use_custom_clahe": bool(pick("use_custom_clahe", False)),
        "clahe_clip_limit": float(pick("clahe_clip_limit", 2.0)),
        "clahe_tile_grid": tuple(pick("clahe_tile_grid", (8, 8)))}
    g = cfg["clahe_tile_grid"]
    if isinstance(g, list):
        g = tuple(g)
    if not (isinstance(g, tuple) and len(g) == 2):
        raise ValueError(f"Invalid clahe_tile_grid in manifest/overrides: {cfg['clahe_tile_grid']}")
    cfg["clahe_tile_grid"] = (int(g[0]), int(g[1]))
    return cfg

def load_model_from_bundle(bundle_dir: Path, device: torch.device) -> torch.nn.Module:
    model_dir = bundle_dir / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Bundle missing model directory: {model_dir}")
    model = mlflow.pytorch.load_model(str(model_dir), map_location="cpu")
    model.eval().to(device)
    return model

# Image preprocessing
def build_infer_transform(img_size: int, use_imagenet_norm: bool, is_rgb: bool) -> A.Compose:
    if use_imagenet_norm and is_rgb:
        norm = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0)
    else:
        norm = A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0)# for 1-channel 
    return A.Compose([A.Resize(img_size, img_size), norm, ToTensorV2()])

def load_grayscale(path: Union[str, Path]) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img

def apply_custom_clahe( img_gray: np.ndarray, use_custom_clahe: bool,
    clahe_clip_limit: float, clahe_tile_grid: Tuple[int, int]) -> np.ndarray:
    if not use_custom_clahe:
        return img_gray
    if check_exposure(img_gray):
        clahe = A.CLAHE(clip_limit=float(clahe_clip_limit), tile_grid_size=tuple(clahe_tile_grid), p=1.0)
        img_gray = clahe(image=img_gray)["image"]
    return img_gray


def to_model_channels(img_gray: np.ndarray, to_rgb: bool) -> np.ndarray:
    if to_rgb:
        return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)  # HWC, C=3
    return img_gray[..., None]  # HWC, C=1

# Inference core
@torch.no_grad()
def infer_one( model: torch.nn.Module, device: torch.device,
    image_path: str, tfm: A.Compose, class_names: List[str], thresholds: Dict[str, float],
    use_custom_clahe: bool, clahe_clip_limit: float,
    clahe_tile_grid: Tuple[int, int], to_rgb: bool) -> Dict:
    img_gray = load_grayscale(image_path)
    img_gray = apply_custom_clahe(img_gray, use_custom_clahe, clahe_clip_limit, clahe_tile_grid)
    img = to_model_channels(img_gray, to_rgb=to_rgb)  # HWC

    out = tfm(image=img)
    x = out["image"].unsqueeze(0).to(device)  # (1,C,H,W)

    logits = model(x)
    probs_t = F.softmax(logits, dim=1)[0]
    probs = probs_t.detach().cpu().numpy()

    pred_id = int(np.argmax(probs))
    conf = float(np.max(probs))

    pred_policy_id = predict_with_policy_idx( probs=probs, class_names=class_names, 
        thresholds=thresholds, priority=("Tuberculosis", "Pneumonia"),fallback="Normal")

    return { "path": str(image_path), "pred_id": pred_id,
        "pred_name": class_names[pred_id] if pred_id < len(class_names) else str(pred_id),
        "confidence": conf, "probs": probs.tolist(),
        "pred_policy_id": int(pred_policy_id), "pred_policy_name": class_names[pred_policy_id] if pred_policy_id < len(class_names) else None,
        "thresholds": thresholds}


def pick_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Production inference using a bundled model at /bundle (or --bundle_dir).")
    parser.add_argument("--bundle_dir", type=str, default=None, help="Model bundle directory. Default: /bundle if present.")
    parser.add_argument("--device", type=str, default=None, help="cpu|cuda (optional)")

    # inputs
    parser.add_argument("--image", type=str, default=None, help="Single image path (inside container filesystem)")
    parser.add_argument("--csv", type=str, default=None, help="CSV file path with an image column")
    parser.add_argument("--img_col", type=str, default="image_path")

    # outputs
    parser.add_argument("--out", type=str, required=True, help="For --image: must be a .json file. For --csv: .csv file or directory.")

    # Optional preprocessing overrides (prefer manifest defaults)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--to_rgb", action="store_true", help="Override: force RGB input")
    parser.add_argument("--use_imagenet_norm", action="store_true", help="Override: force ImageNet normalization")
    parser.add_argument("--use_custom_clahe", action="store_true", help="Override: enable custom CLAHE gating")
    parser.add_argument("--clahe_clip_limit", type=float, default=None)
    parser.add_argument("--clahe_tile_grid", type=str, default=None, help="Override: e.g. 8,8")

    args = parser.parse_args()

    if (args.image is None) == (args.csv is None):
        raise ValueError("Provide exactly one of --image or --csv")

    device = pick_device(args.device)

    # bundle resolution
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else Path("/bundle")
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle dir not found: {bundle_dir}. Pass --bundle_dir or bake to /bundle in the image.")

    validate_bundle(bundle_dir)
    manifest = load_manifest(bundle_dir)
    class_names = load_class_names(bundle_dir)
    thresholds = load_thresholds(bundle_dir)

    # preprocessing overrides
    overrides = {
        "img_size": args.img_size, "to_rgb": True if args.to_rgb else None, "use_imagenet_norm": True if args.use_imagenet_norm else None,
        "use_custom_clahe": True if args.use_custom_clahe else None, "clahe_clip_limit": args.clahe_clip_limit,
        "clahe_tile_grid": tuple(int(x) for x in args.clahe_tile_grid.split(",")) if args.clahe_tile_grid else None}
    pp = resolve_preprocess(manifest, overrides)

    log.info("Using bundle: %s", bundle_dir)
    log.info("Device: %s", device)
    log.info("Classes: %s", class_names)
    log.info("Policy thresholds: %s", thresholds)
    log.info("Preprocess: %s", pp)

    model = load_model_from_bundle(bundle_dir, device)
    tfm = build_infer_transform(pp["img_size"], use_imagenet_norm=pp["use_imagenet_norm"], is_rgb=pp["to_rgb"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # single image
    if args.image:
        if out_path.suffix.lower() != ".json":
            raise ValueError("For --image, --out must be a .json file (e.g. /work/pred.json).")

        res = infer_one( model=model, device=device, image_path=args.image,
            tfm=tfm, class_names=class_names, thresholds=thresholds,
            use_custom_clahe=pp["use_custom_clahe"], clahe_clip_limit=pp["clahe_clip_limit"],clahe_tile_grid=pp["clahe_tile_grid"],
            to_rgb=pp["to_rgb"])
        out_path.write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        log.info("Saved: %s", out_path)
        return

    # batch csv
    df = pd.read_csv(args.csv)
    if args.img_col not in df.columns:
        raise ValueError(f"CSV missing column: {args.img_col}")

    rows = []
    for pth in df[args.img_col].astype(str).tolist():
        try:
            rows.append(
                infer_one( model=model, device=device, image_path=pth, tfm=tfm,
                    class_names=class_names, thresholds=thresholds, use_custom_clahe=pp["use_custom_clahe"],
                    clahe_clip_limit=pp["clahe_clip_limit"], clahe_tile_grid=pp["clahe_tile_grid"],
                    to_rgb=pp["to_rgb"]))
        except Exception as e:
            rows.append({"path": str(pth), "error": str(e)})

    out_df = pd.DataFrame(rows)
    if out_path.suffix.lower() == ".csv":
        out_csv = out_path
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        out_csv = out_path / "inference.csv"

    out_df.to_csv(out_csv, index=False)
    log.info("Saved: %s", out_csv)

if __name__ == "__main__":
    main()
