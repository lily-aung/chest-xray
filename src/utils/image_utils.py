import cv2
import numpy as np, pandas as pd
from PIL import Image

def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0

def global_normalize(img, p_low=1, p_high=99):
    lo, hi = np.percentile(img, (p_low, p_high))
    img = np.clip(img, lo, hi)
    return (img - lo) / (hi - lo + 1e-6)

def gamma_correction(img, gamma=0.9):
    return np.clip(img ** gamma, 0, 1)

def clahe_safe(img, clip=2.0, grid=(8,8)):
    img8 = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(img8) / 255.0

def intensity_stats(df, max_samples=2000):
    vals = []
    for p in df.sample(min(len(df), max_samples), random_state=42)["path"]:
        img = Image.open(p).convert("L")
        vals.append(np.array(img).mean() / 255.0)
    return np.array(vals)

# Saturation fraction / intensity-based exposure metrics
def exposure_metrics(df, max_samples=None, black_thr=0.05, white_thr=0.95):
    d = df.sample(min(len(df), max_samples), random_state=42) if max_samples else df
    rows = []
    for p, lbl, sp in d[["path","label","split"]].itertuples(index=False):
        try:
            arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
            rows.append({
                "path": p, "label": lbl, "split": sp,
                "mean": float(arr.mean()), "std": float(arr.std()),
                "p01": float(np.quantile(arr, 0.01)), "p99": float(np.quantile(arr, 0.99)),
                "black_frac": float((arr <= black_thr).mean()),
                "white_frac": float((arr >= white_thr).mean()),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

def exposure_metrics_single_fromPath(image_path, black_thr=0.05, white_thr=0.95):
    try:
        arr = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
        metrics = {
            "path": image_path,
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p01": float(np.quantile(arr, 0.01)),
            "p99": float(np.quantile(arr, 0.99)),
            "black_frac": float((arr <= black_thr).mean()),
            "white_frac": float((arr >= white_thr).mean())}
        return metrics
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def exposure_metrics_single(image, black_thr=0.05, white_thr=0.95):
    try:
        arr = image
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
            if arr.max() > 1.5:
                arr /= 255.0
            arr = np.clip(arr, 0.0, 1.0)
        return {
            "mean": float(arr.mean()), "std": float(arr.std()),
            "p01": float(np.quantile(arr, 0.01)), "p99": float(np.quantile(arr, 0.99)),
            "black_frac": float((arr <= black_thr).mean()),"white_frac": float((arr >= white_thr).mean())}
    except Exception as e:
        print(f"Exposure metric error (array input): {e}")
        return None

def check_exposure(image, black_thr=0.40, white_thr=0.40, p99_thr=0.50, p01_thr=0.45, contrast_thr=0.35):
    """
    CLAHE application decision based on exposure metrics
    Parameters:
        image (np.ndarray): The input image.
        black_thr (float): Threshold for black pixel fraction.
        white_thr (float): Threshold for white pixel fraction.
        p99_thr (float): Threshold for the 99th percentile.
        p01_thr (float): Threshold for the 1st percentile.
        contrast_thr (float): Threshold for the contrast difference (p99 - p01).
    Returns:
        bool: True if CLAHE should be applied, False otherwise.
    """
    metrics = exposure_metrics_single(image)
    if not metrics:
        return False    
    severe_under = (metrics["black_frac"] > black_thr) or (metrics["p99"] < p99_thr)
    severe_over = (metrics["white_frac"] > white_thr) or (metrics["p01"] > p01_thr)
    low_contrast = (metrics["p99"] - metrics["p01"]) < contrast_thr
    # Return whether CLAHE should be applied #Rule: low contrast and no severe under/over exposure
    return low_contrast and not severe_under and not severe_over

def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return 0.6745 * (x - med) / mad

def img_metrics(orig, proc):
    return ( proc.mean(),proc.std(),
        np.corrcoef(orig.flatten(), proc.flatten())[0,1])

def remove_black_borders(img, thr=0.05, pad=5):
    assert img.ndim == 2, "Expected grayscale image"
    mask = img > thr
    coords = np.column_stack(np.where(mask))
    if len(coords) == 0: return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    y0, x0 = max(0,y0-pad), max(0,x0-pad)
    y1, x1 = min(img.shape[0],y1+pad), min(img.shape[1],x1+pad)
    return img[y0:y1, x0:x1]

def border_black_ratio(img, thr=0.05, border=20):
    """
    img: grayscale float image in [0,1]
    border: pixels from each edge to inspect
    """
    h, w = img.shape
    top ,bottom = img[:border, :], img[-border:, :]
    left, right= img[:, :border], img[:, -border:]
    border_pixels = np.concatenate([
        top.ravel(), bottom.ravel(),
        left.ravel(), right.ravel()])
    return (border_pixels < thr).mean()

def has_black_borders(img, thr=0.05, border=20, frac=0.4):
    return border_black_ratio(img, thr, border) > frac
def black_fraction(img, thr=0.05):
    return (img < thr).mean()