import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd
from src.utils.image_utils import load_gray, remove_black_borders
# Project root = chest-xray/
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # chest-xray/
print(PROJECT_ROOT)
RAW_ROOT = PROJECT_ROOT / "data/raw/archive"
BACKUP_ROOT = PROJECT_ROOT / "data/blackborder"
CSV_PATH = PROJECT_ROOT / "data/eda/black_border_removal_stats.csv"

BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
black_border_true = pd.read_csv(CSV_PATH)

def save_gray(img, path):
    img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img8).save(path)

def resolve_csv_path(p):
    p = Path(p)
    if p.parts[0] == "..":
        p = Path(*p.parts[1:])
    return (PROJECT_ROOT / p).resolve()

black_set = set(black_border_true["path"])
for p in tqdm(black_set, desc="Processing black-border images"):
    src = resolve_csv_path(p) #    # Convert CSV path to absolute
    if not src.exists():
        print("Missing:", src)
        continue
    rel = src.relative_to(RAW_ROOT) #Get relative path inside archive/
    backup_path = BACKUP_ROOT / rel #    # Build backup path that mirrors folder structure
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists(): #    #Back up original image (never overwrite)
        shutil.copy2(src, backup_path)
    print("src path: {} copied to: {}", src, backup_path)
    img = load_gray(src) 
    cropped = remove_black_borders(img)  #Process and overwrite raw image
    save_gray(cropped, src)


