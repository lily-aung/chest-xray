import os
from pathlib import Path

import cv2
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, ConcatDataset

from src.utils.image_utils import check_exposure
from src.data.transforms import train_base_transforms, train_flip_transforms, val_transforms


def infer_class_names_and_num_classes(dataset, fallback_names=None, fallback_num_classes=None):
    """
    Infer class index -> name mapping from common dataset attributes.
      - dataset.class_to_idx (dict name->idx)
      - dataset.classes (list in idx order)
      - dataset.idx_to_class (dict idx->name)
    """
    if hasattr(dataset, "class_to_idx") and isinstance(dataset.class_to_idx, dict) and dataset.class_to_idx:
        mapping = dataset.class_to_idx  # name -> idx
        num_classes = int(max(mapping.values()) + 1)
        class_names = [None] * num_classes
        for name, idx in mapping.items():
            if 0 <= int(idx) < num_classes:
                class_names[int(idx)] = str(name)
        class_names = [cn if cn is not None else str(i) for i, cn in enumerate(class_names)]
        return num_classes, class_names

    if hasattr(dataset, "classes") and isinstance(dataset.classes, (list, tuple)) and len(dataset.classes) > 0:
        class_names = [str(x) for x in dataset.classes]
        return len(class_names), class_names

    if hasattr(dataset, "idx_to_class") and isinstance(dataset.idx_to_class, dict) and dataset.idx_to_class:
        mapping = dataset.idx_to_class  # idx -> name
        num_classes = int(max(mapping.keys()) + 1)
        class_names = [str(mapping.get(i, i)) for i in range(num_classes)]
        return num_classes, class_names

    if fallback_names is not None and len(fallback_names) > 0:
        return len(fallback_names), [str(x) for x in fallback_names]

    if fallback_num_classes is not None:
        n = int(fallback_num_classes)
        return n, [str(i) for i in range(n)]
    raise ValueError("Could not infer class names/num_classes from dataset and no fallback provided.")


class ChestXrayDataset(Dataset):
    """
    CSV must contain columns:
      - image_path : str (relative or absolute)
      - label      : int
    """
    def __init__( self,csv_file,img_dir=None, transforms=None,use_custom_clahe=False,
        clahe_clip_limit=2.0, clahe_tile_grid=(8, 8), to_rgb=False, strict=True):
        self.df = pd.read_csv(csv_file)
        if "image_path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(f"{csv_file} must contain columns: image_path, label")

        self.img_dir = None if img_dir in (None, "null", "None") else str(img_dir)
        self.transforms = transforms
        self.use_custom_clahe = bool(use_custom_clahe)
        self.to_rgb = bool(to_rgb)
        self.strict = bool(strict)
        self.clahe = A.CLAHE( clip_limit=float(clahe_clip_limit), tile_grid_size=tuple(clahe_tile_grid), p=1.0)
        print("ChestXrayDataset : Feature Exraction in Progress >>> " )
        print("Contrast Enhancement : ",  use_custom_clahe)
    def __len__(self):
        return len(self.df)

    def _resolve_path(self, image_path: str) -> str:
        p = str(image_path).strip()
        if os.path.isabs(p): return p
        if self.img_dir: return os.path.join(self.img_dir, p)
        return p

    def __getitem__(self, idx):
        row = self.df.iloc[int(idx)]
        path = self._resolve_path(row["image_path"])

        if not os.path.exists(path):
            msg = f"Image not found: {path}"
            if self.strict: raise FileNotFoundError(msg)
            return self._empty_sample(int(row["label"]))

        if os.path.getsize(path) == 0:
            msg = f"Empty image file: {path}"
            if self.strict:  raise ValueError(msg)
            return self._empty_sample(int(row["label"]))

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            msg = f"Failed to load image (cv2.imread returned None): {path}"
            if self.strict: raise ValueError(msg)
            return self._empty_sample(int(row["label"]))

        # CLAHE on under/over-exposed images
        if self.use_custom_clahe and check_exposure(image):
            
            image = self.clahe(image=image)["image"]
        # HWC formatting for Albumentations
        if self.to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # HWC, C=3
        else:
            image = image[..., None]  # HWC, C=1

        if self.transforms is not None:
            out = self.transforms(image=image)
            image = out["image"]

        return image, int(row["label"]), row["image_path"]

    def _empty_sample(self, label: int):
        import numpy as np
        h = w = 224
        c = 3 if self.to_rgb else 1
        img = np.zeros((h, w, c), dtype=np.uint8)
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        return img, int(label)

def build_datasets(cfg):
    use_imagenet_norm = bool(getattr(cfg.data, "use_imagenet_norm", False))
    # Pretrained backbones expect 3-channel input
    model_name = str(cfg.model.name).lower()
    to_rgb = bool(getattr(cfg.model, "pretrained", False)) and model_name not in {"cnn_baseline", "cnn_attention"}

    train_base = ChestXrayDataset( csv_file=cfg.data.train_csv, img_dir=getattr(cfg.data, "img_dir", None),
        transforms=train_base_transforms(cfg, use_imagenet_norm=use_imagenet_norm, is_rgb=to_rgb),
        use_custom_clahe=cfg.augmentation.custom_clahe,
        clahe_clip_limit=cfg.augmentation.clahe_clip_limit,
        clahe_tile_grid=tuple(cfg.augmentation.clahe_tile_grid),
        to_rgb=to_rgb, strict=True)
    if cfg.augmentation.horizontal_flip:
        train_flip = ChestXrayDataset( csv_file=cfg.data.train_csv,  img_dir=getattr(cfg.data, "img_dir", None),
            transforms=train_flip_transforms(cfg, use_imagenet_norm=use_imagenet_norm, is_rgb=to_rgb),
            use_custom_clahe=cfg.augmentation.custom_clahe,
            clahe_clip_limit=cfg.augmentation.clahe_clip_limit,
            clahe_tile_grid=tuple(cfg.augmentation.clahe_tile_grid),
            to_rgb=to_rgb, strict=True)
        train_dataset = ConcatDataset([train_base, train_flip])
    else:
        train_dataset = train_base

    val_dataset = ChestXrayDataset( csv_file=cfg.data.val_csv,  img_dir=getattr(cfg.data, "img_dir", None),
        transforms=val_transforms(cfg.data.img_size, use_imagenet_norm=use_imagenet_norm, is_rgb=to_rgb),
        use_custom_clahe=cfg.augmentation.custom_clahe, to_rgb=to_rgb, strict=True)

    return train_dataset, val_dataset


def build_test_dataset_bk(cfg, test_csv = None):
    use_imagenet_norm = bool(getattr(cfg.data, "use_imagenet_norm", False))
    model_name = str(cfg.model.name).lower()
    to_rgb = bool(getattr(cfg.model, "pretrained", False)) and model_name not in {"cnn_baseline", "cnn_attention"}
    if test_csv is None:
        raise ValueError("test_csv is required to build test dataset")
    if test_csv:
        test_csv = getattr(cfg.data, "test_csv", None)
    test_dataset = ChestXrayDataset( csv_file=test_csv, img_dir=getattr(cfg.data, "img_dir", None),
        transforms=val_transforms(cfg.data.img_size, use_imagenet_norm=use_imagenet_norm, is_rgb=to_rgb),
        use_custom_clahe=False, to_rgb=to_rgb, strict=True)
    return test_dataset

def build_test_dataset(cfg, test_csv=None):
    use_imagenet_norm = bool(getattr(cfg.data, "use_imagenet_norm", False))
    model_name = str(cfg.model.name).lower()
    to_rgb = bool(getattr(cfg.model, "pretrained", False)) and model_name not in {"cnn_baseline", "cnn_attention"}
    if test_csv is None:
        test_csv = getattr(cfg.data, "test_csv", None)
    if test_csv is None:
        raise ValueError("test_csv is required (passed in or cfg.data.test_csv)")

    test_dataset = ChestXrayDataset(
        csv_file=test_csv, img_dir=getattr(cfg.data, "img_dir", None),
        transforms=val_transforms(cfg.data.img_size, use_imagenet_norm=use_imagenet_norm, is_rgb=to_rgb),
        use_custom_clahe=bool(getattr(cfg.augmentation, "custom_clahe", False)),
        clahe_clip_limit=float(getattr(cfg.augmentation, "clahe_clip_limit", 2.0)),
        clahe_tile_grid=tuple(getattr(cfg.augmentation, "clahe_tile_grid", (8, 8))),
        to_rgb=to_rgb, strict=True)
    return test_dataset
