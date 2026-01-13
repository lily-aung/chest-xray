import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _norm(use_imagenet_norm, is_rgb):
    if use_imagenet_norm and is_rgb:
        return A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0)
    return A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0)
    
def train_base_transforms(cfg, use_imagenet_norm=False, is_rgb=False):
    return A.Compose([
        A.Rotate(limit=cfg.augmentation.rotation_deg, border_mode=cv2.BORDER_REPLICATE, p=0.5),
        A.ShiftScaleRotate(shift_limit=cfg.augmentation.shift_limit, scale_limit=cfg.augmentation.scale_limit, rotate_limit=0, border_mode=cv2.BORDER_REPLICATE, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3) if cfg.augmentation.gaussian_blur else A.NoOp(),
        A.Resize(cfg.data.img_size, cfg.data.img_size),
        _norm(use_imagenet_norm, is_rgb),
        ToTensorV2(),
    ])

def train_flip_transforms(cfg, use_imagenet_norm=False, is_rgb=False):
    return A.Compose([A.HorizontalFlip(p=1.0), *train_base_transforms(cfg, use_imagenet_norm, is_rgb).transforms])

def val_transforms(img_size, use_imagenet_norm=False, is_rgb=False):
    return A.Compose([
        A.Resize(img_size, img_size),
        _norm(use_imagenet_norm, is_rgb),
        ToTensorV2(),
    ])

