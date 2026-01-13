from .cnn import CNNBaseline
from .backbones import (build_resnet50, build_densenet121, build_efficientnet_b0, build_swin_tiny)

def build_model(model_cfg):
    """
    Build model from model config.

    Expected model_cfg:
      - model_cfg.name
      - model_cfg.num_classes
    Optional:
      - model_cfg.pretrained (bool, default True)
      - model_cfg.timm_name (str, optional, for timm-based models like Swin)
    """
    model_name = model_cfg.name.lower()
    num_classes = model_cfg.num_classes

    pretrained = getattr(model_cfg, "pretrained", True)
    timm_name = getattr(model_cfg, "timm_name", None)

    if model_name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes)

    elif model_name == "resnet50":
        return build_resnet50(num_classes=num_classes, pretrained=pretrained)

    elif model_name == "densenet121":
        return build_densenet121(num_classes=num_classes, pretrained=pretrained)

    elif model_name in ["efficientnet_b0", "efficientnetb0"]:
        return build_efficientnet_b0(num_classes=num_classes, pretrained=pretrained)

    elif model_name == "swin_tiny_patch4_window7_224":
        return build_swin_tiny(num_classes=num_classes, pretrained=pretrained)

    else:
        raise ValueError(
            f"Model '{model_name}' not recognized. Available: "
            "['cnn_baseline', 'cnn_attention', 'resnet50', 'densenet121', "
            "'efficientnet_b0', 'swin_tiny_patch4_window7_224']")
