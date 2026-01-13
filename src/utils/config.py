import yaml
from types import SimpleNamespace
from pathlib import Path

# backbone -> yaml filename
BACKBONE_TO_YAML = {
    "resnet50": "resnet50.yaml",
    "densenet121": "densenet121.yaml",
    "efficientnet_b0": "efficientnet_b0.yaml",
    "swin_tiny": "swin_tiny.yaml"}

def get_config_for_backbone(backbone: str, config_dir: str = "configs") -> str:
    """ Return YAML config path for backbone or raise if missing"""
    cfg_dir = Path(config_dir)
    if not cfg_dir.exists():
        raise RuntimeError(f"Config directory not found: {cfg_dir}")
    backbone_lower = backbone.lower()
    if backbone_lower not in BACKBONE_TO_YAML:
        raise RuntimeError(f"No YAML mapping for backbone '{backbone}'. Available: {list(BACKBONE_TO_YAML.keys())}")
    yaml_file = cfg_dir / BACKBONE_TO_YAML[backbone_lower]
    if not yaml_file.exists():
        raise RuntimeError(f"Config file not found: {yaml_file}")
    return str(yaml_file)

def load_config(path: str):
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    # Recursively convert dict -> namespace
    def dict_to_ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_ns(i) for i in d]
        return d
    return dict_to_ns(cfg_dict)

