import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Set random seeds for full reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #  deterministic algorithms (PyTorch ≥1.8) ==> need ot disable it on google colab TPU
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
