import torch

import numpy as np
import torch

def compute_class_weights_from_dataset(dataset, num_classes, device=None, strategy="inv_freq"):
    counts = np.zeros(int(num_classes), dtype=np.int64)
    for _, label in dataset:
        counts[int(label)] += 1
    if counts.sum() == 0:
        raise ValueError("Empty dataset: cannot compute class weights")
    if np.any(counts == 0):
        missing = np.where(counts == 0)[0].tolist()
        raise ValueError(f"Some classes have 0 samples in training split: {missing}")
    if strategy != "inv_freq":
        raise ValueError(f"Unknown strategy '{strategy}'. Supported: ['inv_freq']")
    freq = counts / counts.sum()
    weights = 1.0 / freq
    weights = weights / weights.mean()  # normalize so mean weight = 1.0
    w = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        w = w.to(device)
    return w, counts

def build_cross_entropy_with_weights(class_weights=None):
    """
    Build CrossEntropyLoss
    """
    if class_weights is None:
        return torch.nn.CrossEntropyLoss()
    return torch.nn.CrossEntropyLoss(weight=class_weights)
