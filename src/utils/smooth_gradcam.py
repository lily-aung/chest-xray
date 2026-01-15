# src/utils/smooth_gradcam.py
from __future__ import annotations
import numpy as np
import torch

class SmoothGradCAM:
    """
    SmoothGrad-CAM: average multiple Grad-CAM maps from noisy versions of the input.
    Works with your existing GradCAM engine (callable returning cam, class_id, score).
    """
    def __init__(self, cam_engine, n_samples: int = 25, noise_std: float = 0.10, clamp: bool = True):
        """
        n_samples: number of noisy samples
        noise_std: std of Gaussian noise in normalized tensor units
                  (after Normalize + ToTensorV2). Typical: 0.05 to 0.20
        clamp: clamp noisy inputs to a reasonable range to avoid extreme artifacts
        """
        self.cam_engine = cam_engine
        self.n_samples = int(n_samples)
        self.noise_std = float(noise_std)
        self.clamp = bool(clamp)

    def __call__(self, x1: torch.Tensor, class_id: int | None = None):
        """
        x1: (1,C,H,W) torch tensor on device
        returns: cam_avg (H,W) numpy 0..1, class_id_used, score (from clean pass)
        """
        assert x1.ndim == 4 and x1.shape[0] == 1, "Expected x1 shape (1,C,H,W)"

        cams = []
        # use the clean pass for class selection and score
        cam0, cid, score = self.cam_engine(x1, class_id=class_id)

        for _ in range(self.n_samples):
            noise = torch.randn_like(x1) * self.noise_std
            xn = x1 + noise
            if self.clamp:
                # After normalization, values can be outside 0..1.
                # Clamp gently to avoid adversarial extremes.
                xn = torch.clamp(xn, -3.0, 3.0)

            cam_n, _, _ = self.cam_engine(xn, class_id=cid)
            cams.append(cam_n)

        cam_avg = np.mean(np.stack(cams, axis=0), axis=0)

        # normalize 0..1 again (safety)
        cam_avg = cam_avg - cam_avg.min()
        cam_avg = cam_avg / (cam_avg.max() + 1e-8)

        return cam_avg, cid, float(score)
