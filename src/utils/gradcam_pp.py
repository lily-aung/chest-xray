from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

class GradCAMPlusPlus:
    """
    Grad-CAM++ for CNNs.
    Uses first, second, and third-order gradients to compute per-channel weights.
    Works best when target_layer is a conv layer producing (B,C,h,w).
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.h1 = target_layer.register_forward_hook(self._save_activations)
        self.h2 = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out  # (B,C,h,w)

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]  # (B,C,h,w)

    def close(self):
        self.h1.remove()
        self.h2.remove()

    def __call__(self, x: torch.Tensor, class_id: int | None = None):
        """
        x: (1,C,H,W)
        returns: cam(H,W) numpy 0..1, class_id, prob
        """
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)              # (1,K)
        probs = F.softmax(logits, dim=1)    # (1,K)

        if class_id is None:
            class_id = int(probs.argmax(dim=1).item())

        # Backprop on the logit for stability
        logits[0, class_id].backward(retain_graph=False)

        A = self.activations                # (1,C,h,w)
        dA = self.gradients                 # (1,C,h,w)

        if A is None or dA is None:
            raise RuntimeError("Hooks did not capture activations/gradients. Check target_layer.")

        # Grad-CAM++ weights
        # alpha_ij^k = d2y / (2*d2y + sum(A * d3y))
        # weights_k = sum(alpha_ij^k * relu(dy))
        # We'll compute in a numerically stable way.
        grad_1 = dA
        grad_2 = dA ** 2
        grad_3 = dA ** 3

        # sum over spatial dims
        sum_A_grad_3 = (A * grad_3).sum(dim=(2, 3), keepdim=True)  # (1,C,1,1)

        eps = 1e-8
        alpha = grad_2 / (2.0 * grad_2 + sum_A_grad_3 + eps)       # (1,C,h,w)

        relu_grad = F.relu(grad_1)                                  # (1,C,h,w)
        weights = (alpha * relu_grad).sum(dim=(2, 3), keepdim=True)  # (1,C,1,1)

        cam = (weights * A).sum(dim=1, keepdim=True)                # (1,1,h,w)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy(), class_id, float(probs[0, class_id].item())
