### adapt from  https://github.com/jacobgil/pytorch-grad-cam
import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._save_activations)
        self.h2 = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out  # (B, C, h, w)

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]  # (B, C, h, w)

    def close(self):
        self.h1.remove()
        self.h2.remove()

    def __call__(self, x, class_id=None):
        """
        x: (1,1,H,W)
        returns cam (H,W), class_id, prob
        """
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)              # (1, num_classes)
        probs = F.softmax(logits, dim=1)

        if class_id is None:
            class_id = int(probs.argmax(dim=1).item())

        score = probs[0, class_id]
        logits[0, class_id].backward()      # backprop on the logit

        A = self.activations                # (1,C,h,w)
        dA = self.gradients                 # (1,C,h,w)

        # weights: GAP over spatial dims of gradients
        w = dA.mean(dim=(2, 3), keepdim=True)   # (1,C,1,1)
        cam = (w * A).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam = F.relu(cam)

        # upsample to input size
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]

        # normalize 0..1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy(), class_id, float(score.item())
