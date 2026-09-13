"""Minimal Grad-CAM implementation for ROPNet.

Produces a heatmap over the input image showing which region the model
relied on most for its prediction, purely from the model's own gradients
(no external service involved).
"""

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        self.model.eval()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, logits.softmax(dim=1).detach().cpu().numpy()[0]

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def overlay_heatmap(pil_image, cam: np.ndarray, alpha: float = 0.4):
    """Overlay a Grad-CAM heatmap (values in [0, 1]) onto a PIL image."""
    import matplotlib.cm as cm
    from PIL import Image

    image = pil_image.convert("RGB").resize((cam.shape[1], cam.shape[0]))
    heatmap = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap)
    blended = Image.blend(image, heatmap_img, alpha=alpha)
    return blended
