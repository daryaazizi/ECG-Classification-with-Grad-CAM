import torch
import torch.nn.functional as F


class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class):
        output = self.model(input_tensor)

        self.model.zero_grad()
        target = output[:, target_class]
        target.backward()

        weights = torch.mean(self.gradients, dim=-1, keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)

        cam = F.relu(cam)

        cam_resized = F.interpolate(
            cam.unsqueeze(0),
            size=input_tensor.shape[-1],
            mode="linear",
            align_corners=False,
        )
        cam_resized = cam_resized.squeeze()

        max_cam = torch.max(cam_resized)
        if max_cam > 0:
            cam_resized = cam_resized / max_cam
        return cam_resized
