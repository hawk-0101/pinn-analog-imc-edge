"""
Per-layer physics-loss sensitivity for mixed-precision allocation.
"""

from collections import OrderedDict
import torch
from .physics_losses import oscillator_physics_loss


def compute_layer_sensitivity(
    model: torch.nn.Module,
    t_data: torch.Tensor,
) -> OrderedDict:
    """
    Compute Frobenius norm of physics-loss gradient w.r.t. each layer's weights.
    Higher value → layer is more critical for physics constraint accuracy.
    """
    model.zero_grad()
    loss, _, _ = oscillator_physics_loss(model, t_data.clone())
    loss.backward()

    sens = OrderedDict()
    for name, param in model.named_parameters():
        if param.grad is not None and "weight" in name:
            frob = torch.norm(param.grad, p="fro").item()
            # Group by layer prefix (e.g. "net.0")
            layer_key = ".".join(name.split(".")[:-1])
            sens[layer_key] = sens.get(layer_key, 0.0) + frob

    model.zero_grad()
    return sens
