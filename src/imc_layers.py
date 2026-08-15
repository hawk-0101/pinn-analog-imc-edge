"""
Analog In-Memory Computing linear layer.

Models:
  1. b-bit symmetric uniform weight quantisation
  2. Additive Gaussian conductance noise  (σ_eff = noise_std × mean|W|)

The noise scales with mean absolute weight value to approximate real PCM
behaviour where programming noise is proportional to conductance magnitude.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IMCLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        noise_std: float = 0.05,
        bias: bool = True,
    ):
        super().__init__()
        self.linear    = nn.Linear(in_features, out_features, bias=bias)
        self.bits      = bits
        self.noise_std = noise_std

    def _quantize(self, w: torch.Tensor) -> torch.Tensor:
        if self.bits >= 32:
            return w
        n_levels = 2 ** self.bits - 1          # symmetric: levels / 2 each side
        w_max    = w.abs().amax().clamp(min=1e-8)
        w_norm   = (w / w_max).clamp(-1.0, 1.0)
        w_q      = torch.round(w_norm * (n_levels / 2)) / (n_levels / 2)
        return w_q * w_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._quantize(self.linear.weight)
        if self.noise_std > 0:
            scale = w.abs().mean().clamp(min=1e-8)
            w = w + torch.randn_like(w) * self.noise_std * scale
        return F.linear(x, w, self.linear.bias)
