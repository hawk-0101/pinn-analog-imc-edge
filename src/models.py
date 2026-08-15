"""
Network architectures for PINN-IMC thesis.

All networks share the same hidden layer sizes (32, 32) and Tanh activations
so that any performance difference is due to the loss function or hardware,
not to architectural advantage.
"""

import torch
import torch.nn as nn
from .imc_layers import IMCLinear


# ── Clean (FP32, no hardware noise) ────────────────────────────────────────
class CleanNet(nn.Module):
    """Standard feedforward, full FP32 precision (no IMC simulation)."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


# ── IMC-deployed ────────────────────────────────────────────────────────────
class IMCNet(nn.Module):
    """
    All linear layers replaced with IMCLinear.
    Used as base class for both StdNN and PINN (same architecture, different
    training loss).
    """
    def __init__(self, hidden=32, bits=8, noise_std=0.05):
        super().__init__()
        self.net = nn.Sequential(
            IMCLinear(1, hidden, bits=bits, noise_std=noise_std), nn.Tanh(),
            IMCLinear(hidden, hidden, bits=bits, noise_std=noise_std), nn.Tanh(),
            IMCLinear(hidden, 1, bits=bits, noise_std=noise_std),
        )

    def forward(self, x):
        return self.net(x)


# ── Mixed-precision IMC ─────────────────────────────────────────────────────
class MixedPrecisionIMCNet(nn.Module):
    """
    Different bit-widths per layer.
    Typically used with output layer at higher precision (most
    physics-loss-sensitive) and hidden layers at lower precision.
    """
    def __init__(self, hidden=32, bits_hidden=4, bits_output=8, noise_std=0.05):
        super().__init__()
        self.l0 = IMCLinear(1, hidden, bits=bits_hidden, noise_std=noise_std)
        self.a0 = nn.Tanh()
        self.l1 = IMCLinear(hidden, hidden, bits=bits_hidden, noise_std=noise_std)
        self.a1 = nn.Tanh()
        self.l2 = IMCLinear(hidden, 1, bits=bits_output, noise_std=noise_std)

    def forward(self, x):
        return self.l2(self.a1(self.l1(self.a0(self.l0(x)))))


# ── Variable-input variants (for PDE benchmarks) ─────────────────────────
class CleanNetND(nn.Module):
    """Feedforward with configurable input dimension (for heat eqn: in_dim=2)."""
    def __init__(self, in_dim=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class IMCNetND(nn.Module):
    """IMC-deployed variant with configurable input dimension."""
    def __init__(self, in_dim=2, hidden=32, bits=8, noise_std=0.05):
        super().__init__()
        self.net = nn.Sequential(
            IMCLinear(in_dim, hidden, bits=bits, noise_std=noise_std), nn.Tanh(),
            IMCLinear(hidden, hidden, bits=bits, noise_std=noise_std), nn.Tanh(),
            IMCLinear(hidden, 1, bits=bits, noise_std=noise_std),
        )

    def forward(self, x):
        return self.net(x)
