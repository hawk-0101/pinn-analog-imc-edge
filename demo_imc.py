"""
Live demo: the analog IMC substrate, made visible.

Run:  python demo_imc.py

Shows, in order:
  1. A weight row in FP32 vs. b-bit symmetric quantization (discrete levels).
  2. Two consecutive forward passes -> different stochastic conductance noise,
     proving the noise is resampled per pass (cycle-to-cycle variation), not a
     fixed offset that could simply be calibrated away.
  3. How the effective output drifts across 30 Monte-Carlo passes, which is why
     every reported number is averaged over MC passes and 5 seeds.

This is intentionally instant and dependency-light so it cannot fail in front
of an audience. It uses the exact IMCLinear layer from the thesis (src/).
"""

import torch
from src.imc_layers import IMCLinear

torch.manual_seed(0)
LINE = "-" * 64


def banner(text):
    print("\n" + LINE)
    print(text)
    print(LINE)


def show_quantization():
    banner("1.  WEIGHT QUANTIZATION  (FP32  ->  b-bit discrete levels)")
    w = torch.linspace(-0.9, 0.9, 8)  # a synthetic weight row
    for bits in (8, 4, 2):
        layer = IMCLinear(8, 1, bits=bits, noise_std=0.0)
        with torch.no_grad():
            wq = layer._quantize(w)
        n_levels = 2 ** bits - 1
        print(f"\n  {bits}-bit  ({n_levels} levels):")
        print("    FP32 :", " ".join(f"{v:+.3f}" for v in w.tolist()))
        print("    quant:", " ".join(f"{v:+.3f}" for v in wq.tolist()))


def show_stochastic_noise():
    banner("2.  STOCHASTIC CONDUCTANCE NOISE  (resampled every forward pass)")
    layer = IMCLinear(4, 1, bits=4, noise_std=0.05)
    x = torch.ones(1, 4)
    print("\n  Same input, same weights, 4-bit, 5% noise -> 5 forward passes:")
    for i in range(5):
        y = layer(x).item()
        print(f"    pass {i + 1}:  y = {y:+.5f}")
    print("\n  -> Output changes each pass. The noise is NOT a fixed offset;")
    print("     it models cycle-to-cycle device variation. A PINN must satisfy")
    print("     its physics residual THROUGH this moving target.")


def show_mc_spread():
    banner("3.  WHY WE AVERAGE 30 MONTE-CARLO PASSES")
    layer = IMCLinear(4, 1, bits=4, noise_std=0.05)
    x = torch.ones(1, 4)
    ys = torch.tensor([layer(x).item() for _ in range(30)])
    print(f"\n  30 passes:  mean = {ys.mean():+.5f}   std = {ys.std():.5f}")
    print(f"  single-pass range: [{ys.min():+.5f}, {ys.max():+.5f}]")
    print("\n  -> A single deployment draw is misleading. Reported MSE = E[MSE]")
    print("     over MC passes, then over 5 seeds.")


if __name__ == "__main__":
    show_quantization()
    show_stochastic_noise()
    show_mc_spread()
    print("\n" + LINE)
    print("Next visual: results/figures/osc_trajectory_8b5n.png")
    print("  StdNN diverges past the training horizon; the PINN tracks the")
    print("  analytic solution under the same quantization + noise.")
    print(LINE + "\n")
