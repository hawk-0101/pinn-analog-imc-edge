# PINN on Analog In-Memory Computing (IMC) Hardware — Edge AI

Physics-Informed Neural Networks (PINNs) running on **Analog In-Memory Computing** hardware simulation. This thesis project systematically tests how PINNs behave under the two dominant analog hardware non-idealities: **quantization noise** (limited bit precision) and **stochastic analog hardware noise** (conductance drift).

**Key result:** PINNs deliver **22.5× better extrapolation than standard NNs in FP32**, ~**74% lower L2 error at 8-bit precision**, and physics constraints reduce extrapolation failure from **145× to 2× under hardware noise** — showing that physics priors are a practical defense mechanism for deploying neural networks on noisy edge/analog hardware.

## Overview

Standard neural networks degrade catastrophically when deployed on low-precision analog hardware. This work asks whether embedding physics constraints (the governing ODE/PDE) into the loss function makes networks *robust to analog non-idealities*. We simulate analog IMC layers with configurable bit-width and stochastic noise, train three method families (StdNN, PINN, AdaPINN), and sweep a **180-configuration grid (4 bit-widths × 3 noise levels × 5 seeds × 3 methods)**.

Benchmarks:
- **Damped harmonic oscillator ODE** (interpolation + extrapolation regimes)
- **1D heat equation PDE**

## Key Results

| Metric | Result |
|---|---|
| Extrapolation gain over StdNN (FP32) | **22.5×** |
| L2 error reduction at 8-bit | **~74%** |
| Extrapolation failure under noise | reduced from **145× to 2×** |
| AdaPINN improvement at 2-bit | **95%** |
| Robustness across widths 16–128 | 92–97% extrapolation gain retained |

All physics consistency tests pass. Full sweep runs in ~3 minutes.

## Tech Stack

- **Python** (PyTorch 2.x, NumPy, Matplotlib)
- Physics-Informed Neural Networks (PINN / AdaPINN)
- Analog IMC simulation (quantization + conductance noise layers)
- Hardware: ASUS TUF F16 (RTX 5060, local training only — no cloud)

## How to Run

```bash
pip install torch numpy matplotlib

# Full 180-configuration sweep (~3 min)
python experiments/damped_oscillator.py

# Quick smoke test (1 seed, 500 epochs)
python experiments/damped_oscillator.py --quick

# Benchmark 2: 1D heat equation
python experiments/heat_equation.py

# Phase 3 experiments
python experiments/lambda_ablation.py
python experiments/noise_aware_training.py
python experiments/network_size_ablation.py
python experiments/interp_extrap_report.py
```

## Repository Structure

```
src/
  models.py          — CleanNet, IMCNet, MixedPrecisionIMCNet, ND variants
  imc_layers.py      — IMCLinear (b-bit quantization + Gaussian conductance noise)
  physics_losses.py  — Oscillator + heat equation residual/IC/BC losses
  bit_allocator.py   — Greedy mixed-precision bit allocation
  sensitivity.py     — Per-layer physics-loss sensitivity
experiments/
  damped_oscillator.py     — Canonical sweep experiment
  heat_equation.py         — 1D PDE benchmark
  lambda_ablation.py       — Physics-weight (λ) sensitivity
  noise_aware_training.py  — Clean-train vs noise-injected comparison
  network_size_ablation.py — Width ablation (16–128)
  interp_extrap_report.py  — Dedicated interp/extrap analysis
results/
  tables/   — CSV/JSON results
  figures/  — PNG plots
legacy/     — Archived original (buggy) code
```

## Author

**Harish K** | M.Tech AI, Amrita Vishwa Vidyapeetham
- Email: 11harishjackie@gmail.com
- LinkedIn: https://linkedin.com/in/harish-k-717161186

Academic use. Not yet released as open-source.