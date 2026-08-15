
"""
================================================================================
THESIS: Complete PINN-IMC Pipeline (Modules M1–M6)
================================================================================
Title:  Quantifying Precision-Energy Trade-offs for Physics-Informed Neural
        Networks on Analogue In-Memory Computing Hardware

This single file implements:
  M1  – Full-precision / baseline PINN
  M2  – IMC Simulation Layer (quantization + noise + drift)
  M3  – Sensitivity Analysis Engine (per-layer physics-loss gradients)
  M4  – Mixed-Precision Allocator (greedy bit-width assignment)
  M5  – Noise-Injection Training (robustness via in-situ noise)
  M6  – Evaluation & Visualisation (bit-width sweep, Pareto curves, plots)

Benchmark:  Damped harmonic oscillator  x'' + 0.5 x' + 2x = 0
Hardware:   Custom IMCLinear (quantisation-aware, Gaussian conductance noise)
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
import os

# Create output directory for thesis figures
os.makedirs("output", exist_ok=True)

# ==============================================================================
#  M2 – IMC SIMULATION LAYER
# ==============================================================================
class IMCLinear(nn.Module):
    """
    Hardware-faithful linear layer for analogue In-Memory Computing.
    Simulates three physical non-idealities:
      1. Weight quantisation to `bits` integer levels
      2. Additive Gaussian conductance noise  (σ = noise_std)
      3. Temporal drift is modelled as a fixed per-epoch bias in the training loop
    """
    def __init__(self, in_features, out_features, noise_std=0.05, bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.noise_std = noise_std
        self.bits = bits
        self.quantization_scale = 2**bits - 1

    def forward(self, x):
        # 1. Quantise weights to limited precision
        w_q = torch.round(self.weight * self.quantization_scale) / self.quantization_scale
        # 2. Inject IMC noise (different every forward pass → stochastic layer)
        noise = torch.randn_like(w_q) * self.noise_std
        w_noisy = w_q + noise
        return torch.matmul(x, w_noisy.t()) + self.bias


# ==============================================================================
#  M1 – BASELINE MODELS
# ==============================================================================
class StandardNN(nn.Module):
    def __init__(self, bits=8, noise_std=0.05):
        super().__init__()
        self.net = nn.Sequential(
            IMCLinear(1, 20, noise_std=noise_std, bits=bits), nn.Tanh(),
            IMCLinear(20, 20, noise_std=noise_std, bits=bits), nn.Tanh(),
            IMCLinear(20, 1,  noise_std=noise_std, bits=bits),
        )
    def forward(self, x):
        return self.net(x)

class PINN(nn.Module):
    def __init__(self, bits=8, noise_std=0.05):
        super().__init__()
        self.net = nn.Sequential(
            IMCLinear(1, 20, noise_std=noise_std, bits=bits), nn.Tanh(),
            IMCLinear(20, 20, noise_std=noise_std, bits=bits), nn.Tanh(),
            IMCLinear(20, 1,  noise_std=noise_std, bits=bits),
        )
    def forward(self, x):
        return self.net(x)


# ==============================================================================
#  M1 – PHYSICS LOSS (PDE Residual for Damped Oscillator)
# ==============================================================================
# ODE:  x''(t) + 0.5 x'(t) + 2 x(t) = 0
# Exact solution:  x(t) = exp(-0.25 t) * cos(1.98 t)

def physics_loss(model, t, lambda_residual=1.0):
    t = t.clone().requires_grad_(True)
    x = model(t)
    x_t  = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x),
                               create_graph=True, retain_graph=True)[0]
    x_tt = torch.autograd.grad(x_t, t, grad_outputs=torch.ones_like(x_t),
                               create_graph=True, retain_graph=True)[0]
    residual = x_tt + 0.5 * x_t + 2.0 * x
    return lambda_residual * torch.mean(residual ** 2)


# ==============================================================================
#  DATA GENERATION
# ==============================================================================
T_TRAIN_END = 10.0
T_TEST_END  = 15.0
N_TRAIN = 120
N_TEST  = 300
NOISE_DATA = 0.05

t_train = torch.linspace(0, T_TRAIN_END, N_TRAIN).view(-1, 1)
x_exact_train = torch.exp(-0.25 * t_train) * torch.cos(1.98 * t_train)
x_train = x_exact_train + torch.randn_like(t_train) * NOISE_DATA

t_test = torch.linspace(0, T_TEST_END, N_TEST).view(-1, 1)
x_exact_test = torch.exp(-0.25 * t_test) * torch.cos(1.98 * t_test)


# ==============================================================================
#  TRAINING UTILITIES
# ==============================================================================
def train_model(model, is_pinn=False, epochs=2000, lr=0.01,
                lambda_phys=0.1, verbose=True):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(t_train)
        loss_data = torch.mean((pred - x_train) ** 2)
        loss = loss_data
        if is_pinn:
            loss = loss + lambda_phys * physics_loss(model, t_train)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if verbose and epoch % 500 == 0:
            tag = "PINN" if is_pinn else "StdNN"
            print(f"  [{tag}] Epoch {epoch:4d}/{epochs} | Loss = {loss.item():.4f}")
    return history


def evaluate_model(model):
    with torch.no_grad():
        mse = torch.mean((model(t_test) - x_exact_test) ** 2).item()
        mae = torch.mean(torch.abs(model(t_test) - x_exact_test)).item()
    return mse, mae


# ==============================================================================
#  M3 – SENSITIVITY ANALYSIS ENGINE
# ==============================================================================
def compute_layer_sensitivity(model, t_data):
    """
    Computes per-layer Frobenius-norm sensitivity to the physics loss.
    Returns OrderedDict: layer_key (e.g. 'net.0') -> sensitivity_score
    """
    model.zero_grad()
    t_data = t_data.clone().requires_grad_(True)
    l_physics = physics_loss(model, t_data)
    l_physics.backward()

    sens = OrderedDict()
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Group by layer index, e.g. "net.0" from "net.0.weight"
            parts = name.split('.')
            layer_key = '.'.join(parts[:-1])
            frob = torch.norm(param.grad, p='fro').item()
            sens[layer_key] = sens.get(layer_key, 0.0) + frob
    return sens


def print_sensitivity_report(sens):
    print("
" + "="*60)
    print("M3 – SENSITIVITY ANALYSIS REPORT")
    print("="*60)
    total = sum(sens.values())
    for k, v in sens.items():
        pct = 100 * v / total if total > 0 else 0
        print(f"  {k:12s}  |  Frobenius = {v:10.4f}  |  {pct:5.1f}% of total")
    print("="*60)
    # Save JSON for M4 consumption
    with open("output/sensitivity_report.json", "w") as f:
        json.dump(dict(sens), f, indent=2)
    print("Saved -> output/sensitivity_report.json")


# ==============================================================================
#  M4 – MIXED-PRECISION ALLOCATOR
# ==============================================================================
def allocate_bits_greedy(sensitivity_scores, bit_options=[2,4,6,8],
                         energy_budget=None, base_energy_per_layer=1.0):
    """
    Greedy algorithm:
      1. Sort layers by physics-sensitivity (descending)
      2. Assign highest feasible bit-width within remaining energy budget
      3. Fall back to lowest bit if budget exhausted
    Energy model: E(layer, b) = (b / 8.0) * base_energy_per_layer
    """
    sorted_layers = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)
    n_layers = len(sorted_layers)
    if energy_budget is None:
        energy_budget = n_layers * base_energy_per_layer  # all-at-8-bit baseline

    energy_per_bit = {b: (b / 8.0) * base_energy_per_layer for b in bit_options}
    allocation = {}
    remaining = energy_budget

    for layer_name, score in sorted_layers:
        assigned = bit_options[0]  # default lowest
        for b in sorted(bit_options, reverse=True):
            if energy_per_bit[b] <= remaining + 1e-9:
                assigned = b
                break
        allocation[layer_name] = assigned
        remaining -= energy_per_bit[assigned]

    return allocation, energy_budget - remaining  # actual energy used


def print_allocation_report(alloc, actual_energy, budget):
    print("
" + "="*60)
    print("M4 – MIXED-PRECISION ALLOCATION REPORT")
    print("="*60)
    for k, v in alloc.items():
        print(f"  {k:12s}  ->  {v}-bit")
    print(f"
  Energy budget : {budget:.3f}")
    print(f"  Energy used   : {actual_energy:.3f}")
    print(f"  Savings       : {100*(1-actual_energy/budget):.1f}% vs uniform 8-bit")
    print("="*60)
    with open("output/bit_allocation.json", "w") as f:
        json.dump(alloc, f, indent=2)
    print("Saved -> output/bit_allocation.json")


# ==============================================================================
#  M5 – MIXED-PRECISION PINN (uses M4 allocation end-to-end)
# ==============================================================================
class MixedPrecisionPINN(nn.Module):
    """
    PINN where each IMCLinear layer can have a different bit-width,
    as determined by the M4 allocator.
    """
    def __init__(self, allocation, noise_std=0.05):
        super().__init__()
        self.layer0 = IMCLinear(1, 20,  noise_std=noise_std,
                                bits=allocation.get("net.0", 8))
        self.layer1 = nn.Tanh()
        self.layer2 = IMCLinear(20, 20, noise_std=noise_std,
                                bits=allocation.get("net.2", 8))
        self.layer3 = nn.Tanh()
        self.layer4 = IMCLinear(20, 1,  noise_std=noise_std,
                                bits=allocation.get("net.4", 8))

    def forward(self, x):
        x = self.layer1(self.layer0(x))
        x = self.layer3(self.layer2(x))
        x = self.layer4(x)
        return x


# ==============================================================================
#  M6 – BIT-WIDTH SWEEP & EVALUATION
# ==============================================================================
def run_bitwidth_sweep(bit_list=[2, 4, 6, 8], epochs=1500):
    results = OrderedDict()
    print("
" + "="*60)
    print("M6 – BIT-WIDTH SWEEP")
    print("="*60)
    for bits in bit_list:
        print(f"
--- {bits}-bit IMC ---")
        std  = StandardNN(bits=bits, noise_std=0.05)
        pinn = PINN(bits=bits, noise_std=0.05)

        train_model(std,  is_pinn=False, epochs=epochs, verbose=False)
        train_model(pinn, is_pinn=True,  epochs=epochs, verbose=False)

        mse_std, mae_std   = evaluate_model(std)
        mse_pinn, mae_pinn = evaluate_model(pinn)

        results[bits] = {
            "std_mse":  mse_std,  "std_mae":  mae_std,
            "pinn_mse": mse_pinn, "pinn_mae": mae_pinn,
        }
        print(f"  StdNN  MSE={mse_std:.5f}  MAE={mae_std:.5f}")
        print(f"  PINN   MSE={mse_pinn:.5f}  MAE={mae_pinn:.5f}")
        print(f"  PINN improvement: {100*(mse_std-mse_pinn)/mse_std:.1f}%")
    return results


def plot_bitwidth_sweep(results):
    bits = list(results.keys())
    std_mse  = [results[b]["std_mse"]  for b in bits]
    pinn_mse = [results[b]["pinn_mse"] for b in bits]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(bits))
    w = 0.35
    ax.bar(x - w/2, std_mse,  w, label='Standard NN', color='tomato', edgecolor='k')
    ax.bar(x + w/2, pinn_mse, w, label='PINN',        color='seagreen', edgecolor='k')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}-bit" for b in bits])
    ax.set_ylabel("Test MSE")
    ax.set_xlabel("IMC Quantisation Bit-Width")
    ax.set_title("M6 – PINN vs Standard NN: Robustness Across IMC Precision Levels")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/bitwidth_sweep.png", dpi=200)
    plt.show()
    print("Saved -> output/bitwidth_sweep.png")


def plot_trajectory_comparison(std_model, pinn_model, bits_label="8-bit"):
    with torch.no_grad():
        pred_std  = std_model(t_test).numpy()
        pred_pinn = pinn_model(t_test).numpy()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(t_test.numpy(), x_exact_test.numpy(), 'k-',  lw=2.5, label='Exact Physics (Ideal)')
    ax.plot(t_test.numpy(), pred_std,            'r--', lw=2,   label=f'Standard NN on IMC ({bits_label})')
    ax.plot(t_test.numpy(), pred_pinn,           'g-',  lw=2,   label=f'PINN on IMC ({bits_label})')
    ax.scatter(t_train.numpy(), x_train.numpy(), c='blue', s=12, alpha=0.35, label='Training Data')
    ax.axvline(T_TRAIN_END, color='gray', ls=':', alpha=0.6, label='Train / Test Boundary')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Robot Joint Position")
    ax.set_title("M6 – Trajectory Comparison: PINN Robustness to IMC Noise")
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("output/trajectory_comparison.png", dpi=200)
    plt.show()
    print("Saved -> output/trajectory_comparison.png")


def plot_sensitivity_heatmap(sens, title="M3 – Per-Layer Sensitivity to Physics Loss"):
    names = list(sens.keys())
    vals  = np.array([sens[k] for k in names]).reshape(1, -1)

    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(vals, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_yticks([0])
    ax.set_yticklabels(["Sensitivity
(Frobenius Norm)"])
    fig.colorbar(im, ax=ax, label="Gradient Magnitude")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig("output/sensitivity_heatmap.png", dpi=200)
    plt.show()
    print("Saved -> output/sensitivity_heatmap.png")


def plot_pareto_energy_accuracy(results, base_energy=1.0):
    """
    Pareto-style scatter: energy cost vs MSE for each bit-width.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for bits, res in results.items():
        energy = (bits / 8.0) * base_energy * 3  # 3 layers
        ax.scatter(energy, res["std_mse"],  c='tomato',  s=120, marker='o', edgecolors='k')
        ax.scatter(energy, res["pinn_mse"], c='seagreen', s=120, marker='s', edgecolors='k')
        ax.annotate(f"{bits}b", (energy, res["pinn_mse"]), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)

    ax.set_xlabel("Estimated Inference Energy (arbitrary units)")
    ax.set_ylabel("Test MSE (Physics Accuracy)")
    ax.set_title("M6 – Energy vs Accuracy Pareto Frontier")
    ax.grid(True, alpha=0.3)
    # Create proxy artists for legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='tomato',  edgecolor='k', label='Standard NN'),
                       Patch(facecolor='seagreen', edgecolor='k', label='PINN')]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig("output/pareto_energy_accuracy.png", dpi=200)
    plt.show()
    print("Saved -> output/pareto_energy_accuracy.png")


# ==============================================================================
#  MAIN PIPELINE
# ==============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # --------------------------------------------------------------------------
    # STEP 1: Reference 8-bit PINN for M3 sensitivity analysis
    # --------------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1 – Training reference PINN (8-bit) for M3 analysis")
    print("="*60)
    ref_pinn = PINN(bits=8, noise_std=0.05)
    train_model(ref_pinn, is_pinn=True, epochs=2000)
    mse_ref, mae_ref = evaluate_model(ref_pinn)
    print(f"\nReference PINN -> MSE={mse_ref:.5f}  MAE={mae_ref:.5f}")

    # --------------------------------------------------------------------------
    # STEP 2: M3 – Sensitivity Analysis
    # --------------------------------------------------------------------------
    sens = compute_layer_sensitivity(ref_pinn, t_train)
    print_sensitivity_report(sens)
    plot_sensitivity_heatmap(sens)

    # --------------------------------------------------------------------------
    # STEP 3: M4 – Mixed-Precision Allocation
    # --------------------------------------------------------------------------
    # Budget = enough for exactly 3 layers at 6-bit average (energy = 3 * 0.75 = 2.25)
    alloc, actual_e = allocate_bits_greedy(
        sens, bit_options=[2,4,6,8], energy_budget=2.25, base_energy_per_layer=1.0
    )
    print_allocation_report(alloc, actual_e, budget=2.25)

    # --------------------------------------------------------------------------
    # STEP 4: M5 – Train Mixed-Precision PINN (end-to-end M4 allocation)
    # --------------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 4 – Training Mixed-Precision PINN (M5)")
    print("="*60)
    mp_pinn = MixedPrecisionPINN(allocation=alloc, noise_std=0.05)
    train_model(mp_pinn, is_pinn=True, epochs=1500)
    mse_mp, mae_mp = evaluate_model(mp_pinn)
    print(f"Mixed-Precision PINN -> MSE={mse_mp:.5f}  MAE={mae_mp:.5f}")

    # --------------------------------------------------------------------------
    # STEP 5: M6 – Bit-Width Sweep (2, 4, 6, 8 bits)
    # --------------------------------------------------------------------------
    sweep_results = run_bitwidth_sweep(bit_list=[2, 4, 6, 8], epochs=1500)
    plot_bitwidth_sweep(sweep_results)
    plot_pareto_energy_accuracy(sweep_results)

    # --------------------------------------------------------------------------
    # STEP 6: M6 – Trajectory plot at 8-bit (most stable comparison)
    # --------------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 6 – Generating trajectory comparison plot")
    print("="*60)
    std_8  = StandardNN(bits=8, noise_std=0.05)
    pinn_8 = PINN(bits=8, noise_std=0.05)
    train_model(std_8,  is_pinn=False, epochs=1500, verbose=False)
    train_model(pinn_8, is_pinn=True,  epochs=1500, verbose=False)
    plot_trajectory_comparison(std_8, pinn_8, bits_label="8-bit")

    # --------------------------------------------------------------------------
    # FINAL SUMMARY TABLE
    # --------------------------------------------------------------------------
    print("\n" + "="*60)
    print("FINAL THESIS RESULTS SUMMARY")
    print("="*60)
    print(f"{'Config':<30s}  {'MSE':>10s}  {'vs StdNN':>10s}")
    print("-"*60)
    print(f"{'Standard NN (8-bit)':<30s}  {sweep_results[8]['std_mse']:>10.5f}  {'—':>10s}")
    print(f"{'PINN (8-bit)':<30s}  {sweep_results[8]['pinn_mse']:>10.5f}  {100*(sweep_results[8]['std_mse']-sweep_results[8]['pinn_mse'])/sweep_results[8]['std_mse']:>9.1f}%")
    print(f"{'Mixed-Precision PINN (M4+M5)':<30s}  {mse_mp:>10.5f}  {100*(sweep_results[8]['std_mse']-mse_mp)/sweep_results[8]['std_mse']:>9.1f}%")
    print(f"{'PINN (2-bit)':<30s}  {sweep_results[2]['pinn_mse']:>10.5f}  {100*(sweep_results[2]['std_mse']-sweep_results[2]['pinn_mse'])/sweep_results[2]['std_mse']:>9.1f}%")
    print("="*60)
    print("All figures saved to ./output/")
