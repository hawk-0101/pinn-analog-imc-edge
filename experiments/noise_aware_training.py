"""
experiments/noise_aware_training.py
====================================
Phase 3b: Compare clean-train-deploy vs noise-injected-train-deploy.

Hypothesis: Clean training produces better PINN extrapolation because
noisy autograd corrupts the physics residual.

Setup: PINN trained with and without IMC noise, deployed at 8-bit 5% noise.
5 seeds each.
"""

import sys, os, csv, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import set_seed, generate_data, evaluate_model, T_DATA_END, T_TEST_END
from src.models import CleanNet, IMCNet
from src.imc_layers import IMCLinear
from src.physics_losses import oscillator_physics_loss

ROOT    = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.join(ROOT, "results", "figures")
TAB_DIR = os.path.join(ROOT, "results", "tables")
LOG_DIR = os.path.join(ROOT, "results", "logs")
for d in [FIG_DIR, TAB_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

BITS      = 8
NOISE_STD = 0.05
N_SEEDS   = 5
EPOCHS    = 2000
LR        = 5e-4
WARMUP    = 1000
LAM_MAX   = 0.5
N_EVAL    = 30


def copy_weights(src, dst):
    with torch.no_grad():
        for cp, dp in zip(src.parameters(), dst.parameters()):
            if cp.shape == dp.shape:
                dp.copy_(cp)


class NoisyTrainNet(nn.Module):
    """Network with IMC noise during training (for comparison)."""
    def __init__(self, hidden=32, bits=8, noise_std=0.05):
        super().__init__()
        self.net = nn.Sequential(
            IMCLinear(1, hidden, bits=bits, noise_std=noise_std), nn.Tanh(),
            IMCLinear(hidden, hidden, bits=bits, noise_std=noise_std), nn.Tanh(),
            IMCLinear(hidden, 1, bits=bits, noise_std=noise_std),
        )

    def forward(self, x):
        return self.net(x)


def train_clean_pinn(seed, epochs):
    set_seed(seed)
    t_d, x_d, t_c, _, _ = generate_data(seed)
    model = CleanNet(hidden=32)
    opt   = optim.Adam(model.parameters(), lr=LR)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        opt.zero_grad()
        l_data = torch.mean((model(t_d) - x_d) ** 2)
        lam    = LAM_MAX * min(1.0, ep / max(1, WARMUP))
        l_phys, _, _ = oscillator_physics_loss(model, t_c, w_residual=1.0, w_ic=5.0)
        loss = l_data + lam * l_phys
        loss.backward(); opt.step(); sch.step()
    return model


def train_noisy_pinn(seed, epochs):
    set_seed(seed)
    t_d, x_d, t_c, _, _ = generate_data(seed)
    model = NoisyTrainNet(hidden=32, bits=BITS, noise_std=NOISE_STD)
    opt   = optim.Adam(model.parameters(), lr=LR)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        opt.zero_grad()
        l_data = torch.mean((model(t_d) - x_d) ** 2)
        lam    = LAM_MAX * min(1.0, ep / max(1, WARMUP))
        l_phys, _, _ = oscillator_physics_loss(model, t_c, w_residual=1.0, w_ic=5.0)
        loss = l_data + lam * l_phys
        loss.backward(); opt.step(); sch.step()
    return model


def train_noisy_stdnn(seed, epochs):
    """StdNN trained with noise (data loss only) for completeness."""
    set_seed(seed)
    t_d, x_d, _, _, _ = generate_data(seed)
    model = NoisyTrainNet(hidden=32, bits=BITS, noise_std=NOISE_STD)
    opt   = optim.Adam(model.parameters(), lr=LR)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.mean((model(t_d) - x_d) ** 2)
        loss.backward(); opt.step(); sch.step()
    return model


def deploy_eval(model, seed, is_noisy_trained=False):
    _, _, _, t_test, x_exact = generate_data(seed)
    if is_noisy_trained:
        imc = model
    else:
        imc = IMCNet(hidden=32, bits=BITS, noise_std=NOISE_STD)
        copy_weights(model, imc)
    mse_i, mse_e, _ = evaluate_model(imc, t_test, x_exact, n_passes=N_EVAL)
    return mse_i, mse_e


CONFIGS = [
    ("CleanTrain_PINN",  train_clean_pinn,  False),
    ("NoisyTrain_PINN",  train_noisy_pinn,  True),
    ("NoisyTrain_StdNN", train_noisy_stdnn, True),
]


def main():
    rows = []
    log_path = os.path.join(LOG_DIR, "noise_aware.log")
    with open(log_path, "w") as lf:
        lf.write(f"noise-aware training: bits={BITS} noise={NOISE_STD}\n\n")
        t0 = time.time()
        for name, train_fn, is_noisy in CONFIGS:
            for seed in range(N_SEEDS):
                model = train_fn(seed, EPOCHS)
                mse_i, mse_e = deploy_eval(model, seed, is_noisy_trained=is_noisy)
                row = dict(method=name, seed=seed, mse_interp=mse_i, mse_extrap=mse_e)
                rows.append(row)
                msg = f"{name:20s} seed={seed} interp={mse_i:.6f} extrap={mse_e:.6f}"
                print(msg, flush=True)
                lf.write(msg + "\n")
        elapsed = time.time() - t0
        lf.write(f"\nTotal: {elapsed:.0f}s\n")

    # Save raw
    raw_path = os.path.join(TAB_DIR, "noise_aware_raw.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "seed", "mse_interp", "mse_extrap"])
        w.writeheader(); w.writerows(rows)
    print(f"Saved {raw_path}")

    # Summarise
    summary = []
    for name, _, _ in CONFIGS:
        sub = [r for r in rows if r["method"] == name]
        summary.append(dict(
            method=name,
            mse_interp_mean=np.mean([r["mse_interp"] for r in sub]),
            mse_interp_std =np.std( [r["mse_interp"] for r in sub]),
            mse_extrap_mean=np.mean([r["mse_extrap"] for r in sub]),
            mse_extrap_std =np.std( [r["mse_extrap"] for r in sub]),
        ))
    sum_path = os.path.join(TAB_DIR, "noise_aware_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    # Print
    print("\n" + "="*65)
    print(f"NOISE-AWARE TRAINING  ({BITS}-bit, noise={NOISE_STD})")
    print(f"{'Method':22s} {'extrap_mean':>14} {'extrap_std':>12}")
    print("-"*65)
    for s in summary:
        print(f"{s['method']:22s} {s['mse_extrap_mean']:14.6f} {s['mse_extrap_std']:12.6f}")
    print("="*65)

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    names  = [s["method"] for s in summary]
    means  = [s["mse_extrap_mean"] for s in summary]
    stds   = [s["mse_extrap_std"]  for s in summary]
    colors = ["seagreen", "tomato", "orange"]
    bars   = ax.bar(names, means, yerr=stds, capsize=6, color=colors,
                    edgecolor="k", alpha=0.85)
    ax.set_ylabel("Extrapolation MSE (mean +/- std)")
    ax.set_title(f"Clean vs Noise-Aware Training  ({BITS}-bit IMC, noise={NOISE_STD:.0%})")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "noise_aware_training.png")
    plt.savefig(fig_path, dpi=200); plt.close()
    print(f"Saved {fig_path}")

    json_path = os.path.join(TAB_DIR, "noise_aware.json")
    with open(json_path, "w") as jf:
        json.dump(summary, jf, indent=2)
    print(f"Total elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
