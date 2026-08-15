"""
experiments/network_size_ablation.py
=====================================
Phase 3c: Network size ablation over hidden={16, 32, 64, 128}.

Tests PINN at 8-bit 5% noise to see if larger networks recover
more from physics guidance or if small nets suffice.
"""

import sys, os, csv, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import set_seed, generate_data, evaluate_model
from src.models import CleanNet, IMCNet
from src.physics_losses import oscillator_physics_loss

ROOT    = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.join(ROOT, "results", "figures")
TAB_DIR = os.path.join(ROOT, "results", "tables")
LOG_DIR = os.path.join(ROOT, "results", "logs")
for d in [FIG_DIR, TAB_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

HIDDEN_LIST = [16, 32, 64, 128]
BITS        = 8
NOISE_STD   = 0.05
N_SEEDS     = 5
EPOCHS      = 2000
LR          = 5e-4
WARMUP      = 1000
LAM_MAX     = 0.5
N_EVAL      = 30


def copy_weights(src, dst):
    with torch.no_grad():
        for cp, dp in zip(src.parameters(), dst.parameters()):
            if cp.shape == dp.shape:
                dp.copy_(cp)


def train_std(seed, epochs, hidden):
    set_seed(seed)
    t_d, x_d, _, _, _ = generate_data(seed)
    model = CleanNet(hidden=hidden)
    opt   = optim.Adam(model.parameters(), lr=LR)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.mean((model(t_d) - x_d) ** 2)
        loss.backward(); opt.step(); sch.step()
    return model


def train_pinn(seed, epochs, hidden):
    set_seed(seed)
    t_d, x_d, t_c, _, _ = generate_data(seed)
    model = CleanNet(hidden=hidden)
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


def deploy_eval(model, hidden, seed):
    _, _, _, t_test, x_exact = generate_data(seed)
    imc = IMCNet(hidden=hidden, bits=BITS, noise_std=NOISE_STD)
    copy_weights(model, imc)
    mse_i, mse_e, _ = evaluate_model(imc, t_test, x_exact, n_passes=N_EVAL)
    return mse_i, mse_e


def main():
    rows = []
    log_path = os.path.join(LOG_DIR, "netsize_ablation.log")
    with open(log_path, "w") as lf:
        lf.write(f"network size ablation: bits={BITS} noise={NOISE_STD}\n\n")
        t0 = time.time()
        for hidden in HIDDEN_LIST:
            for method_name, train_fn in [("StdNN", train_std), ("PINN", train_pinn)]:
                for seed in range(N_SEEDS):
                    model = train_fn(seed, EPOCHS, hidden)
                    mse_i, mse_e = deploy_eval(model, hidden, seed)
                    row = dict(
                        method=method_name, hidden=hidden, seed=seed,
                        mse_interp=mse_i, mse_extrap=mse_e,
                    )
                    rows.append(row)
                    msg = (f"{method_name:6s} h={hidden:3d} seed={seed} "
                           f"interp={mse_i:.6f} extrap={mse_e:.6f}")
                    print(msg, flush=True)
                    lf.write(msg + "\n")
        elapsed = time.time() - t0
        lf.write(f"\nTotal: {elapsed:.0f}s\n")

    # Raw CSV
    raw_path = os.path.join(TAB_DIR, "netsize_ablation_raw.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "hidden", "seed", "mse_interp", "mse_extrap"])
        w.writeheader(); w.writerows(rows)
    print(f"Saved {raw_path}")

    # Summarise
    summary = []
    for hidden in HIDDEN_LIST:
        for method in ["StdNN", "PINN"]:
            sub = [r for r in rows if r["hidden"] == hidden and r["method"] == method]
            summary.append(dict(
                method=method, hidden=hidden,
                mse_interp_mean=np.mean([r["mse_interp"] for r in sub]),
                mse_interp_std =np.std( [r["mse_interp"] for r in sub]),
                mse_extrap_mean=np.mean([r["mse_extrap"] for r in sub]),
                mse_extrap_std =np.std( [r["mse_extrap"] for r in sub]),
            ))
    sum_path = os.path.join(TAB_DIR, "netsize_ablation_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    # Print
    print("\n" + "="*70)
    print(f"NETWORK SIZE ABLATION  ({BITS}-bit, noise={NOISE_STD})")
    print(f"{'Hidden':>8} {'StdNN_extrap':>14} {'PINN_extrap':>14} {'Improvement':>12}")
    print("-"*70)
    for hidden in HIDDEN_LIST:
        s_std  = next(s for s in summary if s["hidden"]==hidden and s["method"]=="StdNN")
        s_pinn = next(s for s in summary if s["hidden"]==hidden and s["method"]=="PINN")
        imp = (1 - s_pinn["mse_extrap_mean"] / s_std["mse_extrap_mean"]) * 100
        print(f"{hidden:8d} {s_std['mse_extrap_mean']:14.6f} {s_pinn['mse_extrap_mean']:14.6f} {imp:11.1f}%")
    print("="*70)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, color, marker in [("StdNN", "tomato", "s"), ("PINN", "seagreen", "o")]:
        sub = [s for s in summary if s["method"] == method]
        hs  = [s["hidden"] for s in sub]
        ms  = [s["mse_extrap_mean"] for s in sub]
        es  = [s["mse_extrap_std"]  for s in sub]
        ax.errorbar(hs, ms, yerr=es, fmt=f"{marker}-", color=color,
                    capsize=5, lw=2, markersize=8, label=method)
    ax.set_xlabel("Hidden layer width")
    ax.set_ylabel("Extrapolation MSE")
    ax.set_yscale("log")
    ax.set_xticks(HIDDEN_LIST)
    ax.set_title(f"Network Size Ablation  ({BITS}-bit IMC, noise={NOISE_STD:.0%})")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "netsize_ablation.png")
    plt.savefig(fig_path, dpi=200); plt.close()
    print(f"Saved {fig_path}")

    json_path = os.path.join(TAB_DIR, "netsize_ablation.json")
    with open(json_path, "w") as jf:
        json.dump(summary, jf, indent=2)
    print(f"Total elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
