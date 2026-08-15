"""
experiments/lambda_ablation.py
==============================
Phase 3a: Sweep physics weight lambda in {0.01, 0.1, 0.5, 1.0, 5.0}
at 8-bit, 5% noise, 5 seeds.

Answers: "How sensitive is PINN extrapolation to physics weight?"
"""

import sys, os, csv, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import set_seed, generate_data, evaluate_model, T_DATA_END, T_TEST_END
from src.models import CleanNet, IMCNet
from src.physics_losses import oscillator_physics_loss

ROOT    = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.join(ROOT, "results", "figures")
TAB_DIR = os.path.join(ROOT, "results", "tables")
LOG_DIR = os.path.join(ROOT, "results", "logs")
for d in [FIG_DIR, TAB_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

LAMBDA_LIST = [0.01, 0.1, 0.5, 1.0, 5.0]
BITS        = 8
NOISE_STD   = 0.05
N_SEEDS     = 5
EPOCHS      = 2000
LR          = 5e-4
WARMUP_EPS  = 1000
N_EVAL      = 30


def copy_weights(src, dst):
    with torch.no_grad():
        for cp, dp in zip(src.parameters(), dst.parameters()):
            if cp.shape == dp.shape:
                dp.copy_(cp)


def train_pinn_lam(seed, epochs, lam_max):
    set_seed(seed)
    t_d, x_d, t_c, _, _ = generate_data(seed)
    model = CleanNet(hidden=32)
    opt   = optim.Adam(model.parameters(), lr=LR)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        opt.zero_grad()
        l_data = torch.mean((model(t_d) - x_d) ** 2)
        lam    = lam_max * min(1.0, ep / max(1, WARMUP_EPS))
        l_phys, _, _ = oscillator_physics_loss(model, t_c, w_residual=1.0, w_ic=5.0)
        loss = l_data + lam * l_phys
        loss.backward(); opt.step(); sch.step()
    return model


def deploy_eval(clean_model, seed):
    _, _, _, t_test, x_exact = generate_data(seed)
    imc = IMCNet(hidden=32, bits=BITS, noise_std=NOISE_STD)
    copy_weights(clean_model, imc)
    mse_i, mse_e, _ = evaluate_model(imc, t_test, x_exact, n_passes=N_EVAL)
    return mse_i, mse_e


def main():
    rows = []
    log_path = os.path.join(LOG_DIR, "lambda_ablation.log")
    with open(log_path, "w") as lf:
        lf.write(f"lambda ablation: bits={BITS} noise={NOISE_STD} seeds={N_SEEDS}\n\n")
        t0 = time.time()
        for lam in LAMBDA_LIST:
            for seed in range(N_SEEDS):
                model = train_pinn_lam(seed, EPOCHS, lam)
                mse_i, mse_e = deploy_eval(model, seed)
                row = dict(lam=lam, seed=seed, mse_interp=mse_i, mse_extrap=mse_e)
                rows.append(row)
                msg = f"lam={lam:.2f} seed={seed} interp={mse_i:.6f} extrap={mse_e:.6f}"
                print(msg, flush=True)
                lf.write(msg + "\n")
        elapsed = time.time() - t0
        lf.write(f"\nTotal: {elapsed:.0f}s\n")

    # Save raw
    raw_path = os.path.join(TAB_DIR, "lambda_ablation_raw.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lam", "seed", "mse_interp", "mse_extrap"])
        w.writeheader(); w.writerows(rows)
    print(f"Saved {raw_path}")

    # Summarise
    summary = []
    for lam in LAMBDA_LIST:
        sub = [r for r in rows if r["lam"] == lam]
        summary.append(dict(
            lam=lam,
            mse_interp_mean=np.mean([r["mse_interp"] for r in sub]),
            mse_interp_std =np.std( [r["mse_interp"] for r in sub]),
            mse_extrap_mean=np.mean([r["mse_extrap"] for r in sub]),
            mse_extrap_std =np.std( [r["mse_extrap"] for r in sub]),
        ))
    sum_path = os.path.join(TAB_DIR, "lambda_ablation_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"Saved {sum_path}")

    # Print table
    print("\n" + "="*60)
    print(f"LAMBDA ABLATION  ({BITS}-bit, noise={NOISE_STD})")
    print(f"{'lambda':>8} {'extrap_mean':>14} {'extrap_std':>12} {'interp_mean':>14}")
    print("-"*60)
    for s in summary:
        print(f"{s['lam']:8.2f} {s['mse_extrap_mean']:14.6f} {s['mse_extrap_std']:12.6f} {s['mse_interp_mean']:14.6f}")
    print("="*60)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    lams = [s["lam"] for s in summary]
    means = [s["mse_extrap_mean"] for s in summary]
    stds  = [s["mse_extrap_std"]  for s in summary]
    ax.errorbar(lams, means, yerr=stds, fmt="o-", color="seagreen",
                capsize=5, lw=2, markersize=8, label="Extrapolation MSE")
    interp_means = [s["mse_interp_mean"] for s in summary]
    interp_stds  = [s["mse_interp_std"]  for s in summary]
    ax.errorbar(lams, interp_means, yerr=interp_stds, fmt="s--", color="royalblue",
                capsize=5, lw=1.5, markersize=7, label="Interpolation MSE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physics weight $\\lambda_{max}$")
    ax.set_ylabel("MSE (mean +/- std, 5 seeds)")
    ax.set_title(f"Physics Weight Ablation  ({BITS}-bit IMC, noise={NOISE_STD:.0%})")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "lambda_ablation.png")
    plt.savefig(fig_path, dpi=200); plt.close()
    print(f"Saved {fig_path}")

    # JSON
    json_path = os.path.join(TAB_DIR, "lambda_ablation.json")
    with open(json_path, "w") as jf:
        json.dump(summary, jf, indent=2)
    print(f"Saved {json_path}")
    print(f"Total elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
