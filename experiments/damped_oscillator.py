"""
experiments/damped_oscillator.py
=================================
Phase 1 + 2 canonical experiment.

Methods
-------
A  StdNN  — standard NN, data loss only, deployed on IMC hardware
B  PINN   — physics-informed, clean-trained, deployed on IMC hardware
D  AdaPINN — adaptive physics weight + mixed-precision IMC deployment

Hardware sweep
--------------
  bits   : 2, 4, 6, 8
  noise  : 0.0, 0.05, 0.10  (fraction of mean |W|)
  seeds  : 0–4 (5 seeds)

Outputs
-------
  results/tables/osc_sweep_raw.csv       all individual runs
  results/tables/osc_sweep_summary.csv   mean ± std per config
  results/figures/osc_trajectory_8b5n.png
  results/figures/osc_bitwidth_sweep.png
  results/figures/osc_noise_sweep.png
  results/logs/osc_run.log

Usage
-----
  python experiments/damped_oscillator.py [--epochs N] [--seeds K] [--quick]
"""

import sys, os, csv, time, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import (
    set_seed, generate_data, evaluate_model, exact_solution,
    T_DATA_END, T_TEST_END, OMEGA_D, ALPHA,
)
from src.models import CleanNet, IMCNet, MixedPrecisionIMCNet
from src.physics_losses import (
    oscillator_physics_loss, measure_oscillator_residual,
)
from src.sensitivity import compute_layer_sensitivity
from src.bit_allocator import allocate_bits

# ── Directories ──────────────────────────────────────────────────────────────
ROOT    = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.join(ROOT, "results", "figures")
TAB_DIR = os.path.join(ROOT, "results", "tables")
LOG_DIR = os.path.join(ROOT, "results", "logs")
for d in [FIG_DIR, TAB_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Sweep configuration ──────────────────────────────────────────────────────
BITS_LIST  = [2, 4, 6, 8]
NOISE_LIST = [0.0, 0.05, 0.10]
N_EVAL     = 30    # forward passes for MC evaluation of IMC noise

# Training hyperparameters (shared across methods for fair comparison)
LR          = 5e-4
LAMBDA_MAX  = 0.5
WARMUP_EPS  = 1000
EVAL_PASSES = N_EVAL


# ══════════════════════════════════════════════════════════════════════════════
#  Training functions
# ══════════════════════════════════════════════════════════════════════════════

def copy_weights(src: nn.Module, dst: nn.Module):
    """Transfer weights from a CleanNet to any IMC model (shape-matched)."""
    with torch.no_grad():
        for cp, dp in zip(src.parameters(), dst.parameters()):
            if cp.shape == dp.shape:
                dp.copy_(cp)


def train_std(seed: int, epochs: int) -> CleanNet:
    """Train a standard NN (data loss only, FP32)."""
    set_seed(seed)
    t_d, x_d, _, _, _ = generate_data(seed)
    model = CleanNet(hidden=32)
    opt   = optim.Adam(model.parameters(), lr=LR)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.mean((model(t_d) - x_d) ** 2)
        loss.backward(); opt.step(); sch.step()
    return model


def train_pinn(seed: int, epochs: int, lam_max: float = LAMBDA_MAX) -> CleanNet:
    """Train a PINN (data + physics, FP32, physics on full prediction domain)."""
    set_seed(seed)
    t_d, x_d, t_c, _, _ = generate_data(seed)
    model = CleanNet(hidden=32)
    opt   = optim.Adam(model.parameters(), lr=LR)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        opt.zero_grad()
        l_data = torch.mean((model(t_d) - x_d) ** 2)
        lam    = lam_max * min(1.0, ep / max(1, WARMUP_EPS))
        l_phys, _, _ = oscillator_physics_loss(
            model, t_c, w_residual=1.0, w_ic=5.0
        )
        loss = l_data + lam * l_phys
        loss.backward(); opt.step(); sch.step()
    return model


class _EMA:
    """Exponential moving average for adaptive weight."""
    def __init__(self, alpha=0.99, init=1.0):
        self.val = init; self.alpha = alpha

    def update(self, x):
        self.val = self.alpha * self.val + (1 - self.alpha) * x
        return self.val


def train_adaptive_pinn(seed: int, epochs: int) -> CleanNet:
    """
    Method D: PINN with adaptive physics weight.

    λ_phys tracks the running ratio L_data / L_phys via EMA so that the
    physics term contributes proportionally to the data term.  This prevents
    the physics residual from being swamped or dominating at any precision.
    """
    set_seed(seed)
    t_d, x_d, t_c, _, _ = generate_data(seed)
    model = CleanNet(hidden=32)
    opt   = optim.Adam(model.parameters(), lr=LR)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ema   = _EMA(alpha=0.99, init=1.0)
    for ep in range(epochs):
        opt.zero_grad()
        l_data = torch.mean((model(t_d) - x_d) ** 2)
        l_phys, _, _ = oscillator_physics_loss(
            model, t_c, w_residual=1.0, w_ic=5.0
        )
        # Adaptive λ: keep physics on same scale as data loss
        ratio = (l_data.item() / (l_phys.item() + 1e-8))
        lam   = float(np.clip(ema.update(ratio), 0.01, 50.0))
        # Warm up: no physics for first 200 epochs
        if ep < 200:
            loss = l_data
        else:
            loss = l_data + lam * l_phys
        loss.backward(); opt.step(); sch.step()
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Deploy on IMC and evaluate
# ══════════════════════════════════════════════════════════════════════════════

def deploy_and_eval(clean_model, bits, noise_std, seed, mixed=False):
    """
    Copy clean weights into IMC model, evaluate MSE and physics residual.
    Returns (mse_interp, mse_extrap, l_physics).
    """
    _, _, _, t_test, x_exact = generate_data(seed)

    if mixed:
        # Method D mixed: 4-bit hidden, 8-bit output layer
        imc = MixedPrecisionIMCNet(
            hidden=32, bits_hidden=max(4, bits // 2), bits_output=bits,
            noise_std=noise_std,
        )
    else:
        imc = IMCNet(hidden=32, bits=bits, noise_std=noise_std)

    copy_weights(clean_model, imc)
    mse_i, mse_e, _ = evaluate_model(imc, t_test, x_exact, n_passes=EVAL_PASSES)
    l_phys = measure_oscillator_residual(imc, t_test)
    return mse_i, mse_e, l_phys


# ══════════════════════════════════════════════════════════════════════════════
#  Full sweep
# ══════════════════════════════════════════════════════════════════════════════

METHODS = {
    "A_StdNN"  : {"train_fn": train_std,          "mixed": False},
    "B_PINN"   : {"train_fn": train_pinn,          "mixed": False},
    "D_AdaPINN": {"train_fn": train_adaptive_pinn, "mixed": True},
}


def run_sweep(n_seeds: int, epochs: int, log_handle):
    rows = []

    for method_name, cfg in METHODS.items():
        train_fn = cfg["train_fn"]
        mixed    = cfg["mixed"]
        for seed in range(n_seeds):
            t0 = time.time()
            clean_model = train_fn(seed, epochs)
            train_time  = time.time() - t0

            for bits in BITS_LIST:
                for noise_std in NOISE_LIST:
                    mse_i, mse_e, l_phys = deploy_and_eval(
                        clean_model, bits, noise_std, seed, mixed=mixed
                    )
                    row = dict(
                        method=method_name, seed=seed,
                        bits=bits, noise_std=noise_std,
                        mse_interp=mse_i, mse_extrap=mse_e,
                        l_physics=l_phys, train_s=train_time,
                    )
                    rows.append(row)
                    msg = (
                        f"{method_name:12s} seed={seed} {bits}b "
                        f"n={noise_std:.2f} | "
                        f"i={mse_i:.5f} e={mse_e:.5f} r={l_phys:.5f}"
                    )
                    print(msg, flush=True)
                    log_handle.write(msg + "\n")

    return rows


def summarise(rows):
    """Aggregate raw rows into mean ± std per (method, bits, noise)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        key = (r["method"], r["bits"], r["noise_std"])
        groups[key].append(r)

    summary = []
    for (method, bits, noise_std), group in sorted(groups.items()):
        for col in ["mse_interp", "mse_extrap", "l_physics"]:
            vals = [g[col] for g in group]
        summary.append(dict(
            method=method, bits=bits, noise_std=noise_std,
            mse_interp_mean=np.mean([g["mse_interp"] for g in group]),
            mse_interp_std =np.std( [g["mse_interp"] for g in group]),
            mse_extrap_mean=np.mean([g["mse_extrap"] for g in group]),
            mse_extrap_std =np.std( [g["mse_extrap"] for g in group]),
            l_phys_mean    =np.mean([g["l_physics"]  for g in group]),
            l_phys_std     =np.std( [g["l_physics"]  for g in group]),
            n_seeds        =len(group),
        ))
    return summary


def save_csv(rows, path, fieldnames=None):
    if not rows:
        return
    fn = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader(); w.writerows(rows)
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_trajectory(seed: int, bits: int, noise_std: float, epochs: int):
    """Trajectory comparison figure for a single (bits, noise_std) config."""
    set_seed(seed)
    m_std  = train_std(seed, epochs)
    set_seed(seed)
    m_pinn = train_pinn(seed, epochs)

    _, _, _, t_test, x_exact = generate_data(seed)
    t_data, x_data, _, _, _  = generate_data(seed)

    imc_s = IMCNet(bits=bits, noise_std=noise_std)
    imc_p = IMCNet(bits=bits, noise_std=noise_std)
    copy_weights(m_std, imc_s); copy_weights(m_pinn, imc_p)

    with torch.no_grad():
        preds, predp = [], []
        for _ in range(N_EVAL):
            preds.append(imc_s(t_test)); predp.append(imc_p(t_test))
    pred_s = torch.stack(preds).mean(0).numpy()
    pred_p = torch.stack(predp).mean(0).numpy()

    t_np   = t_test.numpy().squeeze()
    xe_np  = x_exact.numpy().squeeze()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_np, xe_np,   "k-",  lw=2.0, label="Exact solution")
    ax.plot(t_np, pred_s,  "r--", lw=1.8, label=f"StdNN (IMC {bits}-bit)")
    ax.plot(t_np, pred_p,  "g-",  lw=1.8, label=f"PINN  (IMC {bits}-bit)")
    ax.scatter(t_data.numpy().squeeze(), x_data.numpy().squeeze(),
               s=14, c="steelblue", alpha=0.5, label="Training data")
    ax.axvline(T_DATA_END, color="gray", ls=":", lw=1.2, label="Data boundary")
    ax.set_xlabel("Time  t  [s]")
    ax.set_ylabel("x(t)")
    ax.set_title(
        f"Damped Oscillator — {bits}-bit IMC  (noise={noise_std:.0%})\n"
        f"Training: {int(T_DATA_END)}s sparse, Prediction: {int(T_TEST_END)}s"
    )
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"osc_trajectory_{bits}b{int(noise_std*100)}n.png")
    plt.savefig(path, dpi=200); plt.close()
    print(f"Saved {path}")


def plot_bitwidth_bar(summary, noise_std_target=0.05):
    sub = [r for r in summary if abs(r["noise_std"] - noise_std_target) < 1e-6]
    bits   = sorted({r["bits"] for r in sub})
    methods = ["A_StdNN", "B_PINN", "D_AdaPINN"]
    colors  = ["tomato", "seagreen", "royalblue"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(bits)); w = 0.25
    for i, (method, color) in enumerate(zip(methods, colors)):
        vals = []; errs = []
        for b in bits:
            row = next((r for r in sub if r["method"]==method and r["bits"]==b), None)
            vals.append(row["mse_extrap_mean"] if row else 0)
            errs.append(row["mse_extrap_std"]  if row else 0)
        offset = (i - 1) * w
        bars = ax.bar(x + offset, vals, w, label=method.replace("_", " "),
                      color=color, edgecolor="k", alpha=0.85)
        ax.errorbar(x + offset, vals, yerr=errs, fmt="none", capsize=3,
                    color="black", lw=1.2)

    ax.set_xticks(x); ax.set_xticklabels([f"{b}-bit" for b in bits])
    ax.set_ylabel("Extrapolation MSE (mean ± std)")
    ax.set_xlabel("IMC Bit-Width")
    ax.set_title(f"PINN vs StdNN: Extrapolation on IMC Hardware  (noise={noise_std_target:.0%})")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "osc_bitwidth_sweep.png")
    plt.savefig(path, dpi=200); plt.close()
    print(f"Saved {path}")


def plot_noise_sweep(summary, bits_target=8):
    sub     = [r for r in summary if r["bits"] == bits_target]
    noises  = sorted({r["noise_std"] for r in sub})
    methods = ["A_StdNN", "B_PINN", "D_AdaPINN"]
    colors  = ["tomato", "seagreen", "royalblue"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, color in zip(methods, colors):
        vals = []; errs = []; xs = []
        for n in noises:
            row = next((r for r in sub if r["method"]==method and abs(r["noise_std"]-n)<1e-6), None)
            if row:
                vals.append(row["mse_extrap_mean"])
                errs.append(row["mse_extrap_std"])
                xs.append(n * 100)
        ax.plot(xs, vals, "o-", color=color, label=method.replace("_", " "), lw=1.8)
        ax.fill_between(xs,
                        [v-e for v,e in zip(vals,errs)],
                        [v+e for v,e in zip(vals,errs)],
                        color=color, alpha=0.15)

    ax.set_xlabel("IMC Conductance Noise (% of mean |W|)")
    ax.set_ylabel("Extrapolation MSE")
    ax.set_title(f"Effect of IMC Noise on Extrapolation Accuracy  ({bits_target}-bit)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "osc_noise_sweep.png")
    plt.savefig(path, dpi=200); plt.close()
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,  default=2000)
    parser.add_argument("--seeds",  type=int,  default=5)
    parser.add_argument("--quick",  action="store_true",
                        help="Quick run: 1 seed, 500 epochs, 2 bits, 2 noise levels")
    args = parser.parse_args()

    if args.quick:
        global BITS_LIST, NOISE_LIST
        BITS_LIST  = [4, 8]
        NOISE_LIST = [0.0, 0.05]
        args.seeds  = 1
        args.epochs = 500

    log_path = os.path.join(LOG_DIR, "osc_run.log")
    with open(log_path, "w") as log_f:
        log_f.write(
            f"damped_oscillator experiment\n"
            f"epochs={args.epochs}  seeds={args.seeds}\n"
            f"omega_d={OMEGA_D:.6f}  T_data=[0,{T_DATA_END}]  T_pred=[0,{T_TEST_END}]\n\n"
        )
        t_start = time.time()
        rows    = run_sweep(args.seeds, args.epochs, log_f)
        elapsed = time.time() - t_start
        log_f.write(f"\nTotal time: {elapsed:.0f}s\n")

    # Save raw results
    save_csv(rows, os.path.join(TAB_DIR, "osc_sweep_raw.csv"))

    # Summarise
    summary = summarise(rows)
    save_csv(summary, os.path.join(TAB_DIR, "osc_sweep_summary.csv"))

    # Print key table
    print("\n" + "="*70)
    print("EXTRAPOLATION MSE SUMMARY  (noise=5%, mean over seeds)")
    print(f"{'Method':14s} {'2-bit':>9} {'4-bit':>9} {'6-bit':>9} {'8-bit':>9}")
    print("-"*70)
    for method in ["A_StdNN", "B_PINN", "D_AdaPINN"]:
        row_str = f"{method:14s}"
        for bits in [2, 4, 6, 8]:
            r = next(
                (r for r in summary
                 if r["method"]==method and r["bits"]==bits
                 and abs(r["noise_std"]-0.05)<1e-6),
                None,
            )
            val = f"{r['mse_extrap_mean']:.4f}" if r else "N/A"
            row_str += f" {val:>9}"
        print(row_str)
    print("="*70)

    # Plots
    plot_trajectory(seed=0, bits=8, noise_std=0.05, epochs=args.epochs)
    plot_bitwidth_bar(summary, noise_std_target=0.05)
    plot_noise_sweep(summary, bits_target=8)

    # Save summary JSON for quick reference
    json_path = os.path.join(TAB_DIR, "osc_summary.json")
    with open(json_path, "w") as jf:
        json.dump(summary, jf, indent=2)
    print(f"Saved {json_path}")
    print(f"\nAll results in results/tables/ and results/figures/")
    print(f"Total elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
