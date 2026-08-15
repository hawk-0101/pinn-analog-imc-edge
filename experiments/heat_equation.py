"""
experiments/heat_equation.py
=============================
Phase 3d: 1D heat equation benchmark (parabolic PDE).

    u_t = alpha * u_xx    on (x,t) in [0,1] x [0,T]
    u(0,t) = u(1,t) = 0   (Dirichlet BC)
    u(x,0) = sin(pi*x)    (IC)
    Exact:  u(x,t) = exp(-alpha*pi^2*t) * sin(pi*x)

alpha = 0.01, T_train = 0.5, T_test = 1.5 (extrapolation in time).
Sparse training grid: 10x10 = 100 points in [0,1]x[0,0.5].
Dense test grid: 50x50 = 2500 points in [0,1]x[0,1.5].

Methods: StdNN vs PINN, deployed at {4,8}-bit, noise={0,0.05}, 5 seeds.
"""

import sys, os, csv, time, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import set_seed
from src.models import CleanNetND, IMCNetND
from src.physics_losses import heat_physics_loss

ROOT    = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.join(ROOT, "results", "figures")
TAB_DIR = os.path.join(ROOT, "results", "tables")
LOG_DIR = os.path.join(ROOT, "results", "logs")
for d in [FIG_DIR, TAB_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# Physics
ALPHA_HEAT  = 0.01
T_TRAIN_END = 0.5
T_TEST_END  = 1.5

# Grid sizes
NX_TRAIN    = 10
NT_TRAIN    = 10
NX_TEST     = 50
NT_TEST     = 50

# Collocation spans full test domain
NX_COL      = 30
NT_COL      = 30
NX_BC       = 30
NX_IC       = 30

# Sweep
BITS_LIST   = [4, 8]
NOISE_LIST  = [0.0, 0.05]
N_SEEDS     = 5
EPOCHS      = 3000
LR          = 1e-3
WARMUP      = 500
LAM_MAX     = 1.0
N_EVAL      = 30
DATA_NOISE  = 0.05


def exact_heat(x, t):
    return torch.exp(-ALPHA_HEAT * math.pi**2 * t) * torch.sin(math.pi * x)


def generate_heat_data(seed):
    set_seed(seed)
    # Training grid (sparse, only up to T_TRAIN_END)
    xs = torch.linspace(0.05, 0.95, NX_TRAIN)
    ts = torch.linspace(0.0, T_TRAIN_END, NT_TRAIN)
    xx, tt = torch.meshgrid(xs, ts, indexing="ij")
    xt_train = torch.stack([xx.flatten(), tt.flatten()], dim=1)
    u_train  = exact_heat(xt_train[:, 0:1], xt_train[:, 1:2])
    u_train  = u_train + torch.randn_like(u_train) * DATA_NOISE

    # Collocation (full test domain for physics)
    xs_c = torch.linspace(0.01, 0.99, NX_COL)
    ts_c = torch.linspace(0.01, T_TEST_END, NT_COL)
    xxc, ttc = torch.meshgrid(xs_c, ts_c, indexing="ij")
    xt_col = torch.stack([xxc.flatten(), ttc.flatten()], dim=1)

    # BC times (full domain)
    t_bc = torch.linspace(0.01, T_TEST_END, NX_BC).view(-1, 1)

    # IC x-points
    x_ic = torch.linspace(0.05, 0.95, NX_IC).view(-1, 1)

    # Test grid (full domain)
    xs_t = torch.linspace(0.0, 1.0, NX_TEST)
    ts_t = torch.linspace(0.0, T_TEST_END, NT_TEST)
    xxt, ttt = torch.meshgrid(xs_t, ts_t, indexing="ij")
    xt_test  = torch.stack([xxt.flatten(), ttt.flatten()], dim=1)
    u_exact  = exact_heat(xt_test[:, 0:1], xt_test[:, 1:2])

    return xt_train, u_train, xt_col, t_bc, x_ic, xt_test, u_exact


def copy_weights(src, dst):
    with torch.no_grad():
        for cp, dp in zip(src.parameters(), dst.parameters()):
            if cp.shape == dp.shape:
                dp.copy_(cp)


def train_std_heat(seed, epochs):
    xt_train, u_train, _, _, _, _, _ = generate_heat_data(seed)
    set_seed(seed)
    model = CleanNetND(in_dim=2, hidden=32)
    opt   = optim.Adam(model.parameters(), lr=LR)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.mean((model(xt_train) - u_train) ** 2)
        loss.backward(); opt.step(); sch.step()
    return model


def train_pinn_heat(seed, epochs):
    xt_train, u_train, xt_col, t_bc, x_ic, _, _ = generate_heat_data(seed)
    set_seed(seed)
    model = CleanNetND(in_dim=2, hidden=32)
    opt   = optim.Adam(model.parameters(), lr=LR)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        opt.zero_grad()
        l_data = torch.mean((model(xt_train) - u_train) ** 2)
        lam    = LAM_MAX * min(1.0, ep / max(1, WARMUP))
        l_phys, _ = heat_physics_loss(
            model, xt_col, t_bc, x_ic,
            alpha=ALPHA_HEAT, w_pde=1.0, w_bc=10.0, w_ic=10.0,
        )
        loss = l_data + lam * l_phys
        loss.backward(); opt.step(); sch.step()
    return model


def evaluate_heat(imc, xt_test, u_exact, n_passes=N_EVAL):
    """Split into interpolation (t<=T_TRAIN_END) and extrapolation."""
    t_vals = xt_test[:, 1]
    mask_i = t_vals <= T_TRAIN_END
    mask_e = t_vals >  T_TRAIN_END

    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            preds.append(imc(xt_test))
    pred = torch.stack(preds).mean(0)

    mse_i = torch.mean((pred[mask_i] - u_exact[mask_i]) ** 2).item()
    mse_e = torch.mean((pred[mask_e] - u_exact[mask_e]) ** 2).item()
    mse_f = torch.mean((pred - u_exact) ** 2).item()
    return mse_i, mse_e, mse_f


def deploy_eval_heat(model, seed, bits, noise_std):
    _, _, _, _, _, xt_test, u_exact = generate_heat_data(seed)
    imc = IMCNetND(in_dim=2, hidden=32, bits=bits, noise_std=noise_std)
    copy_weights(model, imc)
    return evaluate_heat(imc, xt_test, u_exact)


def main():
    rows = []
    log_path = os.path.join(LOG_DIR, "heat_run.log")
    with open(log_path, "w") as lf:
        lf.write(f"heat equation experiment\n")
        lf.write(f"alpha={ALPHA_HEAT} T_train=[0,{T_TRAIN_END}] T_test=[0,{T_TEST_END}]\n\n")
        t0 = time.time()

        for method_name, train_fn in [("StdNN", train_std_heat), ("PINN", train_pinn_heat)]:
            for seed in range(N_SEEDS):
                model = train_fn(seed, EPOCHS)
                for bits in BITS_LIST:
                    for noise_std in NOISE_LIST:
                        mse_i, mse_e, mse_f = deploy_eval_heat(model, seed, bits, noise_std)
                        row = dict(
                            method=method_name, seed=seed, bits=bits,
                            noise_std=noise_std,
                            mse_interp=mse_i, mse_extrap=mse_e, mse_full=mse_f,
                        )
                        rows.append(row)
                        msg = (f"{method_name:6s} seed={seed} {bits}b n={noise_std:.2f} "
                               f"i={mse_i:.6f} e={mse_e:.6f}")
                        print(msg, flush=True)
                        lf.write(msg + "\n")

        elapsed = time.time() - t0
        lf.write(f"\nTotal: {elapsed:.0f}s\n")

    # Raw CSV
    raw_path = os.path.join(TAB_DIR, "heat_sweep_raw.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Saved {raw_path}")

    # Summarise
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["bits"], r["noise_std"])].append(r)
    summary = []
    for (method, bits, noise_std), grp in sorted(groups.items()):
        summary.append(dict(
            method=method, bits=bits, noise_std=noise_std,
            mse_interp_mean=np.mean([r["mse_interp"] for r in grp]),
            mse_interp_std =np.std( [r["mse_interp"] for r in grp]),
            mse_extrap_mean=np.mean([r["mse_extrap"] for r in grp]),
            mse_extrap_std =np.std( [r["mse_extrap"] for r in grp]),
            mse_full_mean  =np.mean([r["mse_full"]   for r in grp]),
        ))
    sum_path = os.path.join(TAB_DIR, "heat_sweep_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    # Print key table
    print("\n" + "="*70)
    print("HEAT EQUATION: EXTRAPOLATION MSE  (noise=5%)")
    print(f"{'Method':8s} {'4-bit':>14} {'8-bit':>14}")
    print("-"*70)
    for method in ["StdNN", "PINN"]:
        row_str = f"{method:8s}"
        for bits in [4, 8]:
            r = next((s for s in summary
                      if s["method"]==method and s["bits"]==bits
                      and abs(s["noise_std"]-0.05)<1e-6), None)
            if r:
                row_str += f" {r['mse_extrap_mean']:.6f}+/-{r['mse_extrap_std']:.4f}"
            else:
                row_str += f" {'N/A':>14}"
        print(row_str)
    print("="*70)

    # Improvement
    for bits in BITS_LIST:
        s_std  = next((s for s in summary if s["method"]=="StdNN" and s["bits"]==bits and abs(s["noise_std"]-0.05)<1e-6), None)
        s_pinn = next((s for s in summary if s["method"]=="PINN"  and s["bits"]==bits and abs(s["noise_std"]-0.05)<1e-6), None)
        if s_std and s_pinn and s_std["mse_extrap_mean"] > 0:
            imp = (1 - s_pinn["mse_extrap_mean"] / s_std["mse_extrap_mean"]) * 100
            print(f"PINN improvement at {bits}-bit 5% noise: {imp:.1f}%")

    # Heatmap plot (PINN vs StdNN at 8-bit 5% noise)
    _, _, _, _, _, xt_test, u_exact = generate_heat_data(0)
    set_seed(0)
    m_std  = train_std_heat(0, EPOCHS)
    set_seed(0)
    m_pinn = train_pinn_heat(0, EPOCHS)

    imc_s = IMCNetND(in_dim=2, hidden=32, bits=8, noise_std=0.05)
    imc_p = IMCNetND(in_dim=2, hidden=32, bits=8, noise_std=0.05)
    copy_weights(m_std, imc_s); copy_weights(m_pinn, imc_p)

    with torch.no_grad():
        ps, pp = [], []
        for _ in range(N_EVAL):
            ps.append(imc_s(xt_test)); pp.append(imc_p(xt_test))
    pred_s = torch.stack(ps).mean(0).numpy().reshape(NX_TEST, NT_TEST)
    pred_p = torch.stack(pp).mean(0).numpy().reshape(NX_TEST, NT_TEST)
    u_ex   = u_exact.numpy().reshape(NX_TEST, NT_TEST)

    xs_t = np.linspace(0, 1, NX_TEST)
    ts_t = np.linspace(0, T_TEST_END, NT_TEST)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, title in zip(axes,
        [u_ex, np.abs(pred_s - u_ex), np.abs(pred_p - u_ex)],
        ["Exact solution", "StdNN error (8-bit, 5%)", "PINN error (8-bit, 5%)"]):
        im = ax.pcolormesh(ts_t, xs_t, data, shading="auto", cmap="viridis")
        plt.colorbar(im, ax=ax)
        ax.axvline(T_TRAIN_END, color="r", ls="--", lw=1.5)
        ax.set_xlabel("Time t"); ax.set_ylabel("Space x")
        ax.set_title(title)
    plt.suptitle("1D Heat Equation Benchmark", fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "heat_equation.png")
    plt.savefig(fig_path, dpi=200); plt.close()
    print(f"Saved {fig_path}")

    json_path = os.path.join(TAB_DIR, "heat_summary.json")
    with open(json_path, "w") as jf:
        json.dump(summary, jf, indent=2)
    print(f"Total elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
