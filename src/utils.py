"""
Data generation, seeding, and evaluation utilities.

CORRECTED damped oscillator:
    ODE:     x'' + 0.5x' + 2x = 0
    ICs:     x(0) = 1,  x'(0) = 0
    Exact:   x(t) = exp(-0.25t) [cos(wd*t) + (0.25/wd)*sin(wd*t)]
    where    wd = sqrt(7.75)/2 = sqrt(31)/4 ≈ 1.3920  (NOT 1.98)

Experimental design (validated):
    Sparse training data on [0, T_DATA_END] highlights PINN advantage.
    Physics collocation MUST span [0, T_TEST_END] (full prediction domain)
    so the ODE guides the network even in the extrapolation region.
"""

import math
import random
import numpy as np
import torch

# ── Physical constants (fixed, derived from ODE coefficients) ──────────────
ALPHA   = 0.25                          # decay rate = ζ·ω_n
OMEGA_D = math.sqrt(7.75) / 2          # damped frequency ≈ 1.3920 rad/s
B_COEFF = ALPHA / OMEGA_D              # from x'(0)=0: B = α/ω_d ≈ 0.1796

# Training / collocation / test domains
T_DATA_END  = 5.0     # sparse data only up to this point
T_TEST_END  = 15.0    # prediction horizon (extrapolation starts after T_DATA_END)
N_TRAIN     = 40      # sparse training observations
N_COL       = 200     # physics collocation points (span [0, T_TEST_END])
N_TEST      = 300
DATA_NOISE  = 0.10    # measurement noise on training data


def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def exact_solution(t: torch.Tensor) -> torch.Tensor:
    """Exact solution of the corrected damped oscillator."""
    return torch.exp(-ALPHA * t) * (
        torch.cos(OMEGA_D * t) + B_COEFF * torch.sin(OMEGA_D * t)
    )


def generate_data(seed: int = 0):
    """
    Returns (t_data, x_data, t_col, t_test, x_exact_test).

    t_data:  N_TRAIN points in [0, T_DATA_END]  — noisy training observations
    x_data:  corresponding noisy targets
    t_col:   N_COL collocation points in [0, T_TEST_END]  — physics constraint grid
    t_test:  N_TEST points in [0, T_TEST_END]
    x_exact: exact solution at test points
    """
    set_seed(seed)
    t_data = torch.linspace(0, T_DATA_END, N_TRAIN).view(-1, 1)
    x_data = exact_solution(t_data) + torch.randn_like(t_data) * DATA_NOISE

    t_col       = torch.linspace(0, T_TEST_END, N_COL).view(-1, 1)
    t_test      = torch.linspace(0, T_TEST_END, N_TEST).view(-1, 1)
    x_exact     = exact_solution(t_test)
    return t_data, x_data, t_col, t_test, x_exact


def split_test_domains(t_test: torch.Tensor, x_exact: torch.Tensor):
    """Split test into interpolation [0, T_DATA_END] and extrapolation region."""
    mask_i = (t_test.squeeze() <= T_DATA_END)
    mask_e = (t_test.squeeze() >  T_DATA_END)
    return t_test[mask_i], x_exact[mask_i], t_test[mask_e], x_exact[mask_e]


def evaluate_model(model, t_test, x_exact, n_passes: int = 20):
    """
    Average over `n_passes` forward passes to stabilise stochastic IMC noise.
    Returns (mse_interp, mse_extrap, mse_full).
    """
    ti, xi, te, xe = split_test_domains(t_test, x_exact)
    preds_i, preds_e = [], []
    with torch.no_grad():
        for _ in range(n_passes):
            preds_i.append(model(ti))
            preds_e.append(model(te))
    pred_i = torch.stack(preds_i).mean(0)
    pred_e = torch.stack(preds_e).mean(0)
    mse_i = torch.mean((pred_i - xi) ** 2).item()
    mse_e = torch.mean((pred_e - xe) ** 2).item()
    mse_f = (mse_i * len(xi) + mse_e * len(xe)) / (len(xi) + len(xe))
    return mse_i, mse_e, mse_f
