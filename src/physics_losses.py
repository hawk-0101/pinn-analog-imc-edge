"""
Physics loss functions for the thesis benchmarks.

Damped oscillator ODE (corrected):
    x''(t) + 0.5 x'(t) + 2 x(t) = 0
    x(0) = 1,  x'(0) = 0

1D heat equation (Phase 3):
    u_t(x,t) = alpha * u_xx(x,t)
    u(0,t) = u(1,t) = 0
    u(x,0) = sin(pi * x)
"""

import torch
import torch.autograd as ag


# ── Damped oscillator ────────────────────────────────────────────────────────

def oscillator_residual(model: torch.nn.Module, t: torch.Tensor) -> torch.Tensor:
    """PDE residual: x'' + 0.5x' + 2x. Returns tensor of shape (N,1)."""
    t   = t.clone().requires_grad_(True)
    x   = model(t)
    x_t = ag.grad(x, t, torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    x_tt = ag.grad(x_t, t, torch.ones_like(x_t), create_graph=True, retain_graph=True)[0]
    return x_tt + 0.5 * x_t + 2.0 * x


def oscillator_ic_loss(model: torch.nn.Module) -> torch.Tensor:
    """Initial condition loss: (x(0)-1)^2 + (x'(0))^2."""
    t0  = torch.zeros(1, 1, requires_grad=True)
    x0  = model(t0)
    x0t = ag.grad(x0, t0, torch.ones_like(x0), create_graph=True, retain_graph=True)[0]
    return torch.mean((x0 - 1.0) ** 2) + torch.mean(x0t ** 2)


def oscillator_physics_loss(
    model: torch.nn.Module,
    t: torch.Tensor,
    w_residual: float = 1.0,
    w_ic: float = 10.0,
) -> torch.Tensor:
    """Combined physics loss = w_res * L_residual + w_ic * L_ic."""
    res = oscillator_residual(model, t)
    l_res = torch.mean(res ** 2)
    l_ic  = oscillator_ic_loss(model)
    return w_residual * l_res + w_ic * l_ic, l_res.detach(), l_ic.detach()


def measure_oscillator_residual(model: torch.nn.Module, t: torch.Tensor) -> float:
    """Compute mean squared physics residual (for evaluation)."""
    res = oscillator_residual(model, t)
    return torch.mean(res ** 2).item()


# ── 1D Heat equation ─────────────────────────────────────────────────────────

def heat_residual(model: torch.nn.Module, xt: torch.Tensor, alpha: float = 0.01):
    """PDE residual: u_t - alpha * u_xx. xt shape: (N,2)."""
    xt   = xt.clone().requires_grad_(True)
    u    = model(xt)
    grads = ag.grad(u, xt, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_t  = grads[:, 1:2]
    u_x  = grads[:, 0:1]
    u_xx = ag.grad(u_x, xt, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0:1]
    return u_t - alpha * u_xx


def heat_bc_loss(model: torch.nn.Module, t_bc: torch.Tensor) -> torch.Tensor:
    """Boundary condition loss at x=0 and x=1."""
    bc0 = torch.cat([torch.zeros_like(t_bc), t_bc], dim=1)
    bc1 = torch.cat([torch.ones_like(t_bc), t_bc], dim=1)
    return torch.mean(model(bc0) ** 2) + torch.mean(model(bc1) ** 2)


def heat_ic_loss(model: torch.nn.Module, x_ic: torch.Tensor) -> torch.Tensor:
    """Initial condition loss at t=0: u(x,0) = sin(pi*x)."""
    import math
    xt_ic = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1)
    u_pred = model(xt_ic)
    u_true = torch.sin(math.pi * x_ic)
    return torch.mean((u_pred - u_true) ** 2)


def heat_physics_loss(
    model: torch.nn.Module,
    xt_col: torch.Tensor,
    t_bc: torch.Tensor,
    x_ic: torch.Tensor,
    alpha: float = 0.01,
    w_pde: float = 1.0,
    w_bc: float = 10.0,
    w_ic: float = 10.0,
) -> torch.Tensor:
    res   = heat_residual(model, xt_col, alpha)
    l_pde = torch.mean(res ** 2)
    l_bc  = heat_bc_loss(model, t_bc)
    l_ic  = heat_ic_loss(model, x_ic)
    return w_pde * l_pde + w_bc * l_bc + w_ic * l_ic, l_pde.detach()
