"""PINN training loop with manual / grad-norm / NTK loss balancing."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Sequence
import numpy as np
import torch


def _grad(out, inp, create_graph=True):
    return torch.autograd.grad(out, inp, grad_outputs=torch.ones_like(out),
                               create_graph=create_graph, retain_graph=True)[0]


def derivatives(model, t):
    """Return x, x', x'' with autograd."""
    t = t.requires_grad_(True)
    x = model(t)
    dx = _grad(x, t)
    ddx = _grad(dx, t)
    return x, dx, ddx


def duffing_residual(model, t, zeta, w0, alpha, A=0.0, Omega=0.0):
    x, dx, ddx = derivatives(model, t)
    f = A * torch.cos(Omega * t)
    return ddx + 2 * zeta * w0 * dx + (w0**2) * x + alpha * x ** 3 - f


def linear_residual(model, t, zeta, w0, A=0.0, Omega=0.0):
    x, dx, ddx = derivatives(model, t)
    f = A * torch.cos(Omega * t)
    return ddx + 2 * zeta * w0 * dx + (w0**2) * x - f


def vdp_residual(model, t, mu, w0, A=0.0, Omega=0.0):
    x, dx, ddx = derivatives(model, t)
    f = A * torch.cos(Omega * t)
    return ddx - mu * (1 - x ** 2) * dx + (w0**2) * x - f


RESIDUALS = {"linear": linear_residual, "duffing": duffing_residual, "vdp": vdp_residual}


def ic_loss(model, ic, device="cpu"):
    x0, v0 = ic
    t0 = torch.zeros(1, 1, device=device).requires_grad_(True)
    x_pred = model(t0)
    dx_pred = _grad(x_pred, t0)
    return (x_pred - x0).pow(2).mean() + (dx_pred - v0).pow(2).mean()


@dataclass
class TrainConfig:
    iters: int = 10000
    lr: float = 1e-3
    log_every: int = 500
    lambda_ic: float = 100.0
    lambda_data: float = 1.0
    lambda_phys: float = 1.0
    weighting: str = "fixed"      # "fixed" | "gradnorm" | "ntk"
    weighting_every: int = 500
    lbfgs_steps: int = 0          # 0 disables L-BFGS polish
    device: str = "cpu"
    seed: int = 0


def _flat_grads(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([gi.flatten() for gi in g if gi is not None])


def _update_weights(losses: dict, params, cfg: TrainConfig):
    """Adaptive weights via gradient-norm or NTK eigen-trace balancing."""
    if cfg.weighting == "fixed":
        return
    norms = {k: _flat_grads(v, params).abs().mean().item() + 1e-12 for k, v in losses.items()}
    mean = float(np.mean(list(norms.values())))
    if cfg.weighting == "gradnorm":
        cfg.lambda_ic   *= mean / norms.get("ic",   norms.get("phys"))
        cfg.lambda_data *= mean / norms.get("data", mean)
        cfg.lambda_phys *= mean / norms["phys"]
    elif cfg.weighting == "ntk":
        # diagonal NTK proxy: λ_k ∝ 1/||∇L_k||²
        sqr = {k: _flat_grads(v, params).pow(2).mean().item() + 1e-12 for k, v in losses.items()}
        total = sum(1.0 / s for s in sqr.values())
        cfg.lambda_phys = (1.0 / sqr["phys"]) / total
        if "ic" in sqr:   cfg.lambda_ic   = (1.0 / sqr["ic"])   / total
        if "data" in sqr: cfg.lambda_data = (1.0 / sqr["data"]) / total


def train_pinn(model: torch.nn.Module,
               residual_fn: Callable,
               t_colloc: torch.Tensor,
               ic: Sequence[float] | None,
               data: tuple[torch.Tensor, torch.Tensor] | None,
               cfg: TrainConfig,
               extra_params: torch.nn.Module | None = None):
    """Fit a PINN. Returns dict with loss history + final weights + extra_params."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = cfg.device
    model.to(device)
    if extra_params is not None:
        extra_params.to(device)
    params = list(model.parameters())
    if extra_params is not None:
        params += list(extra_params.parameters())
    optim = torch.optim.Adam(params, lr=cfg.lr)

    t_colloc = t_colloc.to(device)
    if data is not None:
        data = (data[0].to(device), data[1].to(device))
    history = {"total": [], "phys": [], "ic": [], "data": [], "step": [],
               "lambda_ic": [], "lambda_phys": [], "lambda_data": []}
    extra_hist: dict[str, list] = {}

    for step in range(cfg.iters):
        optim.zero_grad()
        losses = {}

        r = residual_fn(t_colloc)
        losses["phys"] = r.pow(2).mean()

        is_hard_ic = hasattr(model, "x0") and hasattr(model, "v0")
        if (ic is not None) and (not is_hard_ic):
            losses["ic"] = ic_loss(model, ic, device=device)

        if data is not None:
            t_data, x_data = data
            losses["data"] = (model(t_data) - x_data).pow(2).mean()

        if cfg.weighting != "fixed" and step > 0 and step % cfg.weighting_every == 0:
            _update_weights(losses, params, cfg)

        total = cfg.lambda_phys * losses["phys"]
        if "ic" in losses:   total = total + cfg.lambda_ic   * losses["ic"]
        if "data" in losses: total = total + cfg.lambda_data * losses["data"]
        total.backward()
        optim.step()

        if step % cfg.log_every == 0 or step == cfg.iters - 1:
            history["step"].append(step)
            history["total"].append(float(total.detach()))
            history["phys"].append(float(losses["phys"].detach()))
            history["ic"].append(float(losses.get("ic", torch.tensor(0.)).detach()))
            history["data"].append(float(losses.get("data", torch.tensor(0.)).detach()))
            history["lambda_ic"].append(cfg.lambda_ic)
            history["lambda_phys"].append(cfg.lambda_phys)
            history["lambda_data"].append(cfg.lambda_data)
            if extra_params is not None:
                snap = extra_params.asdict()
                for k, v in snap.items():
                    extra_hist.setdefault(k, []).append(v)

    if cfg.lbfgs_steps > 0:
        lb = torch.optim.LBFGS(params, max_iter=cfg.lbfgs_steps,
                               tolerance_grad=1e-9, tolerance_change=1e-12,
                               line_search_fn="strong_wolfe")
        def closure():
            lb.zero_grad()
            r = residual_fn(t_colloc)
            loss = cfg.lambda_phys * r.pow(2).mean()
            if (ic is not None) and not is_hard_ic:
                loss = loss + cfg.lambda_ic * ic_loss(model, ic, device=device)
            if data is not None:
                loss = loss + cfg.lambda_data * (model(data[0]) - data[1]).pow(2).mean()
            loss.backward()
            return loss
        lb.step(closure)

    return {"history": history, "extra_history": extra_hist}
