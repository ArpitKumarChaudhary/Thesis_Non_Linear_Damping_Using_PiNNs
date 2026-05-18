"""Grouped (parallel-seed) PINN ensemble.

Trains K independent MLPs simultaneously via batched matmul:
  - weights of shape (K, in, out), inputs (K, N, in) -> outputs (K, N, out) via bmm.
  - K independent random seeds -> mean / std error bars for free.
  - Saturates GPU compute and VRAM (the whole point of using a 49 GB A6000).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn


def _xavier_grouped(K, in_dim, out_dim, generator=None):
    a = math.sqrt(6.0 / (in_dim + out_dim))
    return (torch.rand(K, in_dim, out_dim, generator=generator) * 2 - 1) * a


class GroupedLinear(nn.Module):
    def __init__(self, K, in_dim, out_dim, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(_xavier_grouped(K, in_dim, out_dim, g))
        self.b = nn.Parameter(torch.zeros(K, 1, out_dim))

    def forward(self, x):
        return torch.bmm(x, self.W) + self.b


class GroupedSin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class GroupedMLP(nn.Module):
    """K independent MLPs: input (K, N, in) -> output (K, N, out).

    Accepts plain (N, in) inputs and broadcasts. Uses tanh by default.
    """

    def __init__(self, K, in_dim=1, out_dim=1, hidden=512, depth=6,
                 activation="tanh", t_scale: float = 1.0, base_seed: int = 0):
        super().__init__()
        act = {"tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU,
               "sin": GroupedSin, "relu": nn.ReLU}[activation]
        layers = []
        layers.append(GroupedLinear(K, in_dim, hidden, seed=base_seed))
        for i in range(depth - 1):
            layers.append(GroupedLinear(K, hidden, hidden, seed=base_seed + i + 1))
        layers.append(GroupedLinear(K, hidden, out_dim, seed=base_seed + depth + 1))
        self.layers = nn.ModuleList(layers)
        self.act = act()
        self.K = K
        self.register_buffer("t_scale", torch.tensor(float(t_scale)))

    def forward(self, t):
        if t.dim() == 2:                                # (N, 1) -> (K, N, 1)
            t = t.unsqueeze(0).expand(self.K, *t.shape).contiguous()
        x = t / self.t_scale
        for i, lin in enumerate(self.layers):
            x = lin(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x                                        # (K, N, 1)


class GroupedHardIC(nn.Module):
    """x_k(t) = x0 + t v0 + (1 - e^{-t/tau})^2 NN_k(t) — shared IC across the K nets."""

    def __init__(self, base: GroupedMLP, x0: float, v0: float, tau: float = 1.0):
        super().__init__()
        self.base = base
        self.register_buffer("x0", torch.tensor(float(x0)))
        self.register_buffer("v0", torch.tensor(float(v0)))
        self.register_buffer("tau", torch.tensor(float(tau)))

    def forward(self, t):
        if t.dim() == 2:
            t_full = t.unsqueeze(0).expand(self.base.K, *t.shape).contiguous()
        else:
            t_full = t
        gate = (1.0 - torch.exp(-t_full / self.tau)) ** 2
        return self.x0 + t_full * self.v0 + gate * self.base(t_full)


def make_collocation(t_max: float, N: int, K: int, device: str):
    """Return t of shape (K, N, 1) with requires_grad set, all seeds share the grid."""
    t = torch.linspace(0.0, t_max, N, device=device).view(N, 1)
    t = t.unsqueeze(0).expand(K, N, 1).contiguous().requires_grad_(True)
    return t


def grouped_residual_duffing(model, t, zeta, w0, alpha, A=0.0, Omega=0.0):
    x = model(t)
    dx = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
    ddx = torch.autograd.grad(dx.sum(), t, create_graph=True)[0]
    f = A * torch.cos(Omega * t)
    return ddx + 2 * zeta * w0 * dx + (w0**2) * x + alpha * x**3 - f


def grouped_residual_vdp(model, t, mu, w0, A=0.0, Omega=0.0):
    x = model(t)
    dx = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
    ddx = torch.autograd.grad(dx.sum(), t, create_graph=True)[0]
    f = A * torch.cos(Omega * t)
    return ddx - mu * (1 - x**2) * dx + (w0**2) * x - f
