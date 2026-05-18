"""PINN model variants: vanilla MLP, Fourier features, hard-constraint wrapper."""

from __future__ import annotations
import math
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, hidden=64, depth=4, activation="tanh",
                 t_scale: float = 1.0):
        """t_scale: divide input by this before the first layer to map t∈[0,T]→O(1)."""
        super().__init__()
        act = {"tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU,
               "sin": SinAct, "relu": nn.ReLU}[activation]
        layers = [nn.Linear(in_dim, hidden), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("t_scale", torch.tensor(float(t_scale)))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t / self.t_scale)


class SinAct(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class FourierFeatures(nn.Module):
    """Random Fourier feature embedding gamma(t) = [cos(2π B t), sin(2π B t)]."""

    def __init__(self, in_dim=1, n_features=32, sigma=1.0):
        super().__init__()
        B = torch.randn(in_dim, n_features) * sigma
        self.register_buffer("B", B)
        self.out_dim = 2 * n_features

    def forward(self, t):
        proj = 2 * math.pi * t @ self.B
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class FourierMLP(nn.Module):
    def __init__(self, hidden=64, depth=4, n_features=32, sigma=1.0, activation="tanh"):
        super().__init__()
        self.ff = FourierFeatures(in_dim=1, n_features=n_features, sigma=sigma)
        self.mlp = MLP(in_dim=self.ff.out_dim, hidden=hidden, depth=depth,
                       activation=activation)

    def forward(self, t):
        return self.mlp(self.ff(t))


class HardICWrapper(nn.Module):
    """Enforce x(0)=x0 and x'(0)=v0 exactly via a bounded scaling.

      x(t) = x0 + t v0 + (1 - exp(-t/tau))^2 * NN(t)

    The factor is O(t^2) near 0 (so f(0)=0 and f'(0)=0, preserving IC) but
    saturates to O(1) for t >> tau — avoids the t^2 blow-up of the polynomial
    ansatz at long horizons. Default tau = 1.0.
    """

    def __init__(self, base: nn.Module, x0: float, v0: float, tau: float = 1.0):
        super().__init__()
        self.base = base
        self.register_buffer("x0", torch.tensor(float(x0)))
        self.register_buffer("v0", torch.tensor(float(v0)))
        self.register_buffer("tau", torch.tensor(float(tau)))

    def forward(self, t):
        gate = (1.0 - torch.exp(-t / self.tau)) ** 2
        return self.x0 + t * self.v0 + gate * self.base(t)


class LearnableParams(nn.Module):
    """Bundle of trainable physics parameters for inverse problems."""

    def __init__(self, init: dict):
        super().__init__()
        for k, v in init.items():
            setattr(self, k, nn.Parameter(torch.tensor(float(v))))

    def asdict(self):
        return {k: float(getattr(self, k).detach()) for k in self._parameters}
