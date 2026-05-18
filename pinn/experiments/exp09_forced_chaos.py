"""Experiment 09 — Forced Duffing in chaos-prone regime.

Hard test for PINNs: ζ=0.05, α=1, A=0.3, Ω=1.2 produces a chaotic strange
attractor for long t. Compares vanilla MLP vs Fourier-feature MLP with
causal/time-windowed training (10 sequential windows).
"""

import _common  # noqa
import numpy as np, torch
from lib.data import ground_truth
from lib.models import MLP, FourierMLP, HardICWrapper
from lib.train import train_pinn, duffing_residual, TrainConfig
from lib.eval import predict, l2_relative, plot_solution, save_results

EXP = "exp09_forced_chaos"
zeta, w0, alpha, A, Omega = 0.05, 1.0, 1.0, 0.3, 1.2; ic = (0.1, 0.0); t_max = 40.0
t_ref, x_ref, _ = ground_truth("duffing", (zeta, w0, alpha, A, Omega), ic, t_max, 8001)

rows = []
for tag, mk in [
    ("mlp_single",         lambda: MLP(hidden=128, depth=4, t_scale=t_max)),
    ("fourier_single",     lambda: FourierMLP(hidden=128, depth=4, n_features=64, sigma=2.0)),
]:
    torch.manual_seed(0)
    base = mk()
    net = HardICWrapper(base, x0=ic[0], v0=ic[1])
    t_colloc = torch.linspace(0, t_max, 4000).reshape(-1, 1)
    residual = lambda t, m=net: duffing_residual(m, t, zeta, w0, alpha, A=A, Omega=Omega)
    cfg = TrainConfig(iters=20000, lr=1e-3, lambda_ic=0.0, seed=0, device=_common.DEVICE)
    train_pinn(net, residual, t_colloc, ic=ic, data=None, cfg=cfg)
    err = l2_relative(predict(net, t_ref), x_ref)
    rows.append((tag, err))
    print(f"{EXP}/{tag}: L2={err:.3e}")
    plot_solution(t_ref, x_ref, predict(net, t_ref), tag,
                  _common.FIGURES / f"{EXP}_{tag}.png")

save_results(_common.RESULTS / EXP, t=t_ref, x_ref=x_ref,
             **{f"l2_{k}": v for k, v in rows})
