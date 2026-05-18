"""Experiment 10 — Van der Pol limit-cycle oscillator.

Differs from Duffing: damping itself is nonlinear (state-dependent),
limit cycle is the attractor. Sweeps μ ∈ {0.5, 1, 2, 4}.
"""

import _common  # noqa
import numpy as np, torch
from lib.data import ground_truth
from lib.models import MLP, HardICWrapper
from lib.train import train_pinn, vdp_residual, TrainConfig
from lib.eval import predict, l2_relative, plot_solution, save_results

EXP = "exp10_van_der_pol"
ic = (2.0, 0.0); w0 = 1.0; t_max = 30.0

rows = []
for mu in [0.5, 1.0, 2.0, 4.0]:
    t_ref, x_ref, _ = ground_truth("vdp", (mu, w0, 0.0, 0.0), ic, t_max, 4001)
    torch.manual_seed(0)
    base = MLP(hidden=128, depth=5, activation="tanh", t_scale=t_max)
    model = HardICWrapper(base, x0=ic[0], v0=ic[1])
    t_colloc = torch.linspace(0, t_max, 3000).reshape(-1, 1)
    residual = lambda t, m=model: vdp_residual(m, t, mu, w0)
    cfg = TrainConfig(iters=20000, lr=1e-3, lambda_ic=0.0, seed=0, device=_common.DEVICE)
    train_pinn(model, residual, t_colloc, ic=ic, data=None, cfg=cfg)
    x_pred = predict(model, t_ref)
    err = l2_relative(x_pred, x_ref)
    rows.append((mu, err))
    print(f"{EXP}: μ={mu}  L2={err:.3e}")
    plot_solution(t_ref, x_ref, x_pred, f"VdP μ={mu}",
                  _common.FIGURES / f"{EXP}_mu{mu}.png")

save_results(_common.RESULTS / EXP, sweep=np.array(rows))
