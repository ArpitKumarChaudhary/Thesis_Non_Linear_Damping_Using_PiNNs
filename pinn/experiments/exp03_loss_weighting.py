"""Experiment 03 — Loss weighting strategies.

Compares: fixed λ, manual sweep over λ_ic, gradient-norm balancing, NTK-style.
"""

import _common  # noqa
import numpy as np, torch
from lib.data import ground_truth, PRESETS
from lib.models import MLP
from lib.train import train_pinn, duffing_residual, TrainConfig
from lib.eval import predict, l2_relative, plot_loss_curves, save_results

EXP = "exp03_loss_weighting"
zeta, w0, alpha = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]
t_ref, x_ref, _ = ground_truth("duffing", (zeta, w0, alpha, 0.0, 0.0), ic, t_max, 4001)
t_colloc = torch.linspace(0, t_max, 2000).reshape(-1, 1)

configs = [
    ("fixed_lic_1",    dict(weighting="fixed", lambda_ic=1.0)),
    ("fixed_lic_10",   dict(weighting="fixed", lambda_ic=10.0)),
    ("fixed_lic_100",  dict(weighting="fixed", lambda_ic=100.0)),
    ("fixed_lic_1000", dict(weighting="fixed", lambda_ic=1000.0)),
    ("gradnorm",       dict(weighting="gradnorm", lambda_ic=1.0)),
    ("ntk",            dict(weighting="ntk", lambda_ic=1.0)),
]

table = {}
for name, kw in configs:
    torch.manual_seed(0)
    model = MLP(hidden=64, depth=4, t_scale=t_max)
    cfg = TrainConfig(iters=15000, lr=1e-3, seed=0, device=_common.DEVICE, **kw)
    residual = lambda t, m=model: duffing_residual(m, t, zeta, w0, alpha)
    out = train_pinn(model, residual, t_colloc, ic=ic, data=None, cfg=cfg)
    x_pred = predict(model, t_ref)
    err = l2_relative(x_pred, x_ref)
    table[name] = err
    print(f"{EXP}/{name}: L2_rel={err:.3e}")
    plot_loss_curves(out["history"], _common.FIGURES / f"{EXP}_{name}_loss.png", title=name)

save_results(_common.RESULTS / EXP, t=t_ref, x_ref=x_ref,
             **{f"l2_{k}": v for k, v in table.items()})
print(f"{EXP} summary: {table}")
