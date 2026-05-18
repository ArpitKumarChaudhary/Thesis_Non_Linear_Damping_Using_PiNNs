"""Experiment 01 — Baseline PINN on free damped Duffing.

The headline result. Trains a vanilla MLP-PINN with soft IC enforcement and
fixed loss weights, then reports L2-relative error vs the DOP853 ground truth.
"""

import _common  # noqa
import numpy as np, torch
from lib.data import ground_truth, sample_collocation, PRESETS
from lib.models import MLP
from lib.train import train_pinn, duffing_residual, TrainConfig
from lib.eval import predict, l2_relative, rmse, max_err, plot_solution, plot_loss_curves, save_results

EXP = "exp01_baseline"
preset = PRESETS["free_duffing"]                       # zeta=0.1, w0=1, alpha=1
t_max = preset["t_max"]
zeta, w0, alpha = preset["params"]; ic = preset["ic"]

t_ref, x_ref, v_ref = ground_truth("duffing", (zeta, w0, alpha, 0.0, 0.0), ic, t_max, 4001)

model = MLP(hidden=64, depth=4, activation="tanh", t_scale=t_max)
t_colloc = torch.linspace(0, t_max, 2000).reshape(-1, 1)
residual = lambda t: duffing_residual(model, t, zeta, w0, alpha)

cfg = TrainConfig(iters=15000, lr=1e-3, lambda_ic=100.0, lbfgs_steps=500, seed=0,
                  device=_common.DEVICE)
out = train_pinn(model, residual, t_colloc, ic=ic, data=None, cfg=cfg)

x_pred = predict(model, t_ref)
metrics = dict(l2_rel=l2_relative(x_pred, x_ref),
               rmse=rmse(x_pred, x_ref),
               max_err=max_err(x_pred, x_ref))
print(EXP, metrics)

plot_solution(t_ref, x_ref, x_pred,
              f"baseline PINN — Duffing (ζ={zeta}, α={alpha})",
              _common.FIGURES / f"{EXP}_solution.png")
plot_loss_curves(out["history"], _common.FIGURES / f"{EXP}_loss.png", title=EXP)
save_results(_common.RESULTS / EXP, t=t_ref, x_ref=x_ref, x_pred=x_pred,
             **{f"hist_{k}": np.array(v) for k, v in out["history"].items()},
             **{f"metric_{k}": v for k, v in metrics.items()})
