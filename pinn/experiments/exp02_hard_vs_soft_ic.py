"""Experiment 02 — Hard vs soft initial-condition enforcement.

Soft : add (x(0)-x0)^2 + (x'(0)-v0)^2 to the loss with weight lambda_ic.
Hard : reparametrize x(t) = x0 + t v0 + t^2 NN(t), no IC loss term.
"""

import _common  # noqa
import numpy as np, torch
from lib.data import ground_truth, PRESETS
from lib.models import MLP, HardICWrapper
from lib.train import train_pinn, duffing_residual, TrainConfig
from lib.eval import predict, l2_relative, plot_solution, plot_loss_curves, save_results

EXP = "exp02_hard_vs_soft_ic"
zeta, w0, alpha = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]
t_ref, x_ref, _ = ground_truth("duffing", (zeta, w0, alpha, 0.0, 0.0), ic, t_max, 4001)
t_colloc = torch.linspace(0, t_max, 2000).reshape(-1, 1)

results = {}
for mode in ["soft", "hard"]:
    torch.manual_seed(0)
    if mode == "soft":
        model = MLP(hidden=64, depth=4, t_scale=t_max)
        cfg = TrainConfig(iters=15000, lr=1e-3, lambda_ic=100.0, seed=0, device=_common.DEVICE)
    else:
        model = HardICWrapper(MLP(hidden=64, depth=4, t_scale=t_max), x0=ic[0], v0=ic[1])
        cfg = TrainConfig(iters=15000, lr=1e-3, lambda_ic=0.0, seed=0, device=_common.DEVICE)
    residual = lambda t, m=model: duffing_residual(m, t, zeta, w0, alpha)
    out = train_pinn(model, residual, t_colloc, ic=ic, data=None, cfg=cfg)
    x_pred = predict(model, t_ref)
    err = l2_relative(x_pred, x_ref)
    results[mode] = dict(x_pred=x_pred, l2_rel=err, history=out["history"])
    print(f"{EXP}/{mode}: L2_rel={err:.3e}")
    plot_solution(t_ref, x_ref, x_pred, f"{mode} IC", _common.FIGURES / f"{EXP}_{mode}.png")
    plot_loss_curves(out["history"], _common.FIGURES / f"{EXP}_{mode}_loss.png", title=f"{EXP}/{mode}")

save_results(_common.RESULTS / EXP, t=t_ref, x_ref=x_ref,
             x_pred_soft=results["soft"]["x_pred"],
             x_pred_hard=results["hard"]["x_pred"],
             l2_soft=results["soft"]["l2_rel"],
             l2_hard=results["hard"]["l2_rel"])
