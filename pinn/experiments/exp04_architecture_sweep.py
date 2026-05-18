"""Experiment 04 — Architecture sweep: depth × width × activation."""

import _common  # noqa
import itertools, numpy as np, torch
from lib.data import ground_truth, PRESETS
from lib.models import MLP
from lib.train import train_pinn, duffing_residual, TrainConfig
from lib.eval import predict, l2_relative, save_results

EXP = "exp04_architecture_sweep"
zeta, w0, alpha = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]
t_ref, x_ref, _ = ground_truth("duffing", (zeta, w0, alpha, 0.0, 0.0), ic, t_max, 4001)
t_colloc = torch.linspace(0, t_max, 2000).reshape(-1, 1)

depths = [2, 3, 4, 5]
widths = [16, 32, 64, 128]
acts = ["tanh", "sin", "gelu"]

rows = []
for depth, width, act in itertools.product(depths, widths, acts):
    torch.manual_seed(0)
    model = MLP(hidden=width, depth=depth, activation=act, t_scale=t_max)
    residual = lambda t, m=model: duffing_residual(m, t, zeta, w0, alpha)
    cfg = TrainConfig(iters=8000, lr=1e-3, lambda_ic=100.0, seed=0, device=_common.DEVICE)
    train_pinn(model, residual, t_colloc, ic=ic, data=None, cfg=cfg)
    err = l2_relative(predict(model, t_ref), x_ref)
    n_params = sum(p.numel() for p in model.parameters())
    rows.append((depth, width, act, n_params, err))
    print(f"{EXP}: depth={depth} width={width} act={act} params={n_params} L2={err:.3e}")

arr = np.array([(d, w, n, e) for (d, w, _, n, e) in rows], dtype=float)
acts_arr = np.array([a for (_, _, a, _, _) in rows])
save_results(_common.RESULTS / EXP, sweep=arr, activation=acts_arr)
