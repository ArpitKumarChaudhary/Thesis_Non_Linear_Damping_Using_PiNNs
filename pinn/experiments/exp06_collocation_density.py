"""Experiment 06 — How many collocation points are enough?

Sweep N_collocation ∈ {100, 250, 500, 1000, 2000, 4000} and sampling scheme
∈ {uniform, random, latin}. Reports L2 vs collocation budget.
"""

import _common  # noqa
import numpy as np, torch
from lib.data import ground_truth, sample_collocation, PRESETS
from lib.models import MLP
from lib.train import train_pinn, duffing_residual, TrainConfig
from lib.eval import predict, l2_relative, save_results

EXP = "exp06_collocation_density"
zeta, w0, alpha = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]
t_ref, x_ref, _ = ground_truth("duffing", (zeta, w0, alpha, 0.0, 0.0), ic, t_max, 4001)

rows = []
for scheme in ["uniform", "random", "latin"]:
    for N in [100, 250, 500, 1000, 2000, 4000]:
        torch.manual_seed(0)
        t_np = sample_collocation(t_max, N, scheme=scheme, rng=0)
        t_colloc = torch.tensor(t_np, dtype=torch.float32).reshape(-1, 1)
        model = MLP(hidden=64, depth=4, t_scale=t_max)
        residual = lambda t, m=model: duffing_residual(m, t, zeta, w0, alpha)
        cfg = TrainConfig(iters=10000, lr=1e-3, lambda_ic=100.0, seed=0, device=_common.DEVICE)
        train_pinn(model, residual, t_colloc, ic=ic, data=None, cfg=cfg)
        err = l2_relative(predict(model, t_ref), x_ref)
        rows.append((scheme, N, err))
        print(f"{EXP}: scheme={scheme} N={N} L2={err:.3e}")

schemes = np.array([s for (s, _, _) in rows])
arr = np.array([(n, e) for (_, n, e) in rows])
save_results(_common.RESULTS / EXP, scheme=schemes, sweep=arr)
