"""Stable hard-IC tuning: lower lr (1e-3) + Adam warmup + more iters + 5 seeds.

Goal: replace the bimodal seed distribution from tune_seeds.py with a
consistent L2 < 1e-3 mean. Comparison anchor for the thesis.
"""

import _common  # noqa
import time, json, numpy as np, torch
from lib.data import ground_truth, PRESETS
from lib.models import MLP, HardICWrapper
from lib.train import train_pinn, duffing_residual, TrainConfig
from lib.eval import predict, l2_relative, save_results

EXP = "tune_stable"
DEVICE = _common.DEVICE
SEEDS = [0, 1, 2, 3, 4]
zeta, w0, alpha = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]
t_ref, x_ref, _ = ground_truth("duffing", (zeta, w0, alpha, 0.0, 0.0), ic, t_max, 4001)
t_colloc = torch.linspace(0, t_max, 2000).reshape(-1, 1)

CFG = dict(width=64, depth=4, lr=1e-3, iters=15000, lbfgs=500)
print(f"stable cfg: {CFG}, seeds: {SEEDS}\n")

errs = []
for s in SEEDS:
    torch.manual_seed(s)
    base = MLP(hidden=CFG["width"], depth=CFG["depth"], activation="tanh", t_scale=t_max)
    model = HardICWrapper(base, x0=ic[0], v0=ic[1], tau=1.0)
    residual = lambda t, m=model: duffing_residual(m, t, zeta, w0, alpha)
    cfg = TrainConfig(iters=CFG["iters"], lr=CFG["lr"], lambda_ic=0.0,
                      lbfgs_steps=CFG["lbfgs"], seed=s, device=DEVICE, log_every=10**9)
    t0 = time.time()
    train_pinn(model, residual, t_colloc, ic=ic, data=None, cfg=cfg)
    err = l2_relative(predict(model, t_ref), x_ref)
    errs.append(err)
    print(f"  seed={s}  L2={err:.3e}  ({time.time()-t0:.1f}s)")

errs = np.array(errs)
print(f"\nstable HARD: mean={errs.mean():.3e}  std={errs.std():.3e}  "
      f"min={errs.min():.3e}  max={errs.max():.3e}")
save_results(_common.RESULTS / EXP, errs=errs, best_cfg=json.dumps(CFG))
