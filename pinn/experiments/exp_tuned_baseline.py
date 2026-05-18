"""Tuned baseline — free damped Duffing forward problem.

Goal: L2 relative error < 1e-2 on x(t) over t in [0, 20].

Recipe (the one that actually works for ODE PINNs at this scale):
  - Small narrow MLP (depth 4, width 64) — width=1024 was overkill and hurt opt
  - Hard IC reparametrization with tau=1.0
  - Adam(lr=1e-3, betas=(0.9, 0.999)) for 10k iters
  - L-BFGS polish (max_iter=2000, strong wolfe) — this is what closes the last 2 OOM
  - 2000 uniform collocation points (no benefit from more on a 1D ODE)
  - Input normalization t -> t/t_max
"""

import _common  # noqa
import time, numpy as np, torch
from lib.data import ground_truth, PRESETS
from lib.models import MLP, HardICWrapper
from lib.train import train_pinn, duffing_residual, TrainConfig
from lib.eval import predict, l2_relative, rmse, max_err, plot_solution, plot_loss_curves, save_results

EXP = "exp_tuned_baseline"
zeta, w0, alpha = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]
DEVICE = _common.DEVICE

t_ref, x_ref, _ = ground_truth("duffing", (zeta, w0, alpha, 0.0, 0.0), ic, t_max, 4001)

torch.manual_seed(0)
base = MLP(hidden=64, depth=4, activation="tanh", t_scale=t_max)
model = HardICWrapper(base, x0=ic[0], v0=ic[1], tau=1.0)

t_colloc = torch.linspace(0, t_max, 2000).reshape(-1, 1)
residual = lambda t: duffing_residual(model, t, zeta, w0, alpha)

cfg = TrainConfig(
    iters=10000, lr=1e-3,
    lambda_ic=0.0,        # hard-IC, no IC loss term
    lambda_phys=1.0,
    lbfgs_steps=2000,     # the polish is what gets us under 1e-2
    log_every=500,
    device=DEVICE, seed=0,
)

print(f"[{EXP}] device={DEVICE}  iters={cfg.iters}+LBFGS({cfg.lbfgs_steps})  hidden=64 depth=4 tau=1.0")
t0 = time.time()
out = train_pinn(model, residual, t_colloc, ic=ic, data=None, cfg=cfg)
wall = time.time() - t0

x_pred = predict(model, t_ref)
metrics = dict(l2_rel=l2_relative(x_pred, x_ref),
               rmse=rmse(x_pred, x_ref),
               max_err=max_err(x_pred, x_ref))
print(f"[{EXP}] wallclock {wall:.1f}s  metrics={metrics}")
if metrics["l2_rel"] < 1e-2:
    print(f"[{EXP}] ✓ TARGET HIT: L2_rel = {metrics['l2_rel']:.3e} < 1e-2")
else:
    print(f"[{EXP}] ✗ L2_rel = {metrics['l2_rel']:.3e} (target < 1e-2)")

plot_solution(t_ref, x_ref, x_pred,
              f"tuned baseline — Duffing  L2={metrics['l2_rel']:.2e}",
              _common.FIGURES / f"{EXP}_solution.png")
plot_loss_curves(out["history"], _common.FIGURES / f"{EXP}_loss.png", title=EXP)
save_results(_common.RESULTS / EXP, t=t_ref, x_ref=x_ref, x_pred=x_pred, wallclock=wall,
             **{f"hist_{k}": np.array(v) for k, v in out["history"].items()},
             **{f"metric_{k}": v for k, v in metrics.items()})
