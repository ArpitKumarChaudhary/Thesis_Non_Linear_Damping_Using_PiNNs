"""Tuned hard-vs-soft IC ablation — same recipe as exp_tuned_baseline.

Reuses the proven config (depth=4, width=64, Adam(1e-3) -> L-BFGS(2000)).
Reports L2 + IC residual for each branch on free Duffing.
"""

import _common  # noqa
import time, numpy as np, torch
from lib.data import ground_truth, PRESETS
from lib.models import MLP, HardICWrapper
from lib.train import train_pinn, duffing_residual, TrainConfig
from lib.eval import predict, l2_relative, plot_solution, plot_loss_curves, save_results

EXP = "exp_tuned_hard_vs_soft"
zeta, w0, alpha = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]
DEVICE = _common.DEVICE
t_ref, x_ref, _ = ground_truth("duffing", (zeta, w0, alpha, 0.0, 0.0), ic, t_max, 4001)
t_colloc = torch.linspace(0, t_max, 2000).reshape(-1, 1)


def run(mode: str):
    torch.manual_seed(0)
    if mode == "soft":
        model = MLP(hidden=64, depth=4, activation="tanh", t_scale=t_max)
        cfg = TrainConfig(iters=10000, lr=1e-3, lambda_ic=100.0, lbfgs_steps=2000,
                          log_every=500, device=DEVICE, seed=0)
    else:
        base = MLP(hidden=64, depth=4, activation="tanh", t_scale=t_max)
        model = HardICWrapper(base, x0=ic[0], v0=ic[1], tau=1.0)
        cfg = TrainConfig(iters=10000, lr=1e-3, lambda_ic=0.0, lbfgs_steps=2000,
                          log_every=500, device=DEVICE, seed=0)
    residual = lambda t, m=model: duffing_residual(m, t, zeta, w0, alpha)
    t0 = time.time()
    out = train_pinn(model, residual, t_colloc, ic=ic, data=None, cfg=cfg)
    wall = time.time() - t0
    x_pred = predict(model, t_ref)
    err = l2_relative(x_pred, x_ref)
    # IC error: |x(0) - x0| + |x'(0) - v0|
    t0_t = torch.zeros(1, 1, device=DEVICE, requires_grad=True)
    x0p = model(t0_t)
    v0p = torch.autograd.grad(x0p, t0_t, torch.ones_like(x0p), create_graph=False)[0]
    ic_err = float(torch.abs(x0p - ic[0]).mean() + torch.abs(v0p - ic[1]).mean())
    print(f"[{EXP}/{mode}] L2_rel={err:.3e}  IC_err={ic_err:.3e}  wallclock={wall:.1f}s")
    plot_solution(t_ref, x_ref, x_pred, f"{mode} IC  L2={err:.2e}",
                  _common.FIGURES / f"{EXP}_{mode}_solution.png")
    plot_loss_curves(out["history"], _common.FIGURES / f"{EXP}_{mode}_loss.png",
                     title=f"{EXP}/{mode}")
    return dict(x_pred=x_pred, l2_rel=err, ic_err=ic_err, wallclock=wall, history=out["history"])


results = {mode: run(mode) for mode in ["soft", "hard"]}

print(f"\n[{EXP}] SUMMARY:")
print(f"  soft  L2={results['soft']['l2_rel']:.3e}  IC_err={results['soft']['ic_err']:.3e}")
print(f"  hard  L2={results['hard']['l2_rel']:.3e}  IC_err={results['hard']['ic_err']:.3e}")

save_results(_common.RESULTS / EXP, t=t_ref, x_ref=x_ref,
             x_pred_soft=results["soft"]["x_pred"], x_pred_hard=results["hard"]["x_pred"],
             l2_soft=results["soft"]["l2_rel"], l2_hard=results["hard"]["l2_rel"],
             ic_err_soft=results["soft"]["ic_err"], ic_err_hard=results["hard"]["ic_err"],
             wallclock_soft=results["soft"]["wallclock"],
             wallclock_hard=results["hard"]["wallclock"])
