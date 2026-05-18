"""Tuned hard-vs-soft IC ablation, using the winning baseline config:
   width=64, depth=4, tanh, lr=0.01, iters=5000 (Adam) + 500 L-BFGS, N=2000.
   - hard branch:  HardICWrapper(tau=1.0)
   - soft branch:  same MLP, soft IC loss with sweep over lambda_ic.
"""

import _common  # noqa
import time, json, numpy as np, torch
from lib.data import ground_truth, PRESETS
from lib.models import MLP, HardICWrapper
from lib.train import train_pinn, duffing_residual, TrainConfig
from lib.eval import predict, l2_relative, plot_solution, plot_loss_curves, save_results

EXP = "tune_hard_vs_soft"
DEVICE = _common.DEVICE
zeta, w0, alpha = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]
t_ref, x_ref, _ = ground_truth("duffing", (zeta, w0, alpha, 0.0, 0.0), ic, t_max, 4001)
t_colloc = torch.linspace(0, t_max, 2000).reshape(-1, 1)

WINNING = dict(width=64, depth=4, act="tanh", lr=1e-2, iters=5000, lbfgs=500)
print(f"using winning config: {WINNING}")

results = {}

# ------------------------------------------------------------------ HARD branch
print("\n[hard IC, tau=1.0]")
torch.manual_seed(0)
base = MLP(hidden=WINNING["width"], depth=WINNING["depth"],
           activation=WINNING["act"], t_scale=t_max)
hard_model = HardICWrapper(base, x0=ic[0], v0=ic[1], tau=1.0)
residual_h = lambda t: duffing_residual(hard_model, t, zeta, w0, alpha)
cfg_h = TrainConfig(iters=WINNING["iters"], lr=WINNING["lr"], lambda_ic=0.0,
                    lbfgs_steps=WINNING["lbfgs"], seed=0, device=DEVICE,
                    log_every=10**9)
t0 = time.time()
out_h = train_pinn(hard_model, residual_h, t_colloc, ic=ic, data=None, cfg=cfg_h)
x_pred_hard = predict(hard_model, t_ref)
err_hard = l2_relative(x_pred_hard, x_ref)
print(f"  HARD L2 = {err_hard:.4e}  ({time.time()-t0:.1f}s)")
results["hard"] = dict(l2=err_hard, x_pred=x_pred_hard,
                       hist={k: np.array(v) for k, v in out_h["history"].items()})

# ------------------------------------------------------------------ SOFT branch
LAMBDAS = [1.0, 10.0, 100.0, 1000.0]
soft_branch = {}
for lam in LAMBDAS:
    print(f"\n[soft IC, lambda_ic={lam}]")
    torch.manual_seed(0)
    soft_model = MLP(hidden=WINNING["width"], depth=WINNING["depth"],
                     activation=WINNING["act"], t_scale=t_max)
    residual_s = lambda t, m=soft_model: duffing_residual(m, t, zeta, w0, alpha)
    cfg_s = TrainConfig(iters=WINNING["iters"], lr=WINNING["lr"], lambda_ic=lam,
                        lbfgs_steps=WINNING["lbfgs"], seed=0, device=DEVICE,
                        log_every=10**9)
    t0 = time.time()
    out_s = train_pinn(soft_model, residual_s, t_colloc, ic=ic, data=None, cfg=cfg_s)
    x_pred_soft = predict(soft_model, t_ref)
    err_soft = l2_relative(x_pred_soft, x_ref)
    print(f"  SOFT (λ={lam}) L2 = {err_soft:.4e}  ({time.time()-t0:.1f}s)")
    soft_branch[f"lam_{int(lam)}"] = dict(l2=err_soft, x_pred=x_pred_soft,
                                          hist={k: np.array(v) for k, v in out_s["history"].items()})

# best soft over lambdas
best_lam = min(soft_branch, key=lambda k: soft_branch[k]["l2"])
print(f"\nbest soft branch: {best_lam}  L2={soft_branch[best_lam]['l2']:.4e}")
print(f"hard branch:      L2={results['hard']['l2']:.4e}")

plot_solution(t_ref, x_ref, results["hard"]["x_pred"],
              f"hard IC — L2={results['hard']['l2']:.2e}",
              _common.FIGURES / f"{EXP}_hard.png")
plot_solution(t_ref, x_ref, soft_branch[best_lam]["x_pred"],
              f"soft IC ({best_lam}) — L2={soft_branch[best_lam]['l2']:.2e}",
              _common.FIGURES / f"{EXP}_soft_best.png")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 3.4))
labels = ["hard"] + [f"soft λ={int(l)}" for l in LAMBDAS]
errs = [results["hard"]["l2"]] + [soft_branch[f"lam_{int(l)}"]["l2"] for l in LAMBDAS]
bars = ax.bar(labels, errs)
ax.set_yscale("log"); ax.set_ylabel("L2 relative error")
ax.set_title("Hard vs soft IC — tuned config")
ax.axhline(1e-2, color="r", lw=0.5, ls=":", label="target 1e-2")
ax.legend(loc="best")
fig.tight_layout(); fig.savefig(_common.FIGURES / f"{EXP}_summary.png", dpi=120); plt.close(fig)

save_results(_common.RESULTS / EXP, t=t_ref, x_ref=x_ref,
             x_pred_hard=results["hard"]["x_pred"], l2_hard=results["hard"]["l2"],
             lambdas=np.array(LAMBDAS),
             **{f"x_pred_soft_lam{int(l)}": soft_branch[f"lam_{int(l)}"]["x_pred"] for l in LAMBDAS},
             **{f"l2_soft_lam{int(l)}": soft_branch[f"lam_{int(l)}"]["l2"] for l in LAMBDAS},
             best_cfg=json.dumps(WINNING))
print(f"\nDone. Wrote summary plot to figures/{EXP}_summary.png")
