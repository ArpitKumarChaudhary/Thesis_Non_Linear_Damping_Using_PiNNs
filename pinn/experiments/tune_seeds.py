"""5-seed reruns of the two tuned headline experiments — for error bars.

Re-runs the winning config from tune_baseline / tune_hard_vs_soft over
seeds 0..4 and reports mean ± std of L2-relative.
"""

import _common  # noqa
import json, time, numpy as np, torch
from lib.data import ground_truth, PRESETS
from lib.models import MLP, HardICWrapper
from lib.train import train_pinn, duffing_residual, TrainConfig
from lib.eval import predict, l2_relative, save_results

EXP = "tune_seeds"
DEVICE = _common.DEVICE
SEEDS = [0, 1, 2, 3, 4]

zeta, w0, alpha = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]
t_ref, x_ref, _ = ground_truth("duffing", (zeta, w0, alpha, 0.0, 0.0), ic, t_max, 4001)
t_colloc = torch.linspace(0, t_max, 2000).reshape(-1, 1)

WIN = dict(width=64, depth=4, act="tanh", lr=1e-2, iters=5000, lbfgs=500)
print(f"winning config: {WIN}\nseeds: {SEEDS}\n")


def train_once(seed, mode, lam=0.0, tau=1.0):
    torch.manual_seed(seed)
    base = MLP(hidden=WIN["width"], depth=WIN["depth"], activation=WIN["act"], t_scale=t_max)
    if mode == "hard":
        model = HardICWrapper(base, x0=ic[0], v0=ic[1], tau=tau)
        lambda_ic = 0.0
    else:
        model = base
        lambda_ic = lam
    residual = lambda t, m=model: duffing_residual(m, t, zeta, w0, alpha)
    cfg = TrainConfig(iters=WIN["iters"], lr=WIN["lr"], lambda_ic=lambda_ic,
                      lbfgs_steps=WIN["lbfgs"], seed=seed, device=DEVICE, log_every=10**9)
    t0 = time.time()
    train_pinn(model, residual, t_colloc, ic=ic, data=None, cfg=cfg)
    return l2_relative(predict(model, t_ref), x_ref), time.time() - t0


# -------- 1. baseline / hard-IC --------
print("=== exp1: tuned baseline (hard IC, tau=1.0) ===")
hard_errs = []
for s in SEEDS:
    err, dt = train_once(s, "hard", tau=1.0)
    hard_errs.append(err)
    print(f"  seed={s}  L2={err:.3e}  ({dt:.1f}s)")
hard_errs = np.array(hard_errs)
print(f"  HARD L2: mean={hard_errs.mean():.3e}  std={hard_errs.std():.3e}  "
      f"min={hard_errs.min():.3e}  max={hard_errs.max():.3e}")

# -------- 2. hard vs soft ablation --------
print("\n=== exp2: hard vs soft IC, all variants × seeds ===")
LAMBDAS = [1.0, 10.0, 100.0, 1000.0]
soft_errs = {}
for lam in LAMBDAS:
    es = []
    for s in SEEDS:
        err, dt = train_once(s, "soft", lam=lam)
        es.append(err)
        print(f"  λ={lam}  seed={s}  L2={err:.3e}  ({dt:.1f}s)")
    soft_errs[lam] = np.array(es)
    print(f"  SOFT λ={lam} L2: mean={soft_errs[lam].mean():.3e}  std={soft_errs[lam].std():.3e}")

# -------- summary table --------
print("\n=== SUMMARY (mean ± std over 5 seeds) ===")
print(f"  hard IC, tau=1.0    : {hard_errs.mean():.3e} ± {hard_errs.std():.3e}")
for lam in LAMBDAS:
    e = soft_errs[lam]
    print(f"  soft IC, λ={lam:>6.1f}: {e.mean():.3e} ± {e.std():.3e}")

save_results(_common.RESULTS / EXP,
             hard_errs=hard_errs,
             **{f"soft_lam{int(l)}_errs": soft_errs[l] for l in LAMBDAS},
             lambdas=np.array(LAMBDAS),
             best_cfg=json.dumps(WIN))

# bar plot with error bars
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.5, 3.6))
labels = ["hard\nτ=1"] + [f"soft\nλ={int(l)}" for l in LAMBDAS]
means = [hard_errs.mean()] + [soft_errs[l].mean() for l in LAMBDAS]
stds  = [hard_errs.std()] + [soft_errs[l].std() for l in LAMBDAS]
ax.bar(labels, means, yerr=stds, capsize=4)
ax.set_yscale("log"); ax.set_ylabel("L2 relative error  (5 seeds)")
ax.axhline(1e-2, color="r", lw=0.5, ls=":", label="target 1e-2")
ax.set_title("Tuned PINN — hard vs soft IC, error bars over 5 seeds")
ax.legend()
fig.tight_layout(); fig.savefig(_common.FIGURES / f"{EXP}_summary.png", dpi=120); plt.close(fig)
print(f"\nsaved figures/{EXP}_summary.png")
