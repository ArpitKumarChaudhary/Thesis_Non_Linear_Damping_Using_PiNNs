"""Greedy hyperparameter tuner for the free-Duffing forward problem.

Goal: find a config that hits L2-relative < 1e-2 on the headline benchmark
(ζ=0.1, α=1, ic=(1,0), t_max=20). Single non-ensemble MLP, GPU-accelerated.

Strategy: start from a reasonable seed config, then sweep one knob at a time
in a fixed order (lr → arch → IC scheme → L-BFGS). Keep the best config seen.
"""

import _common  # noqa
import time, copy, json, numpy as np, torch
from pathlib import Path
from lib.data import ground_truth, PRESETS
from lib.models import MLP, HardICWrapper
from lib.train import train_pinn, duffing_residual, TrainConfig
from lib.eval import predict, l2_relative, plot_solution, save_results

EXP = "tune_baseline"
DEVICE = _common.DEVICE

zeta, w0, alpha = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]
t_ref, x_ref, _ = ground_truth("duffing", (zeta, w0, alpha, 0.0, 0.0), ic, t_max, 4001)


def run(cfg_dict):
    """Build model+train using cfg_dict; return L2_rel and wallclock seconds."""
    torch.manual_seed(cfg_dict.get("seed", 0))
    width = cfg_dict["width"]; depth = cfg_dict["depth"]
    base = MLP(hidden=width, depth=depth, activation=cfg_dict["act"], t_scale=t_max)
    if cfg_dict["ic_mode"] == "hard":
        model = HardICWrapper(base, x0=ic[0], v0=ic[1], tau=cfg_dict["tau"])
        lambda_ic = 0.0
    else:
        model = base
        lambda_ic = cfg_dict["lambda_ic"]
    n_coll = cfg_dict["n_coll"]
    t_colloc = torch.linspace(0, t_max, n_coll).reshape(-1, 1)
    residual = lambda t, m=model: duffing_residual(m, t, zeta, w0, alpha)
    cfg = TrainConfig(iters=cfg_dict["iters"], lr=cfg_dict["lr"],
                      lambda_ic=lambda_ic, lbfgs_steps=cfg_dict["lbfgs"],
                      seed=cfg_dict.get("seed", 0), device=DEVICE,
                      log_every=10**9)        # silence logging
    t0 = time.time()
    train_pinn(model, residual, t_colloc, ic=ic, data=None, cfg=cfg)
    err = l2_relative(predict(model, t_ref), x_ref)
    return err, time.time() - t0, model


def best(history):
    return min(history, key=lambda r: r["l2"])


seed_cfg = dict(width=64, depth=4, act="tanh", ic_mode="hard", tau=1.0,
                lambda_ic=100.0, n_coll=2000, lr=1e-3, iters=5000, lbfgs=0)
history = []

def trial(name, **overrides):
    c = copy.deepcopy(best_cfg if history else seed_cfg)
    c.update(overrides)
    err, dt, _ = run(c)
    history.append(dict(name=name, l2=err, dt=dt, cfg=c))
    print(f"  {name:30s}  L2={err:.3e}  ({dt:.1f}s)")
    return err

best_cfg = seed_cfg

# Phase 0 — seed
print("[phase 0] seed config")
trial("seed", **seed_cfg)
best_cfg = best(history)["cfg"]
print(f"  best so far: L2={best(history)['l2']:.3e}")

# Phase 1 — learning rate
print("[phase 1] lr sweep")
for lr in [3e-4, 1e-3, 3e-3, 1e-2]:
    trial(f"lr={lr}", lr=lr)
best_cfg = best(history)["cfg"]
print(f"  best so far: lr={best_cfg['lr']}  L2={best(history)['l2']:.3e}")

# Phase 2 — architecture
print("[phase 2] architecture sweep")
for w, d in [(32, 3), (64, 4), (128, 4), (128, 5), (256, 4)]:
    trial(f"w={w} d={d}", width=w, depth=d)
best_cfg = best(history)["cfg"]
print(f"  best so far: width={best_cfg['width']} depth={best_cfg['depth']}  L2={best(history)['l2']:.3e}")

# Phase 3 — hard-IC tau
print("[phase 3] hard-IC tau sweep")
for tau in [0.5, 1.0, 2.0, 4.0]:
    trial(f"tau={tau}", tau=tau)
best_cfg = best(history)["cfg"]
print(f"  best so far: tau={best_cfg['tau']}  L2={best(history)['l2']:.3e}")

# Phase 4 — more collocation + more iters
print("[phase 4] longer training")
for n_c, it in [(2000, 10000), (4000, 10000), (4000, 20000)]:
    trial(f"n={n_c} it={it}", n_coll=n_c, iters=it)
best_cfg = best(history)["cfg"]
print(f"  best so far: n_coll={best_cfg['n_coll']} iters={best_cfg['iters']}  L2={best(history)['l2']:.3e}")

# Phase 5 — L-BFGS polish
print("[phase 5] L-BFGS polish")
for lb in [200, 500, 1000]:
    trial(f"lbfgs={lb}", lbfgs=lb)
best_cfg = best(history)["cfg"]
final = best(history)
print(f"\nFINAL BEST: L2={final['l2']:.4e}")
print(f"            cfg={json.dumps(final['cfg'], indent=2)}")

# Save best model + plot
err, dt, model = run(final["cfg"])
x_pred = predict(model, t_ref)
plot_solution(t_ref, x_ref, x_pred,
              f"tuned baseline — L2={err:.2e}",
              _common.FIGURES / f"{EXP}_solution.png")
save_results(_common.RESULTS / EXP, t=t_ref, x_ref=x_ref, x_pred=x_pred,
             l2_final=err, history=np.array([(h["name"], h["l2"], h["dt"]) for h in history], dtype=object),
             best_cfg=json.dumps(final["cfg"]))
print(f"\nDone. Best L2 = {err:.4e}  ({'reached' if err < 1e-2 else 'did NOT reach'} target 1e-2)")
