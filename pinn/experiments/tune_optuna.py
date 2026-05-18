"""Bayesian hyperparameter optimisation with Optuna.

Replaces the greedy one-knob-at-a-time tuner with a TPE-based Bayesian sweep
over the joint hyperparameter space. Default budget: 80 trials, single seed
each (matches the greedy tuner's seed protocol for direct comparability).

Search space:
  lr         ~ LogUniform(1e-4, 3e-2)
  width      in {32, 64, 128, 256}
  depth      in {3, 4, 5, 6}
  tau        ~ LogUniform(0.3, 4.0)
  iters      in {3000, 5000, 8000, 12000, 15000}
  lbfgs      in {0, 200, 500, 1000}
"""

import _common  # noqa
import time, json, numpy as np, torch
import optuna
from optuna.samplers import TPESampler

from lib.data import ground_truth, PRESETS
from lib.models import MLP, HardICWrapper
from lib.train import train_pinn, duffing_residual, TrainConfig
from lib.eval import predict, l2_relative, save_results

EXP = "tune_optuna"
DEVICE = _common.DEVICE
N_TRIALS = 80
SEED = 0

zeta, w0, alpha = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]
t_ref, x_ref, _ = ground_truth("duffing", (zeta, w0, alpha, 0.0, 0.0), ic, t_max, 4001)
t_colloc = torch.linspace(0, t_max, 2000).reshape(-1, 1)
print(f"[{EXP}] target: minimise L2-rel on free Duffing benchmark")
print(f"[{EXP}] budget: {N_TRIALS} TPE trials, single seed (seed={SEED})")


def objective(trial: optuna.Trial) -> float:
    lr     = trial.suggest_float("lr", 1e-4, 3e-2, log=True)
    width  = trial.suggest_categorical("width", [32, 64, 128, 256])
    depth  = trial.suggest_categorical("depth", [3, 4, 5, 6])
    tau    = trial.suggest_float("tau", 0.3, 4.0, log=True)
    iters  = trial.suggest_categorical("iters", [3000, 5000, 8000, 12000, 15000])
    lbfgs  = trial.suggest_categorical("lbfgs", [0, 200, 500, 1000])
    try:
        torch.manual_seed(SEED)
        base = MLP(hidden=width, depth=depth, activation="tanh", t_scale=t_max)
        model = HardICWrapper(base, x0=ic[0], v0=ic[1], tau=tau)
        residual = lambda t, m=model: duffing_residual(m, t, zeta, w0, alpha)
        cfg = TrainConfig(iters=iters, lr=lr, lambda_ic=0.0, lbfgs_steps=lbfgs,
                          seed=SEED, device=DEVICE, log_every=10**9)
        train_pinn(model, residual, t_colloc, ic=ic, data=None, cfg=cfg)
        err = l2_relative(predict(model, t_ref), x_ref)
        if not np.isfinite(err):
            err = 1e3
    except Exception as e:
        print(f"  trial {trial.number}: FAILED ({e})")
        err = 1e3
    return err


t0 = time.time()
study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42),
                            study_name="pinn_duffing")

# Manual progress callback (printed every trial)
def callback(study_: optuna.Study, trial: optuna.trial.FrozenTrial):
    best = study_.best_value
    print(f"  trial {trial.number:3d}: L2={trial.value:.3e}  best so far={best:.3e}")

study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback], show_progress_bar=False)
wallclock = time.time() - t0

best = study.best_trial
print(f"\n[{EXP}] best L2: {best.value:.4e}")
print(f"[{EXP}] best params: {json.dumps(best.params, indent=2)}")
print(f"[{EXP}] wallclock: {wallclock:.1f} s over {N_TRIALS} trials")

# Hyperparameter importance
importance = optuna.importance.get_param_importances(study)
print(f"\n[{EXP}] hyperparameter importance:")
for k, v in importance.items():
    print(f"  {k}: {v:.3f}")

# Save: per-trial scores, best params, importance scores
trials_arr = np.array([(t.number, t.value, *list(t.params.values()))
                        for t in study.trials if t.value is not None], dtype=object)
save_results(_common.RESULTS / EXP,
             best_l2=best.value,
             best_params=json.dumps(best.params),
             importance=json.dumps(importance),
             n_trials=N_TRIALS,
             wallclock=wallclock,
             trial_l2=np.array([t.value for t in study.trials if t.value is not None]),
             trial_lr=np.array([t.params["lr"] for t in study.trials if t.value is not None]),
             trial_width=np.array([t.params["width"] for t in study.trials if t.value is not None]),
             trial_depth=np.array([t.params["depth"] for t in study.trials if t.value is not None]),
             trial_tau=np.array([t.params["tau"] for t in study.trials if t.value is not None]),
             trial_iters=np.array([t.params["iters"] for t in study.trials if t.value is not None]),
             trial_lbfgs=np.array([t.params["lbfgs"] for t in study.trials if t.value is not None]),
             )

# Plot: importance bars + best-so-far convergence
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))

# importance
keys = list(importance.keys()); vals = [importance[k] for k in keys]
ax[0].barh(keys, vals)
ax[0].set_xlabel("relative importance"); ax[0].set_title("Hyperparameter importance (Optuna fANOVA)")

# best-so-far convergence
l2s = [t.value for t in study.trials if t.value is not None]
best_so_far = np.minimum.accumulate(l2s)
ax[1].semilogy(np.arange(1, len(l2s)+1), l2s, "o", markersize=3, alpha=0.4, label="trial L2")
ax[1].semilogy(np.arange(1, len(best_so_far)+1), best_so_far, "-", lw=2, label="best so far")
ax[1].set_xlabel("trial index"); ax[1].set_ylabel("L2 relative error")
ax[1].set_title(f"TPE convergence ({N_TRIALS} trials)"); ax[1].legend()

fig.tight_layout()
fig.savefig(_common.FIGURES / f"{EXP}_summary.png", dpi=120); plt.close(fig)
print(f"\n[{EXP}] wrote figures/{EXP}_summary.png")
