"""Experiment 07 — Inverse problem: identify ζ, α from sparse noisy data.

We treat (zeta, alpha) as nn.Parameters jointly optimized with the network.
Sparse observations (N_obs ~ 30) drive the data loss; the physics residual
uses the *current learned* (zeta, alpha). Reports parameter convergence trace.
"""

import _common  # noqa
import numpy as np, torch
from lib.data import ground_truth, sparse_observations, PRESETS
from lib.models import MLP, LearnableParams
from lib.train import train_pinn, derivatives, TrainConfig
from lib.eval import predict, l2_relative, plot_solution, save_results

EXP = "exp07_inverse_problem"
zeta_true, w0, alpha_true = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]
t_ref, x_ref, _ = ground_truth("duffing", (zeta_true, w0, alpha_true, 0.0, 0.0), ic, t_max, 4001)
t_obs_np, x_obs_np = sparse_observations(t_ref, x_ref, n_obs=30, noise_sigma=0.02, rng=0)

t_obs = torch.tensor(t_obs_np, dtype=torch.float32).reshape(-1, 1)
x_obs = torch.tensor(x_obs_np, dtype=torch.float32).reshape(-1, 1)
t_colloc = torch.linspace(0, t_max, 2000).reshape(-1, 1)

model = MLP(hidden=64, depth=4, t_scale=t_max)
phys = LearnableParams(dict(zeta=0.5, alpha=0.5))  # poor initial guesses

def residual(t):
    x, dx, ddx = derivatives(model, t)
    return ddx + 2 * phys.zeta * w0 * dx + (w0**2) * x + phys.alpha * x**3

cfg = TrainConfig(iters=20000, lr=2e-3, lambda_ic=100.0, lambda_data=100.0, seed=0,
                  device=_common.DEVICE)
out = train_pinn(model, residual, t_colloc, ic=ic, data=(t_obs, x_obs), cfg=cfg,
                 extra_params=phys)

x_pred = predict(model, t_ref)
err = l2_relative(x_pred, x_ref)
final = phys.asdict()
print(f"{EXP}: L2_rel={err:.3e}  true (ζ={zeta_true}, α={alpha_true})  learned {final}")

plot_solution(t_ref, x_ref, x_pred,
              f"inverse: ζ={final['zeta']:.3f}/{zeta_true} α={final['alpha']:.3f}/{alpha_true}",
              _common.FIGURES / f"{EXP}_solution.png")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5.5, 3.5))
for k in ("zeta", "alpha"):
    if k in out["extra_history"]:
        ax.plot(out["history"]["step"], out["extra_history"][k], label=f"{k} (true={zeta_true if k=='zeta' else alpha_true})")
ax.axhline(zeta_true, color="k", lw=0.5, ls=":")
ax.axhline(alpha_true, color="k", lw=0.5, ls=":")
ax.set_xlabel("step"); ax.set_ylabel("value"); ax.legend()
ax.set_title("parameter convergence")
fig.tight_layout(); fig.savefig(_common.FIGURES / f"{EXP}_params.png", dpi=120); plt.close(fig)

save_results(_common.RESULTS / EXP, t=t_ref, x_ref=x_ref, x_pred=x_pred,
             t_obs=t_obs_np, x_obs=x_obs_np,
             zeta_true=zeta_true, alpha_true=alpha_true,
             zeta_learned=final["zeta"], alpha_learned=final["alpha"], l2_rel=err,
             zeta_hist=np.array(out["extra_history"]["zeta"]),
             alpha_hist=np.array(out["extra_history"]["alpha"]))
