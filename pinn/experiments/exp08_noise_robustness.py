"""Experiment 08 — Noise robustness of the inverse problem.

Sweep observation noise σ ∈ {0, 0.01, 0.02, 0.05, 0.1, 0.2}. Reports parameter
estimate vs σ and L2 error of the reconstructed trajectory.
"""

import _common  # noqa
import numpy as np, torch
from lib.data import ground_truth, sparse_observations, PRESETS
from lib.models import MLP, LearnableParams
from lib.train import train_pinn, derivatives, TrainConfig
from lib.eval import predict, l2_relative, save_results

EXP = "exp08_noise_robustness"
zeta_true, w0, alpha_true = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]
t_ref, x_ref, _ = ground_truth("duffing", (zeta_true, w0, alpha_true, 0.0, 0.0), ic, t_max, 4001)

rows = []
for sigma in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]:
    torch.manual_seed(0)
    t_obs_np, x_obs_np = sparse_observations(t_ref, x_ref, n_obs=40, noise_sigma=sigma, rng=0)
    t_obs = torch.tensor(t_obs_np, dtype=torch.float32).reshape(-1, 1)
    x_obs = torch.tensor(x_obs_np, dtype=torch.float32).reshape(-1, 1)
    t_colloc = torch.linspace(0, t_max, 2000).reshape(-1, 1)

    model = MLP(hidden=64, depth=4, t_scale=t_max)
    phys = LearnableParams(dict(zeta=0.5, alpha=0.5))
    def residual(t, m=model, p=phys):
        x, dx, ddx = derivatives(m, t)
        return ddx + 2 * p.zeta * w0 * dx + (w0**2) * x + p.alpha * x**3
    cfg = TrainConfig(iters=15000, lr=2e-3, lambda_ic=100.0, lambda_data=100.0, seed=0,
                      device=_common.DEVICE)
    train_pinn(model, residual, t_colloc, ic=ic, data=(t_obs, x_obs), cfg=cfg, extra_params=phys)
    final = phys.asdict()
    err = l2_relative(predict(model, t_ref), x_ref)
    rows.append((sigma, final["zeta"], final["alpha"], err))
    print(f"{EXP}: σ={sigma}  ζ={final['zeta']:.4f} (true {zeta_true})"
          f"  α={final['alpha']:.4f} (true {alpha_true})  L2={err:.3e}")

arr = np.array(rows)
save_results(_common.RESULTS / EXP, sweep=arr,
             zeta_true=zeta_true, alpha_true=alpha_true)
