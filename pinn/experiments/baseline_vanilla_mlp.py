"""Vanilla MLP baseline: data-only loss, NO physics residual.

Trained on N_obs sparse observations of the true Duffing trajectory.
Sweep N_obs ∈ {20, 50, 100, 200} to show how much data a no-physics model
needs to compete with PINN. Reports L2 over 5 seeds.
"""

import _common  # noqa
import time, json, numpy as np, torch
from lib.data import ground_truth, sparse_observations, PRESETS
from lib.models import MLP
from lib.eval import predict, l2_relative, save_results

EXP = "baseline_vanilla_mlp"
DEVICE = _common.DEVICE
SEEDS = [0, 1, 2, 3, 4]
zeta, w0, alpha = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]
t_ref, x_ref, _ = ground_truth("duffing", (zeta, w0, alpha, 0.0, 0.0), ic, t_max, 4001)

ITERS = 8000
LR = 1e-3
N_OBS_LIST = [20, 50, 100, 200]

results = {}
for N_obs in N_OBS_LIST:
    seed_errs = []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        idx = np.sort(rng.choice(len(t_ref), size=N_obs, replace=False))
        t_d, x_d = t_ref[idx], x_ref[idx]
        t_d_t = torch.tensor(t_d, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
        x_d_t = torch.tensor(x_d, dtype=torch.float32, device=DEVICE).reshape(-1, 1)

        torch.manual_seed(s)
        model = MLP(hidden=64, depth=4, activation="tanh", t_scale=t_max).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        t0 = time.time()
        for step in range(ITERS):
            opt.zero_grad()
            loss = (model(t_d_t) - x_d_t).pow(2).mean()
            loss.backward(); opt.step()
        # L-BFGS polish
        lb = torch.optim.LBFGS(model.parameters(), max_iter=500,
                               line_search_fn="strong_wolfe",
                               tolerance_grad=1e-9, tolerance_change=1e-12)
        def closure():
            lb.zero_grad()
            l = (model(t_d_t) - x_d_t).pow(2).mean()
            l.backward(); return l
        lb.step(closure)

        err = l2_relative(predict(model, t_ref), x_ref)
        seed_errs.append(err)
        print(f"  N_obs={N_obs:3d} seed={s} L2={err:.3e} ({time.time()-t0:.1f}s)")
    seed_errs = np.array(seed_errs)
    results[f"N{N_obs}"] = seed_errs
    print(f"  N_obs={N_obs}: mean={seed_errs.mean():.3e}  std={seed_errs.std():.3e}")

print("\n=== vanilla MLP (no physics) summary ===")
for N_obs in N_OBS_LIST:
    e = results[f"N{N_obs}"]
    print(f"  N_obs={N_obs:3d}: {e.mean():.3e} ± {e.std():.3e}")

save_results(_common.RESULTS / EXP,
             N_obs_list=np.array(N_OBS_LIST),
             **{f"errs_N{n}": results[f"N{n}"] for n in N_OBS_LIST})
