"""Vanilla MLP baseline: data-only fit on sparse observations, no physics loss.

This is the "is PINN actually better than just fitting an MLP to a few data
points?" comparison every PINN paper needs. Sweeps N_obs ∈ {10, 20, 50, 100}
of clean ground-truth samples; reports L2-rel mean ± std over 5 seeds.
"""

import _common  # noqa
import time, numpy as np, torch
from lib.data import ground_truth, sparse_observations, PRESETS
from lib.models import MLP
from lib.eval import l2_relative, save_results

EXP = "tune_vanilla_mlp"
DEVICE = _common.DEVICE
SEEDS = [0, 1, 2, 3, 4]
N_OBS = [10, 20, 50, 100]

zeta, w0, alpha = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]
t_ref, x_ref, _ = ground_truth("duffing", (zeta, w0, alpha, 0.0, 0.0), ic, t_max, 4001)
t_eval = torch.tensor(t_ref, dtype=torch.float32, device=DEVICE).view(-1, 1)


def fit(seed, n_obs):
    torch.manual_seed(seed)
    t_obs_np, x_obs_np = sparse_observations(t_ref, x_ref, n_obs=n_obs, noise_sigma=0.0, rng=seed)
    t_obs = torch.tensor(t_obs_np, dtype=torch.float32, device=DEVICE).view(-1, 1)
    x_obs = torch.tensor(x_obs_np, dtype=torch.float32, device=DEVICE).view(-1, 1)
    model = MLP(hidden=64, depth=4, activation="tanh", t_scale=t_max).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(8000):
        opt.zero_grad()
        loss = (model(t_obs) - x_obs).pow(2).mean()
        loss.backward(); opt.step()
    # L-BFGS polish on data only
    lb = torch.optim.LBFGS(model.parameters(), max_iter=300,
                           tolerance_grad=1e-9, line_search_fn="strong_wolfe")
    def closure():
        lb.zero_grad()
        l = (model(t_obs) - x_obs).pow(2).mean()
        l.backward(); return l
    lb.step(closure)
    with torch.no_grad():
        x_pred = model(t_eval).cpu().numpy().reshape(-1)
    return l2_relative(x_pred, x_ref)


table = []
for n in N_OBS:
    es = []
    for s in SEEDS:
        t0 = time.time()
        e = fit(s, n)
        es.append(e)
        print(f"  N_obs={n:3d} seed={s} L2={e:.3e} ({time.time()-t0:.1f}s)")
    es = np.array(es)
    table.append((n, es.mean(), es.std()))
    print(f"  N_obs={n:3d}  mean={es.mean():.3e}  std={es.std():.3e}")

print("\nSUMMARY")
for n, m, s in table:
    print(f"  N_obs={n:3d}: L2 = {m:.3e} ± {s:.3e}")

save_results(_common.RESULTS / EXP, sweep=np.array(table), n_obs=np.array(N_OBS), seeds=np.array(SEEDS))
