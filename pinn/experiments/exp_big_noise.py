"""Big-ensemble noise-robustness — sweep observation σ, K seeds per σ.

For each σ ∈ {0, 0.01, 0.02, 0.05, 0.1, 0.2}, train K parallel networks +
per-seed (ζ, α). Reports parameter recovery vs noise with std error bars.
"""

import _common  # noqa
import os, numpy as np, torch, torch.nn as nn
from lib.data import ground_truth, sparse_observations, PRESETS
from lib.grouped import GroupedMLP, make_collocation
from lib.eval import l2_relative, save_results

EXP = "exp_big_noise"
zeta_true, w0, alpha_true = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]

K      = int(os.environ.get("PINN_K", 16))
WIDTH  = int(os.environ.get("PINN_WIDTH", 1024))
DEPTH  = int(os.environ.get("PINN_DEPTH", 6))
N_COLL = int(os.environ.get("PINN_N_COLL", 10000))
ITERS  = int(os.environ.get("PINN_ITERS", 10000))
LR     = float(os.environ.get("PINN_LR", 2e-3))
DEVICE = _common.DEVICE
SIGMAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]


class GP(nn.Module):
    def __init__(self, K, init):
        super().__init__()
        for n, v in init.items():
            self.register_parameter(n, nn.Parameter(torch.full((K, 1, 1), float(v))))


t_ref, x_ref, _ = ground_truth("duffing", (zeta_true, w0, alpha_true, 0.0, 0.0), ic, t_max, 4001)

table = []
all_zeta, all_alpha = [], []
for sigma in SIGMAS:
    torch.manual_seed(0)
    t_obs_np, x_obs_np = sparse_observations(t_ref, x_ref, n_obs=40, noise_sigma=sigma, rng=0)
    t_obs = torch.tensor(t_obs_np, dtype=torch.float32, device=DEVICE).view(-1, 1)
    x_obs = torch.tensor(x_obs_np, dtype=torch.float32, device=DEVICE).view(-1, 1)
    t_obs_e = t_obs.unsqueeze(0).expand(K, -1, -1).contiguous()
    x_obs_e = x_obs.unsqueeze(0).expand(K, -1, -1).contiguous()

    model = GroupedMLP(K=K, hidden=WIDTH, depth=DEPTH, t_scale=t_max).to(DEVICE)
    params = GP(K, dict(zeta=0.5, alpha=0.5)).to(DEVICE)
    t_col = make_collocation(t_max, N_COLL, K, DEVICE)
    opt = torch.optim.Adam(list(model.parameters()) + list(params.parameters()), lr=LR)

    for step in range(ITERS):
        opt.zero_grad()
        x = model(t_col)
        dx = torch.autograd.grad(x.sum(), t_col, create_graph=True)[0]
        ddx = torch.autograd.grad(dx.sum(), t_col, create_graph=True)[0]
        r = ddx + 2 * params.zeta * w0 * dx + (w0**2) * x + params.alpha * x**3
        t0 = torch.zeros(K, 1, 1, device=DEVICE, requires_grad=True)
        x0p = model(t0); v0p = torch.autograd.grad(x0p.sum(), t0, create_graph=True)[0]
        L_ic = (x0p - ic[0]).pow(2).mean() + (v0p - ic[1]).pow(2).mean()
        L_data = (model(t_obs_e) - x_obs_e).pow(2).mean()
        loss = r.pow(2).mean() + 100.0 * L_ic + 100.0 * L_data
        loss.backward(); opt.step()

    z = params.zeta.detach().cpu().numpy().reshape(-1)
    a = params.alpha.detach().cpu().numpy().reshape(-1)
    with torch.no_grad():
        t_eval = torch.tensor(t_ref, dtype=torch.float32, device=DEVICE).view(-1, 1)
        x_pred = model(t_eval).squeeze(-1).cpu().numpy()
    err = np.array([l2_relative(x_pred[k], x_ref) for k in range(K)])
    table.append((sigma, z.mean(), z.std(), a.mean(), a.std(), err.mean(), err.std()))
    all_zeta.append(z); all_alpha.append(a)
    print(f"[{EXP}] σ={sigma}  ζ={z.mean():.4f}±{z.std():.4f}  α={a.mean():.4f}±{a.std():.4f}"
          f"  L2={err.mean():.3e}±{err.std():.2e}")
    del model, params, t_col, opt; torch.cuda.empty_cache()

save_results(_common.RESULTS / EXP, sweep=np.array(table),
             zetas=np.array(all_zeta), alphas=np.array(all_alpha),
             zeta_true=zeta_true, alpha_true=alpha_true,
             K=K, width=WIDTH, depth=DEPTH)
