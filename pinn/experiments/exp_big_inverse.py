"""Big-ensemble inverse problem — learn (ζ, α) from sparse noisy data.

K seeds in parallel. Each seed has its own copy of (ζ, α) trained alongside
the network weights. Sparse obs (N=30) with σ noise common across seeds for
fairness. Reports mean ± std on identified ζ, α and L2-reconstruction.
"""

import _common  # noqa
import os, numpy as np, torch, torch.nn as nn
from lib.data import ground_truth, sparse_observations, PRESETS
from lib.grouped import GroupedMLP, make_collocation
from lib.eval import l2_relative, save_results

EXP = "exp_big_inverse"
zeta_true, w0, alpha_true = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]

K      = int(os.environ.get("PINN_K", 16))
WIDTH  = int(os.environ.get("PINN_WIDTH", 1024))
DEPTH  = int(os.environ.get("PINN_DEPTH", 6))
N_COLL = int(os.environ.get("PINN_N_COLL", 10000))
N_OBS  = int(os.environ.get("PINN_N_OBS", 30))
SIGMA  = float(os.environ.get("PINN_SIGMA", 0.02))
ITERS  = int(os.environ.get("PINN_ITERS", 15000))
LR     = float(os.environ.get("PINN_LR", 2e-3))
DEVICE = _common.DEVICE
print(f"[{EXP}] K={K} W={WIDTH} D={DEPTH} N_obs={N_OBS} sigma={SIGMA} iters={ITERS}")


class GroupedParams(nn.Module):
    """Per-seed learnable physical parameters."""
    def __init__(self, K, init):
        super().__init__()
        for name, v in init.items():
            self.register_parameter(name, nn.Parameter(torch.full((K, 1, 1), float(v))))


t_ref, x_ref, _ = ground_truth("duffing", (zeta_true, w0, alpha_true, 0.0, 0.0), ic, t_max, 4001)
t_obs_np, x_obs_np = sparse_observations(t_ref, x_ref, n_obs=N_OBS, noise_sigma=SIGMA, rng=0)
t_obs = torch.tensor(t_obs_np, dtype=torch.float32, device=DEVICE).view(N_OBS, 1)
x_obs = torch.tensor(x_obs_np, dtype=torch.float32, device=DEVICE).view(N_OBS, 1)
t_obs_e = t_obs.unsqueeze(0).expand(K, N_OBS, 1).contiguous()
x_obs_e = x_obs.unsqueeze(0).expand(K, N_OBS, 1).contiguous()

torch.manual_seed(0)
model = GroupedMLP(K=K, hidden=WIDTH, depth=DEPTH, t_scale=t_max).to(DEVICE)
params = GroupedParams(K, dict(zeta=0.5, alpha=0.5)).to(DEVICE)
t_col = make_collocation(t_max, N_COLL, K, DEVICE)
opt = torch.optim.Adam(list(model.parameters()) + list(params.parameters()), lr=LR)

zeta_hist, alpha_hist = [], []
for step in range(ITERS):
    opt.zero_grad()
    x = model(t_col)
    dx = torch.autograd.grad(x.sum(), t_col, create_graph=True)[0]
    ddx = torch.autograd.grad(dx.sum(), t_col, create_graph=True)[0]
    r = ddx + 2 * params.zeta * w0 * dx + (w0 ** 2) * x + params.alpha * x ** 3
    # soft IC
    t0 = torch.zeros(K, 1, 1, device=DEVICE, requires_grad=True)
    x0p = model(t0)
    v0p = torch.autograd.grad(x0p.sum(), t0, create_graph=True)[0]
    L_ic = (x0p - ic[0]).pow(2).mean() + (v0p - ic[1]).pow(2).mean()
    # data
    L_data = (model(t_obs_e) - x_obs_e).pow(2).mean()
    loss = r.pow(2).mean() + 100.0 * L_ic + 100.0 * L_data
    loss.backward(); opt.step()
    if step % 500 == 0:
        zh = float(params.zeta.detach().mean()); ah = float(params.alpha.detach().mean())
        zeta_hist.append(zh); alpha_hist.append(ah)
        print(f"[{EXP}] step={step:6d} loss={float(loss):.3e}  zeta={zh:.4f} alpha={ah:.4f}")

with torch.no_grad():
    t_eval = torch.tensor(t_ref, dtype=torch.float32, device=DEVICE).view(-1, 1)
    x_pred = model(t_eval).squeeze(-1).cpu().numpy()
errs = np.array([l2_relative(x_pred[k], x_ref) for k in range(K)])
zeta_final = params.zeta.detach().cpu().numpy().reshape(-1)
alpha_final = params.alpha.detach().cpu().numpy().reshape(-1)
print(f"[{EXP}] zeta_true={zeta_true}  learned: {zeta_final.mean():.4f} +/- {zeta_final.std():.4f}")
print(f"[{EXP}] alpha_true={alpha_true} learned: {alpha_final.mean():.4f} +/- {alpha_final.std():.4f}")
print(f"[{EXP}] L2_rel: mean={errs.mean():.3e} +/- {errs.std():.2e}")

save_results(_common.RESULTS / EXP, t=t_ref, x_ref=x_ref, x_pred=x_pred,
             t_obs=t_obs_np, x_obs=x_obs_np,
             zeta_true=zeta_true, alpha_true=alpha_true,
             zeta_learned=zeta_final, alpha_learned=alpha_final,
             zeta_hist=np.array(zeta_hist), alpha_hist=np.array(alpha_hist),
             errs=errs, K=K, sigma=SIGMA)
