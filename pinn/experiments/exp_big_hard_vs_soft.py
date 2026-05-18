"""Big-ensemble version of exp02 — hard vs soft IC, K seeds in parallel.

Trains 2*K nets simultaneously (K hard-IC + K soft-IC) on the same Duffing
forward task, reports mean ± std L2 for each branch.
"""

import _common  # noqa
import os, time, numpy as np, torch
from lib.data import ground_truth, PRESETS
from lib.grouped import GroupedMLP, GroupedHardIC, make_collocation, grouped_residual_duffing
from lib.eval import l2_relative, save_results

EXP = "exp_big_hard_vs_soft"
zeta, w0, alpha = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]

K      = int(os.environ.get("PINN_K", 16))
WIDTH  = int(os.environ.get("PINN_WIDTH", 1024))
DEPTH  = int(os.environ.get("PINN_DEPTH", 6))
N_COLL = int(os.environ.get("PINN_N_COLL", 10000))
ITERS  = int(os.environ.get("PINN_ITERS", 10000))
LR     = float(os.environ.get("PINN_LR", 5e-4))
LAMBDA_IC = float(os.environ.get("PINN_LAMBDA_IC", 100.0))
DEVICE = _common.DEVICE
print(f"[{EXP}] K={K} W={WIDTH} D={DEPTH} N={N_COLL} iters={ITERS} dev={DEVICE}")


def soft_ic_loss(model, device):
    t0 = torch.zeros(1, 1, device=device, requires_grad=True)
    t0e = t0.unsqueeze(0).expand(model.K, 1, 1).contiguous().requires_grad_(True)
    x = model(t0e)
    dx = torch.autograd.grad(x.sum(), t0e, create_graph=True)[0]
    return (x - ic[0]).pow(2).mean() + (dx - ic[1]).pow(2).mean()


def train(model, soft: bool):
    t_colloc = make_collocation(t_max, N_COLL, K, DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []
    for step in range(ITERS):
        opt.zero_grad()
        r = grouped_residual_duffing(model, t_colloc, zeta, w0, alpha)
        loss = r.pow(2).mean()
        if soft:
            loss = loss + LAMBDA_IC * soft_ic_loss(model, DEVICE)
        loss.backward(); opt.step()
        if step % 500 == 0:
            losses.append(float(loss.detach()))
            print(f"[{EXP}/{'soft' if soft else 'hard'}] step={step:6d} loss={float(loss):.3e}")
    return np.array(losses)


t_ref, x_ref, _ = ground_truth("duffing", (zeta, w0, alpha, 0.0, 0.0), ic, t_max, 4001)
t_eval = torch.tensor(t_ref, dtype=torch.float32, device=DEVICE).view(-1, 1)

torch.manual_seed(0)
soft_model = GroupedMLP(K=K, hidden=WIDTH, depth=DEPTH, t_scale=t_max, base_seed=0).to(DEVICE)
soft_loss = train(soft_model, soft=True)
with torch.no_grad():
    x_pred_soft = soft_model(t_eval).squeeze(-1).cpu().numpy()
soft_errs = np.array([l2_relative(x_pred_soft[k], x_ref) for k in range(K)])

torch.cuda.empty_cache()

torch.manual_seed(0)
hard_base = GroupedMLP(K=K, hidden=WIDTH, depth=DEPTH, t_scale=t_max, base_seed=1000)
hard_model = GroupedHardIC(hard_base, x0=ic[0], v0=ic[1], tau=1.0).to(DEVICE)
hard_loss = train(hard_model, soft=False)
with torch.no_grad():
    x_pred_hard = hard_model(t_eval).squeeze(-1).cpu().numpy()
hard_errs = np.array([l2_relative(x_pred_hard[k], x_ref) for k in range(K)])

print(f"[{EXP}] SOFT L2: mean={soft_errs.mean():.3e} +/- {soft_errs.std():.2e}")
print(f"[{EXP}] HARD L2: mean={hard_errs.mean():.3e} +/- {hard_errs.std():.2e}")

save_results(_common.RESULTS / EXP, t=t_ref, x_ref=x_ref,
             x_pred_soft=x_pred_soft, x_pred_hard=x_pred_hard,
             soft_errs=soft_errs, hard_errs=hard_errs,
             soft_loss=soft_loss, hard_loss=hard_loss,
             K=K, width=WIDTH, depth=DEPTH, N_coll=N_COLL, iters=ITERS)
