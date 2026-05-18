"""Big-ensemble loss-weighting ablation.

Each weighting setting gets K seeds trained in parallel. Configs:
  λ_ic ∈ {1, 10, 100, 1000}  (hard-coded since adaptive weighting requires
  per-task gradient stats which are awkward with grouped weights).
Uses hard-IC to keep training stable; the focus here is the *residual*
gradient scale relative to a (zero) IC term — we drop IC loss and instead
add a tiny anchor pulling x'(0)=v0 to study weighting sensitivity. To keep
the comparison faithful to the classic ablation, we run soft-IC with each λ.
"""

import _common  # noqa
import os, numpy as np, torch
from lib.data import ground_truth, PRESETS
from lib.grouped import GroupedMLP, make_collocation, grouped_residual_duffing
from lib.eval import l2_relative, save_results

EXP = "exp_big_loss_weighting"
zeta, w0, alpha = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]

K      = int(os.environ.get("PINN_K", 16))
WIDTH  = int(os.environ.get("PINN_WIDTH", 1024))
DEPTH  = int(os.environ.get("PINN_DEPTH", 6))
N_COLL = int(os.environ.get("PINN_N_COLL", 10000))
ITERS  = int(os.environ.get("PINN_ITERS", 10000))
LR     = float(os.environ.get("PINN_LR", 5e-4))
DEVICE = _common.DEVICE
LAMBDAS = [1.0, 10.0, 100.0, 1000.0]


def soft_ic_loss(model, dev):
    t0 = torch.zeros(1, 1, device=dev).unsqueeze(0).expand(model.K, 1, 1).contiguous()
    t0 = t0.requires_grad_(True)
    x = model(t0)
    dx = torch.autograd.grad(x.sum(), t0, create_graph=True)[0]
    return (x - ic[0]).pow(2).mean() + (dx - ic[1]).pow(2).mean()


t_ref, x_ref, _ = ground_truth("duffing", (zeta, w0, alpha, 0.0, 0.0), ic, t_max, 4001)
t_eval = torch.tensor(t_ref, dtype=torch.float32, device=DEVICE).view(-1, 1)

results = {}
for lam in LAMBDAS:
    torch.manual_seed(0)
    model = GroupedMLP(K=K, hidden=WIDTH, depth=DEPTH, t_scale=t_max).to(DEVICE)
    t_col = make_collocation(t_max, N_COLL, K, DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for step in range(ITERS):
        opt.zero_grad()
        r = grouped_residual_duffing(model, t_col, zeta, w0, alpha)
        loss = r.pow(2).mean() + lam * soft_ic_loss(model, DEVICE)
        loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"[{EXP}/lam={lam}] step={step:6d} loss={float(loss):.3e}")
    with torch.no_grad():
        x_pred = model(t_eval).squeeze(-1).cpu().numpy()
    errs = np.array([l2_relative(x_pred[k], x_ref) for k in range(K)])
    results[f"lam_{int(lam)}"] = errs
    print(f"[{EXP}/lam={lam}] L2: mean={errs.mean():.3e} +/- {errs.std():.2e}")
    del model, t_col, opt; torch.cuda.empty_cache()

save_results(_common.RESULTS / EXP, t=t_ref, x_ref=x_ref,
             lambdas=np.array(LAMBDAS), K=K, width=WIDTH, depth=DEPTH,
             N_coll=N_COLL, iters=ITERS, **results)
