"""Big-ensemble Van der Pol — sweep μ ∈ {0.5, 1, 2, 4}, K seeds each."""

import _common  # noqa
import os, numpy as np, torch
from lib.data import ground_truth
from lib.grouped import GroupedMLP, GroupedHardIC, make_collocation, grouped_residual_vdp
from lib.eval import l2_relative, save_results

EXP = "exp_big_van_der_pol"
ic = (2.0, 0.0); w0 = 1.0; t_max = 30.0

K      = int(os.environ.get("PINN_K", 16))
WIDTH  = int(os.environ.get("PINN_WIDTH", 1024))
DEPTH  = int(os.environ.get("PINN_DEPTH", 6))
N_COLL = int(os.environ.get("PINN_N_COLL", 10000))
ITERS  = int(os.environ.get("PINN_ITERS", 15000))
LR     = float(os.environ.get("PINN_LR", 5e-4))
DEVICE = _common.DEVICE
MUS = [0.5, 1.0, 2.0, 4.0]

results = {}
for mu in MUS:
    t_ref, x_ref, _ = ground_truth("vdp", (mu, w0, 0.0, 0.0), ic, t_max, 4001)
    torch.manual_seed(0)
    base = GroupedMLP(K=K, hidden=WIDTH, depth=DEPTH, t_scale=t_max).to(DEVICE)
    model = GroupedHardIC(base, x0=ic[0], v0=ic[1], tau=1.0).to(DEVICE)
    t_col = make_collocation(t_max, N_COLL, K, DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for step in range(ITERS):
        opt.zero_grad()
        r = grouped_residual_vdp(model, t_col, mu, w0)
        loss = r.pow(2).mean()
        loss.backward(); opt.step()
        if step % 1500 == 0:
            print(f"[{EXP}/μ={mu}] step={step:6d} loss={float(loss):.3e}")
    with torch.no_grad():
        t_eval = torch.tensor(t_ref, dtype=torch.float32, device=DEVICE).view(-1, 1)
        x_pred = model(t_eval).squeeze(-1).cpu().numpy()
    errs = np.array([l2_relative(x_pred[k], x_ref) for k in range(K)])
    results[f"mu_{mu}"] = errs
    results[f"x_pred_mu_{mu}"] = x_pred
    results[f"t_mu_{mu}"] = t_ref
    results[f"x_ref_mu_{mu}"] = x_ref
    print(f"[{EXP}/μ={mu}] L2: mean={errs.mean():.3e} +/- {errs.std():.2e}")
    del base, model, t_col, opt; torch.cuda.empty_cache()

save_results(_common.RESULTS / EXP, mus=np.array(MUS), K=K, **results)
