"""Ensemble PINN — sized to saturate the A6000 (≥30 GB VRAM, 100% GPU util).

Trains K=16 independent damped-Duffing PINNs simultaneously via batched matmul.
Hard-IC reparametrization, large network (depth=6, width=512), 50k collocation
points. Reports mean ± std L2 over the K seeds, dumps per-seed predictions.

Memory back-of-envelope:
  Activations per layer ≈ K · N · hidden · 4 B = 16 · 50000 · 512 · 4 = 1.6 GB
  depth=6, plus 2nd-order autograd (~3× factor) ≈ 28–35 GB total.
"""

import _common  # noqa
import os, time, numpy as np, torch
from lib.data import ground_truth, PRESETS
from lib.grouped import GroupedMLP, GroupedHardIC, make_collocation, grouped_residual_duffing
from lib.eval import l2_relative, save_results, plot_solution

EXP = "exp_big_ensemble"
zeta, w0, alpha = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]

K       = int(os.environ.get("PINN_K", 64))
WIDTH   = int(os.environ.get("PINN_WIDTH", 512))
DEPTH   = int(os.environ.get("PINN_DEPTH", 4))
N_COLL  = int(os.environ.get("PINN_N_COLL", 5000))
ITERS   = int(os.environ.get("PINN_ITERS", 15000))
LR      = float(os.environ.get("PINN_LR", 1e-3))
GRAD_CLIP = float(os.environ.get("PINN_CLIP", 1.0))
DEVICE  = _common.DEVICE

print(f"[{EXP}] K={K} width={WIDTH} depth={DEPTH} N_coll={N_COLL} iters={ITERS} device={DEVICE}")

torch.manual_seed(0)
base = GroupedMLP(K=K, hidden=WIDTH, depth=DEPTH, activation="tanh",
                  t_scale=t_max, base_seed=0)
model = GroupedHardIC(base, x0=ic[0], v0=ic[1], tau=1.0).to(DEVICE)
n_params_per_net = sum(p.numel() for p in base.parameters()) // K
print(f"[{EXP}] params/net={n_params_per_net:,}  total={n_params_per_net*K:,}")

t_colloc = make_collocation(t_max, N_COLL, K, DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)

t0 = time.time()
log = {"step": [], "loss": [], "vram_gb": [], "util": []}
for step in range(ITERS):
    opt.zero_grad()
    r = grouped_residual_duffing(model, t_colloc, zeta, w0, alpha)
    loss = r.pow(2).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    opt.step()
    if step % 500 == 0 or step == ITERS - 1:
        vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        log["step"].append(step); log["loss"].append(float(loss.detach()))
        log["vram_gb"].append(vram)
        print(f"[{EXP}] step={step:6d} loss={float(loss):.4e} VRAM={vram:.2f}GB peak={peak:.2f}GB")

# evaluate
t_ref, x_ref, _ = ground_truth("duffing", (zeta, w0, alpha, 0.0, 0.0), ic, t_max, 4001)
with torch.no_grad():
    t_eval = torch.tensor(t_ref, dtype=torch.float32, device=DEVICE).view(-1, 1)
    x_pred = model(t_eval).squeeze(-1).cpu().numpy()  # (K, N)

errs = np.array([l2_relative(x_pred[k], x_ref) for k in range(K)])
print(f"[{EXP}] L2_rel mean={errs.mean():.3e}  std={errs.std():.3e}  "
      f"min={errs.min():.3e}  max={errs.max():.3e}  wallclock={time.time()-t0:.1f}s")

plot_solution(t_ref, x_ref, x_pred[0],
              f"ensemble seed 0 — L2={errs[0]:.2e}",
              _common.FIGURES / f"{EXP}_seed0.png")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(t_ref, x_ref, "k-", lw=2, label="reference")
for k in range(K):
    ax.plot(t_ref, x_pred[k], alpha=0.25, lw=0.8)
ax.set_xlabel("t"); ax.set_ylabel("x(t)")
ax.set_title(f"K={K} ensemble predictions  (L2 mean={errs.mean():.2e} ± {errs.std():.0e})")
ax.legend()
fig.tight_layout()
fig.savefig(_common.FIGURES / f"{EXP}_all_seeds.png", dpi=120); plt.close(fig)

save_results(_common.RESULTS / EXP, t=t_ref, x_ref=x_ref, x_pred=x_pred,
             errs=errs, K=K, width=WIDTH, depth=DEPTH, N_coll=N_COLL, iters=ITERS,
             loss_hist=np.array(log["loss"]), vram_hist=np.array(log["vram_gb"]))
print(f"[{EXP}] DONE")
