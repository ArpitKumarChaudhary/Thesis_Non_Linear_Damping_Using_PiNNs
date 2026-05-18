"""Architecture sensitivity heatmap: depth x width grid, 5 seeds per cell.

Trains the stable-configuration PINN at every (depth, width) combination on
the free Duffing benchmark and tabulates mean L2 over 5 seeds. Produces
a heatmap so the operating point is visually obvious.
"""

import _common  # noqa
import time, json, numpy as np, torch
from lib.data import ground_truth, PRESETS
from lib.models import MLP, HardICWrapper
from lib.train import train_pinn, duffing_residual, TrainConfig
from lib.eval import predict, l2_relative, save_results

EXP = "tune_arch"
DEVICE = _common.DEVICE
SEEDS = [0, 1, 2, 3, 4]
DEPTHS = [2, 3, 4, 5, 6]
WIDTHS = [16, 32, 64, 128, 256]

zeta, w0, alpha = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]
t_ref, x_ref, _ = ground_truth("duffing", (zeta, w0, alpha, 0.0, 0.0), ic, t_max, 4001)
t_colloc = torch.linspace(0, t_max, 2000).reshape(-1, 1)

print(f"depths={DEPTHS} widths={WIDTHS} seeds={SEEDS}")
print(f"total trials = {len(DEPTHS)*len(WIDTHS)*len(SEEDS)} = {len(DEPTHS)*len(WIDTHS)*len(SEEDS)}\n")


def one(depth, width, seed):
    torch.manual_seed(seed)
    base = MLP(hidden=width, depth=depth, activation="tanh", t_scale=t_max)
    model = HardICWrapper(base, x0=ic[0], v0=ic[1], tau=1.0)
    residual = lambda t, m=model: duffing_residual(m, t, zeta, w0, alpha)
    cfg = TrainConfig(iters=8000, lr=1e-3, lambda_ic=0.0, lbfgs_steps=500,
                      seed=seed, device=DEVICE, log_every=10**9)
    t0 = time.time()
    train_pinn(model, residual, t_colloc, ic=ic, data=None, cfg=cfg)
    return l2_relative(predict(model, t_ref), x_ref), time.time() - t0


mean_grid = np.zeros((len(DEPTHS), len(WIDTHS)))
std_grid  = np.zeros((len(DEPTHS), len(WIDTHS)))
all_errs  = {}
for i, d in enumerate(DEPTHS):
    for j, w in enumerate(WIDTHS):
        es, ts = [], []
        for s in SEEDS:
            e, dt = one(d, w, s)
            es.append(e); ts.append(dt)
        es = np.array(es)
        mean_grid[i, j] = es.mean()
        std_grid[i, j]  = es.std()
        all_errs[(d, w)] = es
        n_par = (1*w) + (w*w*(d-1)) + (w*1) + w*d
        print(f"  depth={d} width={w:3d} (~{n_par:6d} par)  L2 mean={es.mean():.3e} std={es.std():.2e}  ({np.mean(ts):.1f}s/seed)")
    print()

# best cell
bi, bj = np.unravel_index(np.argmin(mean_grid), mean_grid.shape)
print(f"\nbest cell: depth={DEPTHS[bi]}, width={WIDTHS[bj]} -> mean L2={mean_grid[bi,bj]:.3e}")

save_results(_common.RESULTS / EXP,
             depths=np.array(DEPTHS), widths=np.array(WIDTHS),
             seeds=np.array(SEEDS),
             mean_grid=mean_grid, std_grid=std_grid,
             best_depth=DEPTHS[bi], best_width=WIDTHS[bj],
             best_l2=float(mean_grid[bi, bj]),
             cfg=json.dumps(dict(iters=8000, lr=1e-3, lbfgs=500, tau=1.0)))

# heatmap
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.colors as mc

fig, ax = plt.subplots(figsize=(7, 4.2))
im = ax.imshow(np.log10(mean_grid + 1e-12), origin="lower", aspect="auto",
               cmap="viridis", norm=mc.Normalize(vmin=-4, vmax=0))
ax.set_xticks(range(len(WIDTHS))); ax.set_xticklabels(WIDTHS)
ax.set_yticks(range(len(DEPTHS))); ax.set_yticklabels(DEPTHS)
ax.set_xlabel("width"); ax.set_ylabel("depth")
ax.set_title("Architecture heatmap: $\\log_{10}$ mean L2-rel over 5 seeds")
# annotate cells
for i in range(len(DEPTHS)):
    for j in range(len(WIDTHS)):
        ax.text(j, i, f"{mean_grid[i,j]:.1e}", ha="center", va="center",
                color="white" if mean_grid[i,j] > 1e-2 else "black", fontsize=8)
# mark best
ax.plot(bj, bi, "r*", markersize=18, mec="white", mew=1.5)
cb = fig.colorbar(im, ax=ax); cb.set_label("$\\log_{10}$ L2-rel")
fig.tight_layout()
fig.savefig(_common.FIGURES / f"{EXP}_heatmap.png", dpi=120); plt.close(fig)
print(f"\nwrote figures/{EXP}_heatmap.png")
