"""Long-horizon Fourier-feature comparison.

Compares vanilla MLP against FourierMLP on free damped Duffing at horizons
T in {20, 50, 100}, 5 seeds each. Uses the tuned stable configuration as
the baseline; the Fourier branch adds the random-Fourier-feature input
embedding with sigma=2.0 and 64 features.
"""

import _common  # noqa
import time, json, numpy as np, torch
from lib.data import ground_truth
from lib.models import MLP, FourierMLP, HardICWrapper
from lib.train import train_pinn, duffing_residual, TrainConfig
from lib.eval import predict, l2_relative, save_results

EXP = "tune_fourier"
DEVICE = _common.DEVICE
zeta, w0, alpha = 0.05, 1.0, 1.0   # lighter damping so the long-horizon problem is meaningful
ic = (1.0, 0.0)
SEEDS = [0, 1, 2, 3, 4]
HORIZONS = [20.0, 50.0, 100.0]

# Use the stable config plus longer iters for longer T
def cfg_for_T(T):
    # collocation density ~100/period
    n_c = int(100 * T)
    # iter budget scales modestly with T
    iters = int(15000 * (T / 20))
    return n_c, iters


def train_one(seed, T, kind):
    n_c, iters = cfg_for_T(T)
    torch.manual_seed(seed)
    if kind == "mlp":
        base = MLP(hidden=128, depth=4, activation="tanh", t_scale=T)
    else:                                                                    # "fourier"
        base = FourierMLP(hidden=128, depth=4, n_features=64, sigma=2.0)
    model = HardICWrapper(base, x0=ic[0], v0=ic[1], tau=1.0)
    t_colloc = torch.linspace(0, T, n_c).reshape(-1, 1)
    residual = lambda t, m=model: duffing_residual(m, t, zeta, w0, alpha)
    cfg = TrainConfig(iters=iters, lr=1e-3, lambda_ic=0.0, lbfgs_steps=500,
                      seed=seed, device=DEVICE, log_every=10**9)
    t0 = time.time()
    train_pinn(model, residual, t_colloc, ic=ic, data=None, cfg=cfg)
    t_ref, x_ref, _ = ground_truth("duffing", (zeta, w0, alpha, 0.0, 0.0), ic, T, int(200 * T))
    err = l2_relative(predict(model, t_ref), x_ref)
    return err, time.time() - t0


table = []
for T in HORIZONS:
    for kind in ["mlp", "fourier"]:
        es, ts = [], []
        for s in SEEDS:
            e, dt = train_one(s, T, kind)
            es.append(e); ts.append(dt)
            print(f"  T={T:5.0f} {kind:7s} seed={s} L2={e:.3e}  ({dt:.1f}s)")
        es = np.array(es)
        table.append((T, kind, es.mean(), es.std(), np.mean(ts)))
        print(f"  T={T:5.0f} {kind:7s}  mean L2={es.mean():.3e} +/- {es.std():.2e}\n")

print("\n=== SUMMARY ===")
for T, kind, m, s, dt in table:
    print(f"  T={T:5.0f} {kind:7s}: L2={m:.3e} +/- {s:.2e}  ({dt:.1f}s/seed)")

save_results(_common.RESULTS / EXP,
             horizons=np.array(HORIZONS),
             sweep=np.array([(T, m, s, dt) for (T, _, m, s, dt) in table]),
             kinds=np.array([k for (_, k, _, _, _) in table]),
             cfg=json.dumps(dict(zeta=zeta, w0=w0, alpha=alpha, ic=ic, seeds=SEEDS,
                                 width=128, depth=4, lr=1e-3, lbfgs=500)))

# Plot: MLP vs Fourier at each T, log L2 axis
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 3.6))
xs = np.arange(len(HORIZONS))
mlp_m = [m for (T, k, m, _, _) in table if k == "mlp"]
mlp_s = [s for (T, k, _, s, _) in table if k == "mlp"]
fou_m = [m for (T, k, m, _, _) in table if k == "fourier"]
fou_s = [s for (T, k, _, s, _) in table if k == "fourier"]
w = 0.35
ax.bar(xs - w/2, mlp_m, w, yerr=mlp_s, capsize=4, label="vanilla MLP")
ax.bar(xs + w/2, fou_m, w, yerr=fou_s, capsize=4, label="Fourier-feature MLP")
ax.set_xticks(xs); ax.set_xticklabels([f"T={int(T)}" for T in HORIZONS])
ax.set_yscale("log"); ax.set_ylabel("L2 relative error (5 seeds)")
ax.axhline(1e-2, color="r", lw=0.5, ls=":", label="target 1e-2")
ax.set_title("Long-horizon spectral bias: MLP vs Fourier-feature MLP")
ax.legend()
fig.tight_layout()
fig.savefig(_common.FIGURES / f"{EXP}_summary.png", dpi=120); plt.close(fig)
print(f"\nwrote figures/{EXP}_summary.png")
