"""VRAM-sizing smoke test for the ensemble PINN.

Runs 30 iters at progressively larger (K, width, depth, N_coll) until peak
VRAM crosses the target, then prints the recommended environment variables
for the full experiment. Releases the model between trials.
"""

import _common  # noqa
import os, time, gc, torch
from lib.data import PRESETS
from lib.grouped import GroupedMLP, GroupedHardIC, make_collocation, grouped_residual_duffing

DEVICE = _common.DEVICE
assert DEVICE == "cuda", "VRAM smoke only meaningful on CUDA"
TARGET_GB = float(os.environ.get("PINN_TARGET_GB", 40.0))

zeta, w0, alpha = PRESETS["free_duffing"]["params"]
ic = PRESETS["free_duffing"]["ic"]; t_max = PRESETS["free_duffing"]["t_max"]


def trial(K, width, depth, N_coll, iters=30):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); gc.collect()
    torch.manual_seed(0)
    base = GroupedMLP(K=K, hidden=width, depth=depth, t_scale=t_max)
    model = GroupedHardIC(base, x0=ic[0], v0=ic[1], tau=1.0).to(DEVICE)
    n_par = sum(p.numel() for p in base.parameters()) // K
    t_colloc = make_collocation(t_max, N_coll, K, DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for step in range(iters):
        opt.zero_grad()
        r = grouped_residual_duffing(model, t_colloc, zeta, w0, alpha)
        loss = r.pow(2).mean()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"K={K:3d} W={width:5d} D={depth} N={N_coll:7d}  params/net={n_par:>10,}  "
          f"peak={peak:5.1f} GB  {iters/dt:.1f} it/s  loss={float(loss):.2e}")
    del model, base, t_colloc, opt; gc.collect(); torch.cuda.empty_cache()
    return peak


CONFIGS = [
    # Start moderate, then push (K, width, depth, N_coll)
    (16,  512,  4,  20000),
    (16,  512,  6,  20000),
    (16, 1024,  4,  20000),
    (32, 1024,  4,  20000),
    (16, 1024,  6,  20000),
    (32, 1024,  4,  40000),
    (32, 1024,  6,  20000),
    (32, 1024,  6,  40000),
    (64, 1024,  4,  20000),
    (64, 1024,  6,  20000),
    (32, 2048,  4,  20000),
    (32, 2048,  6,  20000),
]

best = None
for cfg in CONFIGS:
    try:
        peak = trial(*cfg)
        if peak >= TARGET_GB:
            best = (cfg, peak)
            print(f"\nTarget {TARGET_GB} GB hit at {cfg} (peak {peak:.1f} GB).")
            break
        if peak > 0.92 * 48.0:
            print(f"Near OOM at {cfg} (peak {peak:.1f} GB) — stopping sweep.")
            best = (cfg, peak); break
    except RuntimeError as e:
        print(f"OOM at {cfg}: {e}")
        break

if best is None:
    print(f"\nNo config in the sweep hit {TARGET_GB} GB. Largest measured peak above.")
else:
    cfg, peak = best
    K, W, D, N = cfg
    print(f"\nRECOMMEND:  PINN_K={K} PINN_WIDTH={W} PINN_DEPTH={D} PINN_N_COLL={N}")
    print(f"            peak VRAM ≈ {peak:.1f} GB on {torch.cuda.get_device_name(0)}")
