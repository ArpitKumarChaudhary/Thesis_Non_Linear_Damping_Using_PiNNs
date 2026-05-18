"""Generate clean thesis-ready figures from the tuned results."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES, FIG = ROOT / "results", ROOT / "figures"

# ---------- baseline: solution + abs error + loss curve ----------
b = np.load(RES / "exp_tuned_baseline" / "results.npz")
t, x_ref, x_pred = b["t"], b["x_ref"], b["x_pred"]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].plot(t, x_ref, "k-", lw=2, label="DOP853 reference")
ax[0].plot(t, x_pred, "r--", lw=1.5, label="PINN")
ax[0].set_xlabel("t"); ax[0].set_ylabel("x(t)"); ax[0].legend()
ax[0].set_title(f"Free Duffing  L2-rel = {float(b['metric_l2_rel']):.2e}")
ax[1].semilogy(t, np.abs(x_pred - x_ref) + 1e-16, "C3-", lw=1)
ax[1].set_xlabel("t"); ax[1].set_ylabel("|PINN − ref|"); ax[1].set_title("absolute error")
steps = b["hist_step"]
for k, c in [("total", "k"), ("phys", "C0"), ("ic", "C1")]:
    arr = b[f"hist_{k}"]
    if np.any(arr > 0):
        ax[2].semilogy(steps, np.maximum(arr, 1e-16), c, label=k)
ax[2].set_xlabel("Adam step"); ax[2].set_ylabel("loss"); ax[2].legend()
ax[2].set_title("training loss")
fig.tight_layout(); fig.savefig(FIG / "tuned_baseline_panel.png", dpi=140); plt.close(fig)
print("wrote tuned_baseline_panel.png")

# ---------- hard vs soft IC: solutions side-by-side + bar chart ----------
h = np.load(RES / "exp_tuned_hard_vs_soft" / "results.npz")
t, x_ref = h["t"], h["x_ref"]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
ax[0].plot(t, x_ref, "k-", lw=2, label="reference")
ax[0].plot(t, h["x_pred_soft"], "r--", lw=1.2, label=f"soft (L2={float(h['l2_soft']):.2e})")
ax[0].plot(t, h["x_pred_hard"], "C0--", lw=1.2, label=f"hard (L2={float(h['l2_hard']):.2e})")
ax[0].set_xlabel("t"); ax[0].set_ylabel("x(t)"); ax[0].legend(loc="upper right", fontsize=9)
ax[0].set_title("Solution overlay")
ax[1].semilogy(t, np.abs(h["x_pred_soft"] - x_ref) + 1e-16, "r-",  lw=1, label="soft")
ax[1].semilogy(t, np.abs(h["x_pred_hard"] - x_ref) + 1e-16, "C0-", lw=1, label="hard")
ax[1].set_xlabel("t"); ax[1].set_ylabel("|err|"); ax[1].legend()
ax[1].set_title("Pointwise abs error")
labels = ["soft IC\n(loss term)", "hard IC\n(reparametrization)"]
l2 = [float(h["l2_soft"]), float(h["l2_hard"])]
ic_err = [float(h["ic_err_soft"]), float(h["ic_err_hard"])]
xpos = np.arange(2)
bars1 = ax[2].bar(xpos - 0.2, l2, 0.4, label="L2 rel", color="C3")
ax[2].set_yscale("log"); ax[2].set_ylabel("L2 relative error", color="C3")
ax[2].tick_params(axis="y", labelcolor="C3")
ax2b = ax[2].twinx()
bars2 = ax2b.bar(xpos + 0.2, np.maximum(ic_err, 1e-16), 0.4, label="IC abs err", color="C0")
ax2b.set_yscale("log"); ax2b.set_ylabel("IC absolute error", color="C0")
ax2b.tick_params(axis="y", labelcolor="C0")
ax[2].set_xticks(xpos); ax[2].set_xticklabels(labels)
ax[2].set_title("L2 + IC error")
fig.tight_layout(); fig.savefig(FIG / "tuned_hard_vs_soft_panel.png", dpi=140); plt.close(fig)
print("wrote tuned_hard_vs_soft_panel.png")

# ---------- one-line summary table ----------
print("\n=== TUNED RESULTS SUMMARY ===")
print(f"  baseline (hard-IC) L2_rel        = {float(b['metric_l2_rel']):.3e}   wall {float(b['wallclock']):.1f}s")
print(f"  hard_vs_soft  hard L2 / IC_err   = {float(h['l2_hard']):.3e}  /  {float(h['ic_err_hard']):.0e}   wall {float(h['wallclock_hard']):.1f}s")
print(f"  hard_vs_soft  soft L2 / IC_err   = {float(h['l2_soft']):.3e}  /  {float(h['ic_err_soft']):.2e}   wall {float(h['wallclock_soft']):.1f}s")
