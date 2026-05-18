"""Cross-phase comparison figure for the thesis."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/home2/arpit_thesis/thesis_v2/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# Figure 1: L2-relative error progression across phases
# ============================================================
phases = ["Phase 0\nIEEE paper\n(reference)",
          "Phase 1\nLinear damped\n(NN/PINN/Grey)",
          "Phase 2\nQuadratic damping\n(PINN soft IC)",
          "Phase 3\nDuffing tuned\n(hard IC + LBFGS)"]
best_l2 = [1e-3, 2.2e-3, 8.66e-2, 5.67e-4]      # representative best L2 per phase
err_low = [1e-3, 2.2e-3, 8.66e-2, 1.17e-4]      # best individual seed
err_high = [1e-3, 6.6e-3, 8.66e-2, 2.19e-3]     # worst (stable) seed
colors = ["#888888", "#1f77b4", "#ff7f0e", "#2ca02c"]

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(phases))
bars = ax.bar(x, best_l2, color=colors, alpha=0.85, edgecolor="black")
# Error bars (asymmetric)
yerr_low = [b - l for b, l in zip(best_l2, err_low)]
yerr_high = [h - b for h, b in zip(err_high, best_l2)]
ax.errorbar(x, best_l2, yerr=[yerr_low, yerr_high], fmt="none",
            ecolor="black", capsize=4, lw=1.5)
ax.axhline(1e-2, color="red", lw=1, linestyle="--", label="Target L2 = 1e-2")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(phases, fontsize=9)
ax.set_ylabel("L2-relative error", fontsize=11)
ax.set_title("Cross-Phase Comparison: L2-Relative Error Evolution",
             fontsize=12, weight="bold")
for i, (b, e_lo, e_hi) in enumerate(zip(best_l2, err_low, err_high)):
    ax.annotate(f"{b:.2e}", xy=(i, b), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=8, weight="bold")
ax.legend(loc="upper right")
ax.set_ylim(1e-5, 1e0)
ax.grid(True, axis="y", alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(OUT / "phase_comparison_l2.png", dpi=130)
plt.close(fig)

# ============================================================
# Figure 2: Method-evolution multi-metric table-as-figure
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.axis("off")
header = ["Aspect", "Phase 1 (Linear)", "Phase 2 (Quadratic)", "Phase 3 (Duffing)"]
rows = [
    ["ODE",
     r"$m\ddot x + c\dot x + kx = F$",
     r"$\ddot x + c|\dot x|\dot x + \omega^2 x = 0$",
     r"$\ddot x + 2\zeta\omega_0\dot x + \omega_0^2 x + \alpha x^3 = 0$"],
    ["Non-linearity", "None (linear)",
     "Quadratic damping (drag)",
     "Cubic stiffness"],
    ["Horizon T (s)", "4 (train 0-2)", "10", "20"],
    ["Hidden size", "(typical 50x4)", r"$160\times 4$", r"$64\times 4$"],
    ["Activation", "tanh", "tanh", "tanh"],
    ["Optimizer", "Adam", "Adam + grad clip", "Adam + L-BFGS polish"],
    ["IC enforcement", "Soft penalty", "Soft (adaptive weight)", "Hard (bounded gate)"],
    ["Reference solver", "Analytical", "RK4 (500 pts)", "DOP853 (4001 pts)"],
    ["Seeds reported", "1", "1", "5 (mean +/- std)"],
    ["MSE", "n.r.", r"$7.86\times 10^{-4}$", r"$\sim 10^{-7}$ (derived)"],
    ["Best L2-rel", r"$2.2\times 10^{-3}$ (in-domain)",
                   r"$8.66\times 10^{-2}$",
                   r"$5.67\times 10^{-4}$ (mean 5 seeds)"],
    ["Max abs error", "n.r.", r"$5.97\times 10^{-2}$", r"$< 10^{-3}$"],
    ["GPU saturation", "n.r.", "<2 GB", "38 GB (grouped K=16)"],
    ["Wall-clock", "minutes", "10 min", "30 s / 1 min stable"],
]
n = len(rows) + 1
table = ax.table(cellText=[header] + rows, cellLoc="left",
                 colWidths=[0.18, 0.27, 0.27, 0.28], loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.5)
# Header style
for j in range(4):
    table[(0, j)].set_facecolor("#3f3f3f")
    table[(0, j)].get_text().set_color("white")
    table[(0, j)].get_text().set_weight("bold")
# Alternate row shading
for i in range(1, n):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor("#f5f5f5")
ax.set_title("Method-Evolution Across Three Implementation Phases",
             fontsize=12, weight="bold", pad=15)
fig.tight_layout()
fig.savefig(OUT / "method_evolution_table.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure 3: Phase 1 NN vs PINN vs Grey-Box bar chart
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4))
intervals = ["0-2 s\n(in domain)", "2-4 s\n(extrapolation)", "0-4 s\n(combined)"]
nn = [0.0147, 0.6254, 0.2875]
pinn = [0.0022, 1.2908, 0.5928]
grey = [0.0066, 0.5214, 0.2395]
x = np.arange(len(intervals))
w = 0.27
ax.bar(x - w, nn, w, label="Pure NN (black-box)", color="#d62728")
ax.bar(x, pinn, w, label="PINN (white-box)", color="#1f77b4")
ax.bar(x + w, grey, w, label="Grey-Box PINN (system ID)", color="#2ca02c")
ax.set_xticks(x)
ax.set_xticklabels(intervals)
ax.set_yscale("log")
ax.set_ylabel("L2-relative error")
ax.set_title("Phase 1: Three-Model Comparison on Linear Damped Oscillator")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, axis="y", alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(OUT / "phase1_nn_pinn_grey.png", dpi=130)
plt.close(fig)

# ============================================================
# Figure 4: Quadratic-damping trajectory (Phase 2 reproduction)
# ============================================================
from scipy.integrate import solve_ivp
def rhs(t, y, c=0.5, w=2.0):
    x, v = y
    return [v, -c * np.sqrt(v**2 + 1e-6) * v - w**2 * x]
sol = solve_ivp(rhs, (0, 10), [1.0, 0.0], t_eval=np.linspace(0, 10, 500),
                method="DOP853", rtol=1e-10, atol=1e-12)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].plot(sol.t, sol.y[0], "k-", lw=2, label="$x(t)$")
axes[0].plot(sol.t, sol.y[1], "r--", lw=1.3, label="$\\dot x(t)$")
axes[0].set_xlabel("time t (s)"); axes[0].set_ylabel("state")
axes[0].set_title("Phase 2 system: $\\ddot x + 0.5|\\dot x|\\dot x + 4x = 0$")
axes[0].legend(); axes[0].grid(True, alpha=0.3)
# Phase plot
axes[1].plot(sol.y[0], sol.y[1], "b-", lw=1)
axes[1].set_xlabel("$x$"); axes[1].set_ylabel("$\\dot x$")
axes[1].set_title("Phase plane (quadratic-damping spiral)")
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "phase2_trajectory.png", dpi=130)
plt.close(fig)

print("Wrote:")
for f in sorted(OUT.glob("*.png")):
    print(" ", f.name)
