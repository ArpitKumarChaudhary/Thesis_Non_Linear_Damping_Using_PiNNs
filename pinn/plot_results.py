"""Aggregate result .npz files into headline comparison plots / tables."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES, FIG = ROOT / "results", ROOT / "figures"


def load(name):
    p = RES / name / "results.npz"
    return np.load(p, allow_pickle=True) if p.exists() else None


def fig_hard_vs_soft():
    d = load("exp02_hard_vs_soft_ic")
    if d is None: return
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["soft IC", "hard IC"], [float(d["l2_soft"]), float(d["l2_hard"])])
    ax.set_yscale("log"); ax.set_ylabel("L2 relative error")
    ax.set_title("Hard vs soft IC")
    fig.tight_layout(); fig.savefig(FIG / "summary_hard_vs_soft.png", dpi=120); plt.close(fig)


def fig_loss_weighting():
    d = load("exp03_loss_weighting")
    if d is None: return
    keys = [k for k in d.files if k.startswith("l2_")]
    vals = [float(d[k]) for k in keys]
    labels = [k.replace("l2_", "") for k in keys]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(labels, vals); ax.set_yscale("log")
    ax.set_ylabel("L2 relative error"); plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title("Loss-weighting strategies")
    fig.tight_layout(); fig.savefig(FIG / "summary_loss_weighting.png", dpi=120); plt.close(fig)


def fig_architecture():
    d = load("exp04_architecture_sweep")
    if d is None: return
    sweep = d["sweep"]; acts = d["activation"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for a in np.unique(acts):
        m = acts == a
        ax.scatter(sweep[m, 2], sweep[m, 3], label=a)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("# parameters"); ax.set_ylabel("L2 relative error")
    ax.set_title("Architecture sweep"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "summary_architecture.png", dpi=120); plt.close(fig)


def fig_fourier():
    d = load("exp05_fourier_features")
    if d is None: return
    sweep, tags = d["sweep"], d["tag"]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for tag in np.unique(tags):
        m = tags == tag
        ax.semilogy(sweep[m, 0], sweep[m, 1], "-o", label=tag)
    ax.set_xlabel("t_max"); ax.set_ylabel("L2 relative error")
    ax.set_title("Fourier features vs t_max"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "summary_fourier.png", dpi=120); plt.close(fig)


def fig_collocation():
    d = load("exp06_collocation_density")
    if d is None: return
    sweep, schemes = d["sweep"], d["scheme"]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for s in np.unique(schemes):
        m = schemes == s
        ax.loglog(sweep[m, 0], sweep[m, 1], "-o", label=s)
    ax.set_xlabel("# collocation points"); ax.set_ylabel("L2 relative error")
    ax.set_title("Collocation density"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "summary_collocation.png", dpi=120); plt.close(fig)


def fig_noise():
    d = load("exp08_noise_robustness")
    if d is None: return
    sweep = d["sweep"]; zt = float(d["zeta_true"]); at = float(d["alpha_true"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(sweep[:, 0], sweep[:, 1], "-o", label="ζ̂")
    ax[0].axhline(zt, color="k", ls=":", label="ζ true")
    ax[0].plot(sweep[:, 0], sweep[:, 2], "-s", label="α̂")
    ax[0].axhline(at, color="k", ls="--", label="α true")
    ax[0].set_xlabel("noise σ"); ax[0].legend()
    ax[1].semilogy(sweep[:, 0], sweep[:, 3], "-o")
    ax[1].set_xlabel("noise σ"); ax[1].set_ylabel("L2 relative error")
    fig.tight_layout(); fig.savefig(FIG / "summary_noise.png", dpi=120); plt.close(fig)


def fig_vdp():
    d = load("exp10_van_der_pol")
    if d is None: return
    sweep = d["sweep"]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.semilogy(sweep[:, 0], sweep[:, 1], "-o")
    ax.set_xlabel("μ"); ax.set_ylabel("L2 relative error")
    ax.set_title("Van der Pol — error vs μ")
    fig.tight_layout(); fig.savefig(FIG / "summary_vdp.png", dpi=120); plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    for fn in (fig_hard_vs_soft, fig_loss_weighting, fig_architecture,
               fig_fourier, fig_collocation, fig_noise, fig_vdp):
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__}: {e}")
    print("wrote summary figures to", FIG)


if __name__ == "__main__":
    main()
