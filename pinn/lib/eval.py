"""Error metrics + matplotlib plotting helpers."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def l2_relative(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))


def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def max_err(pred, true):
    return float(np.max(np.abs(np.asarray(pred) - np.asarray(true))))


def predict(model, t_np: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    t = torch.tensor(t_np, dtype=torch.float32, device=device).reshape(-1, 1)
    with torch.no_grad():
        x = model(t).cpu().numpy().reshape(-1)
    return x


def plot_solution(t, x_true, x_pred, title, out_path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].plot(t, x_true, "k-", lw=2, label="reference")
    ax[0].plot(t, x_pred, "r--", lw=1.5, label="PINN")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("x(t)"); ax[0].legend()
    ax[0].set_title(title)
    ax[1].semilogy(t, np.abs(x_pred - x_true) + 1e-16)
    ax[1].set_xlabel("t"); ax[1].set_ylabel("|x_pred - x_true|")
    ax[1].set_title("absolute error")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_loss_curves(history, out_path, title="loss"):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for key in ("total", "phys", "ic", "data"):
        if key in history and any(v > 0 for v in history[key]):
            ax.semilogy(history["step"], np.maximum(history[key], 1e-16), label=key)
    ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.set_title(title); ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_results(out_dir, **arrays):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "results.npz", **arrays)
