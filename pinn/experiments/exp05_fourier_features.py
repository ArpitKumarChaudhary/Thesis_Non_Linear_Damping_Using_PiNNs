"""Experiment 05 — Fourier features for long-horizon spectral bias.

Vanilla MLPs fail to fit many oscillation periods. Fourier-feature embedding
mitigates the low-frequency bias. We compare both for t_max ∈ {20, 50, 100}.
"""

import _common  # noqa
import numpy as np, torch
from lib.data import ground_truth
from lib.models import MLP, FourierMLP
from lib.train import train_pinn, duffing_residual, TrainConfig
from lib.eval import predict, l2_relative, plot_solution, save_results

EXP = "exp05_fourier_features"
zeta, w0, alpha = 0.05, 1.0, 1.0; ic = (1.0, 0.0)

rows = []
for t_max in [20.0, 50.0, 100.0]:
    t_ref, x_ref, _ = ground_truth("duffing", (zeta, w0, alpha, 0.0, 0.0), ic, t_max, int(200 * t_max))
    t_colloc = torch.linspace(0, t_max, int(200 * t_max)).reshape(-1, 1)
    for tag, mk_model in [
        ("mlp",     lambda tm=t_max: MLP(hidden=128, depth=4, activation="tanh", t_scale=tm)),
        ("fourier", lambda tm=t_max: FourierMLP(hidden=128, depth=4, n_features=64, sigma=2.0)),
    ]:
        torch.manual_seed(0)
        model = mk_model()
        residual = lambda t, m=model: duffing_residual(m, t, zeta, w0, alpha)
        cfg = TrainConfig(iters=20000, lr=1e-3, lambda_ic=100.0, seed=0, device=_common.DEVICE)
        train_pinn(model, residual, t_colloc, ic=ic, data=None, cfg=cfg)
        x_pred = predict(model, t_ref)
        err = l2_relative(x_pred, x_ref)
        rows.append((t_max, tag, err))
        print(f"{EXP}: t_max={t_max} model={tag} L2={err:.3e}")
        plot_solution(t_ref, x_ref, x_pred, f"{tag} (t_max={t_max})",
                      _common.FIGURES / f"{EXP}_T{int(t_max)}_{tag}.png")

arr = np.array([(tm, e) for (tm, _, e) in rows])
tags = np.array([t for (_, t, _) in rows])
save_results(_common.RESULTS / EXP, sweep=arr, tag=tags)
