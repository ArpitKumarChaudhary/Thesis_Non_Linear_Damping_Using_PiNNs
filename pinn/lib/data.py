"""Ground-truth ODE solvers + collocation samplers for damped oscillators."""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp


def linear_damped_rhs(t, y, zeta, w0, A=0.0, Omega=0.0):
    x, v = y
    return [v, -2 * zeta * w0 * v - w0**2 * x + A * np.cos(Omega * t)]


def duffing_rhs(t, y, zeta, w0, alpha, A=0.0, Omega=0.0):
    x, v = y
    return [v, -2 * zeta * w0 * v - w0**2 * x - alpha * x**3 + A * np.cos(Omega * t)]


def van_der_pol_rhs(t, y, mu, w0, A=0.0, Omega=0.0):
    x, v = y
    return [v, mu * (1 - x**2) * v - w0**2 * x + A * np.cos(Omega * t)]


_SYSTEMS = {
    "linear": linear_damped_rhs,
    "duffing": duffing_rhs,
    "vdp": van_der_pol_rhs,
}


def ground_truth(system, params, ic, t_max, n_points, rtol=1e-10, atol=1e-12):
    """Return (t, x, v) of shape (n_points,)."""
    rhs = _SYSTEMS[system]
    t_eval = np.linspace(0.0, t_max, n_points)
    sol = solve_ivp(rhs, (0.0, t_max), ic, args=params, t_eval=t_eval,
                    method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.t, sol.y[0], sol.y[1]


def sample_collocation(t_max, n, scheme="uniform", rng=None):
    rng = np.random.default_rng(rng)
    if scheme == "uniform":
        return np.linspace(0.0, t_max, n)
    if scheme == "random":
        return np.sort(rng.uniform(0.0, t_max, size=n))
    if scheme == "latin":
        edges = np.linspace(0.0, t_max, n + 1)
        return edges[:-1] + rng.uniform(0.0, t_max / n, size=n)
    raise ValueError(scheme)


def sparse_observations(t, x, n_obs, noise_sigma=0.0, rng=None):
    """Subsample a trajectory with optional additive Gaussian noise."""
    rng = np.random.default_rng(rng)
    idx = np.sort(rng.choice(len(t), size=n_obs, replace=False))
    t_obs, x_obs = t[idx], x[idx].copy()
    if noise_sigma > 0:
        x_obs += rng.normal(0.0, noise_sigma, size=x_obs.shape)
    return t_obs, x_obs


PRESETS = {
    "free_duffing": dict(system="duffing", params=(0.1, 1.0, 1.0), ic=(1.0, 0.0), t_max=20.0),
    "free_linear":  dict(system="linear",  params=(0.1, 1.0),       ic=(1.0, 0.0), t_max=20.0),
    "free_vdp":     dict(system="vdp",     params=(1.0, 1.0),       ic=(2.0, 0.0), t_max=20.0),
    "forced_duffing_chaos": dict(system="duffing", params=(0.05, 1.0, 1.0, 0.3, 1.2),
                                  ic=(0.1, 0.0), t_max=40.0),
}
