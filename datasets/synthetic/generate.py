"""
Synthetic non-linear damped oscillator dataset generator for PINN ablations.

Produces an HDF5 file with multiple trajectories sweeping:
  - damping ratio (zeta)
  - nonlinearity (alpha for Duffing, mu for Van der Pol)
  - initial conditions (x0, v0)
  - forcing amplitude / frequency (optional)

Three families:
  1. Linear damped oscillator     :  x'' + 2 zeta w0 x' + w0^2 x = F(t)
  2. Damped Duffing               :  x'' + 2 zeta w0 x' + w0^2 x + alpha x^3 = F(t)
  3. Van der Pol                  :  x'' - mu (1 - x^2) x' + w0^2 x = F(t)

Forcing F(t) = A cos(Omega t)  (set A=0 for free oscillation).
"""

import itertools
import numpy as np
import h5py
from scipy.integrate import solve_ivp
from pathlib import Path


def linear_damped(t, y, zeta, w0, A, Omega):
    x, v = y
    return [v, -2 * zeta * w0 * v - w0**2 * x + A * np.cos(Omega * t)]


def duffing(t, y, zeta, w0, alpha, A, Omega):
    x, v = y
    return [v, -2 * zeta * w0 * v - w0**2 * x - alpha * x**3 + A * np.cos(Omega * t)]


def van_der_pol(t, y, mu, w0, A, Omega):
    x, v = y
    return [v, mu * (1 - x**2) * v - w0**2 * x + A * np.cos(Omega * t)]


def integrate(rhs, args, t_span, n_steps, y0, rtol=1e-10, atol=1e-12):
    t_eval = np.linspace(t_span[0], t_span[1], n_steps)
    sol = solve_ivp(rhs, t_span, y0, args=args, t_eval=t_eval,
                    method="DOP853", rtol=rtol, atol=atol)
    return sol.t, sol.y  # y shape (2, n_steps)


def make_dataset(out_path, t_max=20.0, n_steps=2001, seed=0):
    rng = np.random.default_rng(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Linear damped: 4 zetas x 3 ICs = 12 trajectories ----
    lin_cases = []
    for zeta, (x0, v0) in itertools.product(
        [0.05, 0.1, 0.25, 0.5],
        [(1.0, 0.0), (0.5, 0.5), (-1.0, 1.0)],
    ):
        t, y = integrate(linear_damped, (zeta, 1.0, 0.0, 0.0),
                         (0.0, t_max), n_steps, [x0, v0])
        lin_cases.append((zeta, x0, v0, t, y))

    # ---- Duffing: zeta x alpha x ICs ----
    duf_cases = []
    for zeta, alpha, (x0, v0) in itertools.product(
        [0.05, 0.1, 0.2],
        [0.5, 1.0, 2.0],
        [(1.0, 0.0), (0.5, 0.5)],
    ):
        t, y = integrate(duffing, (zeta, 1.0, alpha, 0.0, 0.0),
                         (0.0, t_max), n_steps, [x0, v0])
        duf_cases.append((zeta, alpha, x0, v0, t, y))

    # ---- Forced Duffing (chaos-prone regime) ----
    forced_cases = []
    for A, Omega in [(0.3, 1.2), (0.5, 1.0), (0.8, 1.4)]:
        t, y = integrate(duffing, (0.05, 1.0, 1.0, A, Omega),
                         (0.0, t_max), n_steps, [0.1, 0.0])
        forced_cases.append((A, Omega, t, y))

    # ---- Van der Pol: mu sweep ----
    vdp_cases = []
    for mu in [0.5, 1.0, 2.0, 4.0]:
        t, y = integrate(van_der_pol, (mu, 1.0, 0.0, 0.0),
                         (0.0, t_max), n_steps, [2.0, 0.0])
        vdp_cases.append((mu, t, y))

    with h5py.File(out_path, "w") as f:
        f.attrs["description"] = "Synthetic damped oscillator dataset for PINN ablations"
        f.attrs["t_max"] = t_max
        f.attrs["n_steps"] = n_steps
        f.attrs["integrator"] = "DOP853, rtol=1e-10, atol=1e-12"

        g = f.create_group("linear_damped")
        for i, (zeta, x0, v0, t, y) in enumerate(lin_cases):
            d = g.create_dataset(f"traj_{i:03d}", data=np.column_stack([t, y[0], y[1]]))
            d.attrs.update(dict(zeta=zeta, w0=1.0, x0=x0, v0=v0))

        g = f.create_group("duffing")
        for i, (zeta, alpha, x0, v0, t, y) in enumerate(duf_cases):
            d = g.create_dataset(f"traj_{i:03d}", data=np.column_stack([t, y[0], y[1]]))
            d.attrs.update(dict(zeta=zeta, w0=1.0, alpha=alpha, x0=x0, v0=v0))

        g = f.create_group("duffing_forced")
        for i, (A, Omega, t, y) in enumerate(forced_cases):
            d = g.create_dataset(f"traj_{i:03d}", data=np.column_stack([t, y[0], y[1]]))
            d.attrs.update(dict(zeta=0.05, w0=1.0, alpha=1.0, A=A, Omega=Omega, x0=0.1, v0=0.0))

        g = f.create_group("van_der_pol")
        for i, (mu, t, y) in enumerate(vdp_cases):
            d = g.create_dataset(f"traj_{i:03d}", data=np.column_stack([t, y[0], y[1]]))
            d.attrs.update(dict(mu=mu, w0=1.0, x0=2.0, v0=0.0))

    print(f"wrote {out_path}")
    print(f"  linear_damped : {len(lin_cases)} trajectories")
    print(f"  duffing       : {len(duf_cases)} trajectories")
    print(f"  duffing_forced: {len(forced_cases)} trajectories")
    print(f"  van_der_pol   : {len(vdp_cases)} trajectories")


if __name__ == "__main__":
    make_dataset("/home2/arpit_thesis/datasets/synthetic/damped_oscillators.h5")
