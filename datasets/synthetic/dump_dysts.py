"""
Cache the oscillator/damped-system subset of DySTS as a single HDF5 file.
Each system gets a trajectory at default params + small param sweeps where useful.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import h5py
from pathlib import Path
import dysts.flows as F

OUT = Path("/home2/arpit_thesis/datasets/synthetic/dysts_oscillators.h5")
N_STEPS = 4000

SYSTEMS = [
    "Duffing",
    "ForcedVanDerPol",
    "DoublePendulum",
    "StickSlipOscillator",
    "GlycolyticOscillation",
    "OscillatingFlow",
]


def dump_traj(grp, name, sys_obj, n_steps=N_STEPS):
    try:
        traj = sys_obj.make_trajectory(n_steps, resample=True)
        if traj is None:
            print(f"  {name}: returned None, skip")
            return
        d = grp.create_dataset(name, data=np.asarray(traj))
        d.attrs["params"] = str(sys_obj.params)
        d.attrs["dt"] = float(getattr(sys_obj, "dt", 0.0))
        d.attrs["shape"] = str(d.shape)
        print(f"  {name}: shape {d.shape}, params {sys_obj.params}")
    except Exception as e:
        print(f"  {name}: FAILED ({e})")


with h5py.File(OUT, "w") as f:
    f.attrs["description"] = "DySTS oscillator/damped subset cached as HDF5"
    f.attrs["source"] = "github.com/williamgilpin/dysts"
    f.attrs["n_steps"] = N_STEPS

    for sys_name in SYSTEMS:
        cls = getattr(F, sys_name, None)
        if cls is None:
            print(f"{sys_name}: not in dysts.flows, skip")
            continue
        g = f.create_group(sys_name)
        print(f"{sys_name}:")
        dump_traj(g, "default", cls())

        # Parameter sweeps for Duffing + ForcedVanDerPol
        if sys_name == "Duffing":
            for gamma in [0.1, 0.3, 0.5]:
                obj = cls()
                if "gamma" in obj.params:
                    obj.gamma = gamma
                    dump_traj(g, f"gamma_{gamma}", obj)
        if sys_name == "ForcedVanDerPol":
            for mu in [0.5, 1.0, 4.0]:
                obj = cls()
                if "mu" in obj.params:
                    obj.mu = mu
                    dump_traj(g, f"mu_{mu}", obj)

print(f"\nWrote {OUT}")
