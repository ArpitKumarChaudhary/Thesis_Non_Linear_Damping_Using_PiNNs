"""Shared bootstrap for experiment scripts."""

import os, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

DEVICE = os.environ.get("PINN_DEVICE",
                       "cuda" if torch.cuda.is_available() else "cpu")
print(f"[_common] using device={DEVICE}")
