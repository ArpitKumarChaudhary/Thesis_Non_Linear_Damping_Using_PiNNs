#!/usr/bin/env bash
# Run every experiment script in order. Results -> pinn/results/, figures -> pinn/figures/.
# Override PYTHON to point at a specific interpreter.
set -euo pipefail
PYTHON="${PYTHON:-/home2/miniconda3/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/experiments"
for f in exp01_*.py exp02_*.py exp03_*.py exp04_*.py exp05_*.py \
         exp06_*.py exp07_*.py exp08_*.py exp09_*.py exp10_*.py; do
    echo "=== $f ==="
    "$PYTHON" "$f"
done
"$PYTHON" "$HERE/plot_results.py"
echo "ALL DONE. See pinn/figures/ and pinn/results/"
