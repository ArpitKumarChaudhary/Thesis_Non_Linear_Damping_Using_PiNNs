#!/usr/bin/env bash
# Run the three additional defensive experiments sequentially.
# Resilient: continues past per-experiment failures.
set +e
PYTHON="${PYTHON:-/home2/miniconda3/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "PID=$$  started $(date '+%H:%M:%S')"
cd "$HERE/experiments"
for f in tune_optuna.py tune_fourier.py tune_arch.py; do
    echo "=================================================================="
    echo "==> $f  start=$(date '+%H:%M:%S')"
    echo "=================================================================="
    if "$PYTHON" "$f"; then
        echo "==> $f  OK   end=$(date '+%H:%M:%S')"
    else
        rc=$?
        echo "==> $f  FAIL rc=$rc end=$(date '+%H:%M:%S')  (continuing)"
    fi
done
echo "ALL DONE at $(date '+%H:%M:%S')"
