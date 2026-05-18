#!/usr/bin/env bash
# Resilient orchestrator. On per-experiment failure: continue, don't abort.
# Targets RTX A6000 (~40 GB VRAM, 100% util).
set +e   # *DO NOT* abort on single-experiment errors

PYTHON="${PYTHON:-/home2/miniconda3/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PINN_K="${PINN_K:-16}"
export PINN_WIDTH="${PINN_WIDTH:-1024}"
export PINN_DEPTH="${PINN_DEPTH:-6}"
export PINN_N_COLL="${PINN_N_COLL:-10000}"
export PINN_ITERS="${PINN_ITERS:-10000}"
export PINN_LR="${PINN_LR:-5e-4}"

echo "Config: K=$PINN_K W=$PINN_WIDTH D=$PINN_DEPTH N=$PINN_N_COLL iters=$PINN_ITERS lr=$PINN_LR"
echo "GPU before: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
echo "PID=$$  started $(date '+%H:%M:%S')"

cd "$HERE/experiments"
for f in exp_big_ensemble.py exp_big_hard_vs_soft.py exp_big_loss_weighting.py \
         exp_big_inverse.py exp_big_noise.py exp_big_van_der_pol.py; do
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

"$PYTHON" "$HERE/plot_results.py" 2>&1 || echo "(plot warnings ignored)"
echo "ALL DONE at $(date '+%H:%M:%S'). results in $HERE/results/  figures in $HERE/figures/"
