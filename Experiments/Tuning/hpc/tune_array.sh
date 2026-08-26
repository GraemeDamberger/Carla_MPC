#!/bin/bash
#SBATCH --account=def-celiasmi
#SBATCH --job-name=carla_hpo_tune
#SBATCH --array=0-9
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%A_%a.out
#
# Parallel Optuna sweep over Experiments/Tuning/tune.py. Each array task boots
# its own -nullrhi CARLA server on a VERIFIED-FREE random port (launch_carla)
# and runs as one Optuna worker against a shared JournalStorage study (see
# HPC_CARLA_HANDOFF.md items 5 and 7).
#
# Only run this after test_single_run.sh has succeeded.
#
# Usage:
#   cd $SCRATCH/carla
#   sbatch ~/Carla_MPC/Experiments/Tuning/hpc/tune_array.sh <method> [n_trials] [n_seeds] [rmse_cap]
#
# e.g.:
#   sbatch ~/Carla_MPC/Experiments/Tuning/hpc/tune_array.sh tube 8 2 5.0
#
# Study naming: "<method>_<STUDY_SUFFIX>" (default sweep_v2). Bump the suffix for
# a clean re-run; the capped objective below is NOT comparable to the uncapped
# v1 studies, so keep them separate. Cost model (HPC_CARLA_HANDOFF.md §5): tube
# ~1 h/trial; the online-learning methods (replay_buffer, residual_dynamics) are
# several times slower because they backprop each step — size n_trials for that.

set -euo pipefail

METHOD="${1:-tube}"
N_TRIALS="${2:-10}"
N_SEEDS="${3:-1}"
RMSE_CAP="${4:-5.0}"                      # normalized-RMSE-ratio upper bound (see tune.py)
STUDY_NAME="${METHOD}_${STUDY_SUFFIX:-sweep_v4}"

module load apptainer python/3.11
source ~/hpo_env/bin/activate
cd ~/Carla_MPC

source Experiments/Tuning/hpc/carla_server.sh

SERVER_LOG="$SCRATCH/carla/server_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

echo "=== [task $SLURM_ARRAY_TASK_ID] Launching CARLA server ==="
if ! launch_carla "$SERVER_LOG"; then
    echo "=== [task $SLURM_ARRAY_TASK_ID] CARLA failed to launch ==="
    tail -n 100 "$SERVER_LOG" || true
    exit 1
fi

cleanup() {
    echo "=== [task $SLURM_ARRAY_TASK_ID] Server log (last 50 lines) ==="
    tail -n 50 "$SERVER_LOG" || true
    kill "${CARLA_PID:-}" 2>/dev/null || true
    wait "${CARLA_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== [task $SLURM_ARRAY_TASK_ID] Port $CARLA_PORT open, starting Optuna worker ==="

CARLA_PORT="$CARLA_PORT" python -m Experiments.Tuning.tune \
    --method "$METHOD" \
    --n_trials "$N_TRIALS" \
    --n_seeds "$N_SEEDS" \
    --rmse_cap "$RMSE_CAP" \
    --model "$SCRATCH/carla/model_trial_0" \
    --study_name "$STUDY_NAME"

echo "=== [task $SLURM_ARRAY_TASK_ID] Worker finished ==="
