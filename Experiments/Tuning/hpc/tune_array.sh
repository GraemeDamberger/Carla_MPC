#!/bin/bash
#SBATCH --account=def-celiasmi
#SBATCH --job-name=carla_hpo_tune
#SBATCH --array=0-9
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#
# Parallel Optuna sweep over Experiments/Tuning/tune.py. Each array task boots
# its own -nullrhi CARLA server on a unique port and runs as one Optuna
# worker against a shared JournalStorage study (see HPC_CARLA_HANDOFF.md
# items 5 and 7). Mechanically the same array + journal-storage pattern as
# Carla_MPC_Hyperparameter/tune_train_array.sh, plus a CARLA server per task.
#
# Only run this after test_single_run.sh has succeeded.
#
# Usage:
#   cd $SCRATCH/carla
#   sbatch ~/Carla_MPC/Experiments/Tuning/hpc/tune_array.sh <method> [n_trials] [n_seeds]
#
# e.g.:
#   sbatch ~/Carla_MPC/Experiments/Tuning/hpc/tune_array.sh tube 10 2
# Cost model (HPC_CARLA_HANDOFF.md §5): one comparison trial (2 seeds x 7
# scenarios) ~= 70 min, so keep n_trials modest relative to --time above.

set -euo pipefail

METHOD="${1:-tube}"
N_TRIALS="${2:-10}"
N_SEEDS="${3:-2}"
STUDY_NAME="${METHOD}_sweep_v1"

module load apptainer python/3.11
source ~/hpo_env/bin/activate
cd ~/Carla_MPC

source Experiments/Tuning/hpc/carla_server.sh

PORT=$((2000 + SLURM_ARRAY_TASK_ID * 10))
SERVER_LOG="$SCRATCH/carla/server_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

echo "=== [task $SLURM_ARRAY_TASK_ID] Starting CARLA server on port $PORT ==="
CARLA_PID=$(start_carla_server "$PORT" "$SERVER_LOG")

cleanup() {
    echo "=== [task $SLURM_ARRAY_TASK_ID] Server log (last 50 lines) ==="
    tail -n 50 "$SERVER_LOG" || true
    kill "$CARLA_PID" 2>/dev/null || true
    wait "$CARLA_PID" 2>/dev/null || true
}
trap cleanup EXIT

if ! wait_for_port "$PORT" 120; then
    tail -n 100 "$SERVER_LOG" || true
    exit 1
fi
echo "=== [task $SLURM_ARRAY_TASK_ID] Port $PORT open, starting Optuna worker ==="

CARLA_PORT="$PORT" python -m Experiments.Tuning.tune \
    --method "$METHOD" \
    --n_trials "$N_TRIALS" \
    --n_seeds "$N_SEEDS" \
    --model "$SCRATCH/carla/model_trial_0" \
    --study_name "$STUDY_NAME"

echo "=== [task $SLURM_ARRAY_TASK_ID] Worker finished ==="
