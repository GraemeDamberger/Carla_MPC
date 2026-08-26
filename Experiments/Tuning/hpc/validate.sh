#!/bin/bash
#SBATCH --account=def-celiasmi
#SBATCH --job-name=carla_validate
#SBATCH --array=0-3
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
#
# Run each method's best parameters over every route x condition at full step
# count, WITH plots (trajectory, speed, error) written per rollout. One array
# task per method.
#
#   cd $SCRATCH/carla
#   sbatch ~/Carla_MPC/Experiments/Tuning/hpc/validate.sh [study_suffix]
#
# Reads Experiments/Tuning/logs/study_<method>_<suffix>/best_params.json, so run
# `tune.py --report --study_suffix <suffix>` first to make sure it is current.

set -euo pipefail

SUFFIX="${1:-sweep_v3}"
METHODS=(tube replay_buffer residual_dynamics tube_adaptive)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"

module load apptainer python/3.11
source ~/hpo_env/bin/activate
cd ~/Carla_MPC

PARAMS="Experiments/Tuning/logs/study_${METHOD}_${SUFFIX}/best_params.json"
if [ ! -f "$PARAMS" ]; then
    echo "No best_params.json for ${METHOD} at ${PARAMS} — skipping."
    exit 0
fi

source Experiments/Tuning/hpc/carla_server.sh
SERVER_LOG="$SCRATCH/carla/server_val_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

echo "=== [${METHOD}] launching CARLA ==="
if ! launch_carla "$SERVER_LOG"; then
    tail -n 100 "$SERVER_LOG" || true
    exit 1
fi
trap 'kill "${CARLA_PID:-}" 2>/dev/null || true' EXIT

echo "=== [${METHOD}] port $CARLA_PORT open; validating ==="
CARLA_PORT="$CARLA_PORT" python -u -m Experiments.Tuning.tune \
    --validate \
    --method "$METHOD" \
    --params_file "$PARAMS" \
    --model "$SCRATCH/carla/model_trial_0"

echo "=== [${METHOD}] validation finished ==="
