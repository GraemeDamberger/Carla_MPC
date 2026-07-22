#!/bin/bash
#SBATCH --account=def-celiasmi
#SBATCH --job-name=carla_hpo_smoketest
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#
# Milestone from HPC_CARLA_HANDOFF.md item 6: boot one -nullrhi CARLA server,
# poll its port, run ONE simulate_carla rollout against it, tear down.
# Run this once and confirm it succeeds before touching tune_array.sh.
#
# Submit from $SCRATCH/carla so the .out file doesn't scatter:
#   cd $SCRATCH/carla && sbatch ~/Carla_MPC/Experiments/Tuning/hpc/test_single_run.sh

set -euo pipefail

module load apptainer python/3.11
source ~/hpo_env/bin/activate
cd ~/Carla_MPC

source Experiments/Tuning/hpc/carla_server.sh

PORT=2000
SERVER_LOG="$SCRATCH/carla/server_${SLURM_JOB_ID}.log"

echo "=== Starting CARLA server on port $PORT ==="
CARLA_PID=$(start_carla_server "$PORT" "$SERVER_LOG")
echo "Server PID: $CARLA_PID"

cleanup() {
    echo "=== Server log (last 50 lines) ==="
    tail -n 50 "$SERVER_LOG" || true
    echo "=== Tearing down CARLA server ==="
    kill "$CARLA_PID" 2>/dev/null || true
    wait "$CARLA_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Waiting for port $PORT ==="
if ! wait_for_port "$PORT" 120; then
    echo "=== Server log (last 100 lines) ==="
    tail -n 100 "$SERVER_LOG" || true
    exit 1
fi
echo "Port $PORT is open."

echo "=== Running one simulate_carla rollout ==="
CARLA_PORT="$PORT" python -c "
from pathlib import Path
from Experiments.Comparison.simulate_carla import simulate_carla

rmse = simulate_carla(
    'hpc_smoketest',
    Path('Experiments/Tuning/logs/hpc_smoketest'),
    method='tube',
    steering_force=0.0,
    wind_force=0.0,
    model_path='$SCRATCH/carla/model_trial_0',
)
print(f'RMSE: {rmse:.4f} m')
"

echo "=== Smoke test finished successfully ==="
