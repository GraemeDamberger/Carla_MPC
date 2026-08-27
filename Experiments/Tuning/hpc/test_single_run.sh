#!/bin/bash
#SBATCH --account=def-celiasmi
#SBATCH --job-name=carla_hpo_smoketest
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
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

SERVER_LOG="$SCRATCH/carla/server_${SLURM_JOB_ID}.log"

echo "=== Launching CARLA server (random free port) ==="
if ! launch_carla "$SERVER_LOG"; then
    echo "=== Server log (last 100 lines) ==="
    tail -n 100 "$SERVER_LOG" || true
    exit 1
fi
PORT="$CARLA_PORT"
echo "Server PID: $CARLA_PID  Port: $PORT"

cleanup() {
    echo "=== Server log (last 50 lines) ==="
    tail -n 50 "$SERVER_LOG" || true
    echo "=== Tearing down CARLA server ==="
    kill "${CARLA_PID:-}" 2>/dev/null || true
    wait "${CARLA_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT
echo "Port $PORT is open."

echo "=== Running one simulate_carla rollout ==="
# Capture Python's stdout+stderr to a dedicated file and cat it back, so a
# traceback can't be lost to Slurm's stderr handling. faulthandler dumps a
# native stack too, in case the carla client crashes below the Python level.
ROLLOUT_LOG="$SCRATCH/carla/rollout_${SLURM_JOB_ID}.log"
set +e
CARLA_PORT="$PORT" python -u -c "
import faulthandler; faulthandler.enable()
from pathlib import Path
from Experiments.Comparison.config import config, CONDITIONS
from Experiments.Comparison.simulate_carla import simulate_carla

# Exercise every v4 code path in one job: each disturbance condition once, on a
# route that has a real corner. Covers apply_wheel_faults (surface + the
# multi-parameter flat tyre), crosswind_force, the Town04 load, the curvature
# velocity profile, the R term, and the tube_adaptive numpy fix.
route = config['route_spawn_indices'][1]
for cond in CONDITIONS:
    rmse = simulate_carla(
        f\"smoke_{cond['name']}\",
        Path('Experiments/Tuning/logs/hpc_smoketest'),
        method='tube_adaptive',
        steering_force=cond['steering_force'],
        flat_tire=cond['flat_tire'],
        surface=cond['surface'],
        wind=cond['wind'],
        spawn_index=route,
        model_path='$SCRATCH/carla/model_trial_0',
    )
    print(f\"RESULT route={route} cond={cond['name']:<10} rmse={rmse:.4f} m\")
" > "$ROLLOUT_LOG" 2>&1
RC=$?
set -e
echo "=== Python rollout output (exit code $RC) ==="
cat "$ROLLOUT_LOG"
echo "=== end rollout output ==="

if [ "$RC" -eq 0 ]; then
    echo "=== Smoke test finished successfully ==="
fi
exit "$RC"
