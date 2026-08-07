#!/bin/bash
#SBATCH --account=def-celiasmi
#SBATCH --job-name=carla_route_survey
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#
# Boot one CARLA server and survey spawn points on the eval map, reporting each
# route's length + curvature so we can choose config['route_spawn_indices'].
# Run once before the v3 sweep:
#   cd $SCRATCH/carla && sbatch ~/Carla_MPC/Experiments/Tuning/hpc/survey_routes.sh

set -euo pipefail

module load apptainer python/3.11
source ~/hpo_env/bin/activate
cd ~/Carla_MPC

source Experiments/Tuning/hpc/carla_server.sh

SERVER_LOG="$SCRATCH/carla/server_survey_${SLURM_JOB_ID}.log"
echo "=== Launching CARLA server ==="
if ! launch_carla "$SERVER_LOG"; then
    tail -n 100 "$SERVER_LOG" || true
    exit 1
fi
trap 'kill "${CARLA_PID:-}" 2>/dev/null || true' EXIT
echo "=== Port $CARLA_PORT open; surveying routes ==="

CARLA_PORT="$CARLA_PORT" python -u -m Experiments.Tuning.hpc.route_survey --step 20 --points 1500
