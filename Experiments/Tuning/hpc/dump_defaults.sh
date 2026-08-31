#!/bin/bash
#SBATCH --account=def-celiasmi
#SBATCH --job-name=carla_defaults
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#
# Print the vehicle's real mass, drag coefficient and wheel parameters, what
# each disturbance condition changes them to, and the crosswind force as a
# fraction of available tire grip. These are the numbers to quote in the paper.
#
#   cd $SCRATCH/carla && sbatch ~/Carla_MPC/Experiments/Tuning/hpc/dump_defaults.sh

set -euo pipefail

module load apptainer python/3.11
source ~/hpo_env/bin/activate
cd ~/Carla_MPC

source Experiments/Tuning/hpc/carla_server.sh
SERVER_LOG="$SCRATCH/carla/server_defaults_${SLURM_JOB_ID}.log"

echo "=== Launching CARLA ==="
if ! launch_carla "$SERVER_LOG"; then
    tail -n 100 "$SERVER_LOG" || true
    exit 1
fi
trap 'kill "${CARLA_PID:-}" 2>/dev/null || true' EXIT

echo "=== Port $CARLA_PORT open; dumping defaults ==="
CARLA_PORT="$CARLA_PORT" python -u -m Experiments.Tuning.hpc.dump_wheel_defaults
