#!/bin/bash
# Shared helpers for launching a -nullrhi CARLA 0.9.16 server on Nibi and
# waiting for it to come up. See HPC_CARLA_HANDOFF.md section 1 for why every
# flag/bind here is required — do not simplify without re-reading it.
#
# Usage (from a Slurm job script, after `module load apptainer`):
#   source Experiments/Tuning/hpc/carla_server.sh
#   start_carla_server <port> <logfile>      # prints the server PID
#   wait_for_port <port> [timeout_seconds]   # returns 0 once the RPC port is open

start_carla_server() {
    local port="$1"
    local logfile="$2"

    local sif="$SCRATCH/carla/carla_0.9.16.sif"
    local nvlibs="$SCRATCH/carla/nvlibs"
    local bin="/workspace/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"

    # Per-port home/Saved dirs: concurrent array tasks must not share Unreal's
    # writable home, or their save-game/log state collides.
    local chome="$SCRATCH/carla/chome_${port}"
    local saved="$SCRATCH/carla/saved_${port}"

    mkdir -p "$nvlibs" "$chome" "$saved"
    cp -u /usr/lib64/libnvidia-*.so.* "$nvlibs"/ 2>/dev/null

    apptainer exec --nv \
        --home "$chome:/carlahome" \
        --bind "$nvlibs:/nvlibs" \
        --bind /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d \
        --bind "$saved:/workspace/CarlaUE4/Saved" \
        --env LD_LIBRARY_PATH=/nvlibs:/.singularity.d/libs \
        "$sif" bash -c "cd /workspace && $bin CarlaUE4 \
            -nullrhi -prefernvidia -RenderOffScreen -nosound -carla-rpc-port=${port}" \
        > "$logfile" 2>&1 &

    echo $!
}

wait_for_port() {
    local port="$1"
    local timeout="${2:-120}"
    local waited=0

    while ! (exec 3<>/dev/tcp/127.0.0.1/"$port") 2>/dev/null; do
        sleep 2
        waited=$((waited + 2))
        if [ "$waited" -ge "$timeout" ]; then
            echo "ERROR: CARLA server did not open port $port within ${timeout}s"
            return 1
        fi
    done
    exec 3<&- 3>&- 2>/dev/null
    return 0
}
