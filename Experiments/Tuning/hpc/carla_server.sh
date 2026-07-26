#!/bin/bash
# Shared helpers for launching a -nullrhi CARLA 0.9.16 server on Nibi and
# waiting for it to come up. See HPC_CARLA_HANDOFF.md section 1 for why every
# flag/bind here is required — do not simplify without re-reading it.
#
# Usage (from a Slurm job script, after `module load apptainer`):
#   source Experiments/Tuning/hpc/carla_server.sh
#   launch_carla <logfile>          # picks a free port, boots CARLA, waits for it
#                                    # -> sets globals CARLA_PORT and CARLA_PID
#
# Why a random free port instead of 2000 + task_id*10: that old scheme was only
# unique WITHIN one array job. When several array jobs run at once and two
# same-index tasks land on the same node, both bind the same port and CARLA dies
# with "bind: Address already in use" (this wiped the tube_adaptive sweep). Here
# each task claims a verified-free random port and retries on any clash.

start_carla_server() {
    local port="$1"
    local logfile="$2"

    local sif="$SCRATCH/carla/carla_0.9.16.sif"
    local nvlibs="$SCRATCH/carla/nvlibs"
    local bin="/workspace/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"

    # Per-port home/Saved dirs: concurrent servers must not share Unreal's
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
        >> "$logfile" 2>&1 &

    echo $!
}

# True (0) when nothing is listening on the given TCP port locally.
port_is_free() {
    ! (exec 3<>/dev/tcp/127.0.0.1/"$1") 2>/dev/null
}

# Random base port in [10000, 50000), stepped by 4 so rpc, rpc+1 (streaming)
# and rpc+2 (secondary) never overlap the next candidate.
pick_base_port() {
    echo $(( (RANDOM % 10000) * 4 + 10000 ))
}

wait_for_port() {
    local port="$1"
    local timeout="${2:-120}"
    local waited=0

    while ! (exec 3<>/dev/tcp/127.0.0.1/"$port") 2>/dev/null; do
        sleep 2
        waited=$((waited + 2))
        if [ "$waited" -ge "$timeout" ]; then
            return 1
        fi
    done
    exec 3<&- 3>&- 2>/dev/null
    return 0
}

# Boot CARLA on a verified-free random port, retrying on any clash/boot failure.
# On success sets globals CARLA_PORT and CARLA_PID and returns 0.
launch_carla() {
    local logfile="$1"
    local max_attempts="${2:-6}"

    # Decorrelate the RNG per process so concurrent tasks don't march in lockstep.
    RANDOM=$(( $$ % 32768 ))

    local attempt=0
    while [ "$attempt" -lt "$max_attempts" ]; do
        attempt=$((attempt + 1))
        local base
        base=$(pick_base_port)

        # Need three consecutive free ports (rpc, rpc+1, rpc+2).
        if ! port_is_free "$base" \
           || ! port_is_free $((base + 1)) \
           || ! port_is_free $((base + 2)); then
            continue
        fi

        echo "[launch_carla] attempt ${attempt}: RPC port ${base}"
        local pid
        pid=$(start_carla_server "$base" "$logfile")

        if wait_for_port "$base" 120; then
            CARLA_PORT="$base"
            CARLA_PID="$pid"
            echo "[launch_carla] CARLA up on port ${base} (pid ${pid})"
            return 0
        fi

        echo "[launch_carla] port ${base} did not come up; killing and retrying"
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done

    echo "[launch_carla] ERROR: no CARLA server after ${max_attempts} attempts"
    return 1
}
