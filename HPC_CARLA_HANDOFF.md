# Carla_MPC — HPC Handoff (Nibi / Digital Research Alliance)

Context for continuing this work in Claude Code. **Read the "Working recipe"
section first — it is the hard-won result of ~15 debugging jobs and everything
else depends on it.**

Goal: run the comparison hyperparameter sweep (`Experiments/Tuning/tune.py`)
in parallel on Nibi, where each Optuna worker needs its own live CARLA server.

---

## 1. THE WORKING RECIPE (verified — do not re-litigate)

CARLA 0.9.16 **does** run on Nibi, but **only with `-nullrhi`** (no graphics
device). Verified: server boots, stays up, and accepts TCP connections on its
RPC port.

```bash
module load apptainer

SIF=$SCRATCH/carla/carla_0.9.16.sif
NVLIBS=$SCRATCH/carla/nvlibs
BIN=/workspace/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping

# One-time per job: stage the NVIDIA userspace libs from the GPU node
mkdir -p "$NVLIBS" "$SCRATCH/carla/chome" "$SCRATCH/carla/saved"
cp -u /usr/lib64/libnvidia-*.so.* "$NVLIBS"/ 2>/dev/null

apptainer exec --nv \
  --home "$SCRATCH/carla/chome:/carlahome" \
  --bind "$NVLIBS:/nvlibs" \
  --bind /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d \
  --bind "$SCRATCH/carla/saved:/workspace/CarlaUE4/Saved" \
  --env LD_LIBRARY_PATH=/nvlibs:/.singularity.d/libs \
  "$SIF" bash -c "cd /workspace && $BIN CarlaUE4 \
      -nullrhi -prefernvidia -RenderOffScreen -nosound -carla-rpc-port=<PORT>"
```

### Non-obvious details that all matter

| Detail | Why |
|---|---|
| `-nullrhi` | **Required.** Without it CARLA exits after ~10s. See §2. |
| Run `CarlaUE4-Linux-Shipping` directly | `CarlaUE4.sh` swallows stderr and hides all errors |
| `cd /workspace` | CARLA lives at `/workspace` in this image, NOT `/home/carla` |
| `--home ...:/carlahome` | Unreal needs a writable home. Use `--home`, NOT `--env HOME=` (Apptainer refuses the latter) |
| Copy `/usr/lib64/libnvidia-*` | `--nv` injects `libGLX_nvidia.so.0` but NOT its private deps (`glsi`, `glcore`, `tls`, `gpucomp`) |
| Do **NOT** bind all of `/usr/lib64` | Drags in the host glibc → container breaks entirely (`GLIBC_2.35 not found`) |
| Do **NOT** use `--writable-tmpfs` | Overlay is ~16MB; CARLA fills it instantly → "No space left on device" |
| `chmod: Read-only file system` warning | Harmless. Appears on every successful run. Ignore. |

### Why `-nullrhi` is correct here, not a compromise

The controller uses **ground-truth state + global waypoints only — no cameras,
no lidar, no rendered sensors.** So there is nothing to render. `-nullrhi` gives
lighter servers, no GPU rendering contention, and likely allows packing many
CARLA servers per GPU (or possibly no GPU at all — worth testing).

---

## 2. What does NOT work (don't retry these)

Full Vulkan rendering is **unfixable from user space** on this cluster:

- Container (Ubuntu 20.04) ships Vulkan loader **1.2.131**; host driver is
  **580.82.07** exposing loader **1.4.304**.
- Failure: `loader_scanned_icd_add: Could not get 'vkCreateInstance' via
  'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0`
- Tried and failed: binding the host ICD json (absolute and relative
  `library_path`); binding host `libvulkan.so.1.4.304`; `-prefernvidia` plus
  binding the whole `/usr/share/vulkan/icd.d`; `apt-get install libvulkan1
  vulkan-tools` in a sandbox (**Ubuntu 20.04's newest loader IS 1.2.131**, so
  apt can never fix it).
- `-opengl` falls back to lavapipe (software) → `GameThread timed out waiting
  for RenderThread` → crash.

A support ticket to support@tech.alliancecan.ca was drafted (ask Graeme if it
was sent). Related upstream: CARLA issue **#8079**, which independently
confirms `-nullrhi` as the workaround.

---

## 3. Cluster facts

- Login: `ssh graemed@nibi.alliancecan.ca` (password + Duo MFA)
- Account: **`def-celiasmi`** → `--account=def-celiasmi`
- `$SCRATCH` = `/scratch/graemed` — data + job I/O (purged ~60 days)
- `/home` — code repos. `~/projects/def-celiasmi/` — backed up, long-term
- Env: `module load python/3.11`; virtualenv `~/hpo_env`
  (numpy 2.4.2, torch 2.12.1, optuna 4.9.0, installed with `pip install --no-index`)
- GPU types that schedule: **`h100`**, MIG **`nvidia_h100_80gb_hbm3_1g.10gb`**.
  `t4` and `a5000` are REJECTED ("node configuration not available").
- Login nodes: ~10 CPU-min / 4 GB cap. No heavy work.
- Container builds: sandbox builds **fail on `/scratch`** (Lustre). Build in
  `$SLURM_TMPDIR` (local NVMe) inside a job instead.
- Slurm `.out` files land in the **submit directory** — always `cd $SCRATCH/carla`
  before `sbatch`, or output scatters.
- Interactive `salloc` queues far worse than `sbatch`. Prefer batch.

---

## 4. Requirements still to do

1. **Clone the repo on Nibi** (shallow — it's heavy with committed data/plots):
   `git clone --depth 1 <Carla_MPC url>`
2. **`scp` the trained base model up** — currently only on Graeme's laptop;
   referenced as `.../logs/run_*/models/model_trial_0`. Put in `$SCRATCH/carla/`.
3. **Install comparison deps** into `~/hpo_env`:
   ```bash
   pip install --no-index scipy opencv-python
   pip install <path>/carla-0.9.16-cp311-manylinux_2_31_x86_64.whl
   ```
   The CARLA wheel is inside the image at
   `/workspace/PythonAPI/carla/dist/` (cp310/cp311/cp312 available; **use
   cp311** to match the env).
4. **Code edit — parameterize the port** in
   `Experiments/Comparison/simulate_carla.py` (~line 221, currently hardcoded
   `carla.Client("localhost", 2000)`):
   ```python
   import os
   port = int(os.environ.get("CARLA_PORT", 2000))
   client = carla.Client("localhost", port)
   client.set_timeout(10.0)
   ```
5. **Code edit — storage** in `Experiments/Tuning/tune.py`: switch SQLite →
   Optuna `JournalStorage` so parallel workers share one study safely (SQLite
   corrupts under concurrent writers on Lustre):
   ```python
   from optuna.storages import JournalStorage
   from optuna.storages.journal import JournalFileBackend
   storage = JournalStorage(JournalFileBackend(str(log_dir / "study.log")))
   ```
6. **Single-run test job** (the next real milestone): start a `-nullrhi` server,
   poll its port until open, run ONE `simulate_carla` against it, tear down.
   Prove one closed-loop rollout works before scaling.
7. **Job array**: each task exports a unique port
   (`CARLA_PORT=$((2000 + SLURM_ARRAY_TASK_ID * 10))`), launches its own CARLA
   server, waits for the port, then runs an Optuna worker against the shared
   journal study. Mechanically identical to the already-working training array.

---

## 5. Reference: the training sweep (DONE, separate repo)

`Carla_MPC_Hyperparameter` — standalone, no CARLA. Already ran successfully as a
10-worker Slurm array on MIG slices, ~200 trials sharing one journal study.
**Use it as the template for the array + journal-storage pattern**; the only new
piece here is launching a CARLA server per task.

Lessons carried over from it:
- Tiny MLPs are **CPU-bound**, not GPU-bound (H100 sat at 5% util). The MPC solve
  (scipy SLSQP) is also CPU-bound — so give each comparison task several CPU
  cores; the GPU matters little once `-nullrhi` removes rendering.
- Cost model: one CARLA run ≈ 5 min; one comparison trial = 2 seeds × 7
  scenarios = 14 runs ≈ 70 min. Parallelism matters a lot.

## 6. Debugging playbook (this saved us repeatedly)

- Poll the port to decide success, don't just check the process is alive.
- Always dump the last N lines of the server log in the job script.
- Bound every `find` — `find /` on Lustre never finishes and eats the walltime.
- Batch-mode diagnostics with several numbered sections per job beat one-off
  interactive pokes, given queue waits.
