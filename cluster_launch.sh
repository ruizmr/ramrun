#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# cluster_launch.sh
# -----------------------------------------------------------------------------
# A reusable Slurm batch script that bootstraps a multi-node Ray cluster on the
# VCU Athena HPC system (or any Slurm cluster) and then executes an arbitrary
# Python entry-point across the allocated nodes.  It is designed to be copied
# once into a research project and re-used for many different experiments
# (Pufferlib RL, DPO fine-tuning, data-processing pipelines, hyper-parameter
# sweeps, …) simply by passing different CLI arguments to the underlying
# Python script.
#
# Main features
#   • Requests GPUs/CPUs/memory according to the Athena wiki limits.
#   • Spawns a Ray head on the first node and Ray workers on the remaining
#     nodes — all within *one* Slurm job (complies with the “one allocation”
#     rule while still letting you use multiple nodes).
#   • Automatic clean-up via shell traps so no orphaned Ray processes are left
#     running after the job ends or is cancelled.
#   • Retries for Ray start-up in case a node is slow to come online.
#   • Optional Apptainer (Singularity) container execution for full
#     reproducibility (set $APPTAINER_IMG to the .sif you want to use).
#   • Uses fast node-local storage (/tmp) for Ray’s object store & spill files
#     and rsyncs results back to $HOME at the end of the run.
#
# Usage
#   sbatch cluster_launch.sh --job-type puffer --env CartPole-v1 --timesteps 1e6
#   sbatch --export=ALL,APPTAINER_IMG=myenv.sif cluster_launch.sh \
#          --job-type dpo --model mistralai/Mixtral-8x22B
# -----------------------------------------------------------------------------

#SBATCH --partition=gpu-a100         # gpu-v100 | gpu-a100 | gpu-h100 | cpu-*
#SBATCH --nodes=4                    # total number of nodes requested
#SBATCH --gres=gpu:4                 # GPUs *per* node (A100 nodes have 4 each)
#SBATCH --cpus-per-task=30           # leave 2 CPUs per node for the OS (Athena rule)
#SBATCH --mem=494G                   # 512 GB node → wiki says max 494 GB
#SBATCH --time=48:00:00              # wall-clock limit (hh:mm:ss)
#SBATCH --job-name=ray_multi_node
#SBATCH --output=logs/%x_%j.out      # auto-create logs/ dir at runtime
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=ALL                 # propagate env vars (e.g. APPTAINER_IMG)

# Fail fast and echo commands (useful for debugging)
set -euo pipefail

###############################################################################
# Configurable parameters (override by exporting before calling sbatch)
###############################################################################

RETRY_COUNT=${RETRY_COUNT:-3}                       # retries for ray start
PORT=${PORT:-6379}                                 # Ray head port
TMP_DIR=${TMP_DIR:-/tmp/$USER/ray_$SLURM_JOB_ID}   # per-job scratch dir
PY_ENTRYPOINT=${PY_ENTRYPOINT:-launch_job.py}      # Python script to run

###############################################################################
# Helper functions
###############################################################################

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

clean_up() {
  log "Cleaning up…"
  ray stop --force > /dev/null 2>&1 || true
  rm -rf "$TMP_DIR"
}

trap clean_up EXIT SIGINT SIGTERM

###############################################################################
# 0.  Preparatory steps — environment modules, containers, dirs
###############################################################################

mkdir -p "$TMP_DIR"        # scratch space
mkdir -p "$HOME/out"       # results (destination)
mkdir -p "$(dirname "$SLURM_OUTPUT")" 2>/dev/null || true

# Load minimal modules (adjust if you prefer Miniconda3/Miniforge3)
module purge 2>/dev/null || true
module load Miniconda3 2>/dev/null || true

if [[ -n "${APPTAINER_IMG:-}" ]]; then
  log "Executing inside Apptainer container: $APPTAINER_IMG"
  # Note: the rest of this script executes *outside* the container. Only the
  # Python entry-point is run inside to guarantee dependency reproducibility.
  APPTAINER_CMD=(apptainer exec --nv "$APPTAINER_IMG")
else
  APPTAINER_CMD=()
fi

###############################################################################
# 1.  Derive useful resource numbers
###############################################################################

# GPUs per node: prefer Slurm-provided var, fall back to nvidia-smi count
GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-0}
if [[ $GPUS_ON_NODE -eq 0 ]]; then
  GPUS_ON_NODE=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
fi

# Cap CPUs below Athena rule (reserve 2 per node)
if [[ -n "${SLURM_CPUS_ON_NODE:-}" && $SLURM_CPUS_ON_NODE -gt 2 ]]; then
  MAX_CPUS=$((SLURM_CPUS_ON_NODE - 2))
else
  MAX_CPUS=${SLURM_CPUS_ON_NODE:-1}
fi

log "Job $SLURM_JOB_ID starting on partition $SLURM_JOB_PARTITION"
log "Nodes: $SLURM_JOB_NUM_NODES, GPUs per node: $GPUS_ON_NODE, CPUs per node used by Ray: $MAX_CPUS"

###############################################################################
# 2.  Gather node list and designate head node
###############################################################################

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node=${nodes[0]}

log "Allocated nodes: ${nodes[*]}"
log "Head node: $head_node  |  Port: $PORT  |  TMP_DIR: $TMP_DIR"

###############################################################################
# 3.  Launch Ray head on the first node (this script already runs there)
###############################################################################

start_ray_head() {
  for ((i=1; i<=RETRY_COUNT; i++)); do
    log "Starting Ray head (attempt $i/$RETRY_COUNT)…"
    ray start --head --port=$PORT \
      --num-cpus=$MAX_CPUS \
      --num-gpus=$GPUS_ON_NODE \
      --temp-dir="$TMP_DIR" \
      --dashboard-host=0.0.0.0 > "$TMP_DIR/ray_head.out" 2>&1 && break
    log "Ray head failed to start, retrying in 10s…"
    sleep 10
  done
}

start_ray_worker() {
  local node=$1
  srun --nodes=1 --ntasks=1 -w "$node" bash -c "\
    for ((j=1; j<=${RETRY_COUNT}; j++)); do
      echo '[$(date '+%H:%M:%S')] Node $node: starting Ray worker (attempt '$j')';\
      ray start --address=$head_node:$PORT \
        --num-cpus=$MAX_CPUS \
        --num-gpus=$GPUS_ON_NODE \
        --temp-dir=$TMP_DIR > $TMP_DIR/ray_worker_$node.out 2>&1 && break;\
      echo 'Ray worker on $node failed, retrying in 10s…';\
      sleep 10;\
    done" &
}

start_ray_head &

# Wait a bit for head to come up before starting workers
sleep 30

###############################################################################
# 4.  Launch Ray workers on the remaining nodes (in parallel)
###############################################################################

for w in "${nodes[@]:1}"; do
  start_ray_worker "$w"
done

wait  # ensure all srun commands finished (workers keep running in background)

###############################################################################
# 5.  Execute the user-provided Python entry-point (single process)
###############################################################################

log "Running Python entry-point: $PY_ENTRYPOINT $*"
${APPTAINER_CMD[@]} python "$PY_ENTRYPOINT" --ray-address "ray://$head_node:$PORT" "$@"

###############################################################################
# 6.  Sync results back to home directory (optional)
###############################################################################

if [[ -d "$TMP_DIR/out" ]]; then
  log "Syncing results from $TMP_DIR/out to $HOME/out/$SLURM_JOB_ID";
  mkdir -p "$HOME/out/$SLURM_JOB_ID"
  rsync -aq "$TMP_DIR/out/" "$HOME/out/$SLURM_JOB_ID/"
fi

log "Job completed successfully."
