#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# submit_job.sh – ergonomic wrapper around `sbatch cluster_launch.sh`
# -----------------------------------------------------------------------------
# This script lets you specify common Slurm resources via simple CLI flags and
# forwards *all remaining arguments* to the underlying batch script exactly as
# you would normally supply them (e.g. `--job-type puffer`).
#
# Example:
#   ./submit_job.sh --nodes 2 --gpus 8 --partition gpu-v100 --time 12:00:00 \
#       --job-type custom --module parallel_test:main
#
# The command above expands internally to:
#   sbatch --nodes=2 --gres=gpu:8 --partition=gpu-v100 --time=12:00:00 \
#          cluster_launch.sh --job-type custom --module parallel_test:main
# -----------------------------------------------------------------------------
set -euo pipefail

# Defaults (edit to taste)
PARTITION="gpu-a100"
NODES=1
GPUS_PER_NODE=1
TIME="48:00:00"  # hh:mm:ss

# ---------------------------------------------------------------------------
# Parse our wrapper flags until we hit "--" or a flag that belongs to the
# batch script (starts with --job- or --module etc.).  We stop parsing when we
# encounter the first unknown flag so that users can write positional args or
# any other flags freely after ours.
# ---------------------------------------------------------------------------
PARSED=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition)
      PARTITION="$2"; shift 2;;
    --nodes)
      NODES="$2"; shift 2;;
    --gpus|--gpus-per-node)
      GPUS_PER_NODE="$2"; shift 2;;
    --time)
      TIME="$2"; shift 2;;
    --help)
      echo "Usage: $0 [--nodes N] [--gpus-per-node G] [--partition P] [--time HH:MM:SS] -- [cluster_launch args...]";
      exit 0;;
    --*)
      # Flag belongs to cluster_launch.sh; stop parsing
      break;;
    *)
      break;;
  esac
done

# Remaining args passed verbatim to cluster_launch.sh
CL_ARGS=("$@")

echo "[submit_job] Requesting $NODES node(s), $GPUS_PER_NODE GPU(s)/node on partition $PARTITION for $TIME"

sbatch --nodes="$NODES" \
       --gres="gpu:$GPUS_PER_NODE" \
       --partition="$PARTITION" \
       --time="$TIME" \
       cluster_launch.sh "${CL_ARGS[@]}"

