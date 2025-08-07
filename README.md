# Multi-Node Ray Utility for VCU Athena (and other Slurm clusters)

This repository contains a **drop-in utility** that lets you launch *any* Python-based research job across **multiple nodes/GPUs** on the VCU Athena cluster while remaining Slurm-compliant and reproducible.

## Files

| File | Purpose |
|------|---------|
| `cluster_launch.sh` | Slurm batch script that allocates resources, boots a Ray head + workers, runs your Python code, and cleans up. |
| `submit_job.sh` | Lightweight wrapper around `sbatch` to make resource flags simpler. |
| `ramrun` | Single-command CLI; parses `--gpus`, `--nodes`, etc. and calls `submit_job.sh`. Place this in your `$PATH` for maximum convenience. |
| `launch_job.py` | Generic entry-point that connects to the Ray cluster and dispatches to specific workloads (`puffer`, `dpo`, or your own `custom` module). |
| `parallel_test.py` | End-to-end diagnostic that launches one Ray task **per GPU** and prints a report. |

---

## Installation (local / login node)

```bash
git clone https://github.com/ruizmr/ramrun.git
cd ramrun
pip install -r requirements.txt   # optional; or use your own conda env
# Add repo root to PATH so the `ramrun` script is discoverable
export PATH="$PWD:$PATH"          # add to ~/.bashrc for permanence
```

---

## Usage examples

```bash
# 1) Run a custom script on 1 node with 2 GPUs
ramrun mytrain.py --gpus 2 --nodes 1 -- --epochs 10 --lr 3e-4

# 2) Same but longer time limit and V100 partition
ramrun mytrain.py --gpus 4 --nodes 2 --partition gpu-v100 --time 72:00:00 -- --batch 64

# 3) Launch built-in Pufferlib RL demo (4 A100 GPUs)
ramrun --job-type puffer --gpus 4 --env CartPole-v1 --timesteps 1e6

# 4) Prove multi-node GPU functionality (diagnostic)
ramrun parallel_test.py --gpus 8 --nodes 2 --partition gpu-v100
```

Notes:
* Put `--` before arguments meant for *your* Python script so `ramrun` stops parsing.
* If `--job-type` is anything other than `custom` you don’t need to provide a target script; `launch_job.py` knows how to launch it.

---

## How it works in one sentence
`ramrun` → `submit_job.sh` → `sbatch cluster_launch.sh` → Ray cluster across nodes → `launch_job.py` → your code.

Everything else (clean-up, object-store caching, rsync-out) is automatic.

---
