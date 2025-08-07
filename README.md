# Multi-Node Ray Utility for VCU Athena (and other Slurm clusters)

This repository contains a **drop-in utility** that lets you launch *any* Python-based research job across **multiple nodes/GPUs** on the VCU Athena cluster while remaining Slurm-compliant and reproducible.

## Files

| File | Purpose |
|------|---------|
| `cluster_launch.sh` | Slurm batch script that allocates resources, boots a Ray head + workers, runs your Python code, and cleans up. |
| `launch_job.py` | Generic entry-point that connects to the Ray cluster and dispatches to specific workloads (`puffer`, `dpo`, or your own `custom` module). |
| `parallel_test.py` | End-to-end diagnostic that launches one Ray task **per GPU** and prints a report. Use it to prove the whole stack works. |
| `submit_job.sh` | *Ergonomic wrapper*: lets you specify nodes/GPUs/partition with simple flags, then calls `sbatch cluster_launch.sh` for you. |

> **Why Ray?**  Using Ray hides multi-node details from your training code, gives you an object store across nodes, and pairs nicely with RLlib, Tune, and HuggingFace TRL.

---

## Quick Start

### 0  Environment (one-time)
```bash
module load Miniconda3        # or Miniforge3
conda create -n athena-env python=3.10 -y
conda activate athena-env
pip install "ray[default]" torch pufferlib trl transformers datasets wandb
chmod +x cluster_launch.sh submit_job.sh
```

### 1  Prove the pipeline with the **parallel test** (wrapper version)
```bash
./submit_job.sh --nodes 4 --gpus 4 --partition gpu-a100 \
    --job-type custom --module parallel_test:main
```
This requests 4 A100 nodes × 4 GPUs each and runs the test.

You can still use raw `sbatch` if you prefer (see earlier README revisions).

### 2  Launch RL or DPO with minimal typing
```bash
# RL example, 2 nodes × 4 GPUs
./submit_job.sh --nodes 2 --gpus 4 --partition gpu-a100 \
    --job-type puffer --env CartPole-v1 --timesteps 1e6

# DPO example on V100 nodes, 8 GPUs each
./submit_job.sh --nodes 1 --gpus 8 --partition gpu-v100 \
    --job-type dpo --model mistralai/Mixtral-8x22B --dataset imdb
```

All flags **after** the resource options are forwarded untouched to `cluster_launch.sh` / `launch_job.py`.

---

## GPU Normalisation / Heterogeneous Clusters
[unchanged section ...]

---
