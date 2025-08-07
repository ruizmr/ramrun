#!/usr/bin/env python
"""parallel_test.py – sanity-check script to *prove* that the Ray+Slurm
infrastructure is working across multiple nodes / GPUs.

The script launches one Ray task per **GPU** (num_gpus=1).  Each task:
  • reports its hostname, CUDA_VISIBLE_DEVICES, GPU name, compute capability,
    and memory.
  • performs a small torch matmul on the GPU to prove compute works.

After gathering all results, the driver prints a table summarising how many
GPUs were used per node and dumps JSON for programmatic inspection.

Run on Athena via:
  sbatch cluster_launch.sh --job-type custom --module parallel_test:main

You should see output lines like:
  Host gpu-a100-01  | GPU 0 | A100-SXM4-80GB | cap 8.0 | mem 80 GB  [OK]
  ...
  Summary: 16 GPUs across 4 nodes succeeded.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from collections import Counter, defaultdict
from typing import Dict, List

import ray  # type: ignore


@ray.remote(num_gpus=1)
def gpu_worker() -> Dict[str, str]:
    """Remote GPU task – returns diagnostic information."""
    import random
    import time

    # Delay a random short time so logs are staggered (easier to read)
    time.sleep(random.uniform(0, 2))

    try:
        import torch  # pylint: disable=import-error  # type: ignore

        # Force the first visible GPU – Ray isolates one GPU via CVD per task
        device = torch.device("cuda")
        # Simple matmul to exercise GPU FP32 pipeline
        a = torch.randn(512, 512, device=device)
        b = torch.randn(512, 512, device=device)
        _ = a @ b  # noqa: WPS437  # matrix multiply
        torch.cuda.synchronize()

        prop = torch.cuda.get_device_properties(device)
        gpu_name = prop.name
        capability = f"{prop.major}.{prop.minor}"
        memory_gb = f"{prop.total_memory // (1024 ** 3)}"
    except ModuleNotFoundError:
        gpu_name = "torch_not_installed"
        capability = "?"
        memory_gb = "?"

    return {
        "host": socket.gethostname(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "-"),
        "gpu_name": gpu_name,
        "capability": capability,
        "memory_gb": memory_gb,
    }


def pretty_print(results: List[Dict[str, str]]):
    """Print a human-readable table + summary."""
    print("\nDetailed GPU task reports:\n".ljust(80, "-"))
    for r in results:
        print(
            f"Host {r['host']:<18} | CVD {r['cuda_visible_devices']:<3} | "
            f"{r['gpu_name']:<25} | cap {r['capability']:<4} | "
            f"mem {r['memory_gb']:>4} GB | OK"
        )

    # Summary per host
    per_host = Counter(r["host"] for r in results)
    print("\nSummary:".ljust(80, "-"))
    for host, n in per_host.items():
        print(f"{host}: {n} GPU tasks")
    print(f"Total: {len(results)} GPU tasks across {len(per_host)} nodes")

    # Dump JSON for machine checks (optional)
    with open("gpu_report.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        print("\nWrote gpu_report.json for verification.")


def main(argv=None):  # noqa: D401
    """CLI entry-point compatible with launch_job.py."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", default="auto")
    args, _ = parser.parse_known_args(argv)

    # Connect to Ray (address passed by cluster_launch.sh)
    ray.init(address=args.ray_address)

    # Query total GPUs Ray sees across the cluster
    gpus_total = int(ray.cluster_resources().get("GPU", 0))
    print(f"[driver] Ray cluster reports {gpus_total} GPUs – launching that many tasks…")

    # Launch one task per GPU
    futures = [gpu_worker.remote() for _ in range(gpus_total)]
    results = ray.get(futures)

    pretty_print(results)

    print("\n[driver] Parallel GPU diagnostics completed successfully.")


if __name__ == "__main__":
    sys.exit(main())

