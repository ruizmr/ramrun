#!/usr/bin/env python
"""launch_job.py

Generic entry-point that connects to an (already running) Ray cluster and then
triggers a workload specified via `--job-type`.  This script is intentionally
minimal so that *any* research task can hook into it by adding a new job type
or by passing additional CLI flags.

Supported job types out-of-the-box
----------------------------------
1. puffer   – Reinforcement-learning runs backed by Pufferlib + Ray RLlib.
2. dpo      – Direct-Preference-Optimization fine-tuning using HuggingFace TRL.
3. custom   – Run an arbitrary Python module path that you provide via
              `--module your.module:entry_fn` (default).

The script auto-detects whether it is running inside a Ray cluster (e.g. when
executed by `cluster_launch.sh`) or locally.  For local debugging you can just
run `python launch_job.py --job-type custom --module myscript.py` and Ray will
fall back to `ray.init()` with `local_mode=True`.

All heavy imports (pufferlib, trl, transformers, etc.) are done lazily inside
job-specific branches so you can use this script even when only a subset of the
libraries are installed in your environment.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Callable, List, Sequence

try:
    import ray  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    print("[ERROR] Ray is required but not installed in this environment.")
    raise exc

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def resolve_callable(target: str) -> Callable[..., None]:
    """Resolve *module.path:callable_name* → Python callable.

    If only a module path is given, we attempt to treat it as a file and exec
    it, falling back to `runpy` semantics.
    """
    if ":" in target:
        module_path, attr = target.split(":", 1)
    else:
        module_path, attr = target, None

    if module_path.endswith(".py") and Path(module_path).is_file():
        # Import from a file path
        import runpy

        def _file_entry(*args, **kwargs):  # noqa: ANN001
            runpy.run_path(module_path, run_name="__main__")
        return _file_entry

    # Regular module import
    mod = importlib.import_module(module_path)
    if attr is None:
        if hasattr(mod, "main") and callable(mod.main):
            return mod.main  # type: ignore[arg-type]
        raise AttributeError(
            f"No callable specified for module '{module_path}'. Use module:func syntax."
        )
    func = getattr(mod, attr)
    if not callable(func):
        raise TypeError(f"{target} is not callable.")
    return func


# ---------------------------------------------------------------------------
# Job-specific launchers
# ---------------------------------------------------------------------------

def run_puffer(extra_args: Sequence[str]):
    """Launch a reinforcement-learning job using Pufferlib + RLlib."""
    try:
        import pufferlib  # type: ignore
        from pufferlib.frameworks.cleanrl import make_rllib_env
        from ray import air, tune  # type: ignore
    except ModuleNotFoundError as exc:
        print("[ERROR] pufferlib or Ray Tune not installed; cannot run RL job.")
        raise exc

    # Example: cartpole training; users should fork & customise
    env_creator = make_rllib_env("CartPole-v1")
    config = {
        "env": env_creator,
        "num_workers": 2,
        "framework": "torch",
        "lr": tune.grid_search([5e-4, 1e-3]),
    }
    print("[INFO] Starting Ray Tune RLlib training with config:", config)
    tuner = tune.Tuner("PPO", run_config=air.RunConfig(stop={"episodes_total": 200}), param_space=config)
    tuner.fit()


def run_dpo(extra_args: Sequence[str]):
    """Fine-tune a language model with Direct Preference Optimisation (TRL)."""
    try:
        from trl import DPOTrainer  # type: ignore
        from datasets import load_dataset  # type: ignore
        import transformers  # pylint: disable=unused-import # noqa: F401
    except ModuleNotFoundError as exc:
        print("[ERROR] HuggingFace TRL / datasets / transformers not installed; cannot run DPO job.")
        raise exc

    from ray import train  # type: ignore
    from ray.train.torch import TorchTrainer  # type: ignore

    def training_loop(config):  # noqa: ANN001
        dataset_name = config.get("dataset", "imdb")
        model_name = config.get("model", "distilbert-base-uncased")
        ds = load_dataset(dataset_name, split="train[:1%]")
        trainer = DPOTrainer(
            model_name,
            reward_model_name_or_path=model_name,
            beta=0.1,
            train_dataset=ds,
            eval_dataset=None,
        )
        trainer.train()
        trainer.save_pretrained("/tmp/model")

    scaling = train.ScalingConfig(num_workers=1, use_gpu=True)
    trainer = TorchTrainer(training_loop, scaling_config=scaling, run_config=train.RunConfig())
    trainer.fit()


def run_custom(module_target: str, extra_args: List[str]):
    callable_fn = resolve_callable(module_target)
    sig = inspect.signature(callable_fn)
    if len(sig.parameters) == 0:
        callable_fn()  # type: ignore[arg-type]
    else:
        callable_fn(extra_args)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Main – CLI parsing + Ray init
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None):  # noqa: D401
    """Entry-point for CLI usage."""
    parser = argparse.ArgumentParser("launch_job.py", add_help=True)

    parser.add_argument(
        "--ray-address",
        default="auto",
        help="Ray cluster address; keep 'auto' when launched via cluster_launch.sh",
    )
    parser.add_argument(
        "--job-type",
        choices=["puffer", "dpo", "custom"],
        default="custom",
        help="Type of workload to run.",
    )
    parser.add_argument(
        "--module",
        default="",
        help="Python module or file path to execute when --job-type=custom (format: pkg.module:callable)",
    )
    parser.add_argument(
        "--wandb-project",
        default=os.getenv("WANDB_PROJECT", "athena-research"),
        help="Weights & Biases project name (optional).",
    )

    # Catch-all for unknown flags to forward downstream
    args, extra = parser.parse_known_args(argv)

    # ---------------------------------------------------------------------
    # Ray initialisation
    # ---------------------------------------------------------------------
    if args.ray_address == "local":
        print("[INFO] Starting LOCAL Ray… (for debugging)")
        ray.init(ignore_reinit_error=True, local_mode=True)
    else:
        print(f"[INFO] Connecting to Ray cluster at {args.ray_address}…")
        ray.init(address=args.ray_address, namespace="default")

    # ---------------------------------------------------------------------
    # Optional Weights & Biases setup
    # ---------------------------------------------------------------------
    if args.wandb_project:
        try:
            import wandb  # type: ignore

            wandb.init(project=args.wandb_project)
        except ModuleNotFoundError:
            print("[WARN] wandb not installed; skipping experiment logging.")

    # ---------------------------------------------------------------------
    # Dispatch to job-specific runner
    # ---------------------------------------------------------------------
    if args.job_type == "puffer":
        run_puffer(extra)
    elif args.job_type == "dpo":
        run_dpo(extra)
    elif args.job_type == "custom":
        module_target = args.module or (extra.pop(0) if extra else "")
        if not module_target:
            parser.error("--module is required when --job-type=custom (or supply as first positional arg).")
        run_custom(module_target, list(extra))
    else:  # pragma: no cover – argparse guards against this
        raise ValueError(f"Unsupported job-type: {args.job_type}")

    print("[INFO] Job completed successfully.")


if __name__ == "__main__":
    main()

