"""Portable formal versioned FFT launcher (legacy module name). Defaults to a read-only plan.

The explicit train action runs the existing Stage-1 preflight and stock
trainer with synchronous checkpoint I/O hooks. This module does not import torch, initialize CUDA, or train on import.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from scripts.biyi import checkpoint_store as store
from scripts.biyi import training_runtime as runtime
from scripts.biyi.checkpoint_schedule import parse_checkpoint_steps


def build_plan(root, revision, settings, *, resume=False, use_environment_paths=False, dataset_root=None):
    root = runtime.artifact_root(root) if os.environ.get("BIYI_STORAGE_MODE") == "backup" else store.storage_root(root)
    store.require(store.SHA.fullmatch(revision), "Use a full immutable dataset revision")
    if settings.get("dataset_revision"):
        store.require(revision == settings["dataset_revision"], "Dataset revision differs from frozen recipe")
    experiment = settings["experiment"]
    dataset = root / "datasets" / revision
    run_dir = root / "checkpoints/pi05_biyi_stage1" / experiment
    base = os.environ.get("BIYI_BASE_CHECKPOINT_DIR", "")
    store.require(base and Path(base).is_absolute(), "Set BIYI_BASE_CHECKPOINT_DIR to the verified base weights")
    environment = {
        "BIYI_DATASET_REPO_ID": store.DATASET_REPO,
        "BIYI_DATASET_LOCAL_ROOT": str(dataset),
        "BIYI_TRAIN_EPISODES": ",".join(map(str, range(settings["dataset_episodes"]))),
        "BIYI_ASSETS_BASE_DIR": str(root / "assets"),
        "BIYI_CHECKPOINT_BASE_DIR": str(root / "checkpoints"),
        "BIYI_BASE_CHECKPOINT_DIR": base,
        "BIYI_EXP_NAME": experiment,
        "BIYI_HF_CHECKPOINT_REPO_ID": store.MODEL_REPO,
        "BIYI_DATASET_REVISION": revision,
    }
    if use_environment_paths:
        for key in (
            "BIYI_DATASET_REPO_ID",
            "BIYI_DATASET_LOCAL_ROOT",
            "BIYI_ASSETS_BASE_DIR",
            "BIYI_CHECKPOINT_BASE_DIR",
        ):
            if os.environ.get(key):
                environment[key] = os.environ[key]
        store.require(environment["BIYI_DATASET_REPO_ID"] == store.DATASET_REPO, "Formal dataset repo changed")
        run_dir = runtime.artifact_root(environment["BIYI_CHECKPOINT_BASE_DIR"]) / "pi05_biyi_stage1" / experiment
    if dataset_root is not None:
        environment["BIYI_DATASET_LOCAL_ROOT"] = str(Path(dataset_root).expanduser().resolve())
    environment["BIYI_CHECKPOINT_IO"] = "1"
    environment["BIYI_MEMORY_SMOKE"] = "0"
    environment["PYTHONPATH"] = str(store.REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    environment["BIYI_DDP_TIMEOUT_SECONDS"] = os.environ.get("BIYI_DDP_TIMEOUT_SECONDS", "3600")
    steps = settings["num_train_steps"]
    warmup = (steps + 5) // 10
    preflight = [
        sys.executable,
        "-m",
        "scripts.biyi.preflight_pi05_stage1",
        "--formal",
        "--wandb-enabled",
        "--project-name",
        "BiyiVLA-Stage1",
        "--exp-name",
        experiment,
        "--num-train-steps",
        str(steps),
        "--warmup-steps",
        str(warmup),
        "--decay-steps",
        str(steps),
        "--peak-lr",
        "2.5e-5",
        "--end-lr",
        "2.5e-6",
    ]
    count = runtime.gpu_count(settings["num_gpus"])
    runtime.validate_batch(settings["batch_size"], count)
    launcher = runtime.launch_prefix(count, torchrun=os.environ.get("BIYI_TORCHRUN"))
    command = [
        *launcher,
        "scripts/train_pytorch.py",
        "pi05_biyi_stage1",
        "--exp-name",
        experiment,
        "--batch-size",
        str(settings["batch_size"]),
        "--num-workers",
        str(settings["num_workers"]),
        "--num-train-steps",
        str(steps),
        "--save-interval",
        str(settings["save_interval"]),
        "--keep-period",
        str(settings.get("keep_period", 5000)),
        "--log-interval",
        str(settings.get("log_interval", 5)),
        "--lr-schedule.warmup-steps",
        str(warmup),
        "--lr-schedule.decay-steps",
        str(steps),
        "--lr-schedule.peak-lr",
        "2.5e-5",
        "--lr-schedule.decay-lr",
        "2.5e-6",
        "--assets-base-dir",
        environment["BIYI_ASSETS_BASE_DIR"],
        "--checkpoint-base-dir",
        environment["BIYI_CHECKPOINT_BASE_DIR"],
        "--wandb-enabled",
        "--project-name",
        "BiyiVLA-Stage1",
    ]
    if settings.get("checkpoint_steps") is not None:
        command.extend(["--checkpoint-steps", *map(str, settings["checkpoint_steps"])])
    if resume:
        command.append("--resume")
    return {
        "environment": environment,
        "preflight": preflight,
        "train": command,
        "run_dir": str(run_dir),
        "dataset_revision": revision,
        "resume": resume,
        "recipe": settings,
        "storage": {
            "mode": os.environ.get("BIYI_STORAGE_MODE", "UNSET"),
            "backup_root": os.environ.get("BIYI_BACKUP_ROOT"),
        },
    }


def validate_storage(plan, settings):
    store.require(not store.git("status", "--porcelain"), "Formal runs require a clean committed source tree")
    env = plan["environment"]
    dataset = store.dataset_identity(
        Path(env["BIYI_DATASET_LOCAL_ROOT"]), plan["dataset_revision"], settings["dataset_episodes"]
    )
    if settings.get("dataset_frames"):
        info = store.read_json(Path(env["BIYI_DATASET_LOCAL_ROOT"]) / "meta/info.json")
        store.require(info["total_frames"] == settings["dataset_frames"], "Frozen frame count changed")
    stats_dir = Path(env["BIYI_ASSETS_BASE_DIR"]) / "pi05_biyi_stage1/biyi_stage1"
    stats_path = store.file_at(stats_dir, "norm_stats.json")
    stats_manifest = store.read_json(store.file_at(stats_dir, "stage1_train_stats_manifest.json"))
    store.require(stats_manifest["norm_stats_sha256"] == store.digest(stats_path), "Stats checksum mismatch")
    store.require(
        stats_manifest["repo_id"] == store.DATASET_REPO
        and stats_manifest["train_episode_indices"] == list(range(settings["dataset_episodes"])),
        "Stats are not from the frozen recipe train episodes",
    )
    for name, expected in stats_manifest["metadata_sha256"].items():
        store.require(
            store.digest(store.file_at(Path(env["BIYI_DATASET_LOCAL_ROOT"]) / "meta", name)) == expected,
            "Stats source metadata changed",
        )
    base = store.file_at(env["BIYI_BASE_CHECKPOINT_DIR"], "model.safetensors")
    record = {
        "schema_version": 1,
        "kind": f"formal_{settings['dataset_episodes']}ep_fft",
        "source_commit": store.git("rev-parse", "HEAD"),
        "recipe": settings,
        "dataset": dataset,
        "norm_stats_sha256": store.digest(stats_path),
        "base_model_sha256": store.digest(base),
        "uv_lock_sha256": store.digest(store.REPO_ROOT / "uv.lock"),
    }
    run_dir = Path(plan["run_dir"])
    if plan["resume"]:
        old = store.read_json(store.file_at(run_dir, "run_manifest.json"))
        store.require(old == record, "Resume source/recipe/data/base/stats differ from the frozen run")
        store.file_at(run_dir, "wandb_id.txt")
        steps = [int(p.name) for p in run_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        store.require(steps, "No completed checkpoint to resume")
        store.seal_checkpoint(run_dir, max(steps))
    else:
        store.require(not run_dir.exists(), "Run already exists; use --resume or a new versioned recipe")
    return record, stats_dir


def resolve_dataset_settings(settings, dataset_root, revision, *, experiment=None):
    """Resolve this run from the supplied snapshot, not historical recipe counts."""
    settings = {**settings, **store.dataset_metadata(dataset_root)}
    settings["dataset_revision"] = revision
    settings["dataset_from_metadata"] = True
    if experiment is not None:
        settings["experiment"] = experiment
    else:
        settings["experiment"] = re.sub(
            r"(?<=pi05_biyi_)[0-9]+(?=ep_fft_v)", str(settings["dataset_episodes"]), settings["experiment"]
        )
    return store.validate_recipe(settings)


def main(default_recipe=store.RECIPE):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "preflight", "train"), nargs="?", default="plan")
    parser.add_argument(
        "--persistent-root", default=os.environ.get("BIYI_RUN_ROOT") or os.environ.get("BIYI_PERSISTENT_ROOT")
    )
    parser.add_argument(
        "--from-stage1-env", action="store_true", help="Honor existing Stage-1 path and recipe environment variables"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print a read-only plan; no preflight, CUDA or storage writes"
    )
    parser.add_argument("--num-gpus", help="Positive count or auto; defaults to the frozen recipe")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dataset-root", type=Path, help="Frozen LeRobot snapshot; counts are read from its metadata")
    parser.add_argument("--exp-name", help="New versioned formal experiment name")
    parser.add_argument("--num-train-steps", type=int)
    parser.add_argument(
        "--checkpoint_step",
        "--checkpoint-step",
        "--checkpoint-steps",
        dest="checkpoint_steps",
        nargs="+",
        help="Explicit save steps, separated by spaces and/or commas; final step is always saved",
    )
    parser.add_argument(
        "--dataset-revision",
        default=os.environ.get("BIYI_DATASET_REVISION"),
        required=not os.environ.get("BIYI_DATASET_REVISION"),
    )
    parser.add_argument("--recipe", type=Path, default=default_recipe)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    settings = store.recipe(args.recipe)
    experiment = args.exp_name
    if args.from_stage1_env:
        for key, variable in {
            "experiment": "BIYI_EXP_NAME",
            "num_train_steps": "BIYI_NUM_TRAIN_STEPS",
            "batch_size": "BIYI_BATCH_SIZE",
            "num_workers": "BIYI_NUM_WORKERS",
            "save_interval": "BIYI_SAVE_INTERVAL",
            "num_gpus": "BIYI_NUM_GPUS",
            "keep_period": "BIYI_KEEP_PERIOD",
            "log_interval": "BIYI_LOG_INTERVAL",
        }.items():
            if os.environ.get(variable):
                value = os.environ[variable]
                store.require(
                    not (args.dry_run or args.action == "plan") or value != "auto",
                    "Dry-run requires explicit GPU count",
                )
                settings[key] = (
                    value if key == "experiment" else (runtime.gpu_count(value) if key == "num_gpus" else int(value))
                )
        experiment = args.exp_name or os.environ.get("BIYI_EXP_NAME")
    if args.num_gpus is not None:
        store.require(
            not (args.dry_run or args.action == "plan") or args.num_gpus != "auto",
            "Dry-run requires explicit GPU count",
        )
        settings["num_gpus"] = runtime.gpu_count(args.num_gpus)
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    if args.num_train_steps is not None:
        settings["num_train_steps"] = args.num_train_steps
    checkpoint_steps = args.checkpoint_steps
    if checkpoint_steps is None and args.from_stage1_env:
        checkpoint_steps = os.environ.get("BIYI_CHECKPOINT_STEPS")
    if checkpoint_steps is not None:
        settings["checkpoint_steps"] = list(parse_checkpoint_steps(checkpoint_steps, settings["num_train_steps"]))
    root = args.persistent_root
    if root is None and args.from_stage1_env:
        root = str(Path(os.environ["BIYI_CHECKPOINT_BASE_DIR"]).parent)
    store.require(root, "Set BIYI_PERSISTENT_ROOT or BIYI_RUN_ROOT")
    dataset_root = args.dataset_root
    if dataset_root is None and args.from_stage1_env:
        dataset_root = os.environ.get("BIYI_DATASET_LOCAL_ROOT")
    if dataset_root is None:
        dataset_root = Path(root) / "datasets" / args.dataset_revision
    plan_only = args.action == "plan" or args.dry_run
    if not plan_only:
        runtime.gpu_count(settings["num_gpus"], verify=True)
        checkpoint_root = (os.environ.get("BIYI_CHECKPOINT_BASE_DIR") if args.from_stage1_env else None) or str(
            Path(root) / "checkpoints"
        )
        policy = runtime.storage_policy(checkpoint_root, probe=True)
    metadata_present = any((Path(dataset_root) / "meta" / name).exists() for name in ("info.json", "episodes.jsonl"))
    # Preserve old filesystem-free command previews when no snapshot was supplied.
    # An explicit --dataset-root or an actual run always requires valid metadata.
    unresolved_preview = plan_only and args.dataset_root is None and not metadata_present
    if unresolved_preview:
        if experiment is not None:
            settings["experiment"] = experiment
        store.validate_recipe(settings)
    else:
        settings = resolve_dataset_settings(settings, dataset_root, args.dataset_revision, experiment=experiment)
    plan = build_plan(
        root,
        args.dataset_revision,
        settings,
        resume=args.resume,
        use_environment_paths=args.from_stage1_env,
        dataset_root=dataset_root,
    )
    plan["dataset_metadata_resolved"] = not unresolved_preview
    if plan_only:
        print(json.dumps(plan, indent=2))
        return
    record, stats_dir = validate_storage(plan, settings)
    environment = {**os.environ, **plan["environment"], "BIYI_STORAGE_DEVICE": policy["device"]}
    runtime.check_memory_smoke(settings["batch_size"], settings["num_gpus"], environment=environment)
    subprocess.run(plan["preflight"], cwd=store.REPO_ROOT, env=environment, check=True)
    if args.action == "preflight":
        print("STORAGE AND EXISTING FORMAL PREFLIGHT PASS; training was not started")
        return
    run_dir = Path(plan["run_dir"])
    if not args.resume:
        run_dir.mkdir(parents=True, exist_ok=False)
        store.write_json(run_dir / "run_manifest.json", record)
        shutil.copyfile(stats_dir / "stage1_train_stats_manifest.json", run_dir / "train_stats_manifest.json")
    if args.resume:
        steps = [int(p.name) for p in run_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        runtime.backup_checkpoint(run_dir, max(steps), environment=environment)
    subprocess.run(plan["train"], cwd=store.REPO_ROOT, env=environment, check=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAILED: {error}" if type(error) is ValueError else f"FAILED: {type(error).__name__}")
        raise SystemExit(1) from None
