"""Portable formal 100-episode FFT launcher. Defaults to a read-only plan.

The explicit train action runs the existing Stage-1 preflight and unmodified stock
trainer. This module does not import torch, initialize CUDA, or train on import.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.biyi import checkpoint_store as store


def build_plan(root, revision, settings, *, resume=False):
    root = store.storage_root(root)
    store.require(store.SHA.fullmatch(revision), "Use a full immutable dataset revision")
    experiment = settings["experiment"]
    dataset = root / "datasets" / revision
    run_dir = root / "checkpoints/pi05_biyi_stage1" / experiment
    base = os.environ.get("BIYI_BASE_CHECKPOINT_DIR", "")
    store.require(base and Path(base).is_absolute(), "Set BIYI_BASE_CHECKPOINT_DIR to the verified base weights")
    environment = {
        "BIYI_DATASET_REPO_ID": store.DATASET_REPO,
        "BIYI_DATASET_LOCAL_ROOT": str(dataset),
        "BIYI_TRAIN_EPISODES": ",".join(map(str, range(100))),
        "BIYI_ASSETS_BASE_DIR": str(root / "assets"),
        "BIYI_CHECKPOINT_BASE_DIR": str(root / "checkpoints"),
        "BIYI_BASE_CHECKPOINT_DIR": base,
        "BIYI_EXP_NAME": experiment,
        "BIYI_HF_CHECKPOINT_REPO_ID": store.MODEL_REPO,
    }
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
    launcher = (
        [sys.executable]
        if settings["num_gpus"] == 1
        else [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={settings['num_gpus']}",
        ]
    )
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
    if resume:
        command.append("--resume")
    return {
        "environment": environment,
        "preflight": preflight,
        "train": command,
        "run_dir": str(run_dir),
        "dataset_revision": revision,
        "resume": resume,
    }


def validate_storage(plan, settings):
    store.require(not store.git("status", "--porcelain"), "Formal runs require a clean committed source tree")
    env = plan["environment"]
    dataset = store.dataset_identity(Path(env["BIYI_DATASET_LOCAL_ROOT"]), plan["dataset_revision"])
    stats_dir = Path(env["BIYI_ASSETS_BASE_DIR"]) / "pi05_biyi_stage1/biyi_stage1"
    stats_path = store.file_at(stats_dir, "norm_stats.json")
    stats_manifest = store.read_json(store.file_at(stats_dir, "stage1_train_stats_manifest.json"))
    store.require(stats_manifest["norm_stats_sha256"] == store.digest(stats_path), "Stats checksum mismatch")
    store.require(
        stats_manifest["repo_id"] == store.DATASET_REPO and stats_manifest["train_episode_indices"] == list(range(100)),
        "Stats are not from the 100 train episodes",
    )
    for name, expected in stats_manifest["metadata_sha256"].items():
        store.require(
            store.digest(store.file_at(Path(env["BIYI_DATASET_LOCAL_ROOT"]) / "meta", name)) == expected,
            "Stats source metadata changed",
        )
    base = store.file_at(env["BIYI_BASE_CHECKPOINT_DIR"], "model.safetensors")
    record = {
        "schema_version": 1,
        "kind": "formal_100ep_fft",
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "preflight", "train"), nargs="?", default="plan")
    parser.add_argument(
        "--persistent-root",
        default=os.environ.get("BIYI_PERSISTENT_ROOT"),
        required=not os.environ.get("BIYI_PERSISTENT_ROOT"),
    )
    parser.add_argument(
        "--dataset-revision",
        default=os.environ.get("BIYI_DATASET_REVISION"),
        required=not os.environ.get("BIYI_DATASET_REVISION"),
    )
    parser.add_argument("--recipe", type=Path, default=store.RECIPE)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    settings = store.recipe(args.recipe)
    plan = build_plan(args.persistent_root, args.dataset_revision, settings, resume=args.resume)
    if args.action == "plan":
        print(json.dumps(plan, indent=2))
        return
    record, stats_dir = validate_storage(plan, settings)
    environment = {**os.environ, **plan["environment"]}
    subprocess.run(plan["preflight"], cwd=store.REPO_ROOT, env=environment, check=True)
    if args.action == "preflight":
        print("STORAGE AND EXISTING FORMAL PREFLIGHT PASS; training was not started")
        return
    run_dir = Path(plan["run_dir"])
    if not args.resume:
        run_dir.mkdir(parents=True, exist_ok=False)
        store.write_json(run_dir / "run_manifest.json", record)
        shutil.copyfile(stats_dir / "stage1_train_stats_manifest.json", run_dir / "train_stats_manifest.json")
    subprocess.run(plan["train"], cwd=store.REPO_ROOT, env=environment, check=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAILED: {error}" if type(error) is ValueError else f"FAILED: {type(error).__name__}")
        raise SystemExit(1) from None
