"""Shared single-host launch and durable checkpoint I/O; no torch import on dry-run."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from scripts.biyi import checkpoint_store as store


def artifact_root(value):
    path = Path(value).expanduser()
    store.require(path.is_absolute(), "Artifact paths must be absolute")
    path = path.resolve()
    store.require(path != store.REPO_ROOT and store.REPO_ROOT not in path.parents, "Artifacts must be outside Git")
    return path


def gpu_count(request="auto", *, verify=False):
    store.require(
        str(request) == "auto" or (str(request).isdigit() and int(request) > 0), "GPU count must be auto or positive"
    )
    if str(request) == "auto" or verify:
        # Only runtime execution probes visibility; plans with explicit counts are CPU-only.
        import torch

        visible = torch.cuda.device_count()
        count = visible if str(request) == "auto" else int(request)
        store.require(0 < count <= visible, f"Requested {count} GPUs, but {visible} are visible")
        return count
    return int(request)


def launch_prefix(count, *, python=None, torchrun=None):
    python = python or sys.executable
    if count == 1:
        return [python]
    runner = [torchrun] if torchrun else [python, "-m", "torch.distributed.run"]
    return [*runner, "--standalone", "--nnodes=1", f"--nproc_per_node={count}"]


def validate_batch(batch, count):
    store.require(
        type(batch) is int and batch > 0 and batch % count == 0,
        "Global batch must be positive and divisible by GPU count",
    )


def storage_policy(checkpoint_base, *, environment=None, probe=False):
    env = os.environ if environment is None else environment
    base = artifact_root(checkpoint_base)
    mode = env.get("BIYI_STORAGE_MODE")
    store.require(
        mode in ("persistent", "backup"),
        "Set BIYI_STORAGE_MODE=persistent or backup after verifying the mount lifetime",
    )
    if mode == "persistent":
        destination = store.storage_root(base)
    else:
        store.require(
            env.get("BIYI_BACKUP_ROOT"), "Temporary runs require BIYI_BACKUP_ROOT on independent persistent storage"
        )
        destination = store.storage_root(env["BIYI_BACKUP_ROOT"])
        store.require(
            base != destination and base not in destination.parents and destination not in base.parents,
            "Checkpoint and backup roots must be independent, non-overlapping paths",
        )
    expected_device = env.get("BIYI_STORAGE_DEVICE")
    if expected_device is not None:
        store.require(
            destination.is_dir() and str(destination.stat().st_dev) == expected_device,
            "Persistent mount disappeared or changed",
        )
    # A path name cannot prove persistence. The operator declares the mount lifetime;
    # this probe proves write/read/copy integrity, not survival of provider deletion.
    if probe:
        base.mkdir(parents=True, exist_ok=True)
        destination.mkdir(parents=True, exist_ok=True)
        with (
            tempfile.TemporaryDirectory(dir=base, prefix=".storage-probe-") as source,
            tempfile.TemporaryDirectory(dir=destination, prefix=".storage-probe-") as target,
        ):
            sample = Path(source) / "probe"
            sample.write_bytes(os.urandom(4096))
            copied = Path(target) / "probe"
            shutil.copyfile(sample, copied)
            with copied.open("rb") as stream:
                os.fsync(stream.fileno())
            store.require(store.digest(sample) == store.digest(copied), "Storage roundtrip failed")
    return {
        "mode": mode,
        "checkpoint_base": str(base),
        "destination": str(destination),
        "device": str(destination.stat().st_dev) if destination.exists() else None,
    }


def backup_checkpoint(run_dir, step, *, environment=None):
    """Synchronously copy a completed save. Failure propagates to the trainer."""
    run_dir = Path(run_dir).resolve()
    policy = storage_policy(run_dir.parent.parent, environment=environment)
    if policy["mode"] != "backup":
        return None
    # Deliberate allowlist: never copy credentials, logs, caches or temporary saves.
    names = ["run_manifest.json", "train_stats_manifest.json", "wandb_id.txt"]
    names += [f"{step}/{name}" for name in ("model.safetensors", "optimizer.pt", "metadata.pt", store.NORM)]
    files = store.inventory(run_dir, names)
    record = store.read_json(run_dir / "run_manifest.json")
    if record.get("kind") in store.FORMAL_KINDS:
        store.seal_checkpoint(run_dir, step)
        names += [f"{step}/checkpoint_manifest.json"]
        files = store.inventory(run_dir, names)
    receipt = {"schema_version": 1, "experiment": run_dir.name, "step": step, "files": files}
    target = Path(policy["destination"]) / run_dir.parent.name / run_dir.name / f"step-{step}"
    if target.exists():
        store.require(
            store.read_json(target / "backup_manifest.json") == receipt, "Existing backup differs; never overwrite"
        )
        store.check_inventory(target, files)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".backup-") as temporary:
        staging = Path(temporary) / "payload"
        for name in names:
            output = staging / name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(store.file_at(run_dir, name), output)
            with output.open("rb") as stream:
                os.fsync(stream.fileno())
        store.write_json(staging / "backup_manifest.json", receipt)
        with (staging / "backup_manifest.json").open("rb") as stream:
            os.fsync(stream.fileno())
        store.check_inventory(staging, files)
        staging.rename(target)
    # Read back the published copy before reporting save success.
    store.check_inventory(target, files)
    print(f"DURABLE BACKUP VERIFIED: {target}", flush=True)
    return target


def restore_backup(backup, run_dir, *, assets_base=None, dataset_root=None):
    """Restore a verified complete save to a new run path, without unpickling."""
    backup, run_dir = Path(backup), Path(run_dir)
    receipt = store.read_json(store.file_at(backup, "backup_manifest.json"))
    step = receipt["step"]
    store.require(type(step) is int and step > 0, "Invalid backup step")
    required = {"run_manifest.json", "train_stats_manifest.json", "wandb_id.txt"}
    required.update(f"{step}/{name}" for name in ("model.safetensors", "optimizer.pt", "metadata.pt", store.NORM))
    allowed = required | {f"{step}/checkpoint_manifest.json"}
    store.require(required <= set(receipt["files"]) <= allowed, "Incomplete or unexpected backup files")
    store.require(run_dir.name == receipt["experiment"], "Restore experiment must match")
    store.check_inventory(backup, receipt["files"])
    store.require(not run_dir.exists(), "Restore never overwrites an existing run")
    store.require(bool(assets_base) == bool(dataset_root), "Restore assets and dataset paths together")
    if assets_base:
        record = store.read_json(backup / "run_manifest.json")
        store.validate_run(record)
        dataset_root = artifact_root(dataset_root)
        store.require(
            store.dataset_identity(dataset_root, record["dataset"]["revision"], record["recipe"]["dataset_episodes"]) == record["dataset"],
            "Restore frozen dataset first",
        )
        norm = store.file_at(backup, f"{step}/{store.NORM}")
        provenance = store.read_json(backup / "train_stats_manifest.json")
        store.require(
            store.digest(norm) == record["norm_stats_sha256"] == provenance["norm_stats_sha256"],
            "Restore norm stats mismatch",
        )
        rebound = {**provenance, "local_root": str(dataset_root)}
        stats = artifact_root(assets_base) / "pi05_biyi_stage1/biyi_stage1"
        stats_path, manifest_path = stats / "norm_stats.json", stats / "stage1_train_stats_manifest.json"
        if stats_path.exists():
            store.require(store.digest(stats_path) == store.digest(norm), "Existing norm stats conflict")
        if manifest_path.exists():
            store.require(store.read_json(manifest_path) == rebound, "Existing stats fingerprint conflicts")
        stats.mkdir(parents=True, exist_ok=True)
        if not stats_path.exists():
            shutil.copyfile(norm, stats_path)
        if not manifest_path.exists():
            store.write_json(manifest_path, rebound)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=run_dir.parent, prefix=".restore-") as temporary:
        staging = Path(temporary) / "payload"
        for name in receipt["files"]:
            output = staging / store.safe_relative(name)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(store.file_at(backup, name), output)
        store.check_inventory(staging, receipt["files"])
        staging.rename(run_dir)
    return run_dir


def smoke_signature(batch, count, *, environment=None):
    env = os.environ if environment is None else environment
    data = Path(env["BIYI_DATASET_LOCAL_ROOT"])
    stats = Path(env["BIYI_ASSETS_BASE_DIR"]) / "pi05_biyi_stage1/biyi_stage1/norm_stats.json"
    return {
        "source_commit": store.git("rev-parse", "HEAD"),
        "batch_size": batch,
        "num_gpus": count,
        "dataset_repo": env["BIYI_DATASET_REPO_ID"],
        "dataset_revision": env.get("BIYI_DATASET_REVISION"),
        "train_episodes": env.get("BIYI_TRAIN_EPISODES", ""),
        "metadata": store.inventory(data / "meta", ["info.json", "episodes.jsonl", "tasks.jsonl"]),
        "norm_stats_sha256": store.digest(stats),
    }


def record_memory_smoke(config, device, step):
    if os.environ.get("BIYI_MEMORY_SMOKE") != "1":
        return
    import torch

    count = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    props = torch.cuda.get_device_properties(device)
    store.require(step >= 2 and not config.wandb_enabled, "Memory smoke must complete at least two W&B-disabled steps")
    record = {
        "signature": smoke_signature(config.batch_size, count),
        "completed_steps": step,
        "gpu_name": props.name,
        "total_memory": props.total_memory,
        "peak_allocated": torch.cuda.max_memory_allocated(device),
        "peak_reserved": torch.cuda.max_memory_reserved(device),
    }
    path = config.checkpoint_dir / f"memory_smoke_rank{rank}.json"
    temp = path.with_suffix(".tmp")
    store.write_json(temp, record)
    temp.replace(path)


def check_memory_smoke(batch, count, *, environment=None):
    """Require completed smoke evidence for the effective GPUs, batch, data and stats."""
    import torch

    env = os.environ if environment is None else environment
    root = env.get("BIYI_MEMORY_SMOKE_RUN_DIR")
    store.require(root, "Run gpu-smoke on these GPUs first; set BIYI_MEMORY_SMOKE_RUN_DIR to its completed run")
    signature = smoke_signature(batch, count, environment=env)
    for rank in range(count):
        record = store.read_json(store.file_at(root, f"memory_smoke_rank{rank}.json"))
        props = torch.cuda.get_device_properties(rank)
        store.require(record["signature"] == signature, "Memory smoke source/batch/GPU count/data/stats differ")
        store.require(
            record["completed_steps"] >= 2
            and record["gpu_name"] == props.name
            and record["total_memory"] == props.total_memory
            and 0 < record["peak_allocated"] <= record["peak_reserved"] <= props.total_memory,
            "Missing or incompatible actual GPU memory smoke evidence",
        )


def generic_run_manifest(run_dir, command, *, resume=False):
    """Legacy Stage-1 provenance; 100-episode releases use the stricter launcher."""
    data = Path(os.environ["BIYI_DATASET_LOCAL_ROOT"])
    info = store.read_json(data / "meta/info.json")
    selected = os.environ.get("BIYI_TRAIN_EPISODES", "")
    count = len(selected.split(",")) if selected else info["total_episodes"]
    store.require(count not in (100, 117), "100/117-episode FFT requires versioned launcher or BIYI_FORMAL_RECIPE")
    revision = os.environ.get("BIYI_DATASET_REVISION", "")
    store.require(store.SHA.fullmatch(revision), "Set the full immutable BIYI_DATASET_REVISION")
    lock = store.read_json(data / "dataset_lock.json")
    store.require(
        lock["revision"] == revision and lock["repo_id"] == os.environ["BIYI_DATASET_REPO_ID"], "Dataset lock mismatch"
    )
    store.check_inventory(data, lock["files"])
    store.require(not store.git("status", "--porcelain"), "Formal runs require clean committed source")
    stats = Path(os.environ["BIYI_ASSETS_BASE_DIR"]) / "pi05_biyi_stage1/biyi_stage1"
    record = {
        "kind": "stage1_formal",
        "source_commit": store.git("rev-parse", "HEAD"),
        "dataset_revision": revision,
        "dataset_lock_sha256": store.digest(data / "dataset_lock.json"),
        "norm_stats_sha256": store.digest(stats / "norm_stats.json"),
        "base_model_sha256": store.digest(Path(os.environ["BIYI_BASE_CHECKPOINT_DIR"]) / "model.safetensors"),
        "uv_lock_sha256": store.digest(store.REPO_ROOT / "uv.lock"),
        "command": command,
    }
    if resume:
        store.require(
            store.read_json(store.file_at(run_dir, "run_manifest.json")) == record, "Resume provenance changed"
        )
        store.file_at(run_dir, "wandb_id.txt")
    else:
        store.require(not run_dir.exists(), "Run already exists")
        run_dir.mkdir(parents=True)
        store.write_json(run_dir / "run_manifest.json", record)
        shutil.copyfile(stats / "stage1_train_stats_manifest.json", run_dir / "train_stats_manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    count = sub.add_parser("gpu-count")
    count.add_argument("count", nargs="?", default="auto")
    count.add_argument("--verify", action="store_true")
    check = sub.add_parser("check-storage")
    check.add_argument("--check-memory", action="store_true")
    restore = sub.add_parser("restore-backup")
    restore.add_argument("--backup", required=True)
    restore.add_argument("--run-dir", required=True)
    restore.add_argument("--assets-base-dir", default=os.environ.get("BIYI_ASSETS_BASE_DIR"))
    restore.add_argument("--dataset-root", default=os.environ.get("BIYI_DATASET_LOCAL_ROOT"))
    launch = sub.add_parser("launch")
    launch.add_argument("--num-gpus", default=os.environ.get("BIYI_NUM_GPUS", "auto"))
    launch.add_argument("--batch-size", type=int, required=True)
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == "gpu-count":
        print(gpu_count(args.count, verify=args.verify))
    elif args.action == "check-storage":
        if args.check_memory:
            count = gpu_count(os.environ.get("BIYI_NUM_GPUS", "auto"), verify=True)
            batch = int(os.environ.get("BIYI_BATCH_SIZE", "32"))
            validate_batch(batch, count)
            check_memory_smoke(batch, count)
        print(json.dumps(storage_policy(os.environ["BIYI_CHECKPOINT_BASE_DIR"], probe=True)))
    elif args.action == "restore-backup":
        print(
            restore_backup(args.backup, args.run_dir, assets_base=args.assets_base_dir, dataset_root=args.dataset_root)
        )
    else:
        store.require(
            not args.dry_run or args.num_gpus != "auto", "Dry-run requires an explicit GPU count; it never probes CUDA"
        )
        count = gpu_count(args.num_gpus, verify=not args.dry_run)
        validate_batch(args.batch_size, count)
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        store.require(command, "Missing trainer command")
        full = [*launch_prefix(count, torchrun=os.environ.get("BIYI_TORCHRUN")), *command]
        if args.dry_run:
            print(json.dumps({"train": full, "gpus": count, "batch_per_gpu": args.batch_size // count}))
            return
        policy = storage_policy(os.environ["BIYI_CHECKPOINT_BASE_DIR"], probe=True)
        check_memory_smoke(args.batch_size, count)
        run_dir = Path(os.environ["BIYI_CHECKPOINT_BASE_DIR"]) / "pi05_biyi_stage1" / os.environ["BIYI_EXP_NAME"]
        resume = "--resume" in command
        generic_run_manifest(run_dir, [x for x in full if x != "--resume"], resume=resume)
        if resume:
            steps = [int(p.name) for p in run_dir.iterdir() if p.is_dir() and p.name.isdigit()]
            store.require(steps, "No completed checkpoint")
            backup_checkpoint(run_dir, max(steps))
        environment = {
            **os.environ,
            "BIYI_CHECKPOINT_IO": "1",
            "BIYI_MEMORY_SMOKE": "0",
            "BIYI_STORAGE_DEVICE": policy["device"],
            "BIYI_DDP_TIMEOUT_SECONDS": os.environ.get("BIYI_DDP_TIMEOUT_SECONDS", "3600"),
            "PYTHONPATH": str(store.REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
        subprocess.run(full, env=environment, check=True)


if __name__ == "__main__":
    main()
