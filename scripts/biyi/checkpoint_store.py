"""Biyi formal checkpoint lifecycle. Offline commands use only the standard library.

No command starts training. Network writes require the upload subcommand and an
exact confirmation phrase. Never load optimizer.pt/metadata.pt during inspection.
"""

import argparse
from datetime import UTC
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_REPO = "PPPPPilot/BiyiVLA-pi05-checkpoints"
DATASET_REPO = "PPPPPilot/BiyiVLA"
RECIPE = REPO_ROOT / "scripts/biyi/configs/pi05_biyi_100ep_fft_v1.json"
RECIPE_117 = REPO_ROOT / "scripts/biyi/configs/pi05_biyi_117ep_fft_v1.json"
FORMAL_KINDS = {"formal_100ep_fft", "formal_117ep_fft"}
NORM = "assets/biyi_stage1/norm_stats.json"
SHA = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def now():
    return datetime.now(UTC).isoformat()


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def safe_relative(name):
    path = Path(name)
    require(not path.is_absolute() and path.parts and ".." not in path.parts, "Unsafe manifest path")
    require(path.as_posix() == name and "\\" not in name, "Non-canonical manifest path")
    return path


def file_at(root, name):
    path = Path(root)
    require(not path.is_symlink(), "Symlink root rejected")
    for part in safe_relative(name).parts:
        path = path / part
        require(not path.is_symlink(), "Symlinks are not permitted in artifacts")
    require(path.is_file() and path.stat().st_size > 0, f"Missing/empty file: {name}")
    return path


def inventory(root, names):
    return {
        name: {"sha256": digest(file_at(root, name)), "bytes": file_at(root, name).stat().st_size}
        for name in sorted(names)
    }


def check_inventory(root, entries):
    require(bool(entries), "Empty inventory")
    require(inventory(root, entries) == entries, "File size/checksum mismatch")


def validate_run(run):
    require(run["kind"] in FORMAL_KINDS and SHA.fullmatch(run["source_commit"]), "Formal source provenance missing")
    # The same recipe contract applies to locally created and downloaded releases.
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "recipe.json"
        write_json(path, run["recipe"])
        recipe(path)
    count = run["recipe"]["dataset_episodes"]
    require(run["kind"] == f"formal_{count}ep_fft", "Formal kind/episode count mismatch")
    if run["recipe"].get("dataset_revision"):
        require(run["dataset"]["revision"] == run["recipe"]["dataset_revision"], "Recipe/data revision mismatch")
    require(
        run["dataset"]["repo_id"] == DATASET_REPO
        and SHA.fullmatch(run["dataset"]["revision"])
        and run["dataset"]["episodes"] == list(range(count)),
        "Expected frozen recipe episode provenance",
    )
    require(
        all(SHA256.fullmatch(run[key]) for key in ("norm_stats_sha256", "base_model_sha256", "uv_lock_sha256"))
        and SHA256.fullmatch(run["dataset"]["lock_sha256"]),
        "Missing immutable artifact hashes",
    )


def recipe(path=RECIPE):
    return validate_recipe(read_json(path))


def validate_recipe(value):
    expected = {
        "schema_version": 1,
        "config_name": "pi05_biyi_stage1",
        "dataset_repo": DATASET_REPO,
        "model_repo": MODEL_REPO,
        "backend": "stock_pytorch_pi05",
        "ahfe": False,
        "lora": False,
        "seed": 42,
    }
    require(
        all(type(value.get(k)) is type(v) and value[k] == v for k, v in expected.items()),
        "Only stock PyTorch pi05 FFT, AHFE off and LoRA off are supported",
    )
    count = value.get("dataset_episodes")
    require(type(count) is int and count in (100, 117), "Supported frozen episode counts are 100 and 117")
    require(re.fullmatch(rf"pi05_biyi_{count}ep_fft_v[1-9][0-9]*", value["experiment"]), "Invalid formal name")
    if count == 117:
        require(value.get("dataset_frames") == 77975, "117-episode recipe requires 77975 frames")
        require(SHA.fullmatch(value.get("dataset_revision", "")), "117-episode recipe requires pinned dataset revision")
    for name in ("num_train_steps", "batch_size", "num_gpus", "save_interval"):
        require(type(value[name]) is int and value[name] > 0, f"Invalid {name}")
    require(
        value["num_train_steps"] >= 10 and value["batch_size"] % value["num_gpus"] == 0,
        "Invalid formal steps / global batch / GPU count",
    )
    require(type(value["num_workers"]) is int and value["num_workers"] >= 0, "Invalid workers")
    return value


def persistent_root(value):
    path = Path(value).expanduser()
    require(path.is_absolute(), "Persistent root must be absolute")
    path = path.resolve()
    require(path != REPO_ROOT and REPO_ROOT not in path.parents, "Persistent artifacts must be outside Git")
    require(
        not any(
            path == p or p in path.parents
            for p in (Path("/tmp"), Path("/content"), Path("/dev/shm"), Path(tempfile.gettempdir()))
        ),
        "Use a persistent mount; /tmp and /content runtime storage are rejected (Drive mount excepted)",
    )
    return path


def storage_root(value):
    # Google Drive may be mounted under /content/drive; it is an explicit persistent exception.
    path = Path(value).expanduser()
    if path.is_absolute() and Path("/content/drive") in path.parents:
        require(
            Path("/content/drive").is_mount(),
            "Mount Google Drive first; an ordinary /content/drive directory is temporary",
        )
        require(Path("/content/drive") in path.resolve().parents, "Drive path escapes its mount")
        require(REPO_ROOT not in path.resolve().parents, "Root must be outside Git")
        return path.resolve()
    return persistent_root(value)


def freeze_dataset(root, revision):
    require(SHA.fullmatch(revision), "Dataset revision must be a full immutable 40-character commit")
    from huggingface_hub import HfApi
    from huggingface_hub import snapshot_download

    api = HfApi()
    info = api.dataset_info(DATASET_REPO, revision=revision)
    require(info.sha == revision, "Dataset revision mismatch")
    destination = root / "datasets" / revision
    require(not destination.exists(), "Frozen dataset destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".freeze-") as staging_name:
        staging = Path(staging_name)
        snapshot_download(DATASET_REPO, repo_type="dataset", revision=revision, local_dir=staging)
        # Local download bookkeeping is not part of the frozen source or a model artifact.
        shutil.rmtree(staging / ".cache", ignore_errors=True)
        names = [p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file()]
        require(len(read_json(staging / "meta/info.json").get("features", {})) > 0, "Invalid LeRobot metadata")
        write_json(
            staging / "dataset_lock.json",
            {"repo_id": DATASET_REPO, "revision": revision, "files": inventory(staging, names)},
        )
        staging.rename(destination)
    print(destination)


def dataset_identity(path, revision, episode_count=100):
    require(type(episode_count) is int and episode_count in (100, 117), "Unsupported episode count")
    lock = read_json(file_at(path, "dataset_lock.json"))
    require(
        SHA.fullmatch(revision) and lock["revision"] == revision and lock["repo_id"] == DATASET_REPO,
        "Dataset lock/revision mismatch; use freeze-dataset",
    )
    check_inventory(path, lock["files"])
    actual = {p.relative_to(path).as_posix() for p in Path(path).rglob("*") if p.is_file() or p.is_symlink()}
    require(actual == set(lock["files"]) | {"dataset_lock.json"}, "Frozen dataset contains unexpected files")
    info = read_json(file_at(path, "meta/info.json"))
    require(info["total_episodes"] == episode_count, f"Formal recipe requires exactly {episode_count} episodes")
    episodes = [json.loads(line) for line in file_at(path, "meta/episodes.jsonl").read_text().splitlines()]
    require(
        len(episodes) == episode_count and [row["episode_index"] for row in episodes] == list(range(episode_count)),
        "Expected contiguous frozen episode IDs",
    )
    return {
        "repo_id": DATASET_REPO,
        "revision": revision,
        "episodes": list(range(episode_count)),
        "lock_sha256": digest(Path(path) / "dataset_lock.json"),
    }


def check_safetensors(path):
    """Validate header, complete offsets and stock-shaped tensors without loading weights."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, "Truncated safetensors")
        length = struct.unpack("<Q", prefix)[0]
        require(0 < length <= min(32 * 1024 * 1024, size - 8), "Invalid safetensors header")
        header = json.loads(stream.read(length))
    widths = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    spans = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        require(not any(word in name.lower() for word in ("lora", "ahfe", "arm_hand")), "Non-stock tensor")
        shape, dtype = tensor["shape"], tensor["dtype"]
        count = 1
        for dimension in shape:
            require(type(dimension) is int and dimension >= 0, "Invalid tensor shape")
            count *= dimension
        lo, hi = tensor["data_offsets"]
        require(dtype in widths and hi - lo == count * widths[dtype] and 0 <= lo <= hi, "Invalid tensor offsets")
        spans.append((lo, hi))
    cursor = 0
    for lo, hi in sorted(spans):
        require(lo == cursor, "Safetensors overlap/gap")
        cursor = hi
    require(cursor == size - 8 - length, "Safetensors payload length mismatch")
    require(
        header.get("action_in_proj.weight", {}).get("shape") == [1024, 32]
        and header.get("action_out_proj.weight", {}).get("shape") == [32, 1024],
        "Not a stock pi05 32D checkpoint",
    )
    require(len(spans) > 100, "Incomplete stock checkpoint")


def inspect_checkpoint(run_dir, step):
    require(type(step) is int and step > 0, "Expected a positive completed step")
    run = read_json(file_at(run_dir, "run_manifest.json"))
    validate_run(run)
    settings = run["recipe"]
    require(settings["experiment"] == Path(run_dir).name, "Experiment mismatch")
    require(step <= settings["num_train_steps"], "Step exceeds frozen training horizon")
    ckpt = Path(run_dir) / str(step)
    names = ["model.safetensors", "optimizer.pt", "metadata.pt", NORM]
    files = inventory(ckpt, names)
    provenance = read_json(file_at(run_dir, "train_stats_manifest.json"))
    require(
        provenance["norm_stats_sha256"] == run["norm_stats_sha256"]
        and provenance["repo_id"] == DATASET_REPO
        and provenance["train_episode_indices"] == run["dataset"]["episodes"],
        "Training stats provenance mismatch",
    )
    check_safetensors(ckpt / "model.safetensors")
    require(files[NORM]["sha256"] == run["norm_stats_sha256"], "Checkpoint norm stats changed")
    stats = read_json(ckpt / NORM)
    require(set(stats["norm_stats"]) >= {"state", "actions"}, "Missing normalization data")
    for key in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            values = stats["norm_stats"][key][field]
            require(
                len(values) == 17 and all(type(v) in (float, int) and math.isfinite(v) for v in values),
                "Normalization must be finite and 17D",
            )
        require(all(v >= 0 for v in stats["norm_stats"][key]["std"]), "Negative normalization std")
    return run, ckpt, files


def seal_checkpoint(run_dir, step):
    run, ckpt, files = inspect_checkpoint(run_dir, step)
    value = {
        "schema_version": 1,
        "step": step,
        "run_manifest_sha256": digest(Path(run_dir) / "run_manifest.json"),
        "files": files,
    }
    path = ckpt / "checkpoint_manifest.json"
    if path.exists():
        require(read_json(path) == value, "Sealed checkpoint changed")
    else:
        write_json(path, value)
    print("OFFLINE INTEGRITY PASS; model quality/inference acceptance is still required")
    print(f"checkpoint_manifest_sha256={digest(path)}")
    return value


def acceptance(report, ckpt):
    report = read_json(report)
    require(
        report.get("accepted") is True and report.get("reviewer") and report.get("reviewed_at"),
        "Explicit reviewer acceptance is required",
    )
    require(
        report.get("checkpoint_manifest_sha256") == digest(ckpt / "checkpoint_manifest.json"),
        "Acceptance report belongs to another checkpoint",
    )
    checks = ("offline_policy_load", "finite_actions", "action_shape_50_17")
    require(all(report.get("checks", {}).get(key) is True for key in checks), "Offline inference checks not accepted")
    evidence = report.get("evidence", [])
    require(evidence and all(isinstance(item, str) for item in evidence), "Attach offline inference evidence files")
    return report


def pack(run_dir, step, report_path, output, *, include_optimizer=False):
    seal_checkpoint(run_dir, step)
    run, ckpt, files = inspect_checkpoint(run_dir, step)
    report = acceptance(report_path, ckpt)
    output = Path(output)
    require(not output.exists(), "Bundle destination exists; never overwrite a release")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".bundle-") as temporary_name:
        temp = Path(temporary_name)
        names = ["model.safetensors", NORM, "checkpoint_manifest.json"]
        if include_optimizer:
            names += ["optimizer.pt", "metadata.pt"]
        for name in names:
            destination = temp / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file_at(ckpt, name), destination)
        shutil.copyfile(file_at(run_dir, "run_manifest.json"), temp / "run_manifest.json")
        shutil.copyfile(file_at(run_dir, "train_stats_manifest.json"), temp / "train_stats_manifest.json")
        if include_optimizer:
            shutil.copyfile(file_at(run_dir, "wandb_id.txt"), temp / "wandb_id.txt")
        evidence_names = []
        for index, name in enumerate(report["evidence"]):
            # Evidence paths are relative to the supplied report; tokens/log trees are never swept up.
            src = file_at(Path(report_path).parent, name)
            require(src.suffix in (".json", ".txt", ".md"), "Evidence must be reviewed JSON/text, without credentials")
            destination = temp / "evidence" / f"{index:03d}{src.suffix}"
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(src, destination)
            evidence_names.append(destination.relative_to(temp).as_posix())
        write_json(temp / "acceptance.json", {**report, "evidence": evidence_names})
        kind = "resume" if include_optimizer else "inference"
        files = inventory(temp, [p.relative_to(temp).as_posix() for p in temp.rglob("*") if p.is_file()])
        manifest = {
            "schema_version": 1,
            "kind": kind,
            "experiment": run["recipe"]["experiment"],
            "step": step,
            "model_repo": MODEL_REPO,
            "files": files,
            "created_at": now(),
        }
        write_json(temp / "bundle_manifest.json", manifest)
        # Verify the staged copy before making a complete bundle visible.
        verify_bundle(temp)
        temp.rename(output)
    verify_bundle(output)
    print(output)


def verify_bundle(path):
    path = Path(path)
    manifest = read_json(file_at(path, "bundle_manifest.json"))
    require(manifest["kind"] in ("inference", "resume") and manifest["model_repo"] == MODEL_REPO, "Invalid bundle type")
    check_inventory(path, manifest["files"])
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() or p.is_symlink()}
    require(actual == set(manifest["files"]) | {"bundle_manifest.json"}, "Unexpected artifact files")
    required = {
        "model.safetensors",
        NORM,
        "checkpoint_manifest.json",
        "run_manifest.json",
        "train_stats_manifest.json",
        "acceptance.json",
    }
    if manifest["kind"] == "resume":
        required |= {"optimizer.pt", "metadata.pt", "wandb_id.txt"}
    require(required <= set(manifest["files"]), "Incomplete bundle")
    require(
        all(name in required or re.fullmatch(r"evidence/[0-9]{3}\.(json|txt|md)", name) for name in manifest["files"]),
        "Non-allowlisted artifact",
    )
    sealed = read_json(path / "checkpoint_manifest.json")
    require(
        sealed["step"] == manifest["step"] and sealed["run_manifest_sha256"] == digest(path / "run_manifest.json"),
        "Checkpoint provenance mismatch",
    )
    for name in ("model.safetensors", NORM, *(["optimizer.pt", "metadata.pt"] if manifest["kind"] == "resume" else [])):
        require(manifest["files"][name] == sealed["files"][name], "Bundle differs from sealed checkpoint")
    run = read_json(path / "run_manifest.json")
    validate_run(run)
    require(
        run["recipe"]["experiment"] == manifest["experiment"] and run["kind"] in FORMAL_KINDS,
        "Wrong formal experiment",
    )
    require(
        type(manifest["step"]) is int and 0 < manifest["step"] <= run["recipe"]["num_train_steps"],
        "Invalid archived step",
    )
    require(manifest["files"][NORM]["sha256"] == run["norm_stats_sha256"], "Normalization provenance mismatch")
    report = acceptance(path / "acceptance.json", path)
    require(
        set(report["evidence"]) == {name for name in manifest["files"] if name.startswith("evidence/")},
        "Acceptance evidence mismatch",
    )
    check_safetensors(path / "model.safetensors")
    return manifest


def upload(path, confirm):
    manifest = verify_bundle(path)
    phrase = f"{MODEL_REPO}:{manifest['experiment']}:{manifest['step']}:{manifest['kind']}"
    require(confirm == phrase, f"Explicit confirmation required: --confirm {phrase}")
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(MODEL_REPO, repo_type="model")
    require(info.private is True, "Upload only to the existing private MODEL repo")
    prefix = f"{manifest['kind']}/{manifest['experiment']}/step-{manifest['step']}"
    existing = api.list_repo_files(MODEL_REPO, repo_type="model", revision=info.sha)
    require(not any(name.startswith(prefix + "/") for name in existing), "Remote release already exists")
    names = [*manifest["files"], "bundle_manifest.json"]
    operations = [
        CommitOperationAdd(path_in_repo=f"{prefix}/{name}", path_or_fileobj=str(file_at(path, name)))
        for name in sorted(names)
    ]
    result = api.create_commit(
        MODEL_REPO,
        repo_type="model",
        operations=operations,
        parent_commit=info.sha,
        commit_message=f"Archive accepted {manifest['kind']} {manifest['experiment']} step {manifest['step']}",
    )
    receipt = {
        "repo_id": MODEL_REPO,
        "repo_type": "model",
        "revision": result.oid,
        "path": prefix,
        "bundle_manifest_sha256": digest(Path(path) / "bundle_manifest.json"),
    }
    write_json(Path(path).with_name(Path(path).name + ".hf-receipt.json"), receipt)
    print(json.dumps(receipt))


def download(root, revision, prefix, output):
    require(SHA.fullmatch(revision), "Download requires immutable HF model commit")
    require(
        re.fullmatch(r"(inference|resume)/pi05_biyi_(100|117)ep_fft_v[1-9][0-9]*/step-[1-9][0-9]*", prefix),
        "Invalid release prefix",
    )
    from huggingface_hub import snapshot_download

    require(not Path(output).exists(), "Destination exists")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=".download-") as temp:
        snapshot_download(
            MODEL_REPO, repo_type="model", revision=revision, allow_patterns=[prefix + "/*"], local_dir=temp
        )
        source = Path(temp) / prefix
        verify_bundle(source)
        shutil.copytree(source, output)
    print(output)


def restore(bundle, root):
    manifest = verify_bundle(bundle)
    require(manifest["kind"] == "resume", "Inference archive has no optimizer; request a full resume backup")
    run = read_json(Path(bundle) / "run_manifest.json")
    require(
        git("rev-parse", "HEAD") == run["source_commit"] and not git("status", "--porcelain"),
        "Restore from the pinned clean source commit",
    )
    dataset_path = root / "datasets" / run["dataset"]["revision"]
    require(
        dataset_identity(dataset_path, run["dataset"]["revision"], run["recipe"]["dataset_episodes"]) == run["dataset"],
        "Restore the frozen dataset before restoring the run",
    )
    destination = root / "checkpoints/pi05_biyi_stage1" / manifest["experiment"]
    require(not destination.exists(), "Restore never merges/overwrites an existing run")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".restore-") as temporary_name:
        temp = Path(temporary_name)
        for name in ("run_manifest.json", "train_stats_manifest.json", "wandb_id.txt"):
            shutil.copyfile(file_at(bundle, name), temp / name)
        ckpt = temp / str(manifest["step"])
        for name in ("model.safetensors", "optimizer.pt", "metadata.pt", NORM, "checkpoint_manifest.json"):
            output = ckpt / name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file_at(bundle, name), output)
        # Inference normalization remains available to the stock data loader on resume.
        stats = root / "assets/pi05_biyi_stage1/biyi_stage1"
        stats.mkdir(parents=True, exist_ok=True)
        for source, name in ((Path(bundle) / NORM, "norm_stats.json"),):
            target = stats / name
            if target.exists():
                require(digest(target) == digest(source), "Existing norm stats conflict")
            else:
                shutil.copyfile(source, target)
        # Stage-1's stats fingerprint includes the machine-local dataset path.
        # Rebind only that path after the complete revision/file lock has passed;
        # keep the original archived manifest in the run directory unchanged.
        rebound = read_json(Path(bundle) / "train_stats_manifest.json")
        rebound["local_root"] = str(dataset_path.resolve())
        target = stats / "stage1_train_stats_manifest.json"
        if target.exists():
            require(read_json(target) == rebound, "Existing stats provenance conflicts")
        else:
            write_json(target, rebound)
        temp.rename(destination)
    inspect_checkpoint(destination, manifest["step"])
    print(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-dataset")
    freeze.add_argument("--persistent-root", required=True)
    freeze.add_argument("--revision", required=True)
    for command in ("seal", "pack"):
        child = commands.add_parser(command)
        child.add_argument("--run-dir", type=Path, required=True)
        child.add_argument("--step", type=int, required=True)
        if command == "pack":
            child.add_argument("--acceptance", type=Path, required=True)
            child.add_argument("--output", type=Path, required=True)
            child.add_argument("--include-optimizer", action="store_true")
    for command in ("verify", "upload", "restore"):
        child = commands.add_parser(command)
        child.add_argument("--bundle", type=Path, required=True)
        if command == "upload":
            child.add_argument("--confirm", required=True)
        if command == "restore":
            child.add_argument("--persistent-root", required=True)
    child = commands.add_parser("download")
    child.add_argument("--persistent-root", required=True)
    child.add_argument("--revision", required=True)
    child.add_argument("--prefix", required=True)
    child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze-dataset":
        freeze_dataset(storage_root(args.persistent_root), args.revision)
    elif args.command == "seal":
        seal_checkpoint(args.run_dir, args.step)
    elif args.command == "pack":
        pack(args.run_dir, args.step, args.acceptance, args.output, include_optimizer=args.include_optimizer)
    elif args.command == "verify":
        print(json.dumps(verify_bundle(args.bundle), indent=2))
    elif args.command == "upload":
        upload(args.bundle, args.confirm)
    elif args.command == "restore":
        restore(args.bundle, storage_root(args.persistent_root))
    elif args.command == "download":
        download(storage_root(args.persistent_root), args.revision, args.prefix, args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # SDK errors may contain HTTP details. Expose only local validation messages.
        print(f"FAILED: {error}" if type(error) is ValueError else f"FAILED: {type(error).__name__}")
        raise SystemExit(1) from None
