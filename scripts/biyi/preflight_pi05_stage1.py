"""Read-only Biyi Stage-1 contract and frozen train stats check."""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi.policies.biyivla_policy import BiyiVLAOutputs
from openpi.training import config as training_config
from openpi.training.data_loader import create_torch_dataset
from openpi.training.data_loader import transform_dataset
from scripts.biyi.check_cloud_auth import WANDB_PROJECT
from scripts.biyi.check_cloud_auth import check_cloud_auth

CONFIG_NAME = "pi05_biyi_stage1"
BASE_WEIGHTS_SHA256 = "dbcbe3ea978ce1d6f791582fa034e7ece8a41feaad2d3c9a5b014c1938610e2a"
BASE_WEIGHTS_SIZE = 7_233_650_408
META_FILES = ("info.json", "episodes.jsonl", "tasks.jsonl")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(root: Path, repo_id: str, selected: tuple[int, ...] | None, episodes: list[dict]) -> dict:
    train_episodes = list(selected) if selected is not None else list(range(len(episodes)))
    return {
        "repo_id": repo_id,
        "local_root": str(root.resolve()),
        "train_episode_indices": train_episodes,
        "train_frame_count": sum(episodes[index]["length"] for index in train_episodes),
        "metadata_sha256": {name: _sha256(root / "meta" / name) for name in META_FILES},
    }


def validate(*, require_stats: bool) -> tuple[training_config.TrainConfig, dict]:
    config = training_config.get_config(CONFIG_NAME)
    model = config.model
    if not model.pi05 or model.action_dim != 32 or model.action_horizon != 50:
        raise ValueError("Stage-1 requires stock pi0.5 with 32D actions and horizon 50")
    if model.paligemma_variant != "gemma_2b" or model.action_expert_variant != "gemma_300m":
        raise ValueError("Stage-1 must use the stock full-finetuning model variants")
    if model.dtype != "bfloat16" or config.pytorch_training_precision != "bfloat16":
        raise ValueError("Stage-1 requires BF16")
    if type(config.optimizer).__name__ != "AdamW" or type(config.freeze_filter).__name__ != "Nothing":
        raise ValueError("Stage-1 requires full AdamW fine-tuning")
    if any("rtc" in field.name.lower() for field in dataclasses.fields(model)):
        raise ValueError("RTC must remain absent")
    base_weights = Path(config.pytorch_weight_path) / "model.safetensors"
    if not base_weights.is_file():
        raise FileNotFoundError(base_weights)
    if base_weights.stat().st_size != BASE_WEIGHTS_SIZE or _sha256(base_weights) != BASE_WEIGHTS_SHA256:
        raise ValueError(f"Stage-1 base checkpoint size or SHA256 mismatch: {base_weights}")

    if not config.data.repo_id or not config.data.base_config.local_root:
        raise ValueError("Set BIYI_DATASET_REPO_ID and BIYI_DATASET_LOCAL_ROOT to the frozen train dataset")
    root = Path(config.data.base_config.local_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    info = json.loads((root / "meta/info.json").read_text())
    features = info["features"]
    for key in ("observation.state", "action"):
        if features[key]["shape"] != [17] or features[key]["dtype"] != "float32":
            raise ValueError(f"{key} must be float32[17]")
    for key in ("observation.images.head", "observation.images.wrist"):
        if key not in features:
            raise ValueError(f"Missing {key}")
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
    tasks = [json.loads(line) for line in (root / "meta/tasks.jsonl").read_text().splitlines()]
    if not tasks or not all(isinstance(task.get("task"), str) and task["task"].strip() for task in tasks):
        raise ValueError("Collection-time task prompts are missing")
    if len(episodes) != info["total_episodes"] or sum(row["length"] for row in episodes) != info["total_frames"]:
        raise ValueError("Episode/frame totals disagree with info.json")
    selected = config.data.base_config.episode_indices
    if selected is None:
        if info.get("splits", {}).get("train") != f"0:{len(episodes)}":
            raise ValueError("Specify BIYI_TRAIN_EPISODES for a frozen train split")
    elif any(index >= len(episodes) for index in selected):
        raise ValueError("BIYI_TRAIN_EPISODES contains an out-of-range episode")

    offset = 0
    task_ids = {task["task_index"] for task in tasks}
    for index, row in enumerate(episodes):
        if row["episode_index"] != index or row["length"] <= 0:
            raise ValueError(f"Invalid episode metadata at {index}")
        chunk = index // info["chunks_size"]
        parquet = root / info["data_path"].format(episode_chunk=chunk, episode_index=index)
        table = pq.read_table(
            parquet, columns=["index", "episode_index", "frame_index", "task_index", "observation.state", "action"]
        )
        if table.num_rows != row["length"]:
            raise ValueError(f"Episode {index} length mismatch")
        frames = np.asarray(table["frame_index"].to_pylist()).reshape(-1)
        episode_ids = np.asarray(table["episode_index"].to_pylist()).reshape(-1)
        global_ids = np.asarray(table["index"].to_pylist()).reshape(-1)
        episode_tasks = np.asarray(table["task_index"].to_pylist()).reshape(-1)
        if (
            not np.array_equal(frames, np.arange(row["length"]))
            or not np.all(episode_ids == index)
            or not np.array_equal(global_ids, np.arange(offset, offset + row["length"]))
            or not set(episode_tasks).issubset(task_ids)
        ):
            raise ValueError(f"Episode {index} frame indexing invalid")
        for key in ("observation.state", "action"):
            values = np.asarray(table[key].to_pylist(), dtype=np.float32)
            if values.shape != (row["length"], 17) or not np.isfinite(values).all():
                raise ValueError(f"Episode {index} {key} violates finite float32[T,17]")
        for key in ("observation.images.head", "observation.images.wrist"):
            video = root / info["video_path"].format(episode_chunk=chunk, episode_index=index, video_key=key)
            if not video.is_file() or video.stat().st_size == 0:
                raise ValueError(f"Missing {video}")
        offset += row["length"]

    data = config.data.create(config.assets_dirs, model)
    if not data.prompt_from_task or data.action_sequence_keys != ("action",):
        raise ValueError("Stage-1 prompt/action mapping changed")
    identity = fingerprint(root, data.repo_id, selected, episodes)
    if require_stats:
        stats_path = config.assets_dirs / data.asset_id / "norm_stats.json"
        manifest_path = stats_path.with_name("stage1_train_stats_manifest.json")
        if data.norm_stats is None or not stats_path.is_file() or not manifest_path.is_file():
            raise ValueError("Stage-1 train norm stats and manifest are required; recompute after dataset freeze")
        manifest = json.loads(manifest_path.read_text())
        if manifest != {**identity, "norm_stats_sha256": _sha256(stats_path)}:
            raise ValueError("Norm stats do not match the frozen train split or metadata")
        for key in ("state", "actions"):
            stats = data.norm_stats[key]
            if any(np.asarray(value).shape != (17,) for value in (stats.mean, stats.std, stats.q01, stats.q99)):
                raise ValueError(f"{key} norm stats are not 17D")
    raw = create_torch_dataset(data, model.action_horizon, model)
    processed = transform_dataset(raw, data, skip_norm_stats=not require_stats)
    for index in sorted({0, len(raw) // 2, len(raw) - 1}):
        sample = raw[index]
        out = processed[index]
        if not str(sample["prompt"]).strip():
            raise ValueError("Empty task prompt")
        if np.asarray(sample["observation.state"]).shape != (17,):
            raise ValueError("Raw state is not 17D")
        if np.asarray(sample["action"]).shape != (50, 17):
            raise ValueError("Raw action is not [50,17]")
        if out["state"].shape != (32,) or out["actions"].shape != (50, 32):
            raise ValueError("Model transform did not pad to 32D")
        if np.any(out["state"][17:]) or np.any(out["actions"][:, 17:]):
            raise ValueError("Padded dimensions are not zero")
        if out["image_mask"]["right_wrist_0_rgb"]:
            raise ValueError("Missing third camera must stay masked")
    cropped = BiyiVLAOutputs()({"actions": np.zeros((50, 32), dtype=np.float32)})["actions"]
    if cropped.shape != (50, 17):
        raise ValueError("Inference output did not crop to 17D")

    return config, identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-missing-stats", action="store_true", help="Only for dataset/config preparation")
    parser.add_argument("--formal", action="store_true", help="Require online cloud authentication before training")
    parser.add_argument("--smoke", action="store_true", help="Check the smoke W&B-disabled contract")
    parser.add_argument("--exp-name")
    parser.add_argument("--project-name")
    wandb_flags = parser.add_mutually_exclusive_group()
    wandb_flags.add_argument("--wandb-enabled", dest="wandb_enabled", action="store_true")
    wandb_flags.add_argument("--no-wandb-enabled", dest="wandb_enabled", action="store_false")
    parser.set_defaults(wandb_enabled=None)
    args = parser.parse_args()
    if args.formal and (args.smoke or args.allow_missing_stats):
        parser.error("Formal preflight cannot be combined with smoke or missing stats")
    if args.formal:
        config = training_config.get_config(CONFIG_NAME)
        exp_name = os.environ.get("BIYI_EXP_NAME", "").strip()
        if not config.wandb_enabled or args.wandb_enabled is not True:
            parser.error("Formal training requires --wandb-enabled")
        if not exp_name or args.exp_name != exp_name:
            parser.error("Formal training requires BIYI_EXP_NAME matching --exp-name")
        if config.project_name != WANDB_PROJECT or args.project_name != WANDB_PROJECT:
            parser.error(f"Formal W&B project must be {WANDB_PROJECT}")
        print("WANDB_ENABLED        = PASS")
        print("WANDB_PROJECT        = PASS")
        if not check_cloud_auth(config.data.repo_id or None, os.environ.get("BIYI_HF_CHECKPOINT_REPO_ID") or None):
            print("FORMAL_PREFLIGHT     = FAIL (cloud authentication)")
            raise SystemExit(1)
    elif args.smoke and args.wandb_enabled is not False:
        parser.error("Smoke preflight requires --no-wandb-enabled")

    try:
        config, identity = validate(require_stats=not args.allow_missing_stats)
    except Exception as exc:
        if args.formal:
            print(f"FORMAL_PREFLIGHT     = FAIL: {exc}")
            raise SystemExit(1) from None
        raise
    if args.formal:
        for gate in (
            "DATASET_CONTRACT",
            "TRAIN_SPLIT",
            "NORM_STATS",
            "BASE_CHECKPOINT",
            "FULL_FINE_TUNING",
            "RTC_DISABLED",
        ):
            print(f"{gate:<20}= PASS")
        print("FORMAL_PREFLIGHT     = PASS")
    print(f"PREFLIGHT PASS: {config.name} {identity['repo_id']} episodes={identity['train_episode_indices']}")


if __name__ == "__main__":
    main()
