"""Compute Stage-1 norm stats from the selected frozen train split only."""

import argparse
import json
from pathlib import Path

import numpy as np
import tqdm

from openpi.shared import normalize
from scripts.biyi.preflight_pi05_stage1 import _sha256
from scripts.biyi.preflight_pi05_stage1 import validate
from scripts.compute_norm_stats import create_torch_dataloader


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--numeric-only", action="store_true", help="Use verified Parquet vectors after normal preflight"
    )
    parser.add_argument("--source-inventory", type=Path, help="Pinned HF file identities; required for numeric-only")
    parser.add_argument("--recipe", type=Path, help="Versioned recipe for numeric-only (default: 117 episodes)")
    args = parser.parse_args(argv)
    if args.numeric_only and args.source_inventory is None:
        parser.error("--numeric-only requires --source-inventory")
    if not args.numeric_only and (args.source_inventory is not None or args.recipe is not None):
        parser.error("--source-inventory and --recipe require --numeric-only")
    config, identity = validate(require_stats=False)
    if args.numeric_only:
        from scripts.biyi import checkpoint_store as store
        from scripts.biyi.compute_stage1_parquet_norm_stats import main as numeric_main

        recipe_path = args.recipe or store.RECIPE_117
        settings = store.recipe(recipe_path)
        store.require(
            identity["train_episode_indices"] == list(range(settings["dataset_episodes"]))
            and identity["train_frame_count"] == settings.get("dataset_frames", identity["train_frame_count"]),
            "Numeric stats recipe must match the effective training split",
        )
        numeric_main(
            [
                "--dataset-root",
                identity["local_root"],
                "--source-inventory",
                str(args.source_inventory),
                "--assets-base-dir",
                str(config.assets_base_dir),
                "--recipe",
                str(recipe_path),
                "--batch-size",
                str(config.batch_size),
            ]
        )
        return
    data = config.data.create(config.assets_dirs, config.model)
    loader, num_batches = create_torch_dataloader(
        data, config.model.action_horizon, config.batch_size, config.model, config.num_workers
    )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(loader, total=num_batches, desc="Computing train stats"):
        for key, running in stats.items():
            running.update(np.asarray(batch[key]))
    # The stock loader drops the final incomplete batch. Include it explicitly
    # so every selected train frame contributes to Stage-1 stats.
    dataset = loader.torch_loader.dataset
    for index in range(num_batches * config.batch_size, len(dataset)):
        sample = dataset[index]
        for key, running in stats.items():
            running.update(np.asarray(sample[key]))
    normalize.save(
        config.assets_dirs / data.asset_id, {key: running.get_statistics() for key, running in stats.items()}
    )
    stats_path = config.assets_dirs / data.asset_id / "norm_stats.json"
    manifest = {**identity, "norm_stats_sha256": _sha256(stats_path)}
    manifest_path = Path(stats_path.with_name("stage1_train_stats_manifest.json"))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"STAGE1 TRAIN STATS PASS: {stats_path}")
    print(f"SOURCE MANIFEST: {manifest_path}")
    print(f"TRAIN EPISODES: {identity['train_episode_indices']}")
    print(f"TRAIN FRAMES: {identity['train_frame_count']}")
    print(f"NORM STATS SHA256: {manifest['norm_stats_sha256']}")


if __name__ == "__main__":
    main()
