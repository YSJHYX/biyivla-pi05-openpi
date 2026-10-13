"""Compute Stage-1 norm stats from the selected frozen train split only."""

import json
from pathlib import Path

import numpy as np
import tqdm

from openpi.shared import normalize
from scripts.biyi.preflight_pi05_stage1 import _sha256
from scripts.biyi.preflight_pi05_stage1 import validate
from scripts.compute_norm_stats import create_torch_dataloader


def main() -> None:
    config, identity = validate(require_stats=False)
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
