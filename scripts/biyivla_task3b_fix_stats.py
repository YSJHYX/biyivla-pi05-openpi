"""Repair degenerate smoke-only quantiles using episode 9 physical action extrema."""

import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq

from openpi.shared.normalize import NormStats, load, save
from openpi.training.config import get_config


def main():
    config = get_config("pi05_biyivla_smoke")
    data = config.data.create(config.assets_dirs, config.model)
    stats_dir = config.assets_dirs / data.asset_id
    stats_path = stats_dir / "norm_stats.json"
    original = stats_dir / "norm_stats_stock.json"
    if original.exists():
        raise FileExistsError(f"Refusing to recalibrate twice: {original}")
    shutil.copy2(stats_path, original)
    parquet = Path(data.local_root) / "data/chunk-000/episode_000009.parquet"
    actions = np.asarray(pq.read_table(parquet, columns=["action"])["action"].to_pylist(), dtype=np.float32)
    assert actions.shape == (1327, 17) and np.isfinite(actions).all()
    stats = load(stats_dir)
    old = stats["actions"]
    q01 = old.q01.copy()
    q99 = old.q99.copy()
    changes = []
    for dim in range(17):
        span = float(q99[dim] - q01[dim])
        raw_min = float(actions[:, dim].min())
        raw_max = float(actions[:, dim].max())
        if span <= 1e-5 and raw_max - raw_min > 1e-4:
            changes.append({"dimension": dim, "old_q01": float(q01[dim]), "old_q99": float(q99[dim]),
                            "new_q01": raw_min, "new_q99": raw_max})
            q01[dim] = raw_min
            q99[dim] = raw_max
    stats["actions"] = NormStats(mean=old.mean, std=old.std, q01=q01, q99=q99)
    save(stats_dir, stats)
    (stats_dir / "norm_stats_calibration.json").write_text(json.dumps({
        "scope": "SMOKE ONLY", "source_episode": 9,
        "source_revision": "7df7a6f967416c9bd7c78e48f8b49f216d504eac",
        "rule": "For quantile span <= 1e-5 and raw physical span > 1e-4, use raw min/max",
        "changes": changes,
    }, indent=2) + "\n")
    print("CALIBRATION", changes)


if __name__ == "__main__":
    main()
