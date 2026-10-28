"""CPU numeric-only stats for the stock absolute-action Biyi Stage-1 adapter.

Uses stock RunningStats and LeRobot's episode-clamped action chunks. Every
metadata/Parquet input is checked against a pinned HF file inventory. Media is
not needed for numeric stats; the normal formal preflight still checks the full
frozen dataset, base, loader, credentials and hardware smoke before training.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi.shared import normalize
from scripts.biyi import checkpoint_store as store

META_FILES = ("info.json", "episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl")


def numeric_batches(episodes, batch_size, horizon=50):
    """Keep the same frame order, batch boundaries and final tail as stock stats."""
    states, actions = [], []
    for state, action in episodes:
        for i in range(len(state)):
            states.append(state[i])
            actions.append(action[np.minimum(np.arange(i, i + horizon), len(action) - 1)])
            if len(states) == batch_size:
                yield {"state": np.stack(states), "actions": np.stack(actions)}
                states, actions = [], []
    # Stock Stage-1 handles dropped final frames one at a time.
    for state, action in zip(states, actions, strict=True):
        yield {"state": state, "actions": action}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--assets-base-dir", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, default=store.RECIPE_117)
    args = parser.parse_args()
    settings = store.recipe(args.recipe)
    source = store.read_json(args.source_inventory)
    root = args.dataset_root.resolve()
    store.require(source["repo_id"] == store.DATASET_REPO, "Wrong numeric source repo")
    store.require(source["revision"] == settings["dataset_revision"], "Wrong pinned numeric source revision")

    def verified(name):
        path = store.file_at(root, name)
        data = path.read_bytes()
        entry = source["files"][name]
        store.require(len(data) == entry["size"], f"Input size mismatch: {name}")
        if entry.get("lfs"):
            actual = hashlib.sha256(data).hexdigest()
            expected = entry["lfs"]["sha256"]
        else:
            actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            expected = entry["blob_id"]
        store.require(actual == expected, f"Input content mismatch: {name}")
        return path

    for name in META_FILES:
        verified(f"meta/{name}")
    info = store.read_json(root / "meta/info.json")
    episodes = [json.loads(x) for x in (root / "meta/episodes.jsonl").read_text().splitlines()]
    count = settings["dataset_episodes"]
    store.require(info["total_episodes"] == len(episodes) == count, "Episode count mismatch")
    store.require(
        info["total_frames"] == sum(x["length"] for x in episodes) == settings["dataset_frames"], "Frame total mismatch"
    )
    store.require([x["episode_index"] for x in episodes] == list(range(count)), "Episode order mismatch")
    store.require(info["splits"] == {"train": f"0:{count}"}, "Expected all-train frozen snapshot")
    for key in ("action", "observation.state"):
        store.require(
            info["features"][key]["dtype"] == "float32" and info["features"][key]["shape"] == [17],
            "Wrong physical schema",
        )

    def vectors():
        offset = 0
        for i, episode in enumerate(episodes):
            name = info["data_path"].format(episode_chunk=i // info["chunks_size"], episode_index=i)
            table = pq.read_table(
                verified(name), columns=["episode_index", "index", "frame_index", "observation.state", "action"]
            )
            n = episode["length"]
            store.require(table.num_rows == n, "Parquet length mismatch")
            store.require(
                table["episode_index"].to_pylist() == [i] * n
                and table["index"].to_pylist() == list(range(offset, offset + n))
                and table["frame_index"].to_pylist() == list(range(n)),
                "Parquet numbering mismatch",
            )
            state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            store.require(
                state.shape == action.shape == (n, 17) and np.isfinite(state).all() and np.isfinite(action).all(),
                "Invalid physical vectors",
            )
            yield state, action
            offset += n

    output = args.assets_base_dir / "pi05_biyi_stage1/biyi_stage1"
    store.require(not output.exists(), "Stats destination already exists; never overwrite")
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    counts = dict.fromkeys(stats, 0)
    for index, batch in enumerate(numeric_batches(vectors(), settings["batch_size"])):
        for key, running in stats.items():
            running.update(batch[key])
            counts[key] += batch[key].reshape(-1, 17).shape[0]
        if index % 1000 == 0:
            print(f"NUMERIC_STATS_BATCH {index}", flush=True)
    result = {key: value.get_statistics() for key, value in stats.items()}
    store.require(
        counts["state"] == info["total_frames"] and counts["actions"] == info["total_frames"] * 50,
        "Missing frames/chunks",
    )
    for stat in result.values():
        store.require(
            all(
                np.asarray(getattr(stat, field)).shape == (17,) and np.isfinite(getattr(stat, field)).all()
                for field in ("mean", "std", "q01", "q99")
            ),
            "Invalid norm stats",
        )
    normalize.save(output, result)
    manifest = {
        "repo_id": store.DATASET_REPO,
        "local_root": str(root),
        "train_episode_indices": list(range(count)),
        "train_frame_count": info["total_frames"],
        "metadata_sha256": {name: store.digest(root / "meta" / name) for name in META_FILES},
        "norm_stats_sha256": store.digest(output / "norm_stats.json"),
    }
    store.write_json(output / "stage1_train_stats_manifest.json", manifest)
    store.write_json(
        output / "numeric_stats_source.json",
        {
            "repo_id": source["repo_id"],
            "revision": source["revision"],
            "inventory_sha256": store.digest(args.source_inventory),
            "source_commit": store.git("rev-parse", "HEAD"),
            "method": "stock_RunningStats_episode_clamped_50_step_actions",
            "state_count": counts["state"],
            "action_count": counts["actions"],
        },
    )
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
