"""Read-only validation of the pinned Task 3B Hugging Face snapshot."""

import dataclasses
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq

from openpi import transforms
from openpi.policies.biyivla_policy import BiyiVLAOutputs
from openpi.training.config import get_config
from openpi.training.data_loader import create_torch_dataset
from openpi.training.data_loader import transform_dataset

ROOT = Path("/data/projects/biyivla_pi05/task3b_dataset")
EXPECTED = ("v2.1", "rm65_sg100", 30, 10, 12285, 2)


def records(name):
    return [json.loads(line) for line in (ROOT / "meta" / name).read_text().splitlines()]


def main():
    info = json.loads((ROOT / "meta/info.json").read_text())
    assert (
        tuple(
            info[k] for k in ("codebase_version", "robot_type", "fps", "total_episodes", "total_frames", "total_tasks")
        )
        == EXPECTED
    )
    assert info["features"]["observation.state"]["shape"] == [17]
    assert info["features"]["action"]["shape"] == [17]
    episode = {r["episode_index"]: r for r in records("episodes.jsonl")}
    source = {r["lerobot_episode_index"]: r for r in records("source_provenance.jsonl")}
    tasks = {r["task_index"]: r["task"] for r in records("tasks.jsonl")}
    assert (episode[8]["length"], episode[9]["length"]) == (2247, 1327)
    assert source[8]["source_episode_id"] == "episode_000015"
    assert source[9]["source_episode_id"] == "episode_000016"
    assert tasks[1] == "place the motor into the fixture"
    assert source[9]["camera_transforms"]["wrist"]["rotation_deg"] == 180
    print("METADATA PASS", EXPECTED)

    for index in (8, 9):
        length = episode[index]["length"]
        chunk = index // info["chunks_size"]
        parquet = ROOT / info["data_path"].format(episode_chunk=chunk, episode_index=index)
        table = pq.read_table(parquet)
        assert table.num_rows == length
        assert parquet.stat().st_size > 0
        for key in ("observation.state", "action"):
            value = np.asarray(table[key].to_pylist(), dtype=np.float32)
            assert value.shape == (length, 17)
            assert np.isfinite(value).all()
        for key in ("observation.images.head", "observation.images.wrist"):
            video = ROOT / info["video_path"].format(episode_chunk=chunk, episode_index=index, video_key=key)
            assert video.stat().st_size > 0
            probe = json.loads(
                subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "stream=width,height,r_frame_rate,nb_frames:format=duration",
                        "-of",
                        "json",
                        str(video),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            )
            stream = probe["streams"][0]
            assert (stream["width"], stream["height"], stream["r_frame_rate"], int(stream["nb_frames"])) == (
                640,
                480,
                "30/1",
                length,
            )
            assert abs(float(probe["format"]["duration"]) - length / 30) < 0.2
            for frame in (0, length // 2, length - 1):
                rgb = subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-ss",
                        str(frame / 30),
                        "-i",
                        str(video),
                        "-frames:v",
                        "1",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-",
                    ],
                    capture_output=True,
                    check=True,
                ).stdout
                assert len(rgb) == 640 * 480 * 3
            print("EPISODE_MEDIA PASS", index, key, "frames", length)

    config = get_config("pi05_biyivla_smoke")
    data = config.data.create(config.assets_dirs, config.model)
    for key in ("state", "actions"):
        stat = data.norm_stats[key]
        assert all(a.shape == (17,) and np.isfinite(a).all() for a in (stat.mean, stat.std, stat.q01, stat.q99))
    raw = create_torch_dataset(data, 50, config.model)
    processed = transform_dataset(raw, data)
    for index in (0, len(raw) // 2, len(raw) - 1):
        sample = raw[index]
        out = processed[index]
        assert int(sample["episode_index"]) == 9
        assert int(sample["task_index"]) == 1
        assert sample["prompt"] == tasks[1]
        assert sample["observation.state"].shape == (17,)
        assert sample["action"].shape == (50, 17)
        assert out["state"].shape == (32,)
        assert out["actions"].shape == (50, 32)
        assert np.isfinite(out["state"]).all()
        assert np.isfinite(out["actions"]).all()
        assert not np.any(out["state"][17:])
        assert not np.any(out["actions"][:, 17:])
        assert out["image_mask"]["right_wrist_0_rgb"] == np.False_
        print("LOADER PASS", index, "timestamp", float(sample["timestamp"]), "prompt", sample["prompt"])

    sample = raw[663]
    physical = transforms.compose([*data.repack_transforms.inputs, *data.data_transforms.inputs])(sample)
    normalized = transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm)(physical)
    padded = transforms.PadStatesAndActions(32)(normalized)
    restored = transforms.Unnormalize(data.norm_stats, use_quantiles=data.use_quantile_norm)(
        {"state": padded["state"], "actions": padded["actions"]}
    )
    cropped = BiyiVLAOutputs()(restored)["actions"]
    assert cropped.shape == (50, 17)
    np.testing.assert_allclose(cropped, physical["actions"], atol=2e-3, rtol=1e-5)
    print("ACTION_ROUNDTRIP PASS", float(np.max(np.abs(cropped - physical["actions"]))))

    multi_base = dataclasses.replace(config.data.base_config, episode_indices=(8, 9))
    multi = dataclasses.replace(config.data, base_config=multi_base)
    multi_data = multi.create(config.assets_dirs, config.model)
    multi_raw = create_torch_dataset(multi_data, 50, config.model)
    assert multi_raw[0]["prompt"] == tasks[0]
    assert multi_raw[2247]["prompt"] == tasks[1]
    print("MULTITASK_ROUTING PASS")


if __name__ == "__main__":
    main()
