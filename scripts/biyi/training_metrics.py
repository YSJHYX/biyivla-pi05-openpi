"""Stage-1 exposure and artifact metrics; no model or optimizer changes."""

import json
from pathlib import Path


def train_frame_count(root, selected=None):
    root = Path(root)
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
    indices = list(range(len(episodes))) if selected is None else list(selected)
    if not indices or len(indices) != len(set(indices)) or any(i < 0 or i >= len(episodes) for i in indices):
        raise ValueError("Invalid train episode selection")
    frames = sum(episodes[i]["length"] for i in indices)
    if frames <= 0:
        raise ValueError("Training requires positive frame count")
    return frames


def exposure_metrics(completed_steps, batch_size, frames):
    if completed_steps < 0 or batch_size <= 0 or frames <= 0:
        raise ValueError("Invalid training exposure")
    samples = completed_steps * batch_size
    return {"global_step": completed_steps, "samples_seen": samples, "equivalent_epochs": samples / frames}


def append_metrics(path, payload):
    # Refuse NaN/Inf before touching the receipt, preserving parseable history.
    text = json.dumps(payload, sort_keys=True, allow_nan=False)
    with Path(path).open("a") as stream:
        stream.write(text + "\n")


def checkpoint_metrics(path, *, backup_verified):
    path = Path(path)
    return {
        "checkpoint_step": int(path.name),
        "checkpoint/model_bytes": (path / "model.safetensors").stat().st_size,
        "checkpoint/optimizer_bytes": (path / "optimizer.pt").stat().st_size,
        "checkpoint/total_bytes": sum(p.stat().st_size for p in path.rglob("*") if p.is_file()),
        "checkpoint/save_complete": 1,
        "checkpoint/backup_verified": int(backup_verified),
    }
