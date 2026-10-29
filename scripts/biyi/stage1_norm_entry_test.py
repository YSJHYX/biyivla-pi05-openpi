"""Ensure the public stats entry preserves its effective split and batch order."""

from types import SimpleNamespace

import pytest

from scripts.biyi import compute_stage1_norm_stats as entry
from scripts.biyi import compute_stage1_parquet_norm_stats as numeric


def test_numeric_entry_keeps_effective_loader_batch_and_paths(monkeypatch, tmp_path):
    identity = {"train_episode_indices": list(range(117)), "train_frame_count": 77975, "local_root": str(tmp_path)}
    config = SimpleNamespace(assets_base_dir=tmp_path / "assets", batch_size=32)
    monkeypatch.setattr(entry, "validate", lambda **kwargs: (config, identity))
    calls = []
    monkeypatch.setattr(numeric, "main", calls.append)
    inventory = tmp_path / "inventory.json"
    entry.main(["--numeric-only", "--source-inventory", str(inventory)])
    options = dict(zip(calls[0][::2], calls[0][1::2], strict=True))
    assert options["--batch-size"] == "32"
    assert options["--dataset-root"] == str(tmp_path)
    assert options["--assets-base-dir"] == str(config.assets_base_dir)
    assert options["--source-inventory"] == str(inventory)


def test_numeric_entry_rejects_a_different_effective_split(monkeypatch, tmp_path):
    identity = {"train_episode_indices": list(range(100)), "train_frame_count": 77975, "local_root": str(tmp_path)}
    monkeypatch.setattr(entry, "validate", lambda **kwargs: (SimpleNamespace(), identity))
    monkeypatch.setattr(numeric, "main", lambda args: pytest.fail("must reject before computing stats"))
    with pytest.raises(ValueError, match="effective training split"):
        entry.main(["--numeric-only", "--source-inventory", str(tmp_path / "inventory.json")])


def test_numeric_entry_requires_pinned_inventory():
    with pytest.raises(SystemExit) as error:
        entry.main(["--numeric-only"])
    assert error.value.code == 2
