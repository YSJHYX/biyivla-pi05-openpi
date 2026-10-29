import json

import pytest

from scripts.biyi.training_metrics import append_metrics
from scripts.biyi.training_metrics import exposure_metrics
from scripts.biyi.training_metrics import train_frame_count


def test_exposure_uses_start_frames_and_completed_optimizer_steps():
    result = exposure_metrics(19494, 4, 77975)
    assert result["global_step"] == 19494
    assert result["samples_seen"] == 77976
    assert result["equivalent_epochs"] == 77976 / 77975


def test_frame_count_respects_explicit_train_selection(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/episodes.jsonl").write_text("\n".join(json.dumps({"length": n}) for n in (12, 17, 21)))
    assert train_frame_count(tmp_path, (0, 2)) == 33
    assert train_frame_count(tmp_path) == 50
    with pytest.raises(ValueError, match="Invalid train episode selection"):
        train_frame_count(tmp_path, (0, 0))


def test_nonfinite_metric_cannot_corrupt_local_history(tmp_path):
    path = tmp_path / "history.jsonl"
    append_metrics(path, {"loss": 0.5})
    with pytest.raises(ValueError, match="Out of range float values"):
        append_metrics(path, {"loss": float("nan")})
    assert path.read_text().splitlines() == ['{"loss": 0.5}']
