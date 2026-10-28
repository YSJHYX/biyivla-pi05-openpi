"""Compare numeric stats inputs to actual LeRobot v2.1 query semantics."""

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch

from scripts.biyi.compute_stage1_parquet_norm_stats import numeric_batches


def test_numeric_batches_match_lerobot_at_boundaries_and_tail():
    state = np.arange(7 * 17, dtype=np.float32).reshape(7, 17)
    action = state * 0.01
    loader = object.__new__(LeRobotDataset)
    loader.episode_data_index = {"from": torch.tensor([0, 3]), "to": torch.tensor([3, 7])}
    loader.delta_indices = {"action": list(range(50))}
    expected = []
    for i in range(7):
        indices, _ = loader._get_query_indices(i, int(i >= 3))  # noqa: SLF001 - audited loader semantics
        expected.append(action[indices["action"]])
    batches = list(numeric_batches([(state[:3], action[:3]), (state[3:], action[3:])], 4))
    # One complete batch crosses the episode boundary, then three individual tails.
    assert [x["state"].shape for x in batches] == [(4, 17), (17,), (17,), (17,)]
    got_state = np.concatenate([x["state"].reshape(-1, 17) for x in batches])
    got_action = np.concatenate([x["actions"].reshape(-1, 50, 17) for x in batches])
    np.testing.assert_array_equal(got_state, state)
    np.testing.assert_array_equal(got_action, np.stack(expected))
