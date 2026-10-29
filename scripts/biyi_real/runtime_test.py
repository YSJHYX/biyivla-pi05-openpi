import time

import numpy as np
import pytest

from scripts.biyi_real import run_pi05_real
from scripts.biyi_real import runtime


def make_chunk(offset: float = 0.0) -> np.ndarray:
    values = np.arange(
        runtime.ACTION_HORIZON * runtime.ACTION_DIM,
        dtype=np.float64,
    ).reshape(runtime.ACTION_HORIZON, runtime.ACTION_DIM)
    return values + offset


def test_split_action_preserves_absolute_radian_values():
    action = np.linspace(-2.0, 2.0, runtime.ACTION_DIM)

    arm, hand = runtime.split_action(action)

    np.testing.assert_array_equal(arm, action[:6])
    np.testing.assert_array_equal(hand, action[6:])
    assert arm.shape == (6,)
    assert hand.shape == (11,)


def test_async_response_discards_rows_consumed_since_observation():
    initial = make_chunk()
    response = make_chunk(10_000.0)
    buffer = runtime.ActionChunkBuffer()
    buffer.install_initial(initial)

    for row in range(8):
        np.testing.assert_array_equal(buffer.next_action(), initial[row])
    request_tick = buffer.executed_total
    prefix = buffer.remaining
    np.testing.assert_array_equal(prefix, initial[8:])

    for row in range(8, 13):
        np.testing.assert_array_equal(buffer.next_action(), initial[row])
    actual_delay = buffer.install_response(response, request_tick=request_tick)

    assert actual_delay == 5
    np.testing.assert_array_equal(buffer.next_action(), response[5])


def test_short_delay_aligned_chunk_requests_again_before_it_drains():
    buffer = runtime.ActionChunkBuffer()
    buffer.install_initial(make_chunk())
    for _ in range(8):
        buffer.next_action()
    request_tick = buffer.executed_total
    for _ in range(43):
        buffer.next_action()

    buffer.install_response(make_chunk(10_000.0), request_tick=request_tick)

    assert len(buffer.remaining) == 7
    assert buffer.request_due(open_loop_horizon=8)


def test_rtc_request_uses_unexecuted_17d_tail_and_bounded_delay():
    base = {"state": np.zeros(17)}
    prefix = make_chunk()[8:]

    request = runtime.add_rtc_request(
        base,
        action_prefix=prefix,
        inference_delay=9,
    )

    assert request["rtc"]["action_prefix"].shape == (42, 17)
    assert request["rtc"]["inference_delay"] == 9
    assert request["rtc"]["prefix_horizon"] == 42
    assert "rtc" not in base


def test_latency_estimator_rounds_up_to_control_ticks_and_caps_to_prefix():
    estimator = runtime.LatencyEstimator(30.0)
    estimator.observe(0.201)

    assert estimator.predicted_ticks(maximum=50) == 7
    assert estimator.predicted_ticks(maximum=4) == 4


def test_chunk_exhaustion_holds_last_absolute_target_without_modification():
    chunk = make_chunk()
    buffer = runtime.ActionChunkBuffer()
    buffer.install_initial(chunk)
    for _ in range(runtime.ACTION_HORIZON):
        buffer.next_action()

    held = buffer.next_action()

    np.testing.assert_array_equal(held, chunk[-1])
    assert buffer.hold_count == 1


@pytest.mark.parametrize(
    "metadata",
    [
        {"action_horizon": 49, "physical_action_dim": 17},
        {"action_horizon": 50, "physical_action_dim": 16},
        {"observation_contract": "other"},
    ],
)
def test_server_metadata_rejects_non_biyivla_contract(metadata):
    with pytest.raises(ValueError, match="must be|Unexpected"):
        run_pi05_real.validate_server_metadata(metadata)


class FakeClient:
    def get_server_metadata(self):
        return {"rtc": True, "action_horizon": 50, "physical_action_dim": 17}

    def infer(self, observation):
        time.sleep(0.001)
        return {"actions": make_chunk(), "echo": observation["prompt"]}


def test_policy_worker_owns_sync_client_and_returns_valid_chunk():
    worker = runtime.AsyncPolicyWorker(FakeClient)
    worker.start()
    assert worker.wait_ready(timeout_s=1.0)["rtc"] is True
    request = worker.submit({"prompt": "move"}, request_tick=12)

    result = worker.wait_result(timeout_s=1.0)

    assert result.request == request
    assert result.response["echo"] == "move"
    np.testing.assert_array_equal(runtime.result_actions(result), make_chunk())
    worker.close()


def test_invalid_policy_actions_are_not_clipped_or_silently_accepted():
    with pytest.raises(ValueError, match="NaN"):
        runtime.validate_action_chunk(np.full((50, 17), np.nan))
    with pytest.raises(ValueError, match="shape"):
        runtime.validate_action_chunk(np.zeros((50, 16)))
