import numpy as np
import pytest
import torch

from openpi import transforms
from openpi.policies import biyivla_rtc_policy


def make_prefix_normalizer():
    stats = transforms.NormStats(
        mean=np.zeros(17),
        std=np.ones(17),
        q01=np.zeros(17),
        q99=np.full(17, 2.0),
    )
    return transforms.Normalize({"actions": stats}, use_quantiles=True)


def prepare(request):
    return biyivla_rtc_policy.prepare_rtc_action_prefix(
        request,
        prefix_normalizer=make_prefix_normalizer(),
        action_horizon=50,
        model_action_dim=32,
        device="cpu",
    )


def test_prepare_rtc_prefix_normalizes_17d_then_pads_to_32d():
    physical = np.ones((7, 17), dtype=np.float32)

    prefix, delay, horizon = prepare({"action_prefix": physical, "inference_delay": 2, "prefix_horizon": 7})

    assert prefix.shape == (1, 50, 32)
    np.testing.assert_allclose(prefix[0, :7, :17].numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(prefix[0, :, 17:].numpy(), 0.0)
    np.testing.assert_allclose(prefix[0, 7:].numpy(), 0.0)
    assert (delay, horizon) == (2, 7)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"action_prefix": np.zeros((3, 16))}, "shape"),
        ({"action_prefix": np.zeros((51, 17))}, "length"),
        ({"action_prefix": np.zeros((3, 17)), "inference_delay": 4}, "inference_delay"),
    ],
)
def test_prepare_rtc_prefix_rejects_invalid_contract(payload, message):
    with pytest.raises(ValueError, match=message):
        prepare(payload)


def test_prepare_rtc_prefix_returns_float_tensor():
    prefix, _, _ = prepare({"action_prefix": np.zeros((1, 17), dtype=np.float64)})
    assert prefix.dtype == torch.float32
