import numpy as np
import pytest
import torch

from openpi.models_pytorch import rtc


@pytest.mark.parametrize("schedule", ["linear", "exp"])
def test_prefix_weights_freeze_then_decay_then_zero(schedule):
    weights = rtc.get_prefix_weights(2, 6, 10, schedule)

    np.testing.assert_allclose(weights[:2].numpy(), 1.0)
    assert torch.all(weights[2:6] < 1)
    assert torch.all(weights[2:6] > 0)
    np.testing.assert_allclose(weights[6:].numpy(), 0.0)


def test_prefix_weights_validate_horizons():
    with pytest.raises(ValueError, match="inference_delay"):
        rtc.get_prefix_weights(4, 3, 10)


def test_guidance_moves_physical_dims_toward_prefix_and_ignores_padding():
    x_t = torch.zeros((1, 3, 4), dtype=torch.float32)
    action_prefix = torch.full_like(x_t, 100.0)
    action_prefix[..., :2] = 1.0
    weights = torch.ones((1, 3, 1), dtype=torch.float32)
    action_dim_mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
    time = torch.tensor(0.5)

    velocity, metrics = rtc.guided_velocity(
        lambda latent: latent * 0,
        x_t,
        action_prefix,
        time,
        weights,
        action_dim_mask,
        max_guidance_weight=5.0,
        collect_metrics=True,
    )
    next_x = x_t - 0.1 * velocity

    assert torch.all(next_x[..., :2] > x_t[..., :2])
    np.testing.assert_allclose(next_x[..., 2:].numpy(), 0.0)
    assert metrics["prefix_error_norm"] > 0
    assert metrics["correction_norm"] > 0
