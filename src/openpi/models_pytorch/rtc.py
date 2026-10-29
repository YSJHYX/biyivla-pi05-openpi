"""Inference-time Real-Time Chunking guidance for the PyTorch pi0 family.

The implementation follows Algorithm 1 of "Real-Time Execution of Action
Chunking Flow Policies" while using openpi's reversed flow-time convention
(t=1 is noise and t=0 is the action distribution).
"""

from collections.abc import Callable
import math
from typing import Literal, Protocol

import torch
from torch import Tensor

from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

PrefixAttentionSchedule = Literal["exp", "linear", "ones", "zeros"]


class Pi0PytorchForRTC(Protocol):
    """The subset of PI0Pytorch used by the RTC sampler."""

    config: object
    paligemma_with_expert: object

    def sample_noise(self, shape: tuple[int, ...], device: str | torch.device) -> Tensor: ...

    def _preprocess_observation(self, observation, *, train: bool): ...

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks): ...

    def _prepare_attention_masks_4d(self, att_2d_masks: Tensor) -> Tensor: ...

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep) -> Tensor: ...


def get_prefix_weights(
    inference_delay: int,
    prefix_horizon: int,
    action_horizon: int,
    schedule: PrefixAttentionSchedule = "exp",
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build the RTC soft mask over action time.

    The first ``inference_delay`` actions are frozen, weights then decay to
    zero at ``prefix_horizon``, and actions beyond the real previous-chunk
    prefix receive no guidance.
    """
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    if not 0 <= inference_delay <= prefix_horizon <= action_horizon:
        raise ValueError(
            "Expected 0 <= inference_delay <= prefix_horizon <= action_horizon, "
            f"got {inference_delay=}, {prefix_horizon=}, {action_horizon=}"
        )
    if schedule not in {"exp", "linear", "ones", "zeros"}:
        raise ValueError(f"Unknown RTC prefix attention schedule: {schedule}")

    positions = torch.arange(action_horizon, device=device, dtype=dtype)
    if schedule == "ones":
        weights = (positions < prefix_horizon).to(dtype)
    elif schedule == "zeros":
        weights = (positions < inference_delay).to(dtype)
    else:
        denominator = prefix_horizon - inference_delay + 1
        weights = torch.clamp((inference_delay - 1 - positions) / denominator + 1, 0, 1)
        if schedule == "exp":
            weights = weights * torch.expm1(weights) / (math.e - 1)
        weights = torch.where(positions < prefix_horizon, weights, 0)
    return weights


def guided_velocity(
    denoise: Callable[[Tensor], Tensor],
    x_t: Tensor,
    action_prefix: Tensor,
    time: Tensor,
    prefix_weights: Tensor,
    action_dim_mask: Tensor,
    max_guidance_weight: float,
    *,
    collect_metrics: bool = False,
) -> tuple[Tensor, dict[str, float]]:
    """Apply one pseudoinverse-guided RTC correction to a velocity field."""
    if max_guidance_weight <= 0:
        raise ValueError("max_guidance_weight must be positive")
    if x_t.shape != action_prefix.shape:
        raise ValueError(f"RTC prefix shape {action_prefix.shape} does not match latent shape {x_t.shape}")

    x_with_grad = x_t.detach().requires_grad_(requires_grad=True)
    with torch.enable_grad():
        base_velocity = denoise(x_with_grad)
        # openpi integrates from t=1 (noise) to t=0 (actions). This is the
        # paper's denoised estimate after converting from tau=1-t.
        denoised_estimate = x_with_grad - time * base_velocity
        error = (action_prefix - denoised_estimate) * prefix_weights * action_dim_mask
        correction = torch.autograd.grad(
            denoised_estimate,
            x_with_grad,
            grad_outputs=error.detach(),
            retain_graph=False,
            create_graph=False,
        )[0]

    tau = 1 - time
    max_weight = torch.as_tensor(max_guidance_weight, device=x_t.device, dtype=x_t.dtype)
    squared_one_minus_tau = (1 - tau) ** 2
    inv_r2 = (squared_one_minus_tau + tau**2) / squared_one_minus_tau
    coefficient = torch.nan_to_num((1 - tau) / tau, posinf=max_guidance_weight)
    guidance_weight = torch.minimum(
        torch.nan_to_num(coefficient * inv_r2, posinf=max_guidance_weight),
        max_weight,
    )
    velocity = base_velocity - guidance_weight * correction
    metrics = {}
    if collect_metrics:
        metrics = {
            "guidance_weight": float(guidance_weight.detach().cpu()),
            "prefix_error_norm": float(torch.linalg.vector_norm(error.detach()).cpu()),
            "correction_norm": float(torch.linalg.vector_norm(correction.detach()).cpu()),
        }
    return velocity.detach(), metrics


def sample_actions_rtc(
    model: Pi0PytorchForRTC,
    device: str | torch.device,
    observation,
    *,
    action_prefix: Tensor,
    physical_action_dim: int,
    inference_delay: int,
    prefix_horizon: int,
    num_steps: int = 10,
    max_guidance_weight: float = 5.0,
    prefix_attention_schedule: PrefixAttentionSchedule = "exp",
    noise: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Sample one pi0.5 chunk with RTC guidance in normalized model space."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    action_horizon = model.config.action_horizon
    model_action_dim = model.config.action_dim
    batch_size = observation.state.shape[0]
    expected_shape = (batch_size, action_horizon, model_action_dim)
    if tuple(action_prefix.shape) != expected_shape:
        raise ValueError(f"Expected normalized RTC prefix {expected_shape}, got {tuple(action_prefix.shape)}")
    if not 0 < physical_action_dim <= model_action_dim:
        raise ValueError(f"Invalid physical action dimension: {physical_action_dim}")
    if not torch.isfinite(action_prefix).all():
        raise ValueError("RTC action prefix contains NaN or Inf")

    images, image_masks, language_tokens, language_masks, state = model._preprocess_observation(  # noqa: SLF001
        observation, train=False
    )
    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, image_masks, language_tokens, language_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

    if noise is None:
        noise = model.sample_noise(expected_shape, device)
    elif tuple(noise.shape) != expected_shape:
        raise ValueError(f"Expected RTC noise {expected_shape}, got {tuple(noise.shape)}")

    weights = get_prefix_weights(
        inference_delay,
        prefix_horizon,
        action_horizon,
        prefix_attention_schedule,
        device=device,
        dtype=torch.float32,
    )[None, :, None]
    action_dim_mask = torch.zeros((1, 1, model_action_dim), device=device, dtype=torch.float32)
    action_dim_mask[..., :physical_action_dim] = 1.0

    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    x_t = noise
    last_metrics: dict[str, float] = {}
    for step in range(num_steps):
        time = torch.tensor(1.0 - step / num_steps, dtype=torch.float32, device=device)
        expanded_time = time.expand(batch_size)

        def denoise(latent: Tensor, step_time: Tensor = expanded_time) -> Tensor:
            return model.denoise_step(state, prefix_pad_masks, past_key_values, latent, step_time)

        velocity, last_metrics = guided_velocity(
            denoise,
            x_t,
            action_prefix,
            time,
            weights,
            action_dim_mask,
            max_guidance_weight,
            collect_metrics=step == num_steps - 1,
        )
        x_t = (x_t + dt * velocity).detach()

    return x_t, last_metrics
