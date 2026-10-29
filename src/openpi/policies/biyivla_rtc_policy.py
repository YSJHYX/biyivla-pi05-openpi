"""BiyiVLA policy wrapper for inference-time Real-Time Chunking."""

from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any

import jax
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from torch import Tensor

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models_pytorch import rtc as _rtc
from openpi.policies import biyivla_policy
from openpi.shared import download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config

logger = logging.getLogger(__name__)

RTC_REQUEST_KEY = "rtc"


class BiyiVLARTCPolicy(_base_policy.BasePolicy):
    """PyTorch pi0.5 policy with optional RTC guidance per request.

    A request without the ``rtc`` key uses the stock sampler. A guided request
    has this additional structure::

        {
            "rtc": {
                "action_prefix": np.ndarray[T, 17],
                "inference_delay": int,
                "prefix_horizon": int,  # defaults to T
            }
        }

    ``action_prefix[0]`` must be aligned with the observation timestamp. The
    robot-side asynchronous controller should discard the actions actually
    consumed while this request was in flight before installing the response.
    """

    def __init__(
        self,
        model,
        *,
        input_transforms: Sequence[_transforms.DataTransformFn],
        output_transforms: Sequence[_transforms.DataTransformFn],
        norm_stats: dict[str, _transforms.NormStats],
        use_quantile_norm: bool,
        pytorch_device: str,
        num_steps: int,
        max_guidance_weight: float,
        prefix_attention_schedule: _rtc.PrefixAttentionSchedule,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._model = model.to(pytorch_device).eval()
        # RTC differentiates only with respect to the action latent. Freezing
        # weights avoids retaining parameter-gradient state during inference.
        self._model.requires_grad_(requires_grad=False)
        self._input_transform = _transforms.compose(input_transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._prefix_normalizer = _transforms.Normalize(norm_stats, use_quantiles=use_quantile_norm)
        self._device = pytorch_device
        self._num_steps = num_steps
        self._max_guidance_weight = max_guidance_weight
        self._prefix_attention_schedule = prefix_attention_schedule
        self._metadata = metadata or {}

    def infer(self, obs: dict) -> dict:  # type: ignore[override]
        request = dict(obs)
        rtc_request = request.pop(RTC_REQUEST_KEY, None)
        inputs = jax.tree.map(lambda value: value, request)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda value: torch.from_numpy(np.asarray(value)).to(self._device)[None, ...], inputs)
        observation = _model.Observation.from_dict(inputs)

        start_time = time.monotonic()
        rtc_metadata: dict[str, Any] = {"enabled": rtc_request is not None}
        if rtc_request is None:
            with torch.no_grad():
                actions = self._model.sample_actions(self._device, observation, num_steps=self._num_steps)
        else:
            prefix, inference_delay, prefix_horizon = prepare_rtc_action_prefix(
                rtc_request,
                prefix_normalizer=self._prefix_normalizer,
                action_horizon=self._model.config.action_horizon,
                model_action_dim=self._model.config.action_dim,
                device=self._device,
            )
            actions, guidance_metrics = _rtc.sample_actions_rtc(
                self._model,
                self._device,
                observation,
                action_prefix=prefix,
                physical_action_dim=biyivla_policy.PHYSICAL_DIM,
                inference_delay=inference_delay,
                prefix_horizon=prefix_horizon,
                num_steps=self._num_steps,
                max_guidance_weight=self._max_guidance_weight,
                prefix_attention_schedule=self._prefix_attention_schedule,
            )
            rtc_metadata.update(
                {
                    "inference_delay": inference_delay,
                    "prefix_horizon": prefix_horizon,
                    "prefix_attention_schedule": self._prefix_attention_schedule,
                    **guidance_metrics,
                }
            )

        model_time = time.monotonic() - start_time
        outputs = {
            "state": np.asarray(inputs["state"][0].detach().cpu()),
            "actions": np.asarray(actions[0].detach().cpu()),
        }
        outputs = self._output_transform(outputs)
        physical_actions = np.asarray(outputs["actions"])
        expected_shape = (self._model.config.action_horizon, biyivla_policy.PHYSICAL_DIM)
        if physical_actions.shape != expected_shape:
            raise ValueError(f"Expected BiyiVLA actions {expected_shape}, got {physical_actions.shape}")
        if not np.isfinite(physical_actions).all():
            raise ValueError("BiyiVLA inference produced NaN or Inf actions")

        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        outputs["rtc"] = rtc_metadata
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def prepare_rtc_action_prefix(
    rtc_request: dict,
    *,
    prefix_normalizer: _transforms.Normalize,
    action_horizon: int,
    model_action_dim: int,
    device: str,
) -> tuple[Tensor, int, int]:
    """Validate, normalize, and 17D -> 32D pad a physical RTC prefix."""
    if not isinstance(rtc_request, dict):
        raise TypeError("rtc must be a mapping")
    if "action_prefix" not in rtc_request:
        raise ValueError("RTC request is missing action_prefix")

    physical_prefix = np.asarray(rtc_request["action_prefix"], dtype=np.float32)
    if physical_prefix.ndim != 2 or physical_prefix.shape[1] != biyivla_policy.PHYSICAL_DIM:
        raise ValueError(
            f"RTC action_prefix must have shape [T,{biyivla_policy.PHYSICAL_DIM}], got {physical_prefix.shape}"
        )
    if not 1 <= physical_prefix.shape[0] <= action_horizon:
        raise ValueError(f"RTC action_prefix length must be in [1,{action_horizon}], got {physical_prefix.shape[0]}")
    if not np.isfinite(physical_prefix).all():
        raise ValueError("RTC action_prefix contains NaN or Inf")

    prefix_horizon = int(rtc_request.get("prefix_horizon", physical_prefix.shape[0]))
    if prefix_horizon > physical_prefix.shape[0]:
        raise ValueError("RTC prefix_horizon cannot exceed the supplied action_prefix length")
    inference_delay = int(rtc_request.get("inference_delay", 0))
    if not 0 <= inference_delay <= prefix_horizon:
        raise ValueError("RTC requires 0 <= inference_delay <= prefix_horizon")

    normalized = prefix_normalizer({"actions": physical_prefix.copy()})["actions"]
    padded = np.zeros((action_horizon, model_action_dim), dtype=np.float32)
    padded[: physical_prefix.shape[0], : biyivla_policy.PHYSICAL_DIM] = normalized
    prefix = torch.from_numpy(padded).to(device)[None, ...]
    return prefix, inference_delay, prefix_horizon


def create_biyivla_rtc_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    default_prompt: str | None = None,
    pytorch_device: str | None = None,
    num_steps: int = 10,
    max_guidance_weight: float = 5.0,
    prefix_attention_schedule: _rtc.PrefixAttentionSchedule = "exp",
) -> BiyiVLARTCPolicy:
    """Load a trained BiyiVLA PyTorch checkpoint as an RTC-capable policy."""
    if not isinstance(train_config.data, _config.LeRobotBiyiVLADataConfig):
        raise ValueError(f"Config {train_config.name!r} is not a BiyiVLA config")
    if not train_config.model.pi05 or train_config.model.action_dim != 32:
        raise ValueError("BiyiVLA RTC requires stock pi0.5 with action_dim=32")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if max_guidance_weight <= 0:
        raise ValueError("max_guidance_weight must be positive")

    local_checkpoint = download.maybe_download(str(checkpoint_dir))
    weight_path = local_checkpoint / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"RTC currently requires a PyTorch checkpoint: {weight_path}")

    logger.info("Loading BiyiVLA pi0.5 PyTorch checkpoint from %s", weight_path)
    model = train_config.model.load_pytorch(train_config, str(weight_path))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("BiyiVLA checkpoint must define an asset_id for norm stats")
    norm_stats = _checkpoints.load_norm_stats(local_checkpoint / "assets", data_config.asset_id)
    if norm_stats is None:
        raise ValueError("BiyiVLA checkpoint does not contain normalization statistics")

    if pytorch_device is None:
        pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
    metadata = {
        **(train_config.policy_metadata or {}),
        "config_name": train_config.name,
        "action_horizon": train_config.model.action_horizon,
        "model_action_dim": train_config.model.action_dim,
        "physical_action_dim": biyivla_policy.PHYSICAL_DIM,
        "rtc_request_key": RTC_REQUEST_KEY,
    }
    return BiyiVLARTCPolicy(
        model,
        input_transforms=[
            _transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        norm_stats=norm_stats,
        use_quantile_norm=data_config.use_quantile_norm,
        pytorch_device=pytorch_device,
        num_steps=num_steps,
        max_guidance_weight=max_guidance_weight,
        prefix_attention_schedule=prefix_attention_schedule,
        metadata=metadata,
    )
