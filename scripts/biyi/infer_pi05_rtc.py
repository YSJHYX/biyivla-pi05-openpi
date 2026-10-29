"""Serve BIYIVLA-PI05-OPENPI with inference-time Real-Time Chunking.

The base observation contract matches ``scripts.biyi.infer_pi05``. Guided
requests add an ``rtc`` mapping::

    {
        ...,
        "rtc": {
            # Previous chunk tail aligned to this observation's timestamp.
            "action_prefix": np.ndarray[T, 17],
            # Predicted actions consumed while this inference is in flight.
            "inference_delay": 4,
            # Number of valid rows in action_prefix; defaults to T.
            "prefix_horizon": T,
        },
    }

The first request may omit ``rtc``. The robot client must keep executing the
previous chunk while awaiting the response and discard the number of returned
actions actually consumed during the request before installing the new chunk.
"""

import dataclasses
import logging

import torch
import tyro

from openpi.models_pytorch import rtc as _rtc
from openpi.policies import biyivla_rtc_policy
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint_dir: str
    """PyTorch checkpoint step directory containing model.safetensors and assets/."""

    config_name: str = "pi05_biyi_stage1"
    default_prompt: str | None = None
    device: str | None = None
    num_steps: int = 10
    max_guidance_weight: float = 5.0
    prefix_attention_schedule: _rtc.PrefixAttentionSchedule = "exp"
    host: str = "0.0.0.0"
    port: int = 8000
    seed: int | None = None


def main(args: Args) -> None:
    if args.seed is not None:
        torch.manual_seed(args.seed)
    config = _config.get_config(args.config_name)
    policy = biyivla_rtc_policy.create_biyivla_rtc_policy(
        config,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.device,
        num_steps=args.num_steps,
        max_guidance_weight=args.max_guidance_weight,
        prefix_attention_schedule=args.prefix_attention_schedule,
    )
    metadata = {
        **policy.metadata,
        "observation_contract": "BIYIVLA-PI05-OPENPI",
        "rtc": True,
        "rtc_prefix_attention_schedule": args.prefix_attention_schedule,
        "rtc_max_guidance_weight": args.max_guidance_weight,
    }
    logging.info(
        "Serving BiyiVLA pi0.5 RTC on ws://%s:%d (32D model space, guidance restricted to 17 physical dims)",
        args.host,
        args.port,
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
