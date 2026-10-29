"""Serve a trained BIYIVLA-PI05-OPENPI checkpoint over WebSocket.

Request observations use the inference-side BiyiVLA contract::

    {
        "state": np.ndarray[17],
        "images": {
            "head": np.ndarray[H, W, 3],
            "wrist": np.ndarray[H, W, 3],
        },
        "prompt": "place the motor into the fixture",
    }

Responses contain ``actions`` with shape ``[50, 17]``. The policy keeps the
stock pi0.5 model action dimension at 32 internally and applies the repository's
17 -> 32 -> 17 BiyiVLA transforms around the model.
"""

import dataclasses
import logging

import torch
import tyro

from openpi.policies import biyivla_policy
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint_dir: str
    """Checkpoint step directory containing model.safetensors and assets/."""

    config_name: str = "pi05_biyi_stage1"
    default_prompt: str | None = None
    device: str | None = None
    num_steps: int = 10
    host: str = "0.0.0.0"
    port: int = 8000
    seed: int | None = None


def main(args: Args) -> None:
    config = _config.get_config(args.config_name)
    if not isinstance(config.data, _config.LeRobotBiyiVLADataConfig):
        raise ValueError(f"Config {args.config_name!r} is not a BiyiVLA config")
    if not config.model.pi05 or config.model.action_dim != 32 or config.model.action_horizon != 50:
        raise ValueError("BIYIVLA-PI05-OPENPI inference requires pi0.5, action_dim=32, action_horizon=50")
    if args.num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        sample_kwargs={"num_steps": args.num_steps},
        default_prompt=args.default_prompt,
        pytorch_device=args.device,
    )
    metadata = {
        **policy.metadata,
        "config_name": config.name,
        "action_horizon": config.model.action_horizon,
        "model_action_dim": config.model.action_dim,
        "physical_action_dim": biyivla_policy.PHYSICAL_DIM,
        "observation_contract": "BIYIVLA-PI05-OPENPI",
        "rtc": False,
    }
    logging.info(
        "Serving BiyiVLA pi0.5 checkpoint on ws://%s:%d (model action_dim=32, physical action_dim=17)",
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
