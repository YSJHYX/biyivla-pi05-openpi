"""Small robot-side transport; imports no Torch, JAX or model code."""

import contextlib
import socket

import numpy as np
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy


class RemotePolicyClient(WebsocketClientPolicy):
    def infer(self, obs):
        # Reuse the old robot client's resize, keeping network images at 224².
        obs = dict(obs)
        obs["images"] = {name: image_tools.resize_with_pad(image, 224, 224) for name, image in obs["images"].items()}
        return super().infer(obs)

    def close(self):
        # Official sync client has no public close(). Interrupt a pending recv
        # before joining the inference thread, including during Ctrl-C.
        with contextlib.suppress(OSError):  # Already disconnected.
            self._ws.socket.shutdown(socket.SHUT_RDWR)
        self._ws.close()


def predict_remote_chunk(policy, observation, *, action_prefix=None, inference_delay=0, prefix_horizon=None):
    request = dict(observation)
    if action_prefix is not None:
        request["rtc"] = {
            "action_prefix": action_prefix,
            "inference_delay": inference_delay,
            "prefix_horizon": prefix_horizon,
        }
    response = policy.infer(request)
    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.shape != (50, 17) or not np.isfinite(actions).all():
        raise ValueError("Model service must return finite actions[50,17]")
    return actions.copy()
