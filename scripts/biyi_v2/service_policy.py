"""WebSocket-facing adapter around the native, checkpoint-loaded Policy."""

from openpi_client import base_policy

from scripts.biyi_v2.rtc_adapter import predict_rtc_chunk


class RTCServicePolicy(base_policy.BasePolicy):
    def __init__(self, policy, max_guidance_weight=10.0):
        self.policy = policy
        self.max_guidance_weight = max_guidance_weight

    def infer(self, observation):
        observation = dict(observation)
        rtc_request = observation.pop("rtc", None)
        if rtc_request is None:
            return self.policy.infer(observation)
        actions = predict_rtc_chunk(
            self.policy,
            observation,
            rtc_request["action_prefix"],
            inference_delay=rtc_request["inference_delay"],
            prefix_horizon=rtc_request["prefix_horizon"],
            max_guidance_weight=self.max_guidance_weight,
        )
        return {"actions": actions}
