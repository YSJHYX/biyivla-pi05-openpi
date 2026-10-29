# BiyiVLA π0.5 inference

Both servers use the `BIYIVLA-PI05-OPENPI` contract. The robot-facing state and
action dimensions are 17; the stock π0.5 model remains 32-dimensional inside
the policy. Checkpoints must be step directories containing:

```text
<checkpoint>/
├── model.safetensors
└── assets/biyi_stage1/norm_stats.json
```

## Standard inference

```bash
python -m scripts.biyi.infer_pi05 \
  --checkpoint-dir ./checkpoints/pi05_biyi_stage1/<run>/<step> \
  --device cuda:0 \
  --port 8000
```

The WebSocket request is:

```python
observation = {
    "state": state_17,
    "images": {"head": head_rgb, "wrist": wrist_rgb},
    "prompt": "place the motor into the fixture",
}
result = client.infer(observation)
assert result["actions"].shape == (50, 17)
```

Images may be HWC or CHW RGB. Floating-point images are interpreted as `[0,1]`;
`uint8` images are interpreted as `[0,255]`. The missing right-wrist camera is
zero-filled and masked by `BiyiVLAInputs`.

## RTC inference

RTC is inference-only guidance and does not require retraining. Start the RTC
server with:

```bash
python -m scripts.biyi.infer_pi05_rtc \
  --checkpoint-dir ./checkpoints/pi05_biyi_stage1/<run>/<step> \
  --device cuda:0 \
  --prefix-attention-schedule exp \
  --max-guidance-weight 5.0 \
  --port 8000
```

The first request has no previous chunk and may use the standard observation.
Subsequent requests add an RTC block:

```python
rtc_observation = {
    **observation,
    "rtc": {
        # Unexecuted previous actions, aligned so row 0 belongs to the same
        # control tick as this observation. Values are physical 17D actions.
        "action_prefix": previous_chunk_tail,
        # Conservative prediction of how many control ticks will be consumed
        # while this request is running.
        "inference_delay": predicted_delay_ticks,
        "prefix_horizon": len(previous_chunk_tail),
    },
}
new_chunk = client.infer(rtc_observation)["actions"]
```

The robot-side controller must be asynchronous:

1. Continue consuming the previous chunk while the RTC request is in flight.
2. Count the actions actually consumed during that request.
3. When the new `[50,17]` chunk arrives, discard that many leading actions from
   the new chunk before installing it.
4. Start the next request after the chosen minimum execution horizon, passing
   the then-unexecuted tail as the next `action_prefix`.

The RTC sampler normalizes the 17D prefix with the checkpoint's action stats,
pads it to 32D, and applies pseudoinverse guidance at every flow step. The
guidance error is explicitly masked to the first 17 dimensions so the 15 model
padding dimensions cannot influence the correction.

`inference_delay` must satisfy:

```text
0 <= inference_delay <= prefix_horizon <= 50
```

Start conservatively. Measure the real warm inference latency at the robot's
control frequency, and ensure the previous chunk cannot run empty while a new
request is outstanding. RTC smooths asynchronous chunk replacement; it is not
a safety controller and does not enforce joint, velocity, collision, or gripper
limits.

## RM65 + SG100 real-hardware client

The standard and RTC 17D outputs can be executed through AVP-SG100's existing
real-hardware backends with `scripts.biyi_real.run_pi05_real`. Setup, launch
commands, action alignment, and the current SG100 state-source limitation are
documented in [`scripts/biyi_real/README.md`](../biyi_real/README.md).
