"""Run v2 native/RTC with the existing RM65 + SG100 startup, over WebSocket.

Terminal 1, MODEL environment, from the OpenPI checkout:
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m scripts.biyi_v2.serve_policy \
    --checkpoint-dir CHECKPOINT --mode rtc --device cuda:0 --host 0.0.0.0 --port 8000
  Native baseline: replace --mode rtc with --mode native.

Terminal 2, existing AVP-SG100 environment, from the same checkout:
  PYTHONPATH=.:packages/openpi-client/src AVP_PYTHON -m scripts.biyi_v2.run_pi05_real \
    --avp-root AVP_ROOT --server-host 127.0.0.1 --server-port 8000 \
    --head-serial D455_SERIAL --wrist-serial D405_SERIAL --prompt "TASK"
  Replace AVP_PYTHON with that environment's Python, e.g.
  /home/khan/miniconda3/envs/avp_sg100/bin/python. CHECKPOINT, AVP_ROOT,
  D455_SERIAL, D405_SERIAL and TASK are user-supplied values, not literal paths.
  On separate computers, use the model computer's reachable IP as --server-host.

The client auto-detects native/RTC from server metadata. --max-control-steps 0
runs until Ctrl-C; a positive value limits actual control ticks. This command
performs the existing startup motion before prompting to begin policy execution.
"""

import argparse
import math
from pathlib import Path

from scripts.biyi_v2.real_robot import open_real_robot
from scripts.biyi_v2.remote_policy import RemotePolicyClient
from scripts.biyi_v2.rtc_runner import run_rtc_robot


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--head-serial", required=True)
    parser.add_argument("--wrist-serial", required=True)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--avp-root", type=Path, default=Path("/home/khan/AVP-SG100"))
    parser.add_argument("--arm-ip", default="192.168.1.18")
    parser.add_argument("--arm-port", type=int, default=8080)
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-baudrate", type=int, default=1_000_000)
    parser.add_argument("--hand-sdk-dir", type=Path, default=None)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--request-interval", type=int, default=10)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--initial-inference-delay", type=int, default=0)
    parser.add_argument("--max-control-steps", type=int, default=0)
    return parser


def validate_args(args):
    if not args.prompt.strip():
        raise ValueError("prompt must not be empty")
    if not math.isfinite(args.control_hz) or args.control_hz <= 0:
        raise ValueError("control_hz must be finite and positive")
    if args.request_interval <= 0 or args.execution_horizon <= 0:
        raise ValueError("request_interval and execution_horizon must be positive")
    if args.max_control_steps < 0 or args.initial_inference_delay < 0:
        raise ValueError("max_control_steps and initial_inference_delay must be nonnegative")


def validate_metadata(metadata):
    for name, expected in (("action_horizon", 50), ("physical_action_dim", 17)):
        if name in metadata and metadata[name] != expected:
            raise ValueError(f"Unexpected model service {name}: {metadata[name]}")
    if metadata.get("observation_contract", "BIYIVLA-PI05-OPENPI") != "BIYIVLA-PI05-OPENPI":
        raise ValueError("Model service observation contract does not match BiyiVLA")
    return bool(metadata.get("rtc", False))


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    validate_args(args)
    policy = RemotePolicyClient(args.server_host, args.server_port)
    try:
        metadata = policy.get_server_metadata()
        rtc_enabled = validate_metadata(metadata)
        print(f"[POLICY] connected; mode={'rtc' if rtc_enabled else 'native'}; metadata={metadata}")
        with open_real_robot(args) as robot:
            # Confirm BEFORE observing/sampling the initial chunk, so that an
            # operator pause never ages an already prepared first action chunk.
            input("启动位姿和 anchor 已完成。按 Enter 获取最新观测并开始推理执行; Ctrl-C 停止: ")
            run_rtc_robot(
                policy,
                robot.arm_backend,
                robot.hand_driver,
                robot.head_camera,
                robot.wrist_camera,
                robot.hand_state_proxy,
                args.prompt,
                num_ticks=args.max_control_steps or None,
                control_hz=args.control_hz,
                request_interval=args.request_interval,
                execution_horizon=args.execution_horizon,
                initial_inference_delay=args.initial_inference_delay,
                before_control=robot.enable,
                on_stop=robot.disable,
                rtc_enabled=rtc_enabled,
            )
    except KeyboardInterrupt:
        print("\n[REAL] Ctrl-C; stopped")
    finally:
        policy.close()


if __name__ == "__main__":
    main()
