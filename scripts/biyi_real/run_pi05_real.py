"""Execute BIYIVLA pi0.5 17D action chunks on RM65 + SG100 hardware.

The policy server stays in the OpenPI environment. This client runs in the
AVP-SG100 environment and imports its already-validated hardware backends.
It accepts either the standard or RTC server; RTC is detected from server
metadata and uses asynchronous prefix alignment automatically.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from scripts.biyi_real import runtime
from scripts.biyi_real.checkpoint_hand_backend import CheckpointSG100Backend

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AVP_ROOT = Path("/home/khan/AVP-SG100")
CONTROL_HZ = 30.0

# Identical to run_avp_arm_hand_sim.py's verified dual-hardware startup.
REAL_STARTUP_ARM_Q_RAD = np.deg2rad(
    np.array(
        [2.867, -105.170, 123.329, -10.844, -95.298, -73.636],
        dtype=np.float64,
    )
)
REAL_STARTUP_ARM_SPEED_PERCENT = 25


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run BIYIVLA pi0.5 17D checkpoint actions on RM65 + SG100 "
            "(arm joint radians; hand SG100 SDK-user radians)."
        )
    )
    parser.add_argument("--prompt", required=True, help="Task instruction sent to pi0.5")
    parser.add_argument("--head-serial", required=True, help="D455 serial number")
    parser.add_argument("--wrist-serial", required=True, help="D405 serial number")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--avp-root", type=Path, default=DEFAULT_AVP_ROOT)
    parser.add_argument("--arm-ip", default="192.168.1.18")
    parser.add_argument("--arm-port", type=int, default=8080)
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-baudrate", type=int, default=1_000_000)
    parser.add_argument(
        "--hand-sdk-dir",
        type=Path,
        default=None,
        help="SG100 SDK root; default uses AVP-SG100's normal lookup",
    )
    parser.add_argument(
        "--open-loop-horizon",
        type=int,
        default=8,
        help="Execute this many rows from each installed chunk before starting the next async request",
    )
    parser.add_argument(
        "--max-control-steps",
        type=int,
        default=0,
        help="Stop after N 30 Hz ticks; 0 runs until Ctrl-C",
    )
    return parser


def _bootstrap_import_paths(avp_root: Path) -> None:
    avp_root = avp_root.expanduser().resolve()
    if not (avp_root / "sg100_teleop").is_dir():
        raise FileNotFoundError(f"AVP-SG100 package not found under {avp_root}")
    openpi_client_src = REPO_ROOT / "packages" / "openpi-client" / "src"
    for path in (openpi_client_src, avp_root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _load_runtime_dependencies() -> dict[str, Any]:
    try:
        from openpi_client import image_tools
        from openpi_client import websocket_client_policy
    except ImportError as exc:
        raise RuntimeError(
            "The robot environment needs the small openpi-client runtime dependencies. "
            "Install scripts/biyi_real/requirements.txt in the AVP-SG100 environment."
        ) from exc

    from sg100_teleop.camera.realsense_camera import RealSenseCamera
    from sg100_teleop.robot.rm65_real import RM65RealBackend
    from sg100_teleop.robot.sg100_sdk_driver import SG100SDKDriver

    return {
        "image_tools": image_tools,
        "WebsocketClientPolicy": websocket_client_policy.WebsocketClientPolicy,
        "RealSenseCamera": RealSenseCamera,
        "RM65RealBackend": RM65RealBackend,
        "SG100SDKDriver": SG100SDKDriver,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.open_loop_horizon < runtime.ACTION_HORIZON:
        raise ValueError(f"--open-loop-horizon must be in [1,{runtime.ACTION_HORIZON - 1}]")
    if args.max_control_steps < 0:
        raise ValueError("--max-control-steps must be >= 0")
    if not args.prompt.strip():
        raise ValueError("--prompt must not be empty")


def validate_server_metadata(metadata: dict[str, Any]) -> bool:
    action_horizon = metadata.get("action_horizon")
    if action_horizon is not None and int(action_horizon) != runtime.ACTION_HORIZON:
        raise ValueError(f"Server action_horizon must be 50, got {action_horizon}")
    physical_dim = metadata.get("physical_action_dim")
    if physical_dim is not None and int(physical_dim) != runtime.ACTION_DIM:
        raise ValueError(f"Server physical_action_dim must be 17, got {physical_dim}")
    contract = metadata.get("observation_contract")
    if contract is not None and contract != "BIYIVLA-PI05-OPENPI":
        raise ValueError(f"Unexpected observation contract: {contract!r}")
    return bool(metadata.get("rtc", False))


def _make_startup_hand_q() -> np.ndarray:
    q = np.zeros(runtime.HAND_DIM, dtype=np.float64)
    # Checkpoint hand order: thumb_j1, thumb_j2, thumb_j3, index_j1,
    # index_j2, index_j3, middle_j1, middle_j2, little_j1, little_j2,
    # little_j3.  Values are SG100 SDK-user radians.
    q[1] = np.deg2rad(-31.0)
    return q


def _make_observation(
    *,
    arm_backend: Any,
    hand_state_proxy: np.ndarray,
    head_camera: Any,
    wrist_camera: Any,
    image_tools: Any,
    prompt: str,
) -> tuple[dict[str, Any], str]:
    now_ns = time.monotonic_ns()
    arm_feedback = arm_backend.joint_state()
    if arm_feedback is None or not arm_feedback.valid:
        raise RuntimeError("RM65 measured joint feedback is unavailable")
    arm_q = np.asarray(arm_feedback.q_rad, dtype=np.float64)
    if arm_q.shape != (runtime.ARM_DIM,) or not np.all(np.isfinite(arm_q)):
        raise RuntimeError("RM65 measured joint feedback is not finite 6D")

    hand_source = "checkpoint_command_proxy"
    hand_q = np.asarray(hand_state_proxy, dtype=np.float64)

    head = head_camera.latest_before(now_ns)
    wrist = wrist_camera.latest_before(now_ns)
    if head is None or wrist is None:
        raise RuntimeError("D455/D405 RGB frames are not available at the observation timestamp")

    state = np.concatenate([arm_q, hand_q]).astype(np.float32, copy=False)
    observation = {
        "state": state,
        "images": {
            # Client-side resize is the same helper used by official OpenPI
            # examples and avoids sending two full 640x480 frames.
            "head": image_tools.resize_with_pad(head.color_rgb, 224, 224),
            "wrist": image_tools.resize_with_pad(wrist.color_rgb, 224, 224),
        },
        "prompt": prompt,
    }
    return observation, hand_source


def _wait_until(deadline_ns: int) -> None:
    remaining_s = (deadline_ns - time.monotonic_ns()) * 1e-9
    if remaining_s > 0:
        time.sleep(remaining_s)


def _run_control_loop(
    *,
    args: argparse.Namespace,
    worker: runtime.AsyncPolicyWorker,
    rtc_enabled: bool,
    chunk_buffer: runtime.ActionChunkBuffer,
    latency: runtime.LatencyEstimator,
    arm_backend: Any,
    hand_backend: Any,
    hand_state_proxy: np.ndarray,
    head_camera: Any,
    wrist_camera: Any,
    image_tools: Any,
) -> None:
    period_ns = round(1e9 / CONTROL_HZ)
    period_s = 1.0 / CONTROL_HZ
    next_tick_ns = time.monotonic_ns()
    completed_steps = 0
    last_hand_reason: str | None = None
    last_hold_count = 0

    while args.max_control_steps == 0 or completed_steps < args.max_control_steps:
        _wait_until(next_tick_ns)
        now_ns = time.monotonic_ns()
        # Do not burst multiple hardware commands when the Python loop was
        # delayed; resume the same 30 Hz cadence from the current tick.
        if now_ns - next_tick_ns > period_ns:
            next_tick_ns = now_ns
        next_tick_ns += period_ns

        inference_result = worker.poll()
        if inference_result is not None:
            actions = runtime.result_actions(inference_result)
            latency.observe(inference_result.latency_s)
            actual_delay = chunk_buffer.install_response(
                actions,
                request_tick=inference_result.request.request_tick,
            )
            policy_timing = (inference_result.response or {}).get("policy_timing", {})
            print(
                "[POLICY] installed chunk",
                f"request={inference_result.request.request_id}",
                f"round_trip_ms={inference_result.latency_s * 1000:.1f}",
                f"discarded_rows={actual_delay}",
                f"policy_timing={policy_timing}",
            )

        if not worker.inflight and chunk_buffer.request_due(args.open_loop_horizon):
            observation, hand_source = _make_observation(
                arm_backend=arm_backend,
                hand_state_proxy=hand_state_proxy,
                head_camera=head_camera,
                wrist_camera=wrist_camera,
                image_tools=image_tools,
                prompt=args.prompt,
            )
            if rtc_enabled:
                prefix = chunk_buffer.remaining
                if len(prefix) == 0:
                    raise RuntimeError("RTC request cannot start after the current chunk is exhausted")
                predicted_delay = latency.predicted_ticks(maximum=len(prefix))
                observation = runtime.add_rtc_request(
                    observation,
                    action_prefix=prefix,
                    inference_delay=predicted_delay,
                )
            else:
                predicted_delay = None
            request = worker.submit(
                observation,
                request_tick=chunk_buffer.executed_total,
            )
            print(
                "[POLICY] request",
                request.request_id,
                f"tick={request.request_tick}",
                f"hand_state={hand_source}",
                f"rtc_delay={predicted_delay}",
            )

        action = chunk_buffer.next_action()
        arm_q, hand_q = runtime.split_action(action)

        # Arm values are RM65 radians. Hand values remain in the checkpoint's
        # SG100 SDK-user-radian domain through the vendor API boundary.
        arm_backend.send_joint_target(arm_q)
        if hand_backend.enabled:
            hand_result = hand_backend.submit(
                hand_q,
                tracking_received_ns=now_ns,
                now_ns=now_ns,
                dt_s=period_s,
            )
            if hand_result.accepted:
                hand_state_proxy[:] = hand_result.effective_checkpoint_qcmd_rad
                last_hand_reason = None
            elif hand_result.reason != last_hand_reason:
                print("[REAL-HAND] command rejected:", hand_result.reason)
                last_hand_reason = hand_result.reason

        if chunk_buffer.hold_count != last_hold_count:
            last_hold_count = chunk_buffer.hold_count
            print("[POLICY] action chunk exhausted; holding its final absolute target")

        completed_steps += 1
        if completed_steps % int(CONTROL_HZ) == 0:
            print(
                "[CONTROL]",
                f"steps={completed_steps}",
                f"arm_enabled={arm_backend.enabled}",
                f"hand_enabled={hand_backend.enabled}",
                f"inflight={worker.inflight}",
                f"holds={chunk_buffer.hold_count}",
            )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    _validate_args(args)
    _bootstrap_import_paths(args.avp_root)
    deps = _load_runtime_dependencies()

    worker: runtime.AsyncPolicyWorker | None = None
    head_camera = None
    wrist_camera = None
    arm_backend = None
    hand_driver = None
    hand_backend = None

    try:
        client_class = deps["WebsocketClientPolicy"]
        worker = runtime.AsyncPolicyWorker(lambda: client_class(args.server_host, args.server_port))
        worker.start()
        metadata = worker.wait_ready()
        rtc_enabled = validate_server_metadata(metadata)
        print(
            "[POLICY] connected",
            f"ws://{args.server_host}:{args.server_port}",
            f"rtc={rtc_enabled}",
            f"metadata={metadata}",
        )

        camera_class = deps["RealSenseCamera"]
        head_camera = camera_class(
            role="head",
            serial=args.head_serial,
            expected_model="RealSense D455",
            width=640,
            height=480,
            fps=30,
            enable_depth=False,
        )
        wrist_camera = camera_class(
            role="right_wrist",
            serial=args.wrist_serial,
            expected_model="RealSense D405",
            width=640,
            height=480,
            fps=30,
            enable_depth=False,
        )
        head_camera.start()
        wrist_camera.start()
        print("[CAMERA] D455 head + D405 right_wrist RGB READY")

        def arm_event(name: str, detail: str) -> None:
            print(f"[REAL-ARM] {name}: {detail}")

        # Exact parameters used by AVP-SG100's current real runner:
        # PROFILE_10, low follow, 100 Hz sender, 150/200 ms watchdogs,
        # 20-degree gross follow error, motion warning without auto-stop.
        arm_backend = deps["RM65RealBackend"](
            args.arm_ip,
            args.arm_port,
            send_hz=100.0,
            follow="low",
            radio=0,
            qdot_max_rad_s=0.5,
            qddot_max_rad_s2=2.0,
            target_timeout_s=0.150,
            feedback_timeout_s=0.200,
            gross_follow_error_rad=np.deg2rad(20.0),
            arm_not_moving_stop=False,
            on_event=arm_event,
        )
        arm_backend.connect()

        hand_driver = deps["SG100SDKDriver"](
            args.hand_port,
            baudrate=args.hand_baudrate,
            sdk_dir=args.hand_sdk_dir,
        )
        hand_backend = CheckpointSG100Backend(
            hand_driver,
            stale_tracking_s=5.00,
            max_velocity_rad_s=4.0,
            # Replanned checkpoint chunks may legitimately disagree at their
            # splice point. Do not latch the hand off on that discontinuity;
            # the velocity slew below still approaches the new absolute target.
            max_cycle_delta_rad=None,
            crc_storm_delta=30_000_000,
        )
        hand_driver.connect()
        hand_driver.start()
        hand_driver.enter_realtime()
        print("[REAL-HAND] connected; realtime MODE_NULL hold")

        startup_hand_q = _make_startup_hand_q()
        print(
            "[STARTUP] applying AVP-SG100 verified pose: hand thumb_j2=-31 deg, "
            "others=0; then RM65 planned startup move"
        )
        startup_result = hand_backend.send_startup_target(startup_hand_q)
        if not startup_result.accepted:
            raise RuntimeError("SG100 rejected the AVP-SG100 startup hand target")
        arm_backend.movej_blocking(
            REAL_STARTUP_ARM_Q_RAD,
            speed_percent=REAL_STARTUP_ARM_SPEED_PERCENT,
        )
        arm_backend.start()

        anchor_result = hand_backend.anchor(startup_hand_q)
        if not anchor_result.accepted:
            raise RuntimeError(f"SG100 MODE7 zero-output anchor failed: {anchor_result.reason}")
        hand_state_proxy = np.asarray(
            anchor_result.effective_checkpoint_qcmd_rad,
            dtype=np.float64,
        ).copy()
        print("[REAL-HAND] startup target anchored with MODE7 zero output")

        initial_observation, hand_source = _make_observation(
            arm_backend=arm_backend,
            hand_state_proxy=hand_state_proxy,
            head_camera=head_camera,
            wrist_camera=wrist_camera,
            image_tools=deps["image_tools"],
            prompt=args.prompt,
        )
        initial_request = worker.submit(initial_observation, request_tick=0)
        print(
            "[POLICY] waiting for initial action chunk with hardware command authority disabled;",
            f"hand_state={hand_source}",
        )
        initial_result = worker.wait_result()
        initial_actions = runtime.result_actions(initial_result)
        if initial_result.request.request_id != initial_request.request_id:
            raise RuntimeError("Initial policy response ID mismatch")

        chunk_buffer = runtime.ActionChunkBuffer()
        chunk_buffer.install_initial(initial_actions)
        latency = runtime.LatencyEstimator(CONTROL_HZ)
        latency.observe(initial_result.latency_s)
        print(
            "[POLICY] initial chunk READY",
            f"round_trip_ms={initial_result.latency_s * 1000:.1f}",
            f"actions={initial_actions.shape}",
        )

        input(
            "已完成相机、推理、初始位姿和 SG100 anchor。"
            "按 Enter 执行 AVP-SG100 的 A-enable, 开始 30 Hz 真机动作; Ctrl-C 停止:"
        )
        # Same enable order as the AVP-SG100 A-key path.
        hand_backend.set_enabled(True)
        arm_backend.set_enabled(True)
        print("[REAL] ARMED; pi0.5 now owns RM65 + SG100 command authority")

        _run_control_loop(
            args=args,
            worker=worker,
            rtc_enabled=rtc_enabled,
            chunk_buffer=chunk_buffer,
            latency=latency,
            arm_backend=arm_backend,
            hand_backend=hand_backend,
            hand_state_proxy=hand_state_proxy,
            head_camera=head_camera,
            wrist_camera=wrist_camera,
            image_tools=deps["image_tools"],
        )
    except KeyboardInterrupt:
        print("\n[REAL] Ctrl-C received; stopping")
    finally:
        # Preserve AVP-SG100's shutdown order and methods exactly.
        if hand_driver is not None:
            try:
                if hand_backend is not None:
                    hand_backend.set_enabled(False)
            finally:
                try:
                    hand_driver.stop()
                finally:
                    hand_driver.disconnect()
            print("[REAL-HAND] disconnected")
        if arm_backend is not None:
            try:
                arm_backend.set_enabled(False)
                arm_backend.stop_smooth()
            finally:
                arm_backend.disconnect()
            print("[REAL-ARM] disconnected")
        if wrist_camera is not None:
            wrist_camera.stop()
        if head_camera is not None:
            head_camera.stop()
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    main()
