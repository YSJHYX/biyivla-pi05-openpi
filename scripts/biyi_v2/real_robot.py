"""Reuse v1's verified device classes, startup pose, anchor and shutdown APIs.

CheckpointSG100Backend is used for startup/anchor only. During inference the
handwritten robot_io.send_action calls the existing SG100 SDK driver directly.
"""

from contextlib import ExitStack
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

from scripts.biyi_real import run_pi05_real as legacy
from scripts.biyi_real.checkpoint_hand_backend import CheckpointSG100Backend


def _close_hand(driver, backend):
    try:
        backend.set_enabled(False)
    finally:
        try:
            driver.stop()
        finally:
            driver.disconnect()


def _close_arm(arm):
    try:
        arm.set_enabled(False)
        arm.stop_smooth()
    finally:
        arm.disconnect()


@contextmanager
def open_real_robot(args, *, dependencies=None):
    if dependencies is None:
        legacy._bootstrap_import_paths(args.avp_root)  # noqa: SLF001 -- reuse the existing startup boundary
        dependencies = legacy._load_runtime_dependencies()  # noqa: SLF001
    with ExitStack() as stack:
        camera_class = dependencies["RealSenseCamera"]
        head = camera_class(
            role="head",
            serial=args.head_serial,
            expected_model="RealSense D455",
            width=640,
            height=480,
            fps=30,
            enable_depth=False,
        )
        stack.callback(head.stop)
        head.start()
        wrist = camera_class(
            role="right_wrist",
            serial=args.wrist_serial,
            expected_model="RealSense D405",
            width=640,
            height=480,
            fps=30,
            enable_depth=False,
        )
        stack.callback(wrist.stop)
        wrist.start()

        arm = dependencies["RM65RealBackend"](
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
            on_event=lambda name, detail: print(f"[REAL-ARM] {name}: {detail}"),
        )
        stack.callback(_close_arm, arm)
        arm.connect()

        hand_driver = dependencies["SG100SDKDriver"](
            args.hand_port,
            baudrate=args.hand_baudrate,
            sdk_dir=args.hand_sdk_dir,
        )
        hand_backend = CheckpointSG100Backend(hand_driver)
        stack.callback(_close_hand, hand_driver, hand_backend)
        hand_driver.connect()
        hand_driver.start()
        hand_driver.enter_realtime()

        startup_hand_q = legacy._make_startup_hand_q()  # noqa: SLF001
        startup = hand_backend.send_startup_target(startup_hand_q)
        if not startup.accepted:
            raise RuntimeError(f"SG100 startup rejected: {startup.reason}")
        arm.movej_blocking(
            legacy.REAL_STARTUP_ARM_Q_RAD,
            speed_percent=legacy.REAL_STARTUP_ARM_SPEED_PERCENT,
        )
        arm.start()
        anchor = hand_backend.anchor(startup_hand_q)
        if not anchor.accepted:
            raise RuntimeError(f"SG100 anchor rejected: {anchor.reason}")
        hand_state_proxy = np.asarray(anchor.effective_checkpoint_qcmd_rad, dtype=np.float32).copy()

        def enable():
            # Refresh a stationary arm target before enabling the existing
            # 150 ms watchdog. Model loading/first inference may take seconds.
            feedback = arm.joint_state()
            if feedback is None or not feedback.valid:
                raise RuntimeError("Arm feedback unavailable before enable")
            arm.send_joint_target(np.asarray(feedback.q_rad).copy())
            hand_backend.set_enabled(True)
            arm.set_enabled(True)
            print("[REAL] ARMED; v2 owns arm/hand command authority")

        def disable():
            try:
                hand_backend.set_enabled(False)
            finally:
                arm.set_enabled(False)
                arm.stop_smooth()

        print("[REAL] cameras, startup pose and hand anchor ready; streaming disabled")
        yield SimpleNamespace(
            arm_backend=arm,
            hand_driver=hand_driver,
            head_camera=head,
            wrist_camera=wrist,
            hand_state_proxy=hand_state_proxy,
            enable=enable,
            disable=disable,
        )
