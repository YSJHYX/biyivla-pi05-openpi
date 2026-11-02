import time

import numpy as np


def build_observation(arm_q, hand_q, head_rgb, wrist_rgb, prompt):
    # TODO:拼接 state,并组织原生 policy 所需的 observation
    return {
        "state": np.concatenate([arm_q, hand_q], dtype=np.float32),
        "images": {"head": head_rgb, "wrist": wrist_rgb},
        "prompt": prompt,
    }


def read_observation(
    arm_backend,
    hand_state_proxy,
    head_camera,
    wrist_camera,
    prompt,
):
    # TODO:读取设备数据,再调用 build_observation
    t = time.monotonic_ns()
    arm_feedback = arm_backend.joint_state()
    if arm_feedback is None or not arm_feedback.valid:
        raise RuntimeError("Arm feedback is invalid, cannot build the observation.")
    arm_q = np.asarray(arm_feedback.q_rad, dtype=np.float32)

    head_t = head_camera.latest_before(t)
    wrist_t = wrist_camera.latest_before(t)

    if head_t is None or wrist_t is None:
        raise RuntimeError("Camera feedback is invalid, cannot build the observation.")

    head_rgb = head_t.color_rgb
    wrist_rgb = wrist_t.color_rgb

    return build_observation(arm_q, hand_state_proxy, head_rgb, wrist_rgb, prompt)


def send_action(action, arm_backend, hand_driver):
    # TODO:拆分一行 17D action,调用现有设备接口
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (17,) or not np.isfinite(action).all():
        raise ValueError("Action must be an finite 17D vector.")
    arm_q = action[:6]
    hand_q = np.rad2deg(action[6:], dtype=np.float32)  # Convert radians to degrees for hand

    hand_sent = hand_driver.send_mode7_active_allowlist(tuple(range(11)), hand_q.tolist())
    if hand_sent is None:
        raise RuntimeError("Failed to send hand action.")
    arm_backend.send_joint_target(arm_q.tolist())
    # TODO:返回手部已接受目标对应的 SDK-user 弧度
    return np.deg2rad(hand_sent.requested_positions_deg, dtype=np.float32)
