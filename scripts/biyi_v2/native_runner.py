import time
from scripts.biyi_v2.robot_io import read_observation, send_action
from scripts.biyi_v2.native import predict_native_chunk

def run_native_once(
    policy,
    arm_backend,
    hand_driver,
    head_camera,
    wrist_camera,
    hand_state_proxy,
    prompt,
    execution_steps=10,
    control_hz=30.0,
):
    # TODO：读取一次观测、推理一次、按频率执行前 N 行
    observation = read_observation(arm_backend, hand_state_proxy, head_camera, wrist_camera, prompt)
    actions = predict_native_chunk(policy, observation)

    if not (execution_steps >= 1 and execution_steps <= len(actions)) or not control_hz > 0:
        raise ValueError("Invalid execution_steps or control_hz.")

    for step in range(execution_steps):

        if step == 0: # 第一次指令直接发送，无需等待
            action = actions[step]
            gap_time = 1.0 / control_hz
            start_time = time.monotonic()
            command = send_action(action, arm_backend, hand_driver)
            end_time = start_time + gap_time # 下一次截止时间＝本次发送开始时间＋1 / control_hz
        else:
            remaining_time = end_time - time.monotonic() # 计算距离下一次截止时间的剩余时间
            if remaining_time > 0:
                time.sleep(remaining_time)
            action = actions[step]
            start_time = time.monotonic()
            command = send_action(action, arm_backend, hand_driver)
            end_time = start_time + gap_time

        if command is None:
            raise RuntimeError(f"Failed to send action at step {step}.")

        hand_state_proxy = command
    
    return hand_state_proxy
    