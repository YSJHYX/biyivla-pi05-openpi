"""Robot-side v2 queue and tick loop. Policy inference crosses WebSocket.

The handwritten sampling code remains in rtc_adapter.py on the model server.
Native mode uses this same timing loop but sends no RTC prefix.
"""

import itertools
import math
import queue
import threading
import time

import numpy as np

from scripts.biyi_v2.action_queue import ActionQueue
from scripts.biyi_v2.remote_policy import predict_remote_chunk
from scripts.biyi_v2.robot_io import read_observation
from scripts.biyi_v2.robot_io import send_action


def refill_rtc_once(
    policy,
    action_queue,
    get_observation,
    *,
    inference_delay,
    execution_horizon=10,
    rtc_enabled=True,
):
    request_tick, remaining_actions = action_queue.snapshot()
    observation = get_observation()
    if not rtc_enabled or len(remaining_actions) == 0:
        actions = predict_remote_chunk(policy, observation)
    else:
        length = len(remaining_actions)
        delay = min(inference_delay, length)
        horizon = min(length, max(delay, execution_horizon))
        actions = predict_remote_chunk(
            policy,
            observation,
            action_prefix=remaining_actions,
            inference_delay=delay,
            prefix_horizon=horizon,
        )
    # Predicted delay guides sampling; actual elapsed ticks trim the response.
    return action_queue.merge(actions, request_tick)


def rtc_inference_worker(
    policy,
    action_queue,
    get_observation,
    request_event,
    stop_event,
    error_queue,
    *,
    control_hz=30.0,
    execution_horizon=10,
    initial_inference_delay=0,
    rtc_enabled=True,
):
    inference_delay = initial_inference_delay
    while not stop_event.is_set():
        if not request_event.wait(timeout=0.1):
            continue
        if stop_event.is_set():
            break
        request_event.clear()
        started = time.monotonic()
        try:
            refill_rtc_once(
                policy,
                action_queue,
                get_observation,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
                rtc_enabled=rtc_enabled,
            )
        except Exception as error:
            if not stop_event.is_set():
                error_queue.put(error)
                stop_event.set()
            break
        # Includes image preparation, transport, server transforms and sampling.
        inference_delay = math.ceil((time.monotonic() - started) * control_hz)


def execute_rtc_tick(action_queue, arm_backend, hand_driver, hand_state_proxy, *, require_action=False):
    action = action_queue.get()
    if action is None:
        if require_action:
            raise RuntimeError("Action queue exhausted before the next policy response")
        return hand_state_proxy
    return send_action(action, arm_backend, hand_driver)


def run_rtc_control_ticks(
    execute_tick,
    request_event,
    stop_event,
    *,
    num_ticks,
    control_hz=30.0,
    request_interval=10,
):
    if num_ticks is not None and num_ticks <= 0:
        raise ValueError("num_ticks must be positive or None (until Ctrl-C)")
    if not math.isfinite(control_hz) or control_hz <= 0 or request_interval <= 0:
        raise ValueError("control_hz and request_interval must be positive")
    period = 1.0 / control_hz
    deadline = time.monotonic()
    for index in itertools.count():
        if (num_ticks is not None and index >= num_ticks) or stop_event.is_set():
            break
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        if stop_event.is_set():
            break
        tick_start = time.monotonic()
        if index % request_interval == 0:
            request_event.set()
        execute_tick()
        deadline = tick_start + period


def run_rtc_session(
    policy,
    action_queue,
    get_observation,
    execute_tick,
    *,
    num_ticks,
    control_hz=30.0,
    request_interval=10,
    execution_horizon=10,
    initial_inference_delay=0,
    rtc_enabled=True,
    on_stop=None,
):
    request_event, stop_event = threading.Event(), threading.Event()
    error_queue = queue.Queue()
    worker = threading.Thread(
        target=rtc_inference_worker,
        args=(policy, action_queue, get_observation, request_event, stop_event, error_queue),
        kwargs={
            "control_hz": control_hz,
            "execution_horizon": execution_horizon,
            "initial_inference_delay": initial_inference_delay,
            "rtc_enabled": rtc_enabled,
        },
        name="biyi-v2-inference",
    )
    worker.start()
    try:
        run_rtc_control_ticks(
            execute_tick,
            request_event,
            stop_event,
            num_ticks=num_ticks,
            control_hz=control_hz,
            request_interval=request_interval,
        )
    finally:
        stop_event.set()
        request_event.set()
        # Stop hardware before waiting for any slow/network-blocked inference.
        try:
            if on_stop is not None:
                on_stop()
        finally:
            try:
                close = getattr(policy, "close", None)
                if close is not None:
                    close()
            finally:
                worker.join()
    try:
        error = error_queue.get_nowait()
    except queue.Empty:
        return
    raise error


def run_rtc_robot(
    policy,
    arm_backend,
    hand_driver,
    head_camera,
    wrist_camera,
    hand_state_proxy,
    prompt,
    *,
    num_ticks,
    control_hz=30.0,
    request_interval=10,
    execution_horizon=10,
    initial_inference_delay=0,
    before_control=None,
    on_stop=None,
    rtc_enabled=True,
):
    hand_state_proxy = np.asarray(hand_state_proxy, dtype=np.float32).copy()

    def get_observation():
        return read_observation(
            arm_backend,
            hand_state_proxy.copy(),
            head_camera,
            wrist_camera,
            prompt,
        )

    # Native first chunk is obtained while command authority remains disabled.
    observation = get_observation()
    chunk = predict_remote_chunk(policy, observation)
    action_queue = ActionQueue(chunk)
    if before_control is not None:
        before_control()

    def execute_tick():
        nonlocal hand_state_proxy
        if not arm_backend.enabled:
            raise RuntimeError("Arm backend stopped; ending policy execution")
        hand_state_proxy = execute_rtc_tick(
            action_queue,
            arm_backend,
            hand_driver,
            hand_state_proxy,
            require_action=True,
        )

    run_rtc_session(
        policy,
        action_queue,
        get_observation,
        execute_tick,
        num_ticks=num_ticks,
        control_hz=control_hz,
        request_interval=request_interval,
        execution_horizon=execution_horizon,
        initial_inference_delay=initial_inference_delay,
        rtc_enabled=rtc_enabled,
        on_stop=on_stop,
    )
    return hand_state_proxy.copy()
