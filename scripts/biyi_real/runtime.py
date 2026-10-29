"""Runtime primitives for executing BIYIVLA 17D action chunks.

This module has no robot-SDK imports so its chunk alignment and dispatch
contract can be tested without connecting to hardware.
"""

from __future__ import annotations

from collections.abc import Callable
import contextlib
import dataclasses
import math
import queue
import threading
import time
from typing import Any

import numpy as np

ACTION_DIM = 17
ARM_DIM = 6
HAND_DIM = 11
ACTION_HORIZON = 50


def validate_action_chunk(actions: Any) -> np.ndarray:
    """Return a copied float64 ``[50,17]`` action chunk."""

    chunk = np.asarray(actions, dtype=np.float64)
    expected = (ACTION_HORIZON, ACTION_DIM)
    if chunk.shape != expected:
        raise ValueError(f"Expected policy action shape {expected}, got {chunk.shape}")
    if not np.all(np.isfinite(chunk)):
        raise ValueError("Policy actions contain NaN or Inf")
    return chunk.copy()


def split_action(action: Any) -> tuple[np.ndarray, np.ndarray]:
    """Split one checkpoint 17D-radian action without changing its values.

    The arm prefix is RM65 joint radians.  The hand suffix is the checkpoint's
    SG100 SDK-user-radian domain and must not be interpreted as canonical hand
    coordinates.
    """

    vector = np.asarray(action, dtype=np.float64)
    if vector.shape != (ACTION_DIM,):
        raise ValueError(f"Expected action[{ACTION_DIM}], got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Action contains NaN or Inf")
    return vector[:ARM_DIM].copy(), vector[ARM_DIM:].copy()


class ActionChunkBuffer:
    """Track asynchronous chunk execution on a 30 Hz control timeline.

    A response is aligned to the observation tick that triggered inference.
    Actions consumed while inference was in flight are therefore discarded
    before the response is installed. This is required for both standard and
    RTC servers; RTC additionally uses :meth:`remaining` as its prefix.
    """

    def __init__(self) -> None:
        self._chunk: np.ndarray | None = None
        self._cursor = 0
        self._executed_total = 0
        self._steps_since_install = 0
        self._hold_count = 0

    @property
    def ready(self) -> bool:
        return self._chunk is not None

    @property
    def executed_total(self) -> int:
        return self._executed_total

    @property
    def steps_since_install(self) -> int:
        return self._steps_since_install

    @property
    def hold_count(self) -> int:
        return self._hold_count

    @property
    def remaining(self) -> np.ndarray:
        if self._chunk is None:
            raise RuntimeError("No action chunk is installed")
        return self._chunk[self._cursor :].copy()

    def request_due(self, open_loop_horizon: int) -> bool:
        """Return whether another request must start before this chunk drains."""

        if open_loop_horizon <= 0:
            raise ValueError("open_loop_horizon must be positive")
        remaining = len(self.remaining)
        return self._steps_since_install >= open_loop_horizon or remaining <= open_loop_horizon

    def install_initial(self, actions: Any) -> None:
        self._chunk = validate_action_chunk(actions)
        self._cursor = 0
        self._steps_since_install = 0

    def install_response(self, actions: Any, *, request_tick: int) -> int:
        """Install a response and return its actual delay in control ticks."""

        response = validate_action_chunk(actions)
        actual_delay = self._executed_total - int(request_tick)
        if actual_delay < 0:
            raise ValueError(f"request_tick {request_tick} is ahead of executed tick {self._executed_total}")
        if actual_delay >= ACTION_HORIZON:
            raise RuntimeError(
                f"Inference response arrived after its entire 50-step horizon ({actual_delay} control ticks)"
            )
        self._chunk = response[actual_delay:].copy()
        self._cursor = 0
        self._steps_since_install = 0
        return actual_delay

    def next_action(self) -> np.ndarray:
        """Consume the next action, holding the final row only if necessary."""

        if self._chunk is None:
            raise RuntimeError("No action chunk is installed")
        if self._cursor < len(self._chunk):
            action = self._chunk[self._cursor]
            self._cursor += 1
        else:
            # Keep publishing a position target so the AVP-SG100 RM65
            # backend's existing target-age watchdog is not starved while an
            # already-issued inference request finishes.
            action = self._chunk[-1]
            self._hold_count += 1
        self._executed_total += 1
        self._steps_since_install += 1
        return action.copy()


class LatencyEstimator:
    """EWMA round-trip latency expressed in control ticks."""

    def __init__(self, control_hz: float, *, alpha: float = 0.5) -> None:
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0,1]")
        self._control_hz = float(control_hz)
        self._alpha = float(alpha)
        self._latency_s: float | None = None

    @property
    def latency_s(self) -> float | None:
        return self._latency_s

    def observe(self, latency_s: float) -> None:
        if not np.isfinite(latency_s) or latency_s < 0:
            raise ValueError("latency_s must be finite and non-negative")
        if self._latency_s is None:
            self._latency_s = float(latency_s)
        else:
            self._latency_s = self._alpha * float(latency_s) + (1 - self._alpha) * self._latency_s

    def predicted_ticks(self, *, maximum: int) -> int:
        if maximum < 0:
            raise ValueError("maximum must be non-negative")
        if self._latency_s is None:
            return 0
        return min(maximum, math.ceil(self._latency_s * self._control_hz))


def add_rtc_request(
    observation: dict[str, Any],
    *,
    action_prefix: Any,
    inference_delay: int,
) -> dict[str, Any]:
    """Add the BIYIVLA RTC block aligned to this observation."""

    prefix = np.asarray(action_prefix, dtype=np.float64)
    if prefix.ndim != 2 or prefix.shape[1] != ACTION_DIM:
        raise ValueError(f"RTC prefix must have shape [T,{ACTION_DIM}], got {prefix.shape}")
    if not 1 <= len(prefix) <= ACTION_HORIZON:
        raise ValueError(f"RTC prefix length must be in [1,{ACTION_HORIZON}]")
    if not np.all(np.isfinite(prefix)):
        raise ValueError("RTC prefix contains NaN or Inf")
    delay = int(inference_delay)
    if not 0 <= delay <= len(prefix):
        raise ValueError("RTC inference_delay must be within the prefix horizon")
    return {
        **observation,
        "rtc": {
            "action_prefix": prefix.copy(),
            "inference_delay": delay,
            "prefix_horizon": len(prefix),
        },
    }


@dataclasses.dataclass(frozen=True)
class InferenceRequest:
    request_id: int
    observation: dict[str, Any]
    request_tick: int
    submitted_ns: int


@dataclasses.dataclass(frozen=True)
class InferenceResult:
    request: InferenceRequest
    response: dict[str, Any] | None
    completed_ns: int
    error: BaseException | None = None

    @property
    def latency_s(self) -> float:
        return (self.completed_ns - self.request.submitted_ns) * 1e-9


class AsyncPolicyWorker:
    """Own one synchronous WebSocket policy client on a worker thread."""

    def __init__(self, client_factory: Callable[[], Any]) -> None:
        self._client_factory = client_factory
        self._requests: queue.Queue[InferenceRequest | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[InferenceResult] = queue.Queue()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="pi05-policy-worker",
            daemon=True,
        )
        self._client: Any | None = None
        self._metadata: dict[str, Any] | None = None
        self._startup_error: BaseException | None = None
        self._inflight = False
        self._next_request_id = 1

    @property
    def metadata(self) -> dict[str, Any]:
        if not self._ready.is_set():
            raise RuntimeError("Policy worker has not connected")
        if self._startup_error is not None:
            raise RuntimeError("Policy worker failed to connect") from self._startup_error
        assert self._metadata is not None
        return dict(self._metadata)

    @property
    def inflight(self) -> bool:
        return self._inflight

    def start(self) -> None:
        self._thread.start()

    def wait_ready(self, timeout_s: float | None = None) -> dict[str, Any]:
        if not self._ready.wait(timeout_s):
            raise TimeoutError("Timed out waiting for the policy server")
        return self.metadata

    def submit(self, observation: dict[str, Any], *, request_tick: int) -> InferenceRequest:
        if self._inflight:
            raise RuntimeError("Only one policy request may be in flight")
        request = InferenceRequest(
            request_id=self._next_request_id,
            observation=observation,
            request_tick=int(request_tick),
            submitted_ns=time.monotonic_ns(),
        )
        self._next_request_id += 1
        self._requests.put_nowait(request)
        self._inflight = True
        return request

    def poll(self) -> InferenceResult | None:
        try:
            result = self._results.get_nowait()
        except queue.Empty:
            return None
        self._inflight = False
        return result

    def wait_result(self, timeout_s: float | None = None) -> InferenceResult:
        result = self._results.get(timeout=timeout_s)
        self._inflight = False
        return result

    def close(self) -> None:
        if self._thread.is_alive() and not self._inflight:
            with contextlib.suppress(queue.Full):
                self._requests.put_nowait(None)
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        try:
            self._client = self._client_factory()
            metadata = self._client.get_server_metadata()
            self._metadata = dict(metadata or {})
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()

        while True:
            request = self._requests.get()
            if request is None:
                return
            try:
                response = self._client.infer(request.observation)
                result = InferenceResult(
                    request=request,
                    response=response,
                    completed_ns=time.monotonic_ns(),
                )
            except BaseException as exc:
                result = InferenceResult(
                    request=request,
                    response=None,
                    completed_ns=time.monotonic_ns(),
                    error=exc,
                )
            self._results.put(result)


def result_actions(result: InferenceResult) -> np.ndarray:
    """Raise a worker error or extract the validated action chunk."""

    if result.error is not None:
        raise RuntimeError(f"Policy request {result.request.request_id} failed") from result.error
    if not isinstance(result.response, dict) or "actions" not in result.response:
        raise ValueError("Policy response is missing actions")
    return validate_action_chunk(result.response["actions"])
