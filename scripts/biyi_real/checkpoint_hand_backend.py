"""SG100 execution backend for the hand domain stored in the pi0.5 checkpoint.

The checkpoint stores the eleven hand values in SG100 SDK-user radians.  This
backend therefore performs only radians-to-degrees conversion at the vendor
boundary.  It deliberately does not apply ``SG100HardwareMap`` or any
canonical-coordinate sign conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

HAND_DOF = 11
MODE_JOINT_POSITION = 7


@dataclass(frozen=True)
class CheckpointHandCommandResult:
    accepted: bool
    reason: str
    checkpoint_qcmd_rad: np.ndarray | None = None
    sdk_target_deg: np.ndarray | None = None
    effective_checkpoint_qcmd_rad: np.ndarray | None = None
    slew_limited: bool = False
    sent_monotonic_ns: int = 0


class CheckpointSG100Backend:
    """Drive SG100 from checkpoint-domain SDK-user radians.

    The AVP transport/realtime/feedback checks are retained.  The old
    ``TRACKING_SPIKE`` comparison between the model target and the last sent
    target is intentionally absent.  Large initial model targets are accepted
    and approached by the existing velocity slew limiter.
    """

    def __init__(
        self,
        driver: Any,
        *,
        max_cycle_delta_rad: float | None = None,
        max_velocity_rad_s: float = 4.0,
        stale_tracking_s: float = 5.0,
        feedback_stall_s: float = 0.10,
        crc_storm_delta: int = 30_000_000,
    ) -> None:
        self.driver = driver
        ranges = driver.vendor_user_ranges
        if ranges is None or len(ranges) != HAND_DOF:
            raise ValueError("11 SG100 SDK user ranges are required")
        self._sdk_min_deg = np.asarray([row[0] for row in ranges], dtype=np.float64)
        self._sdk_max_deg = np.asarray([row[1] for row in ranges], dtype=np.float64)
        if (
            not np.all(np.isfinite(self._sdk_min_deg))
            or not np.all(np.isfinite(self._sdk_max_deg))
            or np.any(self._sdk_min_deg > self._sdk_max_deg)
        ):
            raise ValueError("SG100 SDK user ranges must be finite [min,max] pairs")

        if max_cycle_delta_rad is not None and (
            not np.isfinite(max_cycle_delta_rad) or max_cycle_delta_rad <= 0.0
        ):
            raise ValueError("max_cycle_delta_rad must be None or finite and > 0")
        if not np.isfinite(max_velocity_rad_s) or max_velocity_rad_s <= 0.0:
            raise ValueError("max_velocity_rad_s must be finite and > 0")
        if stale_tracking_s <= 0.0 or feedback_stall_s <= 0.0:
            raise ValueError("tracking/feedback timeouts must be > 0")
        if int(crc_storm_delta) <= 0:
            raise ValueError("crc_storm_delta must be > 0")

        self.max_cycle_delta_rad = max_cycle_delta_rad
        self.max_velocity_rad_s = float(max_velocity_rad_s)
        self.stale_tracking_ns = int(stale_tracking_s * 1e9)
        self.feedback_stall_ns = int(feedback_stall_s * 1e9)
        self.crc_storm_delta = int(crc_storm_delta)
        self.active_joint_indices = tuple(range(HAND_DOF))
        self.anchored = False
        self.enabled = False
        self.estopped = False
        self._last_checkpoint_q: np.ndarray | None = None
        self._last_cycle_candidate: np.ndarray | None = None
        self._crc_baseline = 0

    def _apply_sdk_ranges(self, checkpoint_qcmd_rad: Any) -> tuple[np.ndarray, np.ndarray]:
        requested = np.asarray(checkpoint_qcmd_rad, dtype=np.float64)
        if requested.shape != (HAND_DOF,) or not np.all(np.isfinite(requested)):
            raise ValueError("checkpoint hand command must be finite float[11]")
        sdk_deg = np.clip(
            np.rad2deg(requested), self._sdk_min_deg, self._sdk_max_deg
        )
        return requested.copy(), np.deg2rad(sdk_deg)

    def send_startup_target(self, checkpoint_qcmd_rad: Any) -> CheckpointHandCommandResult:
        requested, effective = self._apply_sdk_ranges(checkpoint_qcmd_rad)
        sdk_deg = np.rad2deg(effective)
        submission = self.driver.send_mode7_active_allowlist(
            self.active_joint_indices, sdk_deg
        )
        if submission is None:
            return CheckpointHandCommandResult(
                accepted=False, reason="SET_HAND_CONTROL_REJECTED"
            )
        return CheckpointHandCommandResult(
            accepted=True,
            reason="STARTUP_TARGET_ACCEPTED",
            checkpoint_qcmd_rad=requested,
            sdk_target_deg=sdk_deg.copy(),
            effective_checkpoint_qcmd_rad=effective.copy(),
            sent_monotonic_ns=int(submission.sent_monotonic_ns),
        )

    def anchor(self, checkpoint_qcmd_rad: Any) -> CheckpointHandCommandResult:
        requested, effective = self._apply_sdk_ranges(checkpoint_qcmd_rad)
        sdk_deg = np.rad2deg(effective)
        submission = self.driver.send_mode7_zero_output_sdk_targets(sdk_deg)
        if submission is None:
            return self._safe_stop("ANCHOR_ZERO_OUTPUT_REJECTED")
        self._last_checkpoint_q = effective.copy()
        self._last_cycle_candidate = None
        self._crc_baseline = int(self.driver.diagnostics.get("crc_error_count", 0))
        self.anchored = True
        self.enabled = False
        return CheckpointHandCommandResult(
            accepted=True,
            reason="ANCHORED_ZERO_OUTPUT",
            checkpoint_qcmd_rad=requested,
            sdk_target_deg=sdk_deg.copy(),
            effective_checkpoint_qcmd_rad=effective.copy(),
            sent_monotonic_ns=int(submission.sent_monotonic_ns),
        )

    def set_enabled(self, enabled: bool) -> None:  # noqa: FBT001
        if enabled and (not self.anchored or self.estopped):
            raise RuntimeError("checkpoint hand anchor is required before enable")
        self.enabled = bool(enabled)
        self._last_cycle_candidate = None
        if not self.enabled:
            self.driver.hold_passive()

    def _safe_stop(self, reason: str) -> CheckpointHandCommandResult:
        try:
            self.driver.hold_passive()
        finally:
            self.enabled = False
            self._last_cycle_candidate = None
        return CheckpointHandCommandResult(accepted=False, reason=reason)

    def submit(
        self,
        checkpoint_qcmd_rad: Any,
        *,
        tracking_received_ns: int,
        now_ns: int | None = None,
        dt_s: float,
    ) -> CheckpointHandCommandResult:
        if self.estopped:
            return CheckpointHandCommandResult(accepted=False, reason="ESTOP_LATCHED")
        if not self.anchored:
            return CheckpointHandCommandResult(accepted=False, reason="ANCHOR_REQUIRED")
        if not self.enabled:
            return CheckpointHandCommandResult(
                accepted=False, reason="NOT_ENABLED_NO_ACTIVE_COMMAND"
            )
        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        if now - int(tracking_received_ns) > self.stale_tracking_ns:
            return self._safe_stop("STALE_TRACKING")
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            return self._safe_stop("INVALID_COMMAND_DT")
        if self.driver.fault_active:
            return self._safe_stop("SG100_FAULT")
        if hasattr(self.driver, "serial_open") and not self.driver.serial_open:
            return self._safe_stop("TRANSPORT_CLOSED")

        feedback = self.driver.read_feedback()
        if feedback is None or not feedback.valid:
            return self._safe_stop("TRANSPORT_FEEDBACK_MISSING")
        if now - int(feedback.received_monotonic_ns) > self.feedback_stall_ns:
            return self._safe_stop("TRANSPORT_STALL")
        if feedback.joint_modes != (MODE_JOINT_POSITION,) * HAND_DOF:
            return self._safe_stop("UNEXPECTED_RX_MODE")
        diagnostics = self.driver.diagnostics
        if "vendor_work_mode" in diagnostics and int(diagnostics["vendor_work_mode"]) != 0:
            return self._safe_stop("REALTIME_MODE_LOSS")
        crc_delta = int(diagnostics.get("crc_error_count", 0)) - self._crc_baseline
        if crc_delta >= self.crc_storm_delta:
            return self._safe_stop("CRC_STORM")

        try:
            requested, candidate = self._apply_sdk_ranges(checkpoint_qcmd_rad)
        except ValueError as exc:
            return self._safe_stop(f"COMMAND_REJECTED:{exc}")

        # No TRACKING_SPIKE gate here.  In particular, candidate is never
        # rejected merely because it is far from the last effective target.
        previous_candidate = self._last_cycle_candidate
        self._last_cycle_candidate = candidate.copy()
        if (
            previous_candidate is not None
            and self.max_cycle_delta_rad is not None
            and np.any(np.abs(candidate - previous_candidate) > self.max_cycle_delta_rad)
        ):
            return self._safe_stop("PER_CYCLE_DELTA_EXCEEDED")

        assert self._last_checkpoint_q is not None
        max_step = self.max_velocity_rad_s * float(dt_s)
        delta = candidate - self._last_checkpoint_q
        effective = self._last_checkpoint_q + np.clip(delta, -max_step, max_step)
        slew_limited = bool(np.any(np.abs(effective - candidate) > 1e-12))
        sdk_deg = np.rad2deg(effective)
        submission = self.driver.send_mode7_active_allowlist(
            self.active_joint_indices, sdk_deg
        )
        if submission is None:
            return self._safe_stop("SET_HAND_CONTROL_REJECTED")
        self._last_checkpoint_q = effective.copy()
        return CheckpointHandCommandResult(
            accepted=True,
            reason="ACCEPTED",
            checkpoint_qcmd_rad=requested,
            sdk_target_deg=sdk_deg.copy(),
            effective_checkpoint_qcmd_rad=effective.copy(),
            slew_limited=slew_limited,
            sent_monotonic_ns=int(submission.sent_monotonic_ns),
        )
