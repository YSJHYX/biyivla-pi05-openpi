from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.biyi_real.checkpoint_hand_backend import CheckpointSG100Backend


@dataclass
class _Submission:
    sent_monotonic_ns: int = 123


@dataclass
class _Feedback:
    received_monotonic_ns: int
    valid: bool = True
    joint_modes: tuple[int, ...] = (7,) * 11


class _FakeDriver:
    def __init__(self) -> None:
        # Checkpoint/SDK-user ranges in degrees. In particular J7 and J9 are
        # positive here; this catches accidental canonical sign conversion.
        self.vendor_user_ranges = [
            (0.0, 125.0),
            (-149.0, -30.0),
            (-60.0, 90.0),
            (-90.0, 0.0),
            (0.0, 125.0),
            (-60.0, 90.0),
            (0.0, 125.0),
            (-60.0, 90.0),
            (0.0, 180.0),
            (0.0, 125.0),
            (-60.0, 90.0),
        ]
        self.diagnostics = {"crc_error_count": 0, "vendor_work_mode": 0}
        self.fault_active = False
        self.serial_open = True
        self.now_ns = 1_000_000_000
        self.sent: list[np.ndarray] = []
        self.passive_count = 0

    def read_feedback(self) -> _Feedback:
        return _Feedback(self.now_ns)

    def send_mode7_zero_output_sdk_targets(self, positions_deg: np.ndarray) -> _Submission:
        self.sent.append(np.asarray(positions_deg).copy())
        return _Submission()

    def send_mode7_active_allowlist(self, indices, positions_deg: np.ndarray) -> _Submission:
        assert tuple(indices) == tuple(range(11))
        self.sent.append(np.asarray(positions_deg).copy())
        return _Submission()

    def hold_passive(self) -> bool:
        self.passive_count += 1
        return True


def _armed_backend(**kwargs) -> tuple[CheckpointSG100Backend, _FakeDriver]:
    driver = _FakeDriver()
    backend = CheckpointSG100Backend(driver, max_cycle_delta_rad=None, **kwargs)
    startup = np.zeros(11)
    startup[1] = np.deg2rad(-31.0)
    assert backend.anchor(startup).accepted
    backend.set_enabled(True)
    return backend, driver


def test_large_model_to_current_delta_does_not_trigger_tracking_spike() -> None:
    backend, driver = _armed_backend()
    target = np.zeros(11)
    target[0] = 1.57
    target[1] = -2.0
    result = backend.submit(
        target,
        tracking_received_ns=driver.now_ns,
        now_ns=driver.now_ns,
        dt_s=1.0 / 30.0,
    )
    assert result.accepted
    assert result.reason == "ACCEPTED"
    assert backend.enabled
    assert result.slew_limited


def test_large_replanned_chunk_delta_does_not_disable_hand() -> None:
    backend, driver = _armed_backend()
    first = np.zeros(11)
    first[1] = np.deg2rad(-31.0)
    assert backend.submit(
        first,
        tracking_received_ns=driver.now_ns,
        now_ns=driver.now_ns,
        dt_s=1.0 / 30.0,
    ).accepted

    replanned = first.copy()
    replanned[0] = 1.57
    replanned[8] = 2.40
    result = backend.submit(
        replanned,
        tracking_received_ns=driver.now_ns,
        now_ns=driver.now_ns,
        dt_s=1.0 / 30.0,
    )
    assert result.accepted
    assert backend.enabled
    assert result.slew_limited


def test_checkpoint_sdk_user_signs_reach_vendor_boundary_unchanged() -> None:
    backend, driver = _armed_backend(max_velocity_rad_s=1000.0)
    target = np.zeros(11)
    target[1] = np.deg2rad(-31.0)
    target[6] = 1.2  # middle_j1 is positive in checkpoint/SDK-user space.
    target[8] = 2.4  # little_j1 is positive in checkpoint/SDK-user space.
    result = backend.submit(
        target,
        tracking_received_ns=driver.now_ns,
        now_ns=driver.now_ns,
        dt_s=1.0 / 30.0,
    )
    assert result.accepted
    np.testing.assert_allclose(result.sdk_target_deg[6], np.rad2deg(1.2))
    np.testing.assert_allclose(result.sdk_target_deg[8], np.rad2deg(2.4))
    np.testing.assert_allclose(result.effective_checkpoint_qcmd_rad, target)


def test_crc_limit_matches_requested_large_threshold() -> None:
    backend, driver = _armed_backend(max_velocity_rad_s=1000.0)
    driver.diagnostics["crc_error_count"] = 29_999_999
    target = np.zeros(11)
    target[1] = np.deg2rad(-31.0)
    result = backend.submit(
        target,
        tracking_received_ns=driver.now_ns,
        now_ns=driver.now_ns,
        dt_s=1.0 / 30.0,
    )
    assert result.accepted
