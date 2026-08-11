"""按论文结构实现的 yaw 角速度 PD 与轮端力矩分配。"""

from __future__ import annotations

import numpy as np

TURN_RATE_MAGNITUDES = {
    "low": np.pi * 0.5,
    "medium": np.pi,
    "high": 10.0,
}


def turn_rate_magnitude(speed: str) -> float:
    """Return the positive yaw-rate magnitude for a named turn speed."""
    return float(TURN_RATE_MAGNITUDES[speed])


def turn_rate_reference(
    turn_direction: str | None,
    turn_speed: str,
    turn_test: bool,
    time_s: float,
) -> float:
    """Return a signed manual or trapezoidal yaw-rate reference."""
    magnitude = turn_rate_magnitude(turn_speed)
    if turn_test:
        # Single trapezoid: start at 1 s, ramp for 150 ms, stop at 5 s.
        time_s = max(time_s, 0.0)
        ramp_time = 0.15
        start_time = 1.0
        stop_time = 5.0
        if time_s <= start_time:
            return 0.0
        if time_s < start_time + ramp_time:
            return magnitude * (time_s - start_time) / ramp_time
        if time_s <= stop_time:
            return magnitude
        if time_s < stop_time + ramp_time:
            return magnitude * (1.0 - (time_s - stop_time) / ramp_time)
        return 0.0
    if turn_direction is None:
        return 0.0
    return magnitude if turn_direction == "left" else -magnitude


def yaw_turn_torque(
    yaw_rate_reference: float,
    yaw_rate: float,
    previous_error: float,
    dt: float,
    kp: float = 3.0,
    kd: float = 0.1,
    error_rate: float | None = None,
    error: float | None = None,
) -> tuple[float, float]:
    """Return differential wheel torque and updated yaw-rate error."""
    error = yaw_rate_reference - yaw_rate if error is None else error
    if error_rate is None:
        error_rate = (error - previous_error) / max(dt, 1e-6)
    torque = float(np.clip(kp * error + kd * error_rate, -2.0, 2.0))
    return torque, error


def split_wheel_torque(total_torque: float, turn_torque: float) -> tuple[float, float]:
    """Article allocation: opposite yaw torque is superposed on total T."""
    return 0.5 * total_torque - turn_torque, 0.5 * total_torque + turn_torque
