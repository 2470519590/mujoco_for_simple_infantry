"""本地轮腿 smoke 测试使用的解析运动参考。"""

from __future__ import annotations


def trapezoid_speed_reference(profile: str | None, time_s: float) -> tuple[float, float]:
    """Return world-frame wheel position and velocity references."""
    if profile is None:
        return 0.0, 0.0
    peak_speed = {"low": 1.00, "medium": 2.00, "high": 3.00}[profile]
    ramp_s = 0.75
    cruise_s = 3.6 if profile == "high" else 6.0
    acceleration = peak_speed / ramp_s
    time_s = max(0.0, time_s)
    if time_s < ramp_s:
        return 0.5 * acceleration * time_s * time_s, acceleration * time_s
    ramp_distance = 0.5 * peak_speed * ramp_s
    if time_s < ramp_s + cruise_s:
        cruise_time = time_s - ramp_s
        return ramp_distance + peak_speed * cruise_time, peak_speed
    if time_s < 2.0 * ramp_s + cruise_s:
        ramp_down_time = time_s - ramp_s - cruise_s
        return (
            ramp_distance + peak_speed * cruise_s + peak_speed * ramp_down_time
            - 0.5 * acceleration * ramp_down_time * ramp_down_time,
            peak_speed - acceleration * ramp_down_time,
        )
    return peak_speed * (ramp_s + cruise_s), 0.0


def is_speed_profile_cruise(profile: str | None, time_s: float) -> bool:
    """Return whether a speed profile is in its constant-velocity segment."""
    if profile is None:
        return False
    ramp_s = 0.75
    cruise_s = 3.6 if profile == "high" else 6.0
    time_s = max(0.0, time_s)
    return ramp_s <= time_s < ramp_s + cruise_s
