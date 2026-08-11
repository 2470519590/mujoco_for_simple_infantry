"""按论文结构实现的横滚参考与左右支撑力差动补偿。"""

from __future__ import annotations

import math

import numpy as np


def clamp_leg_length(length: float, minimum_leg_length: float, maximum_leg_length: float) -> float:
    """Clamp a commanded leg-length reference before it enters VMC."""
    if maximum_leg_length < minimum_leg_length:
        raise ValueError("maximum_leg_length must be greater than or equal to minimum_leg_length")
    return float(np.clip(float(length), float(minimum_leg_length), float(maximum_leg_length)))


def base_roll_angle(qpos: np.ndarray) -> float:
    """Return the floating-base roll angle about its forward axis."""
    w, x, y, z = (float(value) for value in qpos[3:7])
    return math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


from dataclasses import dataclass


@dataclass(frozen=True)
class RollLegCommand:
    """Article 2.2 roll outputs before the two leg VMC instances."""

    left_length: float
    right_length: float
    geometric_offset: float
    error: float
    force: float


def roll_leg_length_targets(
    left_nominal_length: float,
    right_nominal_length: float,
    roll: float,
    roll_reference: float,
    track_width: float,
    force_kp: float,
    minimum_leg_length: float,
    maximum_leg_length: float,
) -> RollLegCommand:
    """Keep terrain geometry and dynamic roll compensation as separate paths.

    On the current flat-ground smoke model the estimated ground inclination is
    zero, so the geometric reference is the track-width projection of
    ``roll_reference``.  The article's dynamic compensation is a proportional
    force added after each leg-length PID, not a length-reference integrator.
    """
    geometric_offset = 0.5 * track_width * math.sin(roll_reference)
    error = roll_reference - roll
    return RollLegCommand(
        left_length=clamp_leg_length(left_nominal_length + geometric_offset, minimum_leg_length, maximum_leg_length),
        right_length=clamp_leg_length(right_nominal_length - geometric_offset, minimum_leg_length, maximum_leg_length),
        geometric_offset=float(geometric_offset),
        error=float(error),
        force=float(force_kp * error),
    )
