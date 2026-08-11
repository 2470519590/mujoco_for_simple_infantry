"""腿长调度 LQR 与支撑前馈表。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class LengthSchedulePoint:
    length: float
    force_ff: float
    lqr_k: np.ndarray
    lqr_x0: np.ndarray
    lqr_u0: np.ndarray


@dataclass(frozen=True)
class ScheduledLengthControl:
    length: float
    force_ff: float
    lqr_k: np.ndarray
    lqr_x0: np.ndarray
    lqr_u0: np.ndarray


class LengthSchedule:
    """Piecewise-linear interpolation table keyed by actual common leg length."""

    def __init__(self, points: tuple[LengthSchedulePoint, ...]) -> None:
        if len(points) < 2:
            raise ValueError("length schedule requires at least two points")
        ordered = tuple(sorted(points, key=lambda item: item.length))
        lengths = np.array([point.length for point in ordered], dtype=float)
        if np.any(np.diff(lengths) <= 0.0):
            raise ValueError("length schedule points must be strictly increasing")
        self.points = ordered
        self.lengths = lengths

    @property
    def min_length(self) -> float:
        return float(self.lengths[0])

    @property
    def max_length(self) -> float:
        return float(self.lengths[-1])

    def evaluate(self, length: float) -> ScheduledLengthControl:
        bounded = float(np.clip(length, self.min_length, self.max_length))
        index = int(np.searchsorted(self.lengths, bounded, side="right") - 1)
        index = int(np.clip(index, 0, len(self.points) - 2))
        left = self.points[index]
        right = self.points[index + 1]
        span = max(right.length - left.length, 1e-12)
        alpha = (bounded - left.length) / span

        def mix(left_value, right_value):
            return (1.0 - alpha) * left_value + alpha * right_value

        return ScheduledLengthControl(
            length=bounded,
            force_ff=float(mix(left.force_ff, right.force_ff)),
            lqr_k=mix(left.lqr_k, right.lqr_k),
            lqr_x0=mix(left.lqr_x0, right.lqr_x0),
            lqr_u0=mix(left.lqr_u0, right.lqr_u0),
        )


def load_length_schedule(path: Path) -> LengthSchedule:
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    rows = raw.get("points")
    if not isinstance(rows, list):
        raise ValueError(f"length schedule requires a 'points' list: {path}")
    points: list[LengthSchedulePoint] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"invalid length schedule row: {row!r}")
        points.append(
            LengthSchedulePoint(
                length=float(row["L0"]),
                force_ff=float(row["F_l0"]),
                lqr_k=np.array(row["K"], dtype=float).reshape(2, 6),
                lqr_x0=np.array(row.get("X0", [0.0] * 6), dtype=float).reshape(6),
                lqr_u0=np.array(row.get("U0", [0.0, 0.0]), dtype=float).reshape(2),
            )
        )
    return LengthSchedule(tuple(points))
