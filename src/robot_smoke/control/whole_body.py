"""按论文控制器使用的整车虚拟控制指令组合。"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class WholeBodyVirtualCommand:
    """Task-space commands passed to the two leg VMC instances."""

    wheel_torque: float
    pitch_torque: float
    left_pitch_torque: float
    right_pitch_torque: float
    left_length_force_bias: float
    right_length_force_bias: float
    roll_force: float


def compose_whole_body_command(
    wheel_torque: float,
    pitch_torque: float,
    sync_torque: float,
    base_length_force: float,
    roll_force: float,
) -> WholeBodyVirtualCommand:
    """Compose the article's central balance, sync, length and roll paths.

    ``pitch_torque`` is the whole-body ``Tp`` output. The sync PD adds an
    equal-and-opposite differential component.  Article 2.2 applies roll
    compensation directly to the post-PID along-leg forces.
    """
    return WholeBodyVirtualCommand(
        wheel_torque=float(wheel_torque),
        pitch_torque=float(pitch_torque),
        left_pitch_torque=float(pitch_torque + sync_torque),
        right_pitch_torque=float(pitch_torque - sync_torque),
        left_length_force_bias=float(base_length_force + roll_force),
        right_length_force_bias=float(base_length_force - roll_force),
        roll_force=float(roll_force),
    )
