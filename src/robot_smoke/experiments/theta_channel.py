"""锁定基座的虚拟腿角力矩直接实验。"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from ..core.mujoco_utils import assert_finite as _assert_finite
from ..core.mujoco_utils import lock_base_to_initial as _lock_base_to_initial
from ..model.actuators import apply_joint_torque_ctrl as _apply_joint_torque_ctrl
from ..model.actuators import leg_drive_actuator_ids as _leg_drive_actuator_ids
from ..model.kinematics import compute_virtual_leg_state as _compute_virtual_leg_state
from ..control.vmc import _leg_shape_jacobian


def run_locked_theta_channel_test(mujoco, model, output_csv: Path, output_png: Path) -> None:
    """Track -20, 0, +20 degree virtual-leg angle targets with base locked."""
    commands_deg = (-40.0, -20.0, 0.0, 20.0, 40.0)
    steps_per_command = int(round(4.0 / float(model.opt.timestep)))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    samples: list[tuple[float, float, float, float]] = []

    for command_deg in commands_deg:
        command_rad = math.radians(command_deg)
        for _ in range(steps_per_command):
            _lock_base_to_initial(mujoco, model, data)
            data.ctrl[:] = 0.0
            for side in ("left", "right"):
                jacobian = _leg_shape_jacobian(mujoco, model, data, side)
                state = _compute_virtual_leg_state(mujoco, model, data, side)
                angle_error = math.atan2(
                    math.sin(command_rad - state.theta),
                    math.cos(command_rad - state.theta),
                )
                theta_force = 25.0 * angle_error - 2.5 * state.theta_rate
                joint_tau = jacobian.T @ np.array([0.0, theta_force], dtype=float)
                _apply_joint_torque_ctrl(model, data, _leg_drive_actuator_ids(mujoco, model, side), joint_tau)
            mujoco.mj_step(model, data)
            _assert_finite("theta-channel qpos", data.qpos)
            _assert_finite("theta-channel qvel", data.qvel)
            left = _compute_virtual_leg_state(mujoco, model, data, "left")
            right = _compute_virtual_leg_state(mujoco, model, data, "right")
            samples.append((float(data.time), command_deg, math.degrees(left.theta), math.degrees(right.theta)))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("time_s", "theta_target_deg", "left_theta_world_deg", "right_theta_world_deg"))
        writer.writerows(samples)

    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    values = np.asarray(samples, dtype=float)
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(values[:, 0], values[:, 2], label="left theta_world")
    axis.plot(values[:, 0], values[:, 3], "--", label="right theta_world")
    axis.step(values[:, 0], values[:, 1], where="post", label="theta target", color="black", alpha=0.55)
    axis.set_xlabel("time (s)")
    axis.set_ylabel("theta (deg)")
    axis.grid(True)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_png, dpi=160)
    plt.close(figure)
    print(f"theta_channel_csv: {output_csv}")
    print(f"theta_channel_plot: {output_png}")


def visualize_locked_theta_channel_test(mujoco, model) -> None:
    """Show the same five target-angle steps in the MuJoCo viewer."""
    commands_deg = (-40.0, -20.0, 0.0, 20.0, 40.0)
    steps_per_command = int(round(4.0 / float(model.opt.timestep)))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    import mujoco.viewer  # pylint: disable=import-outside-toplevel

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for command_deg in commands_deg:
            command_rad = math.radians(command_deg)
            for _ in range(steps_per_command):
                _lock_base_to_initial(mujoco, model, data)
                data.ctrl[:] = 0.0
                for side in ("left", "right"):
                    jacobian = _leg_shape_jacobian(mujoco, model, data, side)
                    state = _compute_virtual_leg_state(mujoco, model, data, side)
                    angle_error = math.atan2(
                        math.sin(command_rad - state.theta),
                        math.cos(command_rad - state.theta),
                    )
                    theta_force = 25.0 * angle_error - 2.5 * state.theta_rate
                    joint_tau = jacobian.T @ np.array([0.0, theta_force], dtype=float)
                    _apply_joint_torque_ctrl(model, data, _leg_drive_actuator_ids(mujoco, model, side), joint_tau)
                mujoco.mj_step(model, data)
                if not viewer.is_running():
                    return
                if int(data.time / float(model.opt.timestep)) % 16 == 0:
                    viewer.sync()
        viewer.sync()
        print("theta_channel_viewer: completed five 4-second angle steps")
