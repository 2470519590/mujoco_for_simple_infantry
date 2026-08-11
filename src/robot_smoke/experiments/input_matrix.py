"""自由基座下 T 与 Tp 局部输入效应测量。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..control.lqr import compute_lqr_state as _compute_lqr_state
from ..control.vmc import _leg_shape_jacobian
from ..experiments.equilibrium import _initialize_equilibrium_data
from ..model.actuators import (
    apply_additive_wheel_torque_ctrl as _apply_wheel_torque,
    apply_joint_torque_ctrl as _apply_joint_torque,
    leg_drive_actuator_ids as _leg_drive_actuator_ids,
)
from ..model.mechanics import _set_base_height_for_wheel_contact
from ..core.mujoco_utils import copy_data as _copy_data
from ..core.mujoco_utils import assert_finite as _assert_finite


def _apply_support_and_input(mujoco, model, data, fl: float, t: float, tp: float) -> None:
    data.ctrl[:] = 0.0
    for side in ("left", "right"):
        jacobian = _leg_shape_jacobian(mujoco, model, data, side)
        tau = jacobian.T @ np.array([fl, tp], dtype=float)
        _apply_joint_torque(model, data, _leg_drive_actuator_ids(mujoco, model, side), tau)
    _apply_wheel_torque(mujoco, model, data, t)


def _pulse_response(mujoco, model, source, fl: float, input_index: int, amplitude: float, steps: int):
    data = _copy_data(mujoco, model, source)
    for _ in range(200):
        _apply_support_and_input(mujoco, model, data, fl, 0.0, 0.0)
        mujoco.mj_step(model, data)
    before = _compute_lqr_state(mujoco, model, data, 0.0, "wheel")
    for _ in range(steps):
        t = amplitude if input_index == 0 else 0.0
        tp = amplitude if input_index == 1 else 0.0
        _apply_support_and_input(mujoco, model, data, fl, t, tp)
        mujoco.mj_step(model, data)
    after = _compute_lqr_state(mujoco, model, data, 0.0, "wheel")
    before_vec = np.array([before.theta, before.theta_rate, before.x, before.x_rate, before.pitch, before.pitch_rate])
    after_vec = np.array([after.theta, after.theta_rate, after.x, after.x_rate, after.pitch, after.pitch_rate])
    return before_vec, after_vec


def run_input_matrix_test(mujoco, model, csv_path: Path, plot_path: Path) -> None:
    base = _initialize_equilibrium_data(mujoco, model, 0.35, 0.0, "normal", 0.25, 5, "upright-ik", 0)
    _set_base_height_for_wheel_contact(mujoco, model, base)
    base.qvel[:] = 0.0
    mujoco.mj_forward(model, base)
    fl = 34.5262
    pulse_steps = 5
    amplitude = 1.0
    dt = float(model.opt.timestep) * pulse_steps
    names = ("theta", "dtheta", "x", "dx", "phi", "dphi")
    rows = []
    # B maps an instantaneous input perturbation to Xdot.  The position
    # rows are kinematic rates and therefore have no direct torque column;
    # only the rate rows are estimated from the finite pulse.
    matrix = np.zeros((6, 2), dtype=float)
    for input_index, input_name in enumerate(("T", "Tp")):
        plus_before, plus_after = _pulse_response(mujoco, model, base, fl, input_index, amplitude, pulse_steps)
        minus_before, minus_after = _pulse_response(mujoco, model, base, fl, input_index, -amplitude, pulse_steps)
        delta = (plus_after - plus_before - (minus_after - minus_before)) / (2.0 * amplitude)
        derivative = np.zeros(6, dtype=float)
        derivative[[1, 3, 5]] = delta[[1, 3, 5]] / dt
        matrix[:, input_index] = derivative
        rows.append((input_name, *delta, *derivative))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("input", *[f"pulse_delta_{name}" for name in names], *[f"B_{name}_per_input_s" for name in names]))
        writer.writerows(rows)
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    indices = np.arange(6)
    width = 0.35
    axes[0].bar(indices - width / 2, matrix[:, 0], width, label="T")
    axes[0].bar(indices + width / 2, matrix[:, 1], width, label="Tp")
    axes[0].set_ylabel("Xdot / (N*m)")
    axes[0].set_xticks(indices, names)
    axes[0].legend()
    axes[0].grid(True, axis="y")
    axes[1].bar(indices - width / 2, matrix[:, 0] * dt, width, label="delta X per T pulse")
    axes[1].bar(indices + width / 2, matrix[:, 1] * dt, width, label="delta X per Tp pulse")
    axes[1].set_ylabel("pulse delta X / (N*m)")
    axes[1].set_xticks(indices, names)
    axes[1].legend()
    axes[1].grid(True, axis="y")
    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    print("input_matrix_B_continuous_local:")
    for name, row in zip(names, matrix):
        print(f"  {name}: T={row[0]:+.8g}, Tp={row[1]:+.8g}")
    print(f"input_matrix_csv: {csv_path}")
    print(f"input_matrix_plot: {plot_path}")
