"""沿腿推力 F_l 通道诊断测试。"""

from __future__ import annotations

import math

import numpy as np

from ..model.kinematics import compute_virtual_leg_state as _compute_virtual_leg_state
from ..model.mechanics import (
    _collect_static_operating_point_sample,
)
from ..control.vmc import (
    _drive_constant_length_force_ctrl,
)
from ..core.mujoco_utils import assert_finite as _assert_finite
from ..core.types import VmcDiagnostics

def _run_fl_channel_test(
    mujoco,
    model,
    forces: tuple[float, ...],
    steps_per_force: int,
) -> None:
    print("fl_channel_test:")
    print("  note: LQR T/Tp off, tau_guard off, constant F_l only")
    for length_force in forces:
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        left_lengths: list[float] = []
        right_lengths: list[float] = []
        left_rates: list[float] = []
        right_rates: list[float] = []
        left_diag = VmcDiagnostics()
        right_diag = VmcDiagnostics()
        for step in range(steps_per_force):
            data.ctrl[:] = 0.0
            _, left_diag = _drive_constant_length_force_ctrl(mujoco, model, data, "left", length_force)
            _, right_diag = _drive_constant_length_force_ctrl(mujoco, model, data, "right", length_force)
            mujoco.mj_step(model, data)
            _assert_finite("qpos", data.qpos)
            _assert_finite("qvel", data.qvel)
            left_state = _compute_virtual_leg_state(mujoco, model, data, "left")
            right_state = _compute_virtual_leg_state(mujoco, model, data, "right")
            left_lengths.append(left_state.length)
            right_lengths.append(right_state.length)
            left_rates.append(left_state.length_rate)
            right_rates.append(right_state.length_rate)
        initial_window = max(1, min(50, len(left_rates)))
        tail_window = max(1, min(200, len(left_lengths)))
        sample = _collect_static_operating_point_sample(
            mujoco,
            model,
            data,
            f"F_l={length_force:.6g}",
            0.0,
            0.0,
            left_diag=left_diag,
            right_diag=right_diag,
        )
        print(f"  - F_l: {length_force:.6g}")
        print(f"    left_L_mean_tail: {float(np.mean(left_lengths[-tail_window:])):.6g}")
        print(f"    right_L_mean_tail: {float(np.mean(right_lengths[-tail_window:])):.6g}")
        print(f"    left_dL_initial_mean: {float(np.mean(left_rates[:initial_window])):.6g}")
        print(f"    right_dL_initial_mean: {float(np.mean(right_rates[:initial_window])):.6g}")
        print(f"    left_tau_support: [{sample.left_tau_front:.6g}, {sample.left_tau_rear:.6g}]")
        print(f"    right_tau_support: [{sample.right_tau_front:.6g}, {sample.right_tau_rear:.6g}]")
        print(
            "    contact_normal_force: "
            f"left={sample.left_contact_normal_force:.6g}, "
            f"right={sample.right_contact_normal_force:.6g}"
        )
        print(f"    final_base_height: {float(data.qpos[2]):.6g}")


def _run_fl_pulse_test(
    mujoco,
    model,
    base_force: float,
    pulse_force: float,
    settle_steps: int,
    pulse_steps: int,
) -> None:
    print("fl_pulse_test:")
    print("  note: LQR T/Tp off, tau_guard off, constant F_l baseline with +/- pulse")
    for sign in (1.0, -1.0):
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        for _ in range(settle_steps):
            data.ctrl[:] = 0.0
            _drive_constant_length_force_ctrl(mujoco, model, data, "left", base_force)
            _drive_constant_length_force_ctrl(mujoco, model, data, "right", base_force)
            mujoco.mj_step(model, data)
        left_before = _compute_virtual_leg_state(mujoco, model, data, "left")
        right_before = _compute_virtual_leg_state(mujoco, model, data, "right")
        dls: list[float] = []
        for _ in range(pulse_steps):
            data.ctrl[:] = 0.0
            force = base_force + sign * pulse_force
            _drive_constant_length_force_ctrl(mujoco, model, data, "left", force)
            _drive_constant_length_force_ctrl(mujoco, model, data, "right", force)
            mujoco.mj_step(model, data)
            left_state = _compute_virtual_leg_state(mujoco, model, data, "left")
            right_state = _compute_virtual_leg_state(mujoco, model, data, "right")
            dls.append(0.5 * (left_state.length_rate + right_state.length_rate))
        left_after = _compute_virtual_leg_state(mujoco, model, data, "left")
        right_after = _compute_virtual_leg_state(mujoco, model, data, "right")
        print(f"  - pulse_delta_F_l: {sign * pulse_force:.6g}")
        print(f"    left_L_before: {left_before.length:.6g}")
        print(f"    right_L_before: {right_before.length:.6g}")
        print(f"    left_L_after: {left_after.length:.6g}")
        print(f"    right_L_after: {right_after.length:.6g}")
        print(f"    dL_mean_during_pulse: {float(np.mean(dls)):.6g}")


def _rms(values: list[float]) -> float:
    if not values:
        return math.nan
    array = np.array(values, dtype=float)
    return float(math.sqrt(float(np.mean(array * array))))


