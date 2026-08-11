"""力学诊断与静态工作点辅助函数。"""

from __future__ import annotations

import math

import numpy as np

from ..model.kinematics import compute_virtual_leg_state as _compute_virtual_leg_state
from ..model.kinematics import wheel_radius as _wheel_radius
from ..control.lqr import compute_lqr_state as _compute_lqr_state
from ..control.lqr import set_base_pitch as _set_base_pitch
from ..core.mujoco_utils import id_by_name as _id_by_name
from ..core.types import MechanicsEquilibriumSeed, StaticOperatingPointSample, VmcDiagnostics
from ..control.vmc import _leg_shape_jacobian


def _format_matrix_rows(matrix: np.ndarray) -> list[str]:
    return ["[" + ", ".join(f"{value:.6g}" for value in row) + "]" for row in matrix]


def _estimated_support_force_per_leg(model) -> float:
    return float(np.sum(model.body_mass) * 9.81 * 0.5)


def support_force_scale_for_length(length: float) -> float:
    """Local static VMC feedforward schedule for the current five-bar geometry."""
    return float(np.clip(0.68 + 0.87 * (length - 0.35), 0.60, 0.76))


def _mechanics_equilibrium_seed(mujoco, model, data, theta_ref: float) -> MechanicsEquilibriumSeed:
    total_mass = float(np.sum(model.body_mass))
    gravity = 9.81
    cos_theta = math.cos(theta_ref)
    if abs(cos_theta) < 1e-3:
        raise RuntimeError(f"theta_ref too close to horizontal for support seed: {theta_ref}")
    com_offset = _model_com_x(mujoco, model, data) - _average_wheel_x(mujoco, model, data)
    length_force_per_leg = total_mass * gravity / (2.0 * cos_theta)
    pitch_torque = -total_mass * gravity * com_offset
    return MechanicsEquilibriumSeed(
        total_mass=total_mass,
        gravity=gravity,
        theta_ref=theta_ref,
        com_offset=com_offset,
        length_force_per_leg=length_force_per_leg,
        pitch_torque=pitch_torque,
        wheel_torque=0.0,
    )


def _contact_normal_force_for_wheel(mujoco, model, data, side: str) -> float:
    wheel_geom = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_wheel_geom")
    total_force = 0.0
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if int(contact.geom1) != wheel_geom and int(contact.geom2) != wheel_geom:
            continue
        contact_force = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, contact_index, contact_force)
        total_force += abs(float(contact_force[0]))
    return total_force


def _model_com_x(mujoco, model, data) -> float:
    total_mass = float(np.sum(model.body_mass))
    if total_mass <= 0.0:
        return math.nan
    weighted_x = float(np.dot(model.body_mass, data.xipos[:, 0]))
    return weighted_x / total_mass


def _average_wheel_x(mujoco, model, data) -> float:
    left_wheel_geom = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_geom")
    right_wheel_geom = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "right_wheel_geom")
    return 0.5 * (float(data.geom_xpos[left_wheel_geom, 0]) + float(data.geom_xpos[right_wheel_geom, 0]))


def _standing_base_qpos_for_virtual_leg_target(
    mujoco,
    model,
    left_target: tuple[float, float],
    right_target: tuple[float, float],
) -> np.ndarray:
    base_qpos = model.qpos0[:7].copy()
    wheel_radius = _wheel_radius(mujoco, model)
    nominal_hip_z_in_base = -0.05
    left_vertical = left_target[0] * math.cos(left_target[1])
    right_vertical = right_target[0] * math.cos(right_target[1])
    base_qpos[2] = wheel_radius - nominal_hip_z_in_base + 0.5 * (left_vertical + right_vertical)
    _set_base_pitch(base_qpos, 0.0)
    return base_qpos


def _set_base_height_for_wheel_contact(mujoco, model, data) -> None:
    wheel_radius = _wheel_radius(mujoco, model)
    left_wheel_geom = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_geom")
    right_wheel_geom = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "right_wheel_geom")
    for _ in range(4):
        wheel_z = 0.5 * (
            float(data.geom_xpos[left_wheel_geom, 2])
            + float(data.geom_xpos[right_wheel_geom, 2])
        )
        data.qpos[2] += wheel_radius - wheel_z
        _set_base_pitch(data.qpos, 0.0)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)


def apply_base_impact(mujoco, model, data, impact_level: str | None, step: int) -> None:
    """Apply one fixed horizontal base impulse for disturbance tests."""
    data.xfrc_applied[:] = 0.0
    if impact_level not in ("small", "medium"):
        return
    timestep = float(model.opt.timestep)
    start = int(round(3.0 / timestep))
    duration = max(1, int(round(0.15 / timestep)))
    if start <= step < start + duration:
        base_id = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "base")
        data.xfrc_applied[base_id, 0] = 40.0 if impact_level == "small" else 80.0


def _collect_static_operating_point_sample(
    mujoco,
    model,
    data,
    label: str,
    wheel_torque: float,
    pitch_torque: float,
    x_source: str = "wheel",
    left_diag: VmcDiagnostics | None = None,
    right_diag: VmcDiagnostics | None = None,
) -> StaticOperatingPointSample:
    left_state = _compute_virtual_leg_state(mujoco, model, data, "left")
    right_state = _compute_virtual_leg_state(mujoco, model, data, "right")
    left_diag = left_diag or VmcDiagnostics()
    right_diag = right_diag or VmcDiagnostics()
    return StaticOperatingPointSample(
        label=label,
        lqr_state=_compute_lqr_state(mujoco, model, data, 0.0, x_source),
        left_length=left_state.length,
        right_length=right_state.length,
        left_length_rate=left_state.length_rate,
        right_length_rate=right_state.length_rate,
        wheel_torque=wheel_torque,
        pitch_torque=pitch_torque,
        left_length_force=left_diag.length_force,
        right_length_force=right_diag.length_force,
        left_length_force_raw=left_diag.length_force_raw,
        right_length_force_raw=right_diag.length_force_raw,
        left_support_tau_front=left_diag.support_tau_front,
        left_support_tau_rear=left_diag.support_tau_rear,
        right_support_tau_front=right_diag.support_tau_front,
        right_support_tau_rear=right_diag.support_tau_rear,
        left_tau_front_raw=left_diag.joint_tau_front_raw,
        left_tau_rear_raw=left_diag.joint_tau_rear_raw,
        right_tau_front_raw=right_diag.joint_tau_front_raw,
        right_tau_rear_raw=right_diag.joint_tau_rear_raw,
        left_tau_front=left_diag.joint_tau_front,
        left_tau_rear=left_diag.joint_tau_rear,
        right_tau_front=right_diag.joint_tau_front,
        right_tau_rear=right_diag.joint_tau_rear,
        left_contact_normal_force=_contact_normal_force_for_wheel(mujoco, model, data, "left"),
        right_contact_normal_force=_contact_normal_force_for_wheel(mujoco, model, data, "right"),
        left_jacobian=_leg_shape_jacobian(mujoco, model, data, "left"),
        right_jacobian=_leg_shape_jacobian(mujoco, model, data, "right"),
        qpos=data.qpos.copy(),
    )


def _print_static_operating_point_sample(sample: StaticOperatingPointSample, force_ff: float | None = None) -> None:
    state = sample.lqr_state
    print(f"static_operating_point: {sample.label}")
    print(
        "  X0: "
        f"[{state.theta:.6g}, {state.theta_rate:.6g}, {state.x:.6g}, "
        f"{state.x_rate:.6g}, {state.pitch:.6g}, {state.pitch_rate:.6g}]"
    )
    print(f"  U0: [{sample.wheel_torque:.6g}, {sample.pitch_torque:.6g}]")
    if force_ff is not None:
        print(f"  F_l0_config: {force_ff:.6g}")
    print(
        "  F_l_cmd: "
        f"left={sample.left_length_force:.6g}, right={sample.right_length_force:.6g}"
    )
    print(
        "  F_l_raw: "
        f"left={sample.left_length_force_raw:.6g}, right={sample.right_length_force_raw:.6g}"
    )
    print(f"  L0: left={sample.left_length:.6g}, right={sample.right_length:.6g}")
    print(f"  dL0: left={sample.left_length_rate:.6g}, right={sample.right_length_rate:.6g}")
    print(
        "  tau_support: "
        f"left=[{sample.left_support_tau_front:.6g}, {sample.left_support_tau_rear:.6g}], "
        f"right=[{sample.right_support_tau_front:.6g}, {sample.right_support_tau_rear:.6g}]"
    )
    print(
        "  tau_total_before_clip: "
        f"left=[{sample.left_tau_front_raw:.6g}, {sample.left_tau_rear_raw:.6g}], "
        f"right=[{sample.right_tau_front_raw:.6g}, {sample.right_tau_rear_raw:.6g}]"
    )
    print(
        "  tau_total_after_clip: "
        f"left=[{sample.left_tau_front:.6g}, {sample.left_tau_rear:.6g}], "
        f"right=[{sample.right_tau_front:.6g}, {sample.right_tau_rear:.6g}]"
    )
    print(
        "  contact_force0: "
        f"left={sample.left_contact_normal_force:.6g}, right={sample.right_contact_normal_force:.6g}"
    )
    print("  J0_left:")
    for row in _format_matrix_rows(sample.left_jacobian):
        print(f"    {row}")
    print("  J0_right:")
    for row in _format_matrix_rows(sample.right_jacobian):
        print(f"    {row}")
    print("  q0:")
    print("    [" + ", ".join(f"{value:.6g}" for value in sample.qpos) + "]")
