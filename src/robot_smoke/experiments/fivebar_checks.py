"""五连杆诊断命令辅助函数。"""

from __future__ import annotations

import numpy as np

from ..model.fivebar import (
    analytic_fivebar_kinematics_from_q as _analytic_fivebar_kinematics_from_q,
    body_local_xz as _body_local_xz,
    compute_leg_branch_metrics as _compute_leg_branch_metrics,
    drive_qpos_for_side as _drive_qpos_for_side,
    equilibrium_analytic_ik_targets as _equilibrium_analytic_ik_targets,
)
from ..model.kinematics import compute_virtual_leg_state as _compute_virtual_leg_state
from ..model.mechanics import (
    _contact_normal_force_for_wheel,
    _estimated_support_force_per_leg,
    _format_matrix_rows,
    _set_base_height_for_wheel_contact,
    _standing_base_qpos_for_virtual_leg_target,
)
from ..core.mujoco_utils import assert_finite as _assert_finite
from ..core.mujoco_utils import lock_base_to_qpos as _lock_base_to_qpos
from ..control.vmc import (
    _drive_constant_length_force_ctrl,
    _drive_joint_position_ctrl,
    _settled_numeric_leg_shape_jacobian,
    _wrapped_angle_delta,
)

def _run_fivebar_kinematics_check(
    mujoco,
    model,
    l_slices: tuple[float, ...],
    theta_refs: tuple[float, ...],
) -> None:
    print("fivebar_kinematics_check:")
    print("  note: analytic IK -> locked-base MuJoCo closed-chain settle -> measured virtual leg")
    for side in ("left", "right"):
        front_anchor = _body_local_xz(mujoco, model, f"{side}_front_upper")
        rear_anchor = _body_local_xz(mujoco, model, f"{side}_rear_upper")
        front_upper = _body_local_xz(mujoco, model, f"{side}_front_elbow")
        rear_upper = _body_local_xz(mujoco, model, f"{side}_rear_elbow")
        front_lower = _body_local_xz(mujoco, model, f"{side}_front_hub")
        rear_lower = _body_local_xz(mujoco, model, f"{side}_rear_hub")
        print(f"  {side}_geometry:")
        print(f"    front_anchor_xz: [{front_anchor[0]:.6g}, {front_anchor[1]:.6g}]")
        print(f"    rear_anchor_xz: [{rear_anchor[0]:.6g}, {rear_anchor[1]:.6g}]")
        print(f"    front_upper_length: {float(np.linalg.norm(front_upper)):.6g}")
        print(f"    rear_upper_length: {float(np.linalg.norm(rear_upper)):.6g}")
        print(f"    front_lower_length: {float(np.linalg.norm(front_lower)):.6g}")
        print(f"    rear_lower_length: {float(np.linalg.norm(rear_lower)):.6g}")

    max_length_error = 0.0
    max_theta_error = 0.0
    max_branch_violation = 0.0
    for l_slice in l_slices:
        for theta_ref in theta_refs:
            target = (l_slice, theta_ref)
            left_targets = _equilibrium_analytic_ik_targets(mujoco, model, "left", target)
            right_targets = _equilibrium_analytic_ik_targets(mujoco, model, "right", target)
            if left_targets is None or right_targets is None:
                print(
                    "  - "
                    f"L0_slice={l_slice:.6g}, theta_ref={theta_ref:.6g}, "
                    f"reachable=False, left_ik={left_targets is not None}, right_ik={right_targets is not None}"
                )
                continue
            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            standing_base_qpos = _standing_base_qpos_for_virtual_leg_target(mujoco, model, target, target)
            data.qpos[:7] = standing_base_qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            for _ in range(240):
                _lock_base_to_qpos(mujoco, model, data, standing_base_qpos)
                data.ctrl[:] = 0.0
                _drive_joint_position_ctrl(mujoco, model, data, "left", left_targets[0], left_targets[1], 55.0, 2.0)
                _drive_joint_position_ctrl(mujoco, model, data, "right", right_targets[0], right_targets[1], 55.0, 2.0)
                mujoco.mj_step(model, data)
                _assert_finite("fivebar_kinematics_check qpos", data.qpos)
                _assert_finite("fivebar_kinematics_check qvel", data.qvel)
            _lock_base_to_qpos(mujoco, model, data, standing_base_qpos)
            left_state = _compute_virtual_leg_state(mujoco, model, data, "left")
            right_state = _compute_virtual_leg_state(mujoco, model, data, "right")
            left_branch = _compute_leg_branch_metrics(mujoco, model, data, "left")
            right_branch = _compute_leg_branch_metrics(mujoco, model, data, "right")
            left_l_error = left_state.length - l_slice
            right_l_error = right_state.length - l_slice
            left_theta_error = _wrapped_angle_delta(left_state.theta, theta_ref)
            right_theta_error = _wrapped_angle_delta(right_state.theta, theta_ref)
            max_length_error = max(max_length_error, abs(left_l_error), abs(right_l_error))
            max_theta_error = max(max_theta_error, abs(left_theta_error), abs(right_theta_error))
            max_branch_violation = max(max_branch_violation, left_branch.violation, right_branch.violation)
            print(
                "  - "
                f"L0_slice={l_slice:.6g}, theta_ref={theta_ref:.6g}, reachable=True, "
                f"left_q=[{left_targets[0]:.6g}, {left_targets[1]:.6g}], "
                f"right_q=[{right_targets[0]:.6g}, {right_targets[1]:.6g}], "
                f"left_L={left_state.length:.6g}, right_L={right_state.length:.6g}, "
                f"left_dL={left_l_error:.6g}, right_dL={right_l_error:.6g}, "
                f"left_theta={left_state.theta:.6g}, right_theta={right_state.theta:.6g}, "
                f"left_dtheta={left_theta_error:.6g}, right_dtheta={right_theta_error:.6g}, "
                f"branch_vio=[{left_branch.violation:.6g}, {right_branch.violation:.6g}]"
            )
    print("  summary:")
    print(f"    max_abs_length_error: {max_length_error:.6g}")
    print(f"    max_abs_theta_error: {max_theta_error:.6g}")
    print(f"    max_branch_violation: {max_branch_violation:.6g}")


def _settle_analytic_fivebar_pose(
    mujoco,
    model,
    l_slice: float,
    theta_ref: float,
):
    target = (l_slice, theta_ref)
    left_targets = _equilibrium_analytic_ik_targets(mujoco, model, "left", target)
    right_targets = _equilibrium_analytic_ik_targets(mujoco, model, "right", target)
    if left_targets is None or right_targets is None:
        return None
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    standing_base_qpos = _standing_base_qpos_for_virtual_leg_target(mujoco, model, target, target)
    data.qpos[:7] = standing_base_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(240):
        _lock_base_to_qpos(mujoco, model, data, standing_base_qpos)
        data.ctrl[:] = 0.0
        _drive_joint_position_ctrl(mujoco, model, data, "left", left_targets[0], left_targets[1], 55.0, 2.0)
        _drive_joint_position_ctrl(mujoco, model, data, "right", right_targets[0], right_targets[1], 55.0, 2.0)
        mujoco.mj_step(model, data)
        _assert_finite("fivebar_jacobian_pose qpos", data.qpos)
        _assert_finite("fivebar_jacobian_pose qvel", data.qvel)
    _lock_base_to_qpos(mujoco, model, data, standing_base_qpos)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    _set_base_height_for_wheel_contact(mujoco, model, data)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    return data


def _run_fivebar_jacobian_check(
    mujoco,
    model,
    l_slices: tuple[float, ...],
    theta_refs: tuple[float, ...],
    force_scale: float,
    load_steps: int,
) -> None:
    support_force = force_scale * _estimated_support_force_per_leg(model)
    print("fivebar_jacobian_check:")
    print(
        "  note: analytic IK/Jacobian vs settled numeric Jacobian; "
        "load segment is support-only and is diagnostic, not a balance claim"
    )
    print(f"  support_force_per_leg: {support_force:.6g}")
    print(f"  load_steps: {load_steps}")
    for l_slice in l_slices:
        for theta_ref in theta_refs:
            target = (l_slice, theta_ref)
            data = _settle_analytic_fivebar_pose(mujoco, model, l_slice, theta_ref)
            if data is None:
                print(f"  - L0_slice={l_slice:.6g}, theta_ref={theta_ref:.6g}, reachable=False")
                continue
            print(f"  - L0_slice={l_slice:.6g}, theta_ref={theta_ref:.6g}")
            for side in ("left", "right"):
                ik_targets = _equilibrium_analytic_ik_targets(mujoco, model, side, target)
                q_front, q_rear = _drive_qpos_for_side(mujoco, model, data, side)
                state = _compute_virtual_leg_state(mujoco, model, data, side)
                numeric_j = _settled_numeric_leg_shape_jacobian(mujoco, model, data, side)
                analytic_at_actual = _analytic_fivebar_kinematics_from_q(mujoco, model, side, q_front, q_rear)
                analytic_at_target = (
                    _analytic_fivebar_kinematics_from_q(mujoco, model, side, ik_targets[0], ik_targets[1])
                    if ik_targets is not None
                    else None
                )
                if analytic_at_actual is None:
                    print(f"    {side}: analytic_from_actual_q_failed=True")
                    continue
                analytic_j = analytic_at_actual.jacobian
                jacobian_diff = numeric_j - analytic_j
                tau_numeric = numeric_j.T @ np.array([support_force, 0.0], dtype=float)
                tau_analytic = analytic_j.T @ np.array([support_force, 0.0], dtype=float)
                print(f"    {side}:")
                if ik_targets is not None:
                    print(f"      q_ik: [{ik_targets[0]:.6g}, {ik_targets[1]:.6g}]")
                print(f"      q_settled: [{q_front:.6g}, {q_rear:.6g}]")
                print(
                    f"      measured_shape: L={state.length:.6g}, theta={state.theta:.6g}, "
                    f"dL={state.length_rate:.6g}, dtheta={state.theta_rate:.6g}"
                )
                print(
                    f"      analytic_shape_actual_q: L={analytic_at_actual.length:.6g}, "
                    f"theta={analytic_at_actual.theta:.6g}"
                )
                if analytic_at_target is not None:
                    print(
                        f"      analytic_shape_ik_q: L={analytic_at_target.length:.6g}, "
                        f"theta={analytic_at_target.theta:.6g}"
                    )
                print("      analytic_J_actual_q:")
                for row in _format_matrix_rows(analytic_j):
                    print(f"        {row}")
                print("      numeric_J_settled:")
                for row in _format_matrix_rows(numeric_j):
                    print(f"        {row}")
                print("      numeric_minus_analytic_J:")
                for row in _format_matrix_rows(jacobian_diff):
                    print(f"        {row}")
                print(f"      max_abs_J_diff: {float(np.max(np.abs(jacobian_diff))):.6g}")
                print(f"      tau_support_analytic_Fl: [{tau_analytic[0]:.6g}, {tau_analytic[1]:.6g}]")
                print(f"      tau_support_numeric_Fl: [{tau_numeric[0]:.6g}, {tau_numeric[1]:.6g}]")

            if load_steps > 0:
                load_data = mujoco.MjData(model)
                load_data.qpos[:] = data.qpos
                load_data.qvel[:] = 0.0
                load_data.act[:] = data.act
                load_data.ctrl[:] = 0.0
                mujoco.mj_forward(model, load_data)
                initial_left = _compute_virtual_leg_state(mujoco, model, load_data, "left")
                initial_right = _compute_virtual_leg_state(mujoco, model, load_data, "right")
                max_abs_tau = 0.0
                saturated_steps = 0
                for _ in range(load_steps):
                    load_data.ctrl[:] = 0.0
                    left_saturated, left_diag = _drive_constant_length_force_ctrl(
                        mujoco, model, load_data, "left", support_force
                    )
                    right_saturated, right_diag = _drive_constant_length_force_ctrl(
                        mujoco, model, load_data, "right", support_force
                    )
                    max_abs_tau = max(
                        max_abs_tau,
                        abs(left_diag.joint_tau_front),
                        abs(left_diag.joint_tau_rear),
                        abs(right_diag.joint_tau_front),
                        abs(right_diag.joint_tau_rear),
                    )
                    if left_saturated or right_saturated:
                        saturated_steps += 1
                    mujoco.mj_step(model, load_data)
                    _assert_finite("fivebar_jacobian_load qpos", load_data.qpos)
                    _assert_finite("fivebar_jacobian_load qvel", load_data.qvel)
                final_left = _compute_virtual_leg_state(mujoco, model, load_data, "left")
                final_right = _compute_virtual_leg_state(mujoco, model, load_data, "right")
                print("    support_only_load:")
                print(f"      initial_L: left={initial_left.length:.6g}, right={initial_right.length:.6g}")
                print(f"      final_L: left={final_left.length:.6g}, right={final_right.length:.6g}")
                print(f"      final_dL: left={final_left.length_rate:.6g}, right={final_right.length_rate:.6g}")
                print(
                    "      contact_normal_force: "
                    f"left={_contact_normal_force_for_wheel(mujoco, model, load_data, 'left'):.6g}, "
                    f"right={_contact_normal_force_for_wheel(mujoco, model, load_data, 'right'):.6g}"
                )
                print(f"      saturated_steps: {saturated_steps}")
                print(f"      max_abs_joint_tau: {max_abs_tau:.6g}")
                print(f"      final_base_height: {float(load_data.qpos[2]):.6g}")


