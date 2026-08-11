"""虚拟腿 IK 目标辅助函数。"""

from __future__ import annotations

import math

import numpy as np

from ..model.actuators import leg_drive_actuator_ids as _leg_drive_actuator_ids
from ..model.actuators import leg_drive_joint_ids as _leg_drive_joint_ids
from ..model.fivebar import compute_leg_branch_metrics as _compute_leg_branch_metrics
from ..model.kinematics import compute_virtual_leg_state as _compute_virtual_leg_state
from ..core.mujoco_utils import assert_finite as _assert_finite
from ..core.mujoco_utils import lock_base_to_qpos as _lock_base_to_qpos
from ..core.types import LegBranchMetrics, VirtualLegState, VmcDiagnostics, VmcSideMemory
from ..control.vmc import (
    _drive_joint_position_ctrl,
    _drive_virtual_rod_vmc_ctrl,
    _leg_shape_jacobian,
    _wrapped_angle_delta,
)


def _virtual_rod_ik_targets(
    mujoco,
    model,
    data,
    side: str,
    current_length: float,
    current_theta: float,
    target_length: float,
    target_theta: float,
) -> tuple[float, float]:
    shape_jacobian = _leg_shape_jacobian(mujoco, model, data, side)
    shape_error = np.array(
        [
            target_length - current_length,
            _wrapped_angle_delta(target_theta, current_theta),
        ],
        dtype=float,
    )
    joint_delta = np.linalg.pinv(shape_jacobian) @ shape_error
    front_joint_id, rear_joint_id = _leg_drive_joint_ids(mujoco, model, side)
    front_qpos_id = int(model.jnt_qposadr[front_joint_id])
    rear_qpos_id = int(model.jnt_qposadr[rear_joint_id])
    front_target = float(data.qpos[front_qpos_id] + joint_delta[0])
    rear_target = float(data.qpos[rear_qpos_id] + joint_delta[1])
    return front_target, rear_target


def _settle_ik_candidate(
    mujoco,
    model,
    source_data,
    side: str,
    front_target: float,
    rear_target: float,
    steps: int = 80,
) -> tuple[VirtualLegState, LegBranchMetrics]:
    base_qpos = source_data.qpos[:7].copy()
    data = mujoco.MjData(model)
    data.qpos[:] = source_data.qpos
    data.qvel[:] = 0.0
    data.act[:] = source_data.act
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(steps):
        _lock_base_to_qpos(mujoco, model, data, base_qpos)
        data.ctrl[:] = 0.0
        _drive_joint_position_ctrl(mujoco, model, data, side, front_target, rear_target, 40.0, 1.4)
        mujoco.mj_step(model, data)
        _assert_finite("candidate qpos", data.qpos)
        _assert_finite("candidate qvel", data.qvel)
    _lock_base_to_qpos(mujoco, model, data, base_qpos)
    return (
        _compute_virtual_leg_state(mujoco, model, data, side),
        _compute_leg_branch_metrics(mujoco, model, data, side),
    )


def _equilibrium_global_ik_targets(
    mujoco,
    model,
    data,
    side: str,
    target: tuple[float, float],
    ik_search_samples: int,
) -> tuple[float, float]:
    front_joint_id, rear_joint_id = _leg_drive_joint_ids(mujoco, model, side)
    reset_front = float(model.qpos0[int(model.jnt_qposadr[front_joint_id])])
    reset_rear = float(model.qpos0[int(model.jnt_qposadr[rear_joint_id])])
    low_front, high_front = (float(value) for value in model.jnt_range[front_joint_id])
    low_rear, high_rear = (float(value) for value in model.jnt_range[rear_joint_id])
    coarse_samples = min(25, max(15, 2 * int(ik_search_samples) + 1))

    def score_candidate(front_target: float, rear_target: float, settle_steps: int) -> tuple[float, VirtualLegState, LegBranchMetrics]:
        state, branch = _settle_ik_candidate(
            mujoco,
            model,
            data,
            side,
            front_target,
            rear_target,
            steps=settle_steps,
        )
        length_error = target[0] - state.length
        theta_error = _wrapped_angle_delta(target[1], state.theta)
        target_score = (length_error / 0.008) ** 2 + (theta_error / 0.025) ** 2
        branch_score = (branch.violation / 0.001) ** 2 * 200.0
        reset_score = 0.002 * ((front_target - reset_front) ** 2 + (rear_target - reset_rear) ** 2)
        return target_score + branch_score + reset_score, state, branch

    best_score = math.inf
    best_targets = (reset_front, reset_rear)
    front_values = np.linspace(low_front, high_front, coarse_samples)
    rear_values = np.linspace(low_rear, high_rear, coarse_samples)
    for front_target in front_values:
        for rear_target in rear_values:
            score, _, _ = score_candidate(float(front_target), float(rear_target), 35)
            if score < best_score:
                best_score = score
                best_targets = (float(front_target), float(rear_target))

    front_step = (high_front - low_front) / max(1, coarse_samples - 1)
    rear_step = (high_rear - low_rear) / max(1, coarse_samples - 1)
    refine_front = np.linspace(
        max(low_front, best_targets[0] - front_step),
        min(high_front, best_targets[0] + front_step),
        9,
    )
    refine_rear = np.linspace(
        max(low_rear, best_targets[1] - rear_step),
        min(high_rear, best_targets[1] + rear_step),
        9,
    )
    for front_target in refine_front:
        for rear_target in refine_rear:
            score, _, _ = score_candidate(float(front_target), float(rear_target), 55)
            if score < best_score:
                best_score = score
                best_targets = (float(front_target), float(rear_target))
    return best_targets


def _branch_aware_ik_targets(
    mujoco,
    model,
    data,
    side: str,
    reset_state: VirtualLegState,
    target: tuple[float, float],
    leg_branch: str,
    ik_search_radius: float,
    ik_search_samples: int,
) -> tuple[float, float]:
    current_state = _compute_virtual_leg_state(mujoco, model, data, side)
    base_front, base_rear = _virtual_rod_ik_targets(
        mujoco,
        model,
        data,
        side,
        current_state.length,
        current_state.theta,
        target[0],
        target[1],
    )
    if leg_branch == "off":
        return base_front, base_rear

    ik_search_samples = max(1, ik_search_samples)
    if ik_search_samples == 1 or ik_search_radius <= 0.0:
        offsets = np.array([0.0], dtype=float)
    else:
        offsets = np.linspace(-ik_search_radius, ik_search_radius, ik_search_samples)

    front_joint_id, rear_joint_id = _leg_drive_joint_ids(mujoco, model, side)
    front_qpos_id = int(model.jnt_qposadr[front_joint_id])
    rear_qpos_id = int(model.jnt_qposadr[rear_joint_id])
    current_front = float(data.qpos[front_qpos_id])
    current_rear = float(data.qpos[rear_qpos_id])
    reset_front = float(model.qpos0[front_qpos_id])
    reset_rear = float(model.qpos0[rear_qpos_id])
    low_front, high_front = model.jnt_range[front_joint_id]
    low_rear, high_rear = model.jnt_range[rear_joint_id]

    best_score = math.inf
    best_targets = (base_front, base_rear)
    best_violation = math.inf
    for front_offset in offsets:
        for rear_offset in offsets:
            front_target = float(np.clip(base_front + front_offset, low_front, high_front))
            rear_target = float(np.clip(base_rear + rear_offset, low_rear, high_rear))
            state, branch = _settle_ik_candidate(
                mujoco,
                model,
                data,
                side,
                front_target,
                rear_target,
            )
            length_error = target[0] - state.length
            theta_error = target[1] - state.theta
            target_score = (length_error / 0.02) ** 2 + (theta_error / 0.05) ** 2
            branch_score = (branch.violation / 0.001) ** 2 * 100.0
            reset_score = 0.04 * ((front_target - reset_front) ** 2 + (rear_target - reset_rear) ** 2)
            jump_score = 0.02 * ((front_target - current_front) ** 2 + (rear_target - current_rear) ** 2)
            score = target_score + branch_score + reset_score + jump_score
            if branch.violation < 1e-9 and best_violation > 1e-9:
                best_score = score
                best_targets = (front_target, rear_target)
                best_violation = branch.violation
            elif (branch.violation <= best_violation + 1e-12 and score < best_score) or score < best_score:
                best_score = score
                best_targets = (front_target, rear_target)
                best_violation = branch.violation
    return best_targets


def _virtual_rod_ik_ctrl(
    mujoco,
    model,
    data,
    left_target: tuple[float, float],
    right_target: tuple[float, float],
    virtual_rod_control: str,
    leg_branch: str,
    ik_search_radius: float,
    ik_search_samples: int,
    kp: float,
    kd: float,
    theta_kp: float,
    theta_kd: float,
    joint_kd: float,
    ik_target_cache: dict[tuple[object, ...], tuple[float, float]] | None = None,
    theta_force_offset: float | tuple[float, float] = 0.0,
    length_force_ff: float | tuple[float, float] = 0.0,
    length_ki: float = 0.0,
    length_integral_limit: float = 0.3,
    length_force_rate_limit: float = 0.0,
    branch_guard_enabled: bool = True,
    vmc_memory: dict[str, VmcSideMemory] | None = None,
    vmc_diagnostics: dict[str, VmcDiagnostics] | None = None,
) -> tuple[float, bool, float, float]:
    data.ctrl[:] = 0.0
    max_length_error = 0.0
    max_theta_error = 0.0
    saturated = False
    if isinstance(theta_force_offset, tuple):
        side_theta_offsets = dict(zip(("left", "right"), theta_force_offset))
    else:
        side_theta_offsets = {"left": theta_force_offset, "right": theta_force_offset}
    if isinstance(length_force_ff, tuple):
        side_length_force_ff = dict(zip(("left", "right"), length_force_ff))
    else:
        side_length_force_ff = {"left": length_force_ff, "right": length_force_ff}
    for side, target in (("left", left_target), ("right", right_target)):
        if virtual_rod_control == "vmc":
            # VMC applies task-space force through J^T. Its IK output is only
            # a normal-branch recovery posture for the branch guard, so it
            # must not be recomputed whenever roll integration changes L_d.
            cache_key: tuple[object, ...] = (
                side,
                "vmc_branch_safe_pose",
                leg_branch,
                round(ik_search_radius, 4),
                ik_search_samples,
            )
        else:
            cache_key = (
                side,
                round(target[0], 4),
                round(target[1] / 0.01) * 0.01,
                leg_branch,
                round(ik_search_radius, 4),
                ik_search_samples,
            )
        if ik_target_cache is not None and cache_key in ik_target_cache:
            front_target, rear_target = ik_target_cache[cache_key]
        else:
            reset_data = mujoco.MjData(model)
            mujoco.mj_resetData(model, reset_data)
            mujoco.mj_forward(model, reset_data)
            reset_state = _compute_virtual_leg_state(mujoco, model, reset_data, side)
            front_target, rear_target = _branch_aware_ik_targets(
                mujoco,
                model,
                data,
                side,
                reset_state,
                target,
                leg_branch,
                ik_search_radius,
                ik_search_samples,
            )
            if ik_target_cache is not None:
                ik_target_cache[cache_key] = (front_target, rear_target)
        if virtual_rod_control == "position":
            _drive_joint_position_ctrl(mujoco, model, data, side, front_target, rear_target, kp, kd)
            if vmc_diagnostics is not None:
                position_state = _compute_virtual_leg_state(mujoco, model, data, side)
                vmc_diagnostics[side] = VmcDiagnostics(length_error=target[0] - position_state.length)
        elif virtual_rod_control == "off":
            # Wheel-only balance diagnostic: keep this leg's actuator controls
            # at zero. The upright pose is provided only by initialization.
            if vmc_diagnostics is not None:
                passive_state = _compute_virtual_leg_state(mujoco, model, data, side)
                vmc_diagnostics[side] = VmcDiagnostics(
                    length_error=target[0] - passive_state.length,
                    length_rate=passive_state.length_rate,
                )
        elif virtual_rod_control == "vmc":
            side_memory = None
            if vmc_memory is not None:
                side_memory = vmc_memory.setdefault(side, VmcSideMemory())
            side_saturated, diagnostics = _drive_virtual_rod_vmc_ctrl(
                mujoco,
                model,
                data,
                side,
                target,
                kp,
                kd,
                theta_kp,
                theta_kd,
                kp,
                max(joint_kd, kd),
                (front_target, rear_target),
                side_theta_offsets[side],
                length_force_ff=side_length_force_ff[side],
                length_ki=length_ki,
                length_integral_limit=length_integral_limit,
                length_force_rate_limit=length_force_rate_limit,
                branch_guard_enabled=branch_guard_enabled,
                memory=side_memory,
            )
            saturated = side_saturated or saturated
            if vmc_diagnostics is not None:
                vmc_diagnostics[side] = diagnostics
        else:
            raise RuntimeError(f"unknown virtual rod control mode: {virtual_rod_control}")
        for actuator_id in _leg_drive_actuator_ids(mujoco, model, side):
            low, high = model.actuator_ctrlrange[actuator_id]
            if abs(float(data.ctrl[actuator_id]) - low) < 1e-12:
                saturated = True
            if abs(float(data.ctrl[actuator_id]) - high) < 1e-12:
                saturated = True
        state = _compute_virtual_leg_state(mujoco, model, data, side)
        max_length_error = max(max_length_error, abs(target[0] - state.length))
        max_theta_error = max(max_theta_error, abs(target[1] - state.theta))
    return float(np.max(np.abs(data.ctrl))), saturated, max_length_error, max_theta_error
