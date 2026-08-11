"""虚拟模型控制与腿部任务空间映射。"""

from __future__ import annotations

import math

import numpy as np

from ..model.actuators import (
    apply_joint_torque_ctrl as _apply_joint_torque_ctrl,
    leg_drive_actuator_ids as _leg_drive_actuator_ids,
    leg_drive_joint_ids as _leg_drive_joint_ids,
)
from ..model.fivebar import (
    analytic_fivebar_kinematics_from_q as _analytic_fivebar_kinematics_from_q,
    compute_leg_branch_metrics as _compute_leg_branch_metrics,
    drive_qpos_for_side as _drive_qpos_for_side,
    leg_branch_guard_error as _leg_branch_guard_error,
)
from ..model.kinematics import (
    compute_virtual_leg_shape as _compute_virtual_leg_shape,
    compute_virtual_leg_state as _compute_virtual_leg_state,
)
from ..core.mujoco_utils import assert_finite as _assert_finite
from ..core.mujoco_utils import lock_base_to_qpos as _lock_base_to_qpos
from ..core.types import VmcDiagnostics, VmcSideMemory

def _drive_joint_position_ctrl(
    mujoco,
    model,
    data,
    side: str,
    front_target: float,
    rear_target: float,
    kp: float,
    kd: float,
) -> None:
    actuator_ids = _leg_drive_actuator_ids(mujoco, model, side)
    joint_ids = _leg_drive_joint_ids(mujoco, model, side)
    for actuator_id, joint_id, target in zip(actuator_ids, joint_ids, (front_target, rear_target)):
        qpos_id = int(model.jnt_qposadr[joint_id])
        dof_id = int(model.jnt_dofadr[joint_id])
        gear = float(model.actuator_gear[actuator_id, 0])
        raw_tau = kp * (target - float(data.qpos[qpos_id])) - kd * float(data.qvel[dof_id])
        low, high = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = float(np.clip(raw_tau / gear, low, high))


def _wrapped_angle_delta(target: float, reference: float) -> float:
    return math.atan2(math.sin(target - reference), math.cos(target - reference))


def _settled_leg_shape_for_drive_targets(
    mujoco,
    model,
    source_data,
    side: str,
    front_target: float,
    rear_target: float,
    steps: int = 60,
) -> np.ndarray:
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
        _assert_finite("settled_jacobian qpos", data.qpos)
        _assert_finite("settled_jacobian qvel", data.qvel)
    _lock_base_to_qpos(mujoco, model, data, base_qpos)
    return _compute_virtual_leg_shape(mujoco, model, data, side)


def _settled_numeric_leg_shape_jacobian(mujoco, model, data, side: str, epsilon: float = 1e-4) -> np.ndarray:
    front_joint_id, rear_joint_id = _leg_drive_joint_ids(mujoco, model, side)
    front_qpos_id = int(model.jnt_qposadr[front_joint_id])
    rear_qpos_id = int(model.jnt_qposadr[rear_joint_id])
    current_front = float(data.qpos[front_qpos_id])
    current_rear = float(data.qpos[rear_qpos_id])
    jacobian = np.zeros((2, 2), dtype=float)
    for column, (front_offset, rear_offset) in enumerate(((epsilon, 0.0), (0.0, epsilon))):
        plus_shape = _settled_leg_shape_for_drive_targets(
            mujoco,
            model,
            data,
            side,
            current_front + front_offset,
            current_rear + rear_offset,
        )
        minus_shape = _settled_leg_shape_for_drive_targets(
            mujoco,
            model,
            data,
            side,
            current_front - front_offset,
            current_rear - rear_offset,
        )
        jacobian[0, column] = (plus_shape[0] - minus_shape[0]) / (2.0 * epsilon)
        theta_delta = _wrapped_angle_delta(float(plus_shape[1]), float(minus_shape[1]))
        jacobian[1, column] = theta_delta / (2.0 * epsilon)
    if float(abs(np.linalg.det(jacobian))) < 1e-8:
        raise RuntimeError(f"near-singular leg shape Jacobian on {side}: {jacobian}")
    return jacobian


def _leg_shape_jacobian(mujoco, model, data, side: str) -> np.ndarray:
    q_front, q_rear = _drive_qpos_for_side(mujoco, model, data, side)
    analytic = _analytic_fivebar_kinematics_from_q(mujoco, model, side, q_front, q_rear)
    if analytic is not None and abs(float(np.linalg.det(analytic.jacobian))) >= 1e-8:
        return analytic.jacobian
    return _settled_numeric_leg_shape_jacobian(mujoco, model, data, side)


def _runtime_leg_shape_jacobian(
    mujoco,
    model,
    data,
    side: str,
    memory: VmcSideMemory | None,
    refresh_steps: int = 25,
) -> np.ndarray:
    if memory is None:
        return _leg_shape_jacobian(mujoco, model, data, side)
    if memory.shape_jacobian is None or memory.shape_jacobian_age >= refresh_steps:
        memory.shape_jacobian = _leg_shape_jacobian(mujoco, model, data, side)
        memory.shape_jacobian_age = 0
    else:
        memory.shape_jacobian_age += 1
    return memory.shape_jacobian


def _drive_virtual_rod_vmc_ctrl(
    mujoco,
    model,
    data,
    side: str,
    target: tuple[float, float],
    length_kp: float,
    length_kd: float,
    theta_kp: float,
    theta_kd: float,
    branch_kp: float,
    branch_kd: float,
    safe_joint_targets: tuple[float, float],
    theta_force_offset: float = 0.0,
    length_force_limit: float = 250.0,
    theta_force_limit: float = 80.0,
    joint_tau_limit: float = 45.0,
    length_force_ff: float = 0.0,
    length_ki: float = 0.0,
    length_integral_limit: float = 0.3,
    length_force_rate_limit: float = 0.0,
    branch_guard_enabled: bool = True,
    memory: VmcSideMemory | None = None,
) -> tuple[bool, VmcDiagnostics]:
    state = _compute_virtual_leg_state(mujoco, model, data, side)
    if memory is None:
        memory = VmcSideMemory()
    shape_jacobian = _runtime_leg_shape_jacobian(mujoco, model, data, side, memory)
    length_error = target[0] - state.length
    theta_error = target[1] - state.theta
    theta_force_raw = theta_kp * theta_error - theta_kd * state.theta_rate + theta_force_offset
    dt = float(model.opt.timestep)
    if length_ki > 0.0 and length_integral_limit > 0.0:
        memory.length_integral = float(
            np.clip(
                memory.length_integral + length_error * dt,
                -length_integral_limit,
                length_integral_limit,
            )
        )
    else:
        memory.length_integral = 0.0
    length_force_raw = (
        length_force_ff
        + length_kp * length_error
        + length_ki * memory.length_integral
        - length_kd * state.length_rate
    )
    length_force = float(np.clip(length_force_raw, -length_force_limit, length_force_limit))
    if length_force_rate_limit > 0.0:
        max_delta = length_force_rate_limit * dt
        length_force = float(
            np.clip(
                length_force,
                memory.previous_length_force - max_delta,
                memory.previous_length_force + max_delta,
            )
        )
    memory.previous_length_force = length_force
    branch = _compute_leg_branch_metrics(mujoco, model, data, side)
    branch_guard_error = _leg_branch_guard_error(branch)
    tau_support = shape_jacobian.T @ np.array([length_force, 0.0], dtype=float)
    tau_guard = np.zeros(2, dtype=float)
    if branch_guard_enabled and branch_guard_error > 0.0:
        joint_ids = _leg_drive_joint_ids(mujoco, model, side)
        qpos_ids = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
        dof_ids = [int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids]
        guard_scale = min(1.0, branch_guard_error / 0.02)
        for index, target_q in enumerate(safe_joint_targets):
            tau_guard[index] += guard_scale * branch_kp * (target_q - float(data.qpos[qpos_ids[index]]))
            tau_guard[index] -= guard_scale * branch_kd * float(data.qvel[dof_ids[index]])

    theta_force_limited = float(np.clip(theta_force_raw, -theta_force_limit, theta_force_limit))
    tau_theta_unit = shape_jacobian.T @ np.array([0.0, 1.0], dtype=float)
    tau_reserved = tau_support + tau_guard
    theta_scale = 1.0
    for reserved, theta_per_unit in zip(tau_reserved, tau_theta_unit):
        required = abs(theta_per_unit * theta_force_limited)
        if required <= 1e-12:
            continue
        available = max(0.0, joint_tau_limit - abs(float(reserved)))
        theta_scale = min(theta_scale, available / required)
    theta_scale = float(np.clip(theta_scale, 0.0, 1.0))
    theta_force = theta_force_limited * theta_scale
    tau_theta = tau_theta_unit * theta_force
    joint_tau = tau_support + tau_guard + tau_theta

    joint_tau_raw = joint_tau.copy()
    joint_tau = np.clip(joint_tau, -joint_tau_limit, joint_tau_limit)
    saturated = _apply_joint_torque_ctrl(model, data, _leg_drive_actuator_ids(mujoco, model, side), joint_tau)
    diagnostics = VmcDiagnostics(
        length_error=length_error,
        length_rate=state.length_rate,
        length_force_raw=float(length_force_raw),
        length_force=length_force,
        length_integral=memory.length_integral,
        theta_force_raw=float(theta_force_raw),
        theta_force=theta_force,
        theta_force_scale=theta_scale,
        branch_guard_error=branch_guard_error,
        support_tau_front=float(tau_support[0]),
        support_tau_rear=float(tau_support[1]),
        guard_tau_front=float(tau_guard[0]),
        guard_tau_rear=float(tau_guard[1]),
        theta_tau_front=float(tau_theta[0]),
        theta_tau_rear=float(tau_theta[1]),
        joint_tau_front_raw=float(joint_tau_raw[0]),
        joint_tau_rear_raw=float(joint_tau_raw[1]),
        joint_tau_front=float(joint_tau[0]),
        joint_tau_rear=float(joint_tau[1]),
    )
    return saturated, diagnostics


def _drive_constant_length_force_ctrl(
    mujoco,
    model,
    data,
    side: str,
    length_force: float,
    joint_tau_limit: float = 45.0,
) -> tuple[bool, VmcDiagnostics]:
    state = _compute_virtual_leg_state(mujoco, model, data, side)
    shape_jacobian = _leg_shape_jacobian(mujoco, model, data, side)
    tau_support = shape_jacobian.T @ np.array([length_force, 0.0], dtype=float)
    joint_tau_raw = tau_support.copy()
    joint_tau = np.clip(joint_tau_raw, -joint_tau_limit, joint_tau_limit)
    saturated = _apply_joint_torque_ctrl(model, data, _leg_drive_actuator_ids(mujoco, model, side), joint_tau)
    diagnostics = VmcDiagnostics(
        length_error=0.0,
        length_rate=state.length_rate,
        length_force_raw=length_force,
        length_force=length_force,
        length_integral=0.0,
        theta_force_raw=0.0,
        theta_force=0.0,
        theta_force_scale=0.0,
        support_tau_front=float(tau_support[0]),
        support_tau_rear=float(tau_support[1]),
        joint_tau_front_raw=float(joint_tau_raw[0]),
        joint_tau_rear_raw=float(joint_tau_raw[1]),
        joint_tau_front=float(joint_tau[0]),
        joint_tau_rear=float(joint_tau[1]),
    )
    return saturated, diagnostics


