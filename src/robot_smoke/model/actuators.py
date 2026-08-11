"""MuJoCo smoke 脚本使用的执行器与力矩映射辅助函数。"""

from __future__ import annotations

import numpy as np

from ..core.mujoco_utils import id_by_name as _id_by_name
from ..core.mujoco_utils import name as _name


def leg_drive_joint_ids(mujoco, model, side: str) -> tuple[int, int]:
    front = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_front_drive")
    rear = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_rear_drive")
    return front, rear


def leg_drive_actuator_ids(mujoco, model, side: str) -> tuple[int, int]:
    front = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_front_motor")
    rear = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_rear_motor")
    return front, rear


def wheel_actuator_ids(mujoco, model) -> tuple[int, int]:
    left = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_motor")
    right = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_motor")
    return left, right


def apply_joint_torque_ctrl(
    model,
    data,
    actuator_ids: tuple[int, int],
    joint_tau: np.ndarray,
) -> bool:
    saturated = False
    for actuator_id, tau in zip(actuator_ids, joint_tau):
        gear = float(model.actuator_gear[actuator_id, 0])
        low, high = model.actuator_ctrlrange[actuator_id]
        raw_ctrl = float(tau / gear)
        clipped = float(np.clip(raw_ctrl, low, high))
        if abs(raw_ctrl - clipped) > 1e-12:
            saturated = True
        data.ctrl[actuator_id] = clipped
    return saturated


def apply_additive_wheel_torque_ctrl(
    mujoco,
    model,
    data,
    wheel_torque: float,
    wheel_ctrl_deadzone: float = 0.0,
) -> bool:
    saturated = False
    per_wheel_tau = 0.5 * wheel_torque
    for actuator_id in wheel_actuator_ids(mujoco, model):
        gear = float(model.actuator_gear[actuator_id, 0])
        low, high = model.actuator_ctrlrange[actuator_id]
        raw_ctrl = float(per_wheel_tau / gear)
        clipped = float(np.clip(raw_ctrl, low, high))
        if abs(raw_ctrl - clipped) > 1e-12:
            saturated = True
        if abs(clipped) < wheel_ctrl_deadzone:
            clipped = 0.0
        data.ctrl[actuator_id] = clipped
    return saturated


def apply_wheel_torque_pair_ctrl(
    mujoco,
    model,
    data,
    left_torque: float,
    right_torque: float,
    wheel_ctrl_deadzone: float = 0.0,
) -> bool:
    """Apply independent left/right wheel torques for yaw control."""
    saturated = False
    for actuator_id, torque in zip(wheel_actuator_ids(mujoco, model), (left_torque, right_torque)):
        gear = float(model.actuator_gear[actuator_id, 0])
        low, high = model.actuator_ctrlrange[actuator_id]
        raw_ctrl = float(torque / gear)
        clipped = float(np.clip(raw_ctrl, low, high))
        if abs(raw_ctrl - clipped) > 1e-12:
            saturated = True
        data.ctrl[actuator_id] = 0.0 if abs(clipped) < wheel_ctrl_deadzone else clipped
    return saturated


def actuator_ctrl_map(mujoco, model, data) -> dict[str, float]:
    return {
        _name(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id): float(data.ctrl[actuator_id])
        for actuator_id in range(model.nu)
    }
