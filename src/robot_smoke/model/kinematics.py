"""从 MuJoCo 状态读取运行时运动学测量量。"""

from __future__ import annotations

import math

import numpy as np

from ..core.mujoco_utils import body_pos_vel as _body_pos_vel
from ..core.mujoco_utils import body_linear_velocity as _body_linear_velocity
from ..core.mujoco_utils import id_by_name as _id_by_name
from ..core.mujoco_utils import site_pos as _site_pos
from ..core.mujoco_utils import site_pos_vel as _site_pos_vel
from ..core.types import SimulatedOdometry, VirtualLegState


def base_forward_heading(mujoco, model, data) -> np.ndarray:
    """Return the base local +X direction projected onto the world XY plane."""
    base_id = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "base")
    forward = data.xmat[base_id].reshape(3, 3)[:, 0].copy()
    forward[2] = 0.0
    norm = float(np.linalg.norm(forward))
    if norm <= 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return forward / norm


def lower_loop_error(mujoco, model, data, side: str) -> float:
    rear_hub = _site_pos(mujoco, model, data, f"{side}_rear_hub_site")
    carrier = _site_pos(mujoco, model, data, f"{side}_carrier_site")
    return float(np.linalg.norm(rear_hub - carrier))


def compute_virtual_leg_state(mujoco, model, data, side: str) -> VirtualLegState:
    front_hip_pos, front_hip_vel = _body_pos_vel(mujoco, model, data, f"{side}_front_upper")
    rear_hip_pos, rear_hip_vel = _body_pos_vel(mujoco, model, data, f"{side}_rear_upper")
    wheel_pos, wheel_vel = _site_pos_vel(mujoco, model, data, f"{side}_carrier_site")
    hip_pos = 0.5 * (front_hip_pos + rear_hip_pos)
    hip_vel = 0.5 * (front_hip_vel + rear_hip_vel)

    rod = wheel_pos - hip_pos
    rod_vel = wheel_vel - hip_vel
    heading = base_forward_heading(mujoco, model, data)
    rx = float(np.dot(rod, heading))
    rz = float(rod[2])
    yaw_rate = float(data.qvel[5]) if model.nv > 5 else 0.0
    heading_rate = yaw_rate * np.array([-heading[1], heading[0], 0.0], dtype=float)
    vx = float(np.dot(rod_vel, heading) + np.dot(rod, heading_rate))
    vz = float(rod_vel[2])
    length = max(math.hypot(rx, rz), 1e-9)
    length_rate = (rx * vx + rz * vz) / length
    theta = math.atan2(rx, -rz)
    theta_rate = (-rz * vx + rx * vz) / (length * length)
    return VirtualLegState(
        name=side,
        length=length,
        length_rate=length_rate,
        theta=theta,
        theta_rate=theta_rate,
    )


def compute_virtual_leg_shape(mujoco, model, data, side: str) -> np.ndarray:
    state = compute_virtual_leg_state(mujoco, model, data, side)
    return np.array([state.length, state.theta], dtype=float)


def wheel_speeds(mujoco, model, data) -> tuple[float, float]:
    left_joint = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")
    right_joint = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")
    left_dof = int(model.jnt_dofadr[left_joint])
    right_dof = int(model.jnt_dofadr[right_joint])
    return float(data.qvel[left_dof]), float(data.qvel[right_dof])


def wheel_center_positions(mujoco, model, data) -> tuple[float, float]:
    """World-frame x positions of the two driven-wheel centers."""
    left_id = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "left_wheel")
    right_id = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "right_wheel")
    return float(data.xpos[left_id, 0]), float(data.xpos[right_id, 0])


def wheel_center_heights(mujoco, model, data) -> tuple[float, float]:
    """World-frame z positions of the two driven-wheel centers."""
    left_id = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "left_wheel")
    right_id = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "right_wheel")
    return float(data.xpos[left_id, 2]), float(data.xpos[right_id, 2])


def wheel_center_speeds(mujoco, model, data) -> tuple[float, float]:
    """World-frame x velocities of the driven-wheel centers.

    This is the inertial wheel measurement used by the balance state.  It is
    obtained from each wheel body's translational Jacobian, not base qvel and
    not the wheel hinge's relative qvel.
    """
    speeds = []
    for body_name in ("left_wheel", "right_wheel"):
        body_id = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        speeds.append(float((jacp @ data.qvel)[0]))
    return tuple(speeds)


def wheel_center_forward_speed(mujoco, model, data) -> float:
    """World-truth wheel-center speed expressed along the current vehicle heading."""
    heading = base_forward_heading(mujoco, model, data)
    velocity = np.zeros(3, dtype=float)
    for body_name in ("left_wheel", "right_wheel"):
        body_id = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        velocity += _body_linear_velocity(mujoco, model, data, body_id)
    return float(np.dot(0.5 * velocity, heading))


def update_simulated_odometry(mujoco, model, data, odometry: SimulatedOdometry) -> SimulatedOdometry:
    """Integrate a heading-frame odometer from world-frame wheel-center truth."""
    speed = wheel_center_forward_speed(mujoco, model, data)
    time_s = float(data.time)
    if odometry.previous_time is not None:
        dt = max(0.0, time_s - odometry.previous_time)
        odometry.position += 0.5 * (odometry.speed + speed) * dt
    odometry.speed = speed
    odometry.previous_time = time_s
    return odometry


def wheel_positions(mujoco, model, data) -> tuple[float, float]:
    left_joint = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")
    right_joint = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")
    left_qpos = int(model.jnt_qposadr[left_joint])
    right_qpos = int(model.jnt_qposadr[right_joint])
    return float(data.qpos[left_qpos]), float(data.qpos[right_qpos])


def wheel_radius(mujoco, model) -> float:
    wheel_geom = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_geom")
    return float(model.geom_size[wheel_geom, 0])
