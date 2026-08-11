"""LQR 状态构造与数学辅助函数。"""

from __future__ import annotations

import math

import numpy as np

from ..model.kinematics import (
    compute_virtual_leg_state as _compute_virtual_leg_state,
    wheel_center_positions as _wheel_center_positions,
    wheel_radius as _wheel_radius,
    wheel_center_speeds as _wheel_center_speeds,
)
from ..core.types import LqrState, SimulatedOdometry


def base_pitch_from_qpos(qpos: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in qpos[3:7])
    # Use atan2 so the body angle remains observable after crossing +/-90 deg.
    # asin would clamp a falling body to +/-pi/2 and hide the actual attitude.
    sin_pitch = 2.0 * (w * y - z * x)
    cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    return math.atan2(sin_pitch, cos_pitch)


def lqr_state_from_measurements(
    left_theta: float,
    left_theta_rate: float,
    right_theta: float,
    right_theta_rate: float,
    wheel_positions: tuple[float, float],
    wheel_speeds: tuple[float, float],
    wheel_radius: float,
    left_length: float,
    left_length_rate: float,
    right_length: float,
    right_length_rate: float,
    qpos: np.ndarray,
    qvel: np.ndarray,
    x_reference: float,
    x_source: str,
    x_reference_rate: float = 0.0,
    odometry: SimulatedOdometry | None = None,
) -> LqrState:
    if x_source == "wheel":
        if odometry is None:
            left_wheel_x, right_wheel_x = wheel_positions
            left_wheel_v, right_wheel_v = wheel_speeds
            x_value = 0.5 * (left_wheel_x + right_wheel_x) - x_reference
            x_rate = 0.5 * (left_wheel_v + right_wheel_v) - x_reference_rate
        else:
            x_value = odometry.position - x_reference
            x_rate = odometry.speed - x_reference_rate
    elif x_source == "base":
        x_value = float(qpos[0]) - x_reference
        x_rate = float(qvel[0]) - x_reference_rate
    else:
        raise RuntimeError(f"unknown LQR x source: {x_source}")
    pitch = base_pitch_from_qpos(qpos)
    pitch_rate = float(qvel[4]) if len(qvel) > 4 else 0.0
    return LqrState(
        theta=0.5 * (left_theta + right_theta),
        theta_rate=0.5 * (left_theta_rate + right_theta_rate),
        x=x_value,
        x_rate=x_rate,
        pitch=pitch,
        pitch_rate=pitch_rate,
        length=0.5 * (left_length + right_length),
        length_rate=0.5 * (left_length_rate + right_length_rate),
    )


def lqr_state_vector(state: LqrState) -> np.ndarray:
    return np.array(
        [
            state.theta,
            state.theta_rate,
            state.x,
            state.x_rate,
            state.pitch,
            state.pitch_rate,
        ],
        dtype=float,
    )


def compute_lqr_state(
    mujoco, model, data, x_reference: float, x_source: str = "wheel", x_reference_rate: float = 0.0,
    odometry: SimulatedOdometry | None = None,
) -> LqrState:
    left = _compute_virtual_leg_state(mujoco, model, data, "left")
    right = _compute_virtual_leg_state(mujoco, model, data, "right")
    return lqr_state_from_measurements(
        left.theta,
        left.theta_rate,
        right.theta,
        right.theta_rate,
        _wheel_center_positions(mujoco, model, data),
        _wheel_center_speeds(mujoco, model, data),
        _wheel_radius(mujoco, model),
        left.length,
        left.length_rate,
        right.length,
        right.length_rate,
        data.qpos,
        data.qvel,
        x_reference,
        x_source,
        x_reference_rate,
        odometry,
    )


def set_base_pitch(qpos: np.ndarray, pitch: float) -> None:
    half_pitch = 0.5 * pitch
    qpos[3:7] = np.array([math.cos(half_pitch), 0.0, math.sin(half_pitch), 0.0], dtype=float)


def solve_discrete_are(
    a_matrix: np.ndarray,
    b_matrix: np.ndarray,
    q_matrix: np.ndarray,
    r_matrix: np.ndarray,
    max_iterations: int = 10000,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, int]:
    p_matrix = q_matrix.copy()
    for iteration in range(1, max_iterations + 1):
        middle = r_matrix + b_matrix.T @ p_matrix @ b_matrix
        gain = np.linalg.solve(middle, b_matrix.T @ p_matrix @ a_matrix)
        next_p = a_matrix.T @ p_matrix @ a_matrix - a_matrix.T @ p_matrix @ b_matrix @ gain + q_matrix
        next_p = 0.5 * (next_p + next_p.T)
        if np.max(np.abs(next_p - p_matrix)) < tolerance:
            return next_p, iteration
        p_matrix = next_p
    return p_matrix, max_iterations


def solve_continuous_are_hamiltonian(
    a_matrix: np.ndarray,
    b_matrix: np.ndarray,
    q_matrix: np.ndarray,
    r_matrix: np.ndarray,
) -> np.ndarray:
    state_count = a_matrix.shape[0]
    r_inv = np.linalg.inv(r_matrix)
    hamiltonian = np.block(
        [
            [a_matrix, -(b_matrix @ r_inv @ b_matrix.T)],
            [-q_matrix, -a_matrix.T],
        ]
    )
    eigenvalues, eigenvectors = np.linalg.eig(hamiltonian)
    stable_indices = np.flatnonzero(np.real(eigenvalues) < -1e-9)
    if len(stable_indices) != state_count:
        raise RuntimeError(
            f"continuous CARE has {len(stable_indices)} stable Hamiltonian modes; expected {state_count}"
        )
    stable_vectors = eigenvectors[:, stable_indices]
    upper = stable_vectors[:state_count, :]
    lower = stable_vectors[state_count:, :]
    p_matrix = np.real(lower @ np.linalg.inv(upper))
    return 0.5 * (p_matrix + p_matrix.T)


def apply_theta_pitch_feedforward(
    target: tuple[float, float],
    pitch: float,
    gain: float,
) -> tuple[float, float]:
    if gain == 0.0:
        return target
    return target[0], target[1] - gain * pitch


def smooth_deadzone_gate(value: float, threshold: float) -> float:
    if threshold <= 0.0:
        return 0.0
    ratio = max(0.0, min(1.0, value / threshold))
    return 1.0 - ratio * ratio * (3.0 - 2.0 * ratio)


def lowpass_value(raw_value: float, previous_value: float, cutoff_hz: float, dt: float) -> float:
    if cutoff_hz <= 0.0:
        return raw_value
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    alpha = dt / (tau + dt)
    return previous_value + alpha * (raw_value - previous_value)


def lqr_middle_control_from_state(
    state: LqrState,
    gain_scale: float,
    lqr_k: np.ndarray,
    lqr_x0: np.ndarray,
    lqr_u0: np.ndarray,
    wheel_sign: float,
    pitch_sign: float,
    lqr_t_limit: float,
    lqr_tp_limit: float,
    lqr_x_outer_kp: float,
    lqr_x_outer_max_v: float,
) -> tuple[float, float, float, float]:
    x_velocity_reference = 0.0
    inner_x_error = state.x
    inner_x_rate_error = state.x_rate
    if lqr_x_outer_kp > 0.0:
        position_error = state.x - float(lqr_x0[2])
        x_velocity_reference = float(
            np.clip(-lqr_x_outer_kp * position_error, -lqr_x_outer_max_v, lqr_x_outer_max_v)
        )
        inner_x_error = float(lqr_x0[2])
        inner_x_rate_error = state.x_rate - x_velocity_reference + float(lqr_x0[3])
    state_vector = np.array(
        [
            state.theta,
            state.theta_rate,
            inner_x_error,
            inner_x_rate_error,
            state.pitch,
            state.pitch_rate,
        ],
        dtype=float,
    )
    control = lqr_u0 - gain_scale * (lqr_k @ (state_vector - lqr_x0))
    wheel_torque = wheel_sign * float(np.clip(control[0], -lqr_t_limit, lqr_t_limit))
    virtual_pitch_torque = pitch_sign * float(np.clip(control[1], -lqr_tp_limit, lqr_tp_limit))
    length_force_delta = 0.0
    return wheel_torque, virtual_pitch_torque, length_force_delta, x_velocity_reference
