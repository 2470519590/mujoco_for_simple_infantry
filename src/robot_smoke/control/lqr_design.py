"""本地 LQR 工作点准备与有限差分设计工具。"""

from __future__ import annotations

import math

import numpy as np

from ..model.actuators import (
    apply_additive_wheel_torque_ctrl as _apply_additive_wheel_torque_ctrl,
    leg_drive_actuator_ids as _leg_drive_actuator_ids,
    wheel_actuator_ids as _wheel_actuator_ids,
)
from ..model.fivebar import compute_leg_branch_metrics as _compute_leg_branch_metrics
from ..model.kinematics import (
    compute_virtual_leg_state as _compute_virtual_leg_state,
    wheel_center_heights as _wheel_center_heights,
    wheel_radius as _wheel_radius,
    wheel_speeds as _wheel_speeds,
)
from ..control.lqr import (
    compute_lqr_state as _compute_lqr_state,
    lqr_middle_control_from_state as _lqr_middle_control_from_state,
    lqr_state_vector as _lqr_state_vector,
    set_base_pitch as _set_base_pitch,
    solve_discrete_are as _solve_discrete_are,
    solve_continuous_are_hamiltonian as _solve_continuous_are_hamiltonian,
)
from ..control.ik import (
    _branch_aware_ik_targets,
    _virtual_rod_ik_ctrl,
)
from ..model.mechanics import _collect_static_operating_point_sample, _contact_normal_force_for_wheel
from ..core.mujoco_utils import assert_finite as _assert_finite
from ..core.mujoco_utils import copy_data as _copy_data
from ..core.mujoco_utils import id_by_name as _id_by_name
from ..core.mujoco_utils import lock_base_to_initial as _lock_base_to_initial
from ..core.types import LqrDesignResult, LqrHistorySample, LqrState, VmcDiagnostics, VmcSideMemory
from ..control.vmc import _drive_joint_position_ctrl, _drive_virtual_rod_vmc_ctrl
from ..control.roll import base_roll_angle as _base_roll_angle

def _prepare_lqr_operating_point(
    mujoco,
    model,
    left_target: tuple[float, float],
    right_target: tuple[float, float],
    leg_branch: str,
    ik_search_radius: float,
    ik_search_samples: int,
) -> object:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    left_reset = _compute_virtual_leg_state(mujoco, model, data, "left")
    right_reset = _compute_virtual_leg_state(mujoco, model, data, "right")
    left_front_target, left_rear_target = _branch_aware_ik_targets(
        mujoco,
        model,
        data,
        "left",
        left_reset,
        left_target,
        leg_branch,
        ik_search_radius,
        ik_search_samples,
    )
    right_front_target, right_rear_target = _branch_aware_ik_targets(
        mujoco,
        model,
        data,
        "right",
        right_reset,
        right_target,
        leg_branch,
        ik_search_radius,
        ik_search_samples,
    )
    for _ in range(120):
        _lock_base_to_initial(mujoco, model, data)
        data.ctrl[:] = 0.0
        _drive_joint_position_ctrl(
            mujoco,
            model,
            data,
            "left",
            left_front_target,
            left_rear_target,
            40.0,
            1.4,
        )
        _drive_joint_position_ctrl(
            mujoco,
            model,
            data,
            "right",
            right_front_target,
            right_rear_target,
            40.0,
            1.4,
        )
        mujoco.mj_step(model, data)
    _lock_base_to_initial(mujoco, model, data)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return data


def _apply_lqr_theta_state_perturbation(
    mujoco,
    model,
    data,
    theta_value: float,
    length_value: float,
    left_target: tuple[float, float],
    right_target: tuple[float, float],
    leg_branch: str,
    ik_search_radius: float,
    ik_search_samples: int,
) -> None:
    base_qpos = data.qpos[:7].copy()
    reference = _copy_data(mujoco, model, data)
    left_reset = _compute_virtual_leg_state(mujoco, model, reference, "left")
    right_reset = _compute_virtual_leg_state(mujoco, model, reference, "right")
    left_front_target, left_rear_target = _branch_aware_ik_targets(
        mujoco,
        model,
        reference,
        "left",
        left_reset,
        (length_value, theta_value),
        leg_branch,
        min(ik_search_radius, 0.12),
        ik_search_samples,
    )
    right_front_target, right_rear_target = _branch_aware_ik_targets(
        mujoco,
        model,
        reference,
        "right",
        right_reset,
        (length_value, theta_value),
        leg_branch,
        min(ik_search_radius, 0.12),
        ik_search_samples,
    )
    for _ in range(80):
        data.qpos[:7] = base_qpos
        data.qvel[:6] = 0.0
        mujoco.mj_forward(model, data)
        data.ctrl[:] = 0.0
        _drive_joint_position_ctrl(
            mujoco,
            model,
            data,
            "left",
            left_front_target,
            left_rear_target,
            40.0,
            1.4,
        )
        _drive_joint_position_ctrl(
            mujoco,
            model,
            data,
            "right",
            right_front_target,
            right_rear_target,
            40.0,
            1.4,
        )
        mujoco.mj_step(model, data)
    data.qpos[:7] = base_qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def _apply_lqr_state_perturbation(
    mujoco,
    model,
    operating_data,
    target_state: np.ndarray,
    left_target: tuple[float, float],
    right_target: tuple[float, float],
    leg_branch: str,
    ik_search_radius: float,
    ik_search_samples: int,
    x_source: str,
) -> object:
    data = _copy_data(mujoco, model, operating_data)
    operating_state = _compute_lqr_state(mujoco, model, operating_data, 0.0, x_source)
    if x_source == "wheel":
        radius = _wheel_radius(mujoco, model)
        x_delta = float(target_state[2] - operating_state.x)
        data.qpos[0] += x_delta
        wheel_q_delta = x_delta / radius
        for side in ("left", "right"):
            wheel_joint = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wheel_joint")
            data.qpos[int(model.jnt_qposadr[wheel_joint])] += wheel_q_delta
    else:
        data.qpos[0] += float(target_state[2] - operating_state.x)
    _set_base_pitch(data.qpos, float(target_state[4]))
    mujoco.mj_forward(model, data)
    current_state = _compute_lqr_state(mujoco, model, data, 0.0, x_source)
    if abs(target_state[0] - current_state.theta) > 1e-12:
        _apply_lqr_theta_state_perturbation(
            mujoco,
            model,
            data,
            float(target_state[0]),
            0.5 * (left_target[0] + right_target[0]),
            left_target,
            right_target,
            leg_branch,
            ik_search_radius,
            ik_search_samples,
        )
    if x_source == "wheel":
        radius = _wheel_radius(mujoco, model)
        current_state = _compute_lqr_state(mujoco, model, data, 0.0, x_source)
        x_delta = float(target_state[2] - current_state.x)
        data.qpos[0] += x_delta
        wheel_q_delta = x_delta / radius
        for side in ("left", "right"):
            wheel_joint = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wheel_joint")
            data.qpos[int(model.jnt_qposadr[wheel_joint])] += wheel_q_delta
    mujoco.mj_forward(model, data)

    # Build a second valid closed-chain pose and differentiate the two qpos
    # arrays. Setting only the drive-joint qvel leaves passive five-bar joint
    # velocities inconsistent with the equality constraints.
    velocity_dt = 0.01
    future = _copy_data(mujoco, model, data)
    future_pitch = float(target_state[4] + velocity_dt * target_state[5])
    _set_base_pitch(future.qpos, future_pitch)
    if x_source == "wheel":
        radius = _wheel_radius(mujoco, model)
        x_delta = velocity_dt * float(target_state[3])
        future.qpos[0] += x_delta
        wheel_q_delta = x_delta / radius
        for side in ("left", "right"):
            wheel_joint = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wheel_joint")
            future.qpos[int(model.jnt_qposadr[wheel_joint])] += wheel_q_delta
    else:
        future.qpos[0] += velocity_dt * float(target_state[3])
    _apply_lqr_theta_state_perturbation(
        mujoco,
        model,
        future,
        float(target_state[0]) + velocity_dt * float(target_state[1]),
        0.5 * (left_target[0] + right_target[0]),
        left_target,
        right_target,
        leg_branch,
        ik_search_radius,
        ik_search_samples,
    )
    if x_source == "wheel":
        radius = _wheel_radius(mujoco, model)
        current_future_state = _compute_lqr_state(mujoco, model, future, 0.0, x_source)
        x_delta = float(target_state[2] + velocity_dt * target_state[3]) - current_future_state.x
        future.qpos[0] += x_delta
        wheel_q_delta = x_delta / radius
        for side in ("left", "right"):
            wheel_joint = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wheel_joint")
            future.qpos[int(model.jnt_qposadr[wheel_joint])] += wheel_q_delta
    data.qvel[:] = 0.0
    mujoco.mj_differentiatePos(model, data.qvel, velocity_dt, data.qpos, future.qpos)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    return data


def _simulate_lqr_dynamics_step(
    mujoco,
    model,
    data,
    left_target: tuple[float, float],
    right_target: tuple[float, float],
    leg_branch: str,
    ik_search_radius: float,
    ik_search_samples: int,
    virtual_rod_length_kp: float,
    virtual_rod_length_kd: float,
    virtual_rod_length_ki: float,
    virtual_rod_length_force_ff: float,
    virtual_rod_length_integral_limit: float,
    virtual_rod_length_force_rate_limit: float,
    virtual_rod_joint_kd: float,
    wheel_torque: float,
    pitch_torque: float,
    design_steps: int,
    x_source: str,
    wheel_sign: float,
    pitch_sign: float,
    leg_control_enabled: bool = True,
    branch_guard_enabled: bool = True,
) -> np.ndarray:
    ik_target_cache: dict[tuple[str, float, float, str, float, int], tuple[float, float]] = {}
    vmc_memory: dict[str, VmcSideMemory] = {}
    for _ in range(max(1, design_steps)):
        _virtual_rod_ik_ctrl(
            mujoco,
            model,
            data,
            left_target,
            right_target,
            "vmc" if leg_control_enabled else "off",
            leg_branch,
            ik_search_radius,
            ik_search_samples,
            virtual_rod_length_kp,
            virtual_rod_length_kd,
            0.0,
            0.0,
            virtual_rod_joint_kd,
            ik_target_cache,
            theta_force_offset=pitch_sign * pitch_torque,
            length_force_ff=virtual_rod_length_force_ff,
            length_ki=virtual_rod_length_ki,
            length_integral_limit=virtual_rod_length_integral_limit,
            length_force_rate_limit=virtual_rod_length_force_rate_limit,
            branch_guard_enabled=branch_guard_enabled,
            vmc_memory=vmc_memory,
        )
        _apply_additive_wheel_torque_ctrl(mujoco, model, data, wheel_sign * wheel_torque)
        mujoco.mj_step(model, data)
    return _lqr_state_vector(_compute_lqr_state(mujoco, model, data, 0.0, x_source))


def _design_lqr_gain(
    mujoco,
    model,
    left_target: tuple[float, float],
    right_target: tuple[float, float],
    leg_branch: str,
    ik_search_radius: float,
    ik_search_samples: int,
    virtual_rod_length_kp: float,
    virtual_rod_length_kd: float,
    virtual_rod_length_ki: float,
    virtual_rod_length_force_ff: float,
    virtual_rod_length_integral_limit: float,
    virtual_rod_length_force_rate_limit: float,
    virtual_rod_joint_kd: float,
    q_diag: np.ndarray,
    r_diag: np.ndarray,
    state_eps: np.ndarray,
    input_eps: np.ndarray,
    design_steps: int,
    x_source: str,
    wheel_sign: float,
    pitch_sign: float,
    leg_control_enabled: bool = True,
    branch_guard_enabled: bool = True,
    operating_data: object | None = None,
    operating_u0: np.ndarray | None = None,
) -> LqrDesignResult:
    if operating_data is None:
        operating_data = _prepare_lqr_operating_point(
            mujoco,
            model,
            left_target,
            right_target,
            leg_branch,
            ik_search_radius,
            ik_search_samples,
        )
    if operating_u0 is None:
        operating_u0 = np.zeros(2, dtype=float)
    operating_state = _compute_lqr_state(mujoco, model, operating_data, 0.0, x_source)
    operating_point_sample = _collect_static_operating_point_sample(
        mujoco,
        model,
        operating_data,
        "lqr_design_operating_point",
        float(operating_u0[0]),
        float(operating_u0[1]),
        x_source=x_source,
    )
    x0 = _lqr_state_vector(operating_state)
    state_count = len(x0)
    if len(q_diag) != state_count or len(state_eps) != state_count:
        raise RuntimeError(
            f"LQR state dimension mismatch: state={state_count}, Q={len(q_diag)}, eps={len(state_eps)}"
        )
    a_matrix = np.zeros((state_count, state_count), dtype=float)
    input_count = len(operating_u0)
    if len(r_diag) != input_count or len(input_eps) != input_count:
        raise RuntimeError(
            f"LQR input dimension mismatch: input={input_count}, R={len(r_diag)}, eps={len(input_eps)}"
        )
    b_matrix = np.zeros((state_count, input_count), dtype=float)
    initial_state_deltas = np.zeros((state_count, state_count), dtype=float)
    next_state_deltas = np.zeros((state_count, state_count), dtype=float)
    for state_index in range(state_count):
        delta = np.zeros(state_count, dtype=float)
        delta[state_index] = float(state_eps[state_index])
        plus_data = _apply_lqr_state_perturbation(
            mujoco,
            model,
            operating_data,
            x0 + delta,
            left_target,
            right_target,
            leg_branch,
            ik_search_radius,
            ik_search_samples,
            x_source,
        )
        minus_data = _apply_lqr_state_perturbation(
            mujoco,
            model,
            operating_data,
            x0 - delta,
            left_target,
            right_target,
            leg_branch,
            ik_search_radius,
            ik_search_samples,
            x_source,
        )
        plus_initial = _lqr_state_vector(_compute_lqr_state(mujoco, model, plus_data, 0.0, x_source))
        minus_initial = _lqr_state_vector(_compute_lqr_state(mujoco, model, minus_data, 0.0, x_source))
        plus = _simulate_lqr_dynamics_step(
            mujoco,
            model,
            plus_data,
            left_target,
            right_target,
            leg_branch,
            ik_search_radius,
            ik_search_samples,
            virtual_rod_length_kp,
            virtual_rod_length_kd,
            virtual_rod_length_ki,
            virtual_rod_length_force_ff,
            virtual_rod_length_integral_limit,
            virtual_rod_length_force_rate_limit,
            virtual_rod_joint_kd,
            float(operating_u0[0]),
            float(operating_u0[1]),
            design_steps,
            x_source,
            wheel_sign,
            pitch_sign,
            leg_control_enabled,
            branch_guard_enabled,
        )
        minus = _simulate_lqr_dynamics_step(
            mujoco,
            model,
            minus_data,
            left_target,
            right_target,
            leg_branch,
            ik_search_radius,
            ik_search_samples,
            virtual_rod_length_kp,
            virtual_rod_length_kd,
            virtual_rod_length_ki,
            virtual_rod_length_force_ff,
            virtual_rod_length_integral_limit,
            virtual_rod_length_force_rate_limit,
            virtual_rod_joint_kd,
            float(operating_u0[0]),
            float(operating_u0[1]),
            design_steps,
            x_source,
            wheel_sign,
            pitch_sign,
            leg_control_enabled,
            branch_guard_enabled,
        )
        initial_state_deltas[:, state_index] = 0.5 * (plus_initial - minus_initial)
        next_state_deltas[:, state_index] = 0.5 * (plus - minus)
    if np.linalg.matrix_rank(initial_state_deltas) != state_count:
        singular_values = np.linalg.svd(initial_state_deltas, compute_uv=False)
        raise RuntimeError(
            "LQR state perturbations do not span the augmented state; "
            f"rank={np.linalg.matrix_rank(initial_state_deltas)}/{state_count}, "
            f"singular_values={singular_values.tolist()}, "
            f"actual_deltas={initial_state_deltas.tolist()}"
        )
    a_matrix = next_state_deltas @ np.linalg.inv(initial_state_deltas)
    for input_index in range(input_count):
        delta = np.zeros(input_count, dtype=float)
        delta[input_index] = float(input_eps[input_index])
        plus = _simulate_lqr_dynamics_step(
            mujoco,
            model,
            _copy_data(mujoco, model, operating_data),
            left_target,
            right_target,
            leg_branch,
            ik_search_radius,
            ik_search_samples,
            virtual_rod_length_kp,
            virtual_rod_length_kd,
            virtual_rod_length_ki,
            virtual_rod_length_force_ff,
            virtual_rod_length_integral_limit,
            virtual_rod_length_force_rate_limit,
            virtual_rod_joint_kd,
            float(operating_u0[0] + delta[0]),
            float(operating_u0[1] + delta[1]),
            design_steps,
            x_source,
            wheel_sign,
            pitch_sign,
            leg_control_enabled,
            branch_guard_enabled,
        )
        minus = _simulate_lqr_dynamics_step(
            mujoco,
            model,
            _copy_data(mujoco, model, operating_data),
            left_target,
            right_target,
            leg_branch,
            ik_search_radius,
            ik_search_samples,
            virtual_rod_length_kp,
            virtual_rod_length_kd,
            virtual_rod_length_ki,
            virtual_rod_length_force_ff,
            virtual_rod_length_integral_limit,
            virtual_rod_length_force_rate_limit,
            virtual_rod_joint_kd,
            float(operating_u0[0] - delta[0]),
            float(operating_u0[1] - delta[1]),
            design_steps,
            x_source,
            wheel_sign,
            pitch_sign,
            leg_control_enabled,
            branch_guard_enabled,
        )
        b_matrix[:, input_index] = (plus - minus) / (2.0 * float(input_eps[input_index]))
    baseline_next = _simulate_lqr_dynamics_step(
        mujoco,
        model,
        _copy_data(mujoco, model, operating_data),
        left_target,
        right_target,
        leg_branch,
        ik_search_radius,
        ik_search_samples,
        virtual_rod_length_kp,
        virtual_rod_length_kd,
        virtual_rod_length_ki,
        virtual_rod_length_force_ff,
        virtual_rod_length_integral_limit,
        virtual_rod_length_force_rate_limit,
        virtual_rod_joint_kd,
        float(operating_u0[0]),
        float(operating_u0[1]),
        design_steps,
        x_source,
        wheel_sign,
        pitch_sign,
        leg_control_enabled,
        branch_guard_enabled,
    )
    # `operating_data` and `operating_u0` come from the static-contact search.
    # Do not algebraically trim U0 here: that would create a new input without
    # settling the constrained mechanism at the corresponding new state.
    affine_residual = baseline_next - x0
    q_matrix = np.diag(q_diag)
    r_matrix = np.diag(r_diag)
    riccati_iterations = 0
    try:
        from scipy.linalg import solve_discrete_are as scipy_solve_discrete_are

        p_matrix = scipy_solve_discrete_are(a_matrix, b_matrix, q_matrix, r_matrix)
        k_matrix = np.linalg.solve(
            r_matrix + b_matrix.T @ p_matrix @ b_matrix,
            b_matrix.T @ p_matrix @ a_matrix,
        )
    except (ImportError, np.linalg.LinAlgError, ValueError) as scipy_error:
        controllability = np.hstack(
            [np.linalg.matrix_power(a_matrix, power) @ b_matrix for power in range(state_count)]
        )
        controllability_rank = np.linalg.matrix_rank(controllability)
        if controllability_rank < state_count:
            raise RuntimeError(
                "augmented LQR model is not controllable at the operating point: "
                f"rank={controllability_rank}/{state_count}; scipy={scipy_error}"
            ) from scipy_error
        p_matrix, riccati_iterations = _solve_discrete_are(
            a_matrix, b_matrix, q_matrix, r_matrix
        )
    if riccati_iterations >= 10000:
        dt = float(model.opt.timestep) * max(1, design_steps)
        continuous_a = (a_matrix - np.eye(a_matrix.shape[0])) / dt
        continuous_b = b_matrix / dt
        p_matrix = _solve_continuous_are_hamiltonian(
            continuous_a, continuous_b, q_matrix, r_matrix
        )
        k_matrix = np.linalg.solve(r_matrix, continuous_b.T @ p_matrix)
    elif riccati_iterations > 0:
        k_matrix = np.linalg.solve(
            r_matrix + b_matrix.T @ p_matrix @ b_matrix,
            b_matrix.T @ p_matrix @ a_matrix,
        )
    closed_loop_eigs = np.linalg.eigvals(a_matrix - b_matrix @ k_matrix)
    return LqrDesignResult(
        operating_state=operating_state,
        operating_u0=operating_u0.copy(),
        affine_residual=affine_residual.copy(),
        operating_point_sample=operating_point_sample,
        q_diag=tuple(float(value) for value in q_diag),
        r_diag=tuple(float(value) for value in r_diag),
        state_eps=tuple(float(value) for value in state_eps),
        input_eps=tuple(float(value) for value in input_eps),
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        k_matrix=k_matrix,
        riccati_iterations=riccati_iterations,
        closed_loop_max_abs_eig=float(np.max(np.abs(closed_loop_eigs))),
    )


def _lqr_middle_control(
    mujoco,
    model,
    data,
    x_reference: float,
    x_source: str,
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
    x_reference_rate: float = 0.0,
    odometry=None,
) -> tuple[float, float, float, LqrState, float]:
    state = _compute_lqr_state(mujoco, model, data, x_reference, x_source, x_reference_rate, odometry)
    wheel_torque, virtual_pitch_torque, length_force_delta, x_velocity_reference = _lqr_middle_control_from_state(
        state,
        gain_scale,
        lqr_k,
        lqr_x0,
        lqr_u0,
        wheel_sign,
        pitch_sign,
        lqr_t_limit,
        lqr_tp_limit,
        lqr_x_outer_kp,
        lqr_x_outer_max_v,
    )
    return wheel_torque, virtual_pitch_torque, length_force_delta, state, x_velocity_reference + x_reference_rate


def _collect_lqr_history_sample(
    mujoco,
    model,
    data,
    step: int,
    wheel_torque: float,
    pitch_torque: float,
    left_pitch_torque: float,
    right_pitch_torque: float,
    yaw_rate_reference: float,
    yaw_rate: float,
    yaw_rate_filtered: float,
    yaw_error: float,
    yaw_error_rate_raw: float,
    yaw_error_rate: float,
    yaw_p_torque: float,
    yaw_d_torque: float,
    turn_torque: float,
    sync_error_raw: float,
    sync_error: float,
    sync_error_rate_raw: float,
    sync_error_rate: float,
    sync_p_torque: float,
    sync_d_torque: float,
    sync_torque: float,
    x_reference: float,
    x_source: str,
    left_target: tuple[float, float],
    right_target: tuple[float, float],
    vmc_diagnostics: dict[str, VmcDiagnostics] | None = None,
    x_velocity_reference: float = 0.0,
    odometry=None,
    roll_reference: float = 0.0,
    roll_length_geometric_offset: float = 0.0,
    roll_force: float = 0.0,
    side_length_force_ff: tuple[float, float] = (0.0, 0.0),
    airborne: bool = False,
    landing_phase: str = "ground",
) -> LqrHistorySample:
    lqr_state = _compute_lqr_state(mujoco, model, data, x_reference, x_source, odometry=odometry)
    left = _compute_virtual_leg_state(mujoco, model, data, "left")
    right = _compute_virtual_leg_state(mujoco, model, data, "right")
    left_branch = _compute_leg_branch_metrics(mujoco, model, data, "left")
    right_branch = _compute_leg_branch_metrics(mujoco, model, data, "right")
    left_front, left_rear = _leg_drive_actuator_ids(mujoco, model, "left")
    right_front, right_rear = _leg_drive_actuator_ids(mujoco, model, "right")
    left_wheel, right_wheel = _wheel_actuator_ids(mujoco, model)
    left_wheel_speed, right_wheel_speed = _wheel_speeds(mujoco, model, data)
    left_diag = (vmc_diagnostics or {}).get("left", VmcDiagnostics(length_error=left_target[0] - left.length))
    right_diag = (vmc_diagnostics or {}).get("right", VmcDiagnostics(length_error=right_target[0] - right.length))
    left_contact_force = _contact_normal_force_for_wheel(mujoco, model, data, "left")
    right_contact_force = _contact_normal_force_for_wheel(mujoco, model, data, "right")
    left_wheel_z, right_wheel_z = _wheel_center_heights(mujoco, model, data)
    left_front_ctrl = float(data.ctrl[left_front])
    left_rear_ctrl = float(data.ctrl[left_rear])
    left_wheel_ctrl = float(data.ctrl[left_wheel])
    right_front_ctrl = float(data.ctrl[right_front])
    right_rear_ctrl = float(data.ctrl[right_rear])
    right_wheel_ctrl = float(data.ctrl[right_wheel])
    return LqrHistorySample(
        step=step,
        time_s=step * float(model.opt.timestep),
        theta=lqr_state.theta,
        theta_rate=lqr_state.theta_rate,
        x=lqr_state.x,
        x_rate=lqr_state.x_rate,
        pitch=lqr_state.pitch,
        pitch_rate=lqr_state.pitch_rate,
        x_velocity_reference=x_velocity_reference,
        wheel_torque=wheel_torque,
        pitch_torque=pitch_torque,
        left_pitch_torque=left_pitch_torque,
        right_pitch_torque=right_pitch_torque,
        yaw_rate_reference=yaw_rate_reference,
        yaw_rate=yaw_rate,
        yaw_rate_filtered=yaw_rate_filtered,
        yaw_error=yaw_error,
        yaw_error_rate_raw=yaw_error_rate_raw,
        yaw_error_rate=yaw_error_rate,
        yaw_p_torque=yaw_p_torque,
        yaw_d_torque=yaw_d_torque,
        turn_torque=turn_torque,
        sync_error_raw=sync_error_raw,
        sync_error=sync_error,
        sync_error_rate_raw=sync_error_rate_raw,
        sync_error_rate=sync_error_rate,
        sync_p_torque=sync_p_torque,
        sync_d_torque=sync_d_torque,
        sync_torque=sync_torque,
        base_height=float(data.qpos[2]) if model.nq >= 3 else math.nan,
        max_abs_ctrl=float(np.max(np.abs(data.ctrl))),
        left_length=left.length,
        right_length=right.length,
        left_length_reference=left_target[0],
        right_length_reference=right_target[0],
        left_length_rate=left.length_rate,
        right_length_rate=right.length_rate,
        left_length_error=left_target[0] - left.length,
        right_length_error=right_target[0] - right.length,
        left_length_force_raw=left_diag.length_force_raw,
        right_length_force_raw=right_diag.length_force_raw,
        left_length_force=left_diag.length_force,
        right_length_force=right_diag.length_force,
        left_length_integral=left_diag.length_integral,
        right_length_integral=right_diag.length_integral,
        left_branch_guard_error=left_diag.branch_guard_error,
        right_branch_guard_error=right_diag.branch_guard_error,
        left_guard_tau_front=left_diag.guard_tau_front,
        left_guard_tau_rear=left_diag.guard_tau_rear,
        right_guard_tau_front=right_diag.guard_tau_front,
        right_guard_tau_rear=right_diag.guard_tau_rear,
        left_theta_tau_front=left_diag.theta_tau_front,
        left_theta_tau_rear=left_diag.theta_tau_rear,
        right_theta_tau_front=right_diag.theta_tau_front,
        right_theta_tau_rear=right_diag.theta_tau_rear,
        roll=_base_roll_angle(data.qpos),
        roll_reference=roll_reference,
        roll_length_geometric_offset=roll_length_geometric_offset,
        roll_force=roll_force,
        left_length_force_ff=side_length_force_ff[0],
        right_length_force_ff=side_length_force_ff[1],
        left_theta_force_scale=left_diag.theta_force_scale,
        right_theta_force_scale=right_diag.theta_force_scale,
        left_front_tau=left_diag.joint_tau_front,
        left_rear_tau=left_diag.joint_tau_rear,
        right_front_tau=right_diag.joint_tau_front,
        right_rear_tau=right_diag.joint_tau_rear,
        left_theta=left.theta,
        right_theta=right.theta,
        left_theta_rate=left.theta_rate,
        right_theta_rate=right.theta_rate,
        left_branch_violation=left_branch.violation,
        right_branch_violation=right_branch.violation,
        left_front_ctrl=left_front_ctrl,
        left_rear_ctrl=left_rear_ctrl,
        left_wheel_ctrl=left_wheel_ctrl,
        right_front_ctrl=right_front_ctrl,
        right_rear_ctrl=right_rear_ctrl,
        right_wheel_ctrl=right_wheel_ctrl,
        left_front_motor_tau=left_front_ctrl * float(model.actuator_gear[left_front, 0]),
        left_rear_motor_tau=left_rear_ctrl * float(model.actuator_gear[left_rear, 0]),
        left_wheel_motor_tau=left_wheel_ctrl * float(model.actuator_gear[left_wheel, 0]),
        right_front_motor_tau=right_front_ctrl * float(model.actuator_gear[right_front, 0]),
        right_rear_motor_tau=right_rear_ctrl * float(model.actuator_gear[right_rear, 0]),
        right_wheel_motor_tau=right_wheel_ctrl * float(model.actuator_gear[right_wheel, 0]),
        left_wheel_speed=left_wheel_speed,
        right_wheel_speed=right_wheel_speed,
        left_contact_normal_force=left_contact_force,
        right_contact_normal_force=right_contact_force,
        airborne=airborne,
        vertical_speed=float(data.qvel[2]) if len(data.qvel) > 2 else 0.0,
        left_wheel_z=left_wheel_z,
        right_wheel_z=right_wheel_z,
        landing_phase=landing_phase,
    )


