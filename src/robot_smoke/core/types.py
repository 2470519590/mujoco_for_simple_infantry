"""机器人 smoke 检查使用的数据容器。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActuatorProbe:
    actuator: str
    joint: str
    ctrl: float
    positive_velocity: float
    negative_velocity: float
    opposite_sign: bool


@dataclass(frozen=True)
class PdHoldResult:
    steps: int
    kp: float
    kd: float
    max_abs_ctrl: float
    final_qpos_norm: float
    final_qvel_norm: float
    final_base_height: float
    saturated_steps: int
    max_abs_joint_error: float
    final_abs_joint_error: float


@dataclass(frozen=True)
class VirtualLegState:
    name: str
    length: float
    length_rate: float
    theta: float
    theta_rate: float


@dataclass(frozen=True)
class LegBranchMetrics:
    name: str
    front_dx: float
    rear_dx: float
    below_front: float
    below_rear: float
    elbow_span: float
    violation: float


@dataclass(frozen=True)
class ActuatorCtrlStats:
    actuator: str
    min_ctrl: float
    max_ctrl: float
    negative_saturated_steps: int
    positive_saturated_steps: int


@dataclass(frozen=True)
class LqrState:
    theta: float
    theta_rate: float
    x: float
    x_rate: float
    pitch: float
    pitch_rate: float
    length: float
    length_rate: float


@dataclass
class SimulatedOdometry:
    """Forward odometry synthesized from world-frame wheel-center motion."""

    position: float = 0.0
    speed: float = 0.0
    previous_time: float | None = None


@dataclass(frozen=True)
class LqrControlDebug:
    state: LqrState
    x_velocity_reference: float


@dataclass(frozen=True)
class LqrHistorySample:
    step: int
    time_s: float
    theta: float
    theta_rate: float
    x: float
    x_rate: float
    pitch: float
    pitch_rate: float
    x_velocity_reference: float
    wheel_torque: float
    pitch_torque: float
    left_pitch_torque: float
    right_pitch_torque: float
    yaw_rate_reference: float
    yaw_rate: float
    yaw_rate_filtered: float
    yaw_error: float
    yaw_error_rate_raw: float
    yaw_error_rate: float
    yaw_p_torque: float
    yaw_d_torque: float
    turn_torque: float
    sync_error_raw: float
    sync_error: float
    sync_error_rate_raw: float
    sync_error_rate: float
    sync_p_torque: float
    sync_d_torque: float
    sync_torque: float
    base_height: float
    max_abs_ctrl: float
    left_length: float
    right_length: float
    left_length_reference: float
    right_length_reference: float
    left_length_rate: float
    right_length_rate: float
    left_length_error: float
    right_length_error: float
    left_length_force_raw: float
    right_length_force_raw: float
    left_length_force: float
    right_length_force: float
    left_length_integral: float
    right_length_integral: float
    left_branch_guard_error: float
    right_branch_guard_error: float
    left_guard_tau_front: float
    left_guard_tau_rear: float
    right_guard_tau_front: float
    right_guard_tau_rear: float
    left_theta_tau_front: float
    left_theta_tau_rear: float
    right_theta_tau_front: float
    right_theta_tau_rear: float
    roll: float
    roll_reference: float
    roll_length_geometric_offset: float
    roll_force: float
    left_length_force_ff: float
    right_length_force_ff: float
    left_theta_force_scale: float
    right_theta_force_scale: float
    left_front_tau: float
    left_rear_tau: float
    right_front_tau: float
    right_rear_tau: float
    left_theta: float
    right_theta: float
    left_theta_rate: float
    right_theta_rate: float
    left_branch_violation: float
    right_branch_violation: float
    left_front_ctrl: float
    left_rear_ctrl: float
    left_wheel_ctrl: float
    right_front_ctrl: float
    right_rear_ctrl: float
    right_wheel_ctrl: float
    left_front_motor_tau: float
    left_rear_motor_tau: float
    left_wheel_motor_tau: float
    right_front_motor_tau: float
    right_rear_motor_tau: float
    right_wheel_motor_tau: float
    left_wheel_speed: float
    right_wheel_speed: float
    left_contact_normal_force: float
    right_contact_normal_force: float
    airborne: bool
    vertical_speed: float
    left_wheel_z: float
    right_wheel_z: float
    landing_phase: str


@dataclass(frozen=True)
class VmcDiagnostics:
    length_error: float = 0.0
    length_rate: float = 0.0
    length_force_raw: float = 0.0
    length_force: float = 0.0
    length_integral: float = 0.0
    theta_force_raw: float = 0.0
    theta_force: float = 0.0
    theta_force_scale: float = 1.0
    branch_guard_error: float = 0.0
    support_tau_front: float = 0.0
    support_tau_rear: float = 0.0
    guard_tau_front: float = 0.0
    guard_tau_rear: float = 0.0
    theta_tau_front: float = 0.0
    theta_tau_rear: float = 0.0
    joint_tau_front_raw: float = 0.0
    joint_tau_rear_raw: float = 0.0
    joint_tau_front: float = 0.0
    joint_tau_rear: float = 0.0


@dataclass
class VmcSideMemory:
    length_integral: float = 0.0
    previous_length_force: float = 0.0
    shape_jacobian: np.ndarray | None = None
    shape_jacobian_age: int = 0


@dataclass(frozen=True)
class StaticOperatingPointSample:
    label: str
    lqr_state: LqrState
    left_length: float
    right_length: float
    left_length_rate: float
    right_length_rate: float
    wheel_torque: float
    pitch_torque: float
    left_length_force: float
    right_length_force: float
    left_length_force_raw: float
    right_length_force_raw: float
    left_support_tau_front: float
    left_support_tau_rear: float
    right_support_tau_front: float
    right_support_tau_rear: float
    left_tau_front_raw: float
    left_tau_rear_raw: float
    right_tau_front_raw: float
    right_tau_rear_raw: float
    left_tau_front: float
    left_tau_rear: float
    right_tau_front: float
    right_tau_rear: float
    left_contact_normal_force: float
    right_contact_normal_force: float
    left_jacobian: np.ndarray
    right_jacobian: np.ndarray
    qpos: np.ndarray


@dataclass(frozen=True)
class EquilibriumSearchResult:
    init_mode: str
    l_slice: float
    theta_ref: float
    fl0_scale: float
    tp_bias: float
    wheel_com_kp: float
    wheel_damping: float
    fl0_center: float
    tp_center: float
    fl0: float
    steps: int
    eval_steps: int
    qualified: bool
    L_mean: float
    length_target_error: float
    com_offset_mean: float
    phi_mean: float
    theta_mean: float
    theta_diff_rms: float
    dtheta_diff_rms: float
    length_diff_rms: float
    dx_rms: float
    dphi_rms: float
    dtheta_rms: float
    dL_rms: float
    T_mean: float
    Tp_mean: float
    T_sat_ratio: float
    Tp_sat_ratio: float
    joint_sat_ratio: float
    contact_force_mean: float
    contact_force_min: float
    slip_indicator: float
    final_sample: StaticOperatingPointSample
    final_data: object | None = None


@dataclass(frozen=True)
class MechanicsEquilibriumSeed:
    total_mass: float
    gravity: float
    theta_ref: float
    com_offset: float
    length_force_per_leg: float
    pitch_torque: float
    wheel_torque: float


@dataclass(frozen=True)
class AnalyticFivebarKinematics:
    length: float
    theta: float
    carrier_xz: np.ndarray
    front_elbow_xz: np.ndarray
    rear_elbow_xz: np.ndarray
    jacobian: np.ndarray


@dataclass(frozen=True)
class LqrDesignResult:
    operating_state: LqrState
    operating_u0: np.ndarray
    affine_residual: np.ndarray
    operating_point_sample: StaticOperatingPointSample
    q_diag: tuple[float, ...]
    r_diag: tuple[float, ...]
    state_eps: tuple[float, ...]
    input_eps: tuple[float, ...]
    a_matrix: np.ndarray
    b_matrix: np.ndarray
    k_matrix: np.ndarray
    riccati_iterations: int
    closed_loop_max_abs_eig: float


@dataclass(frozen=True)
class VirtualRodResult:
    steps: int
    lock_base: bool
    left_target_length: float
    left_target_theta: float
    right_target_length: float
    right_target_theta: float
    final_left_length: float
    final_left_theta: float
    final_right_length: float
    final_right_theta: float
    max_length_error: float
    max_theta_error: float
    max_left_branch_violation: float
    max_right_branch_violation: float
    final_left_branch_violation: float
    final_right_branch_violation: float
    max_abs_ctrl: float
    saturated_steps: int
    actuator_ctrl_stats: tuple[ActuatorCtrlStats, ...]
    final_lqr_state: LqrState | None
    history: tuple[LqrHistorySample, ...]
    max_abs_lqr_wheel_torque: float
    max_abs_lqr_pitch_torque: float
    max_abs_left_wheel_speed: float
    max_abs_right_wheel_speed: float
    final_left_wheel_speed: float
    final_right_wheel_speed: float
    final_base_height: float
    max_base_height: float
    min_left_contact_force: float
    min_right_contact_force: float
    airborne_steps: int
    first_airborne_time: float | None
    last_airborne_time: float | None
    final_operating_point: StaticOperatingPointSample | None
    final_data: object | None = None


@dataclass(frozen=True)
class ConstraintCheckResult:
    steps: int
    max_left_error: float
    max_right_error: float
    final_left_error: float
    final_right_error: float
    max_left_branch_violation: float
    max_right_branch_violation: float
    final_left_branch_violation: float
    final_right_branch_violation: float
    final_base_height: float
