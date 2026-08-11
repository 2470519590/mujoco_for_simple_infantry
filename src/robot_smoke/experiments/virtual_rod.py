"""虚拟杆 smoke 与约束检查流程。"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from ..model.actuators import apply_additive_wheel_torque_ctrl as _apply_additive_wheel_torque_ctrl
from ..model.actuators import apply_wheel_torque_pair_ctrl as _apply_wheel_torque_pair_ctrl
from ..model.fivebar import compute_leg_branch_metrics as _compute_leg_branch_metrics
from ..control.lqr import (
    apply_theta_pitch_feedforward as _apply_theta_pitch_feedforward,
    base_pitch_from_qpos as _base_pitch_from_qpos,
    lowpass_value as _lowpass_value,
)
from ..control.lqr_design import (
    _collect_lqr_history_sample,
    _lqr_middle_control,
    _prepare_lqr_operating_point,
)
from ..model.kinematics import (
    compute_virtual_leg_state as _compute_virtual_leg_state,
    lower_loop_error as _lower_loop_error,
    update_simulated_odometry as _update_simulated_odometry,
    wheel_speeds as _wheel_speeds,
)
from ..control.ik import _branch_aware_ik_targets
from ..control.ik import _virtual_rod_ik_ctrl
from ..model.mechanics import _collect_static_operating_point_sample
from ..model.mechanics import _contact_normal_force_for_wheel
from ..model.mechanics import apply_base_impact as _apply_base_impact
from ..control.trajectory import trapezoid_speed_reference as _trapezoid_speed_reference
from ..control.roll import base_roll_angle as _base_roll_angle
from ..control.roll import clamp_leg_length as _clamp_leg_length
from ..control.roll import roll_leg_length_targets as _roll_leg_length_targets
from ..control.length_schedule import LengthSchedule
from ..control.whole_body import compose_whole_body_command as _compose_whole_body_command
from ..control.turning import split_wheel_torque as _split_wheel_torque
from ..control.turning import turn_rate_magnitude as _turn_rate_magnitude
from ..control.turning import turn_rate_reference as _turn_rate_reference
from ..control.turning import yaw_turn_torque as _yaw_turn_torque
from ..core.mujoco_utils import assert_finite as _assert_finite
from ..core.mujoco_utils import copy_data as _copy_data
from ..core.mujoco_utils import id_by_name as _id_by_name
from ..core.mujoco_utils import lock_base_to_initial as _lock_base_to_initial
from ..core.mujoco_utils import name as _name
from ..core.types import (
    ActuatorCtrlStats,
    ConstraintCheckResult,
    LqrHistorySample,
    LqrState,
    SimulatedOdometry,
    VirtualRodResult,
    VmcDiagnostics,
    VmcSideMemory,
)
from ..core.config import RuntimeControlConfig


def _leg_height_test_target(
    target: tuple[float, float],
    time_s: float,
    enabled: bool,
    levels: tuple[float, float, float],
) -> tuple[float, float]:
    """Return the scheduled length while preserving the target theta."""
    if not enabled or time_s < 1.0:
        return target
    level_index = min(2, int((time_s - 1.0) // 3.0))
    return levels[level_index], target[1]


def _leg_length_sine_target(
    target: tuple[float, float],
    time_s: float,
    enabled: bool,
    minimum_leg_length: float,
    maximum_leg_length: float,
    period_s: float,
) -> tuple[float, float]:
    """Return a full-range sinusoidal length target while preserving theta."""
    if not enabled:
        return target
    if period_s <= 0.0:
        raise ValueError("leg_length_sine_period must be positive")
    mid_length = 0.5 * (minimum_leg_length + maximum_leg_length)
    amplitude = 0.5 * (maximum_leg_length - minimum_leg_length)
    if amplitude <= 0.0:
        return mid_length, target[1]
    initial_ratio = float(np.clip((target[0] - mid_length) / amplitude, -1.0, 1.0))
    phase = math.asin(initial_ratio)
    length = mid_length + amplitude * math.sin(2.0 * math.pi * time_s / period_s + phase)
    return length, target[1]


def _jump_length_target(
    target: tuple[float, float],
    time_s: float,
    enabled: bool,
    jump_time_s: float,
    crouch_leg_length: float,
    maximum_leg_length: float,
) -> tuple[float, float]:
    """Command crouch after jump trigger; extension waits for measured crouch completion."""
    if not enabled:
        return target
    if time_s < jump_time_s:
        length = target[0]
    else:
        length = crouch_leg_length
    return length, target[1]


def _synchronized_turn_rate_reference(turn_speed: str, time_s: float, duration_s: float) -> float:
    """High-rate turn reference that starts and ends with the dynamic length test."""
    duration_s = max(float(duration_s), 0.0)
    if duration_s <= 0.0:
        return 0.0
    time_s = float(np.clip(time_s, 0.0, duration_s))
    magnitude = _turn_rate_magnitude(turn_speed)
    ramp_time = min(0.15, 0.5 * duration_s)
    if ramp_time <= 0.0:
        return magnitude
    if time_s < ramp_time:
        return magnitude * time_s / ramp_time
    if time_s > duration_s - ramp_time:
        return magnitude * max(duration_s - time_s, 0.0) / ramp_time
    return magnitude


def _startup_length_target(target: tuple[float, float], time_s: float, ramp_seconds: float) -> tuple[float, float]:
    if ramp_seconds <= 0.0:
        return target
    start_length = 0.35
    ratio = float(np.clip(time_s / ramp_seconds, 0.0, 1.0))
    return start_length + ratio * (target[0] - start_length), target[1]


def _airborne_lqr_torque(
    state: LqrState,
    gain_scale: float,
    lqr_k: np.ndarray,
    lqr_x0: np.ndarray,
    lqr_u0: np.ndarray,
    pitch_sign: float,
    lqr_tp_limit: float,
) -> float:
    """Paper section 3 mode: keep only Tp feedback from theta and dtheta."""
    theta_error = float(state.theta - lqr_x0[0])
    theta_rate_error = float(state.theta_rate - lqr_x0[1])
    pitch_control = float(lqr_u0[1]) - gain_scale * (
        float(lqr_k[1, 0]) * theta_error + float(lqr_k[1, 1]) * theta_rate_error
    )
    return pitch_sign * float(np.clip(pitch_control, -lqr_tp_limit, lqr_tp_limit))


def _landing_state_is_balanced(state: LqrState | None) -> bool:
    if state is None:
        return False
    return (
        abs(state.theta) < 0.08
        and abs(state.theta_rate) < 0.5
        and abs(state.x_rate) < 0.20
        and abs(state.pitch) < 0.08
        and abs(state.pitch_rate) < 0.5
    )


def _position_ik_cache_key(
    side: str,
    target: tuple[float, float],
    leg_branch: str,
    ik_search_radius: float,
    ik_search_samples: int,
) -> tuple[object, ...]:
    return (
        side,
        round(target[0], 4),
        round(target[1] / 0.01) * 0.01,
        leg_branch,
        round(ik_search_radius, 4),
        ik_search_samples,
    )


def _prewarm_jump_position_ik(
    mujoco,
    model,
    data,
    ik_target_cache: dict[tuple[object, ...], tuple[float, float]],
    left_target: tuple[float, float],
    right_target: tuple[float, float],
    leg_branch: str,
    ik_search_radius: float,
    ik_search_samples: int,
    maximum_leg_length: float,
) -> None:
    """Move jump extension IK search before viewer timing-sensitive rollout."""
    for side, base_target in (("left", left_target), ("right", right_target)):
        target = (maximum_leg_length, base_target[1])
        cache_key = _position_ik_cache_key(
            side,
            target,
            leg_branch,
            ik_search_radius,
            ik_search_samples,
        )
        if cache_key in ik_target_cache:
            continue
        reset_data = mujoco.MjData(model)
        mujoco.mj_resetData(model, reset_data)
        mujoco.mj_forward(model, reset_data)
        reset_state = _compute_virtual_leg_state(mujoco, model, reset_data, side)
        ik_target_cache[cache_key] = _branch_aware_ik_targets(
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


def _reset_vmc_memory(vmc_memory: dict[str, VmcSideMemory]) -> None:
    for memory in vmc_memory.values():
        memory.length_integral = 0.0
        memory.previous_length_force = 0.0


def _run_virtual_rod_test(
    mujoco,
    model,
    steps: int,
    lock_base: bool,
    left_target: tuple[float, float],
    right_target: tuple[float, float],
    virtual_rod_control: str,
    leg_branch: str,
    ik_search_radius: float,
    ik_search_samples: int,
    length_kp: float,
    length_kd: float,
    length_ki: float,
    length_force_ff: float,
    length_integral_limit: float,
    length_force_rate_limit: float,
    theta_kp: float,
    theta_kd: float,
    joint_kd: float,
    theta_pitch_ff: float,
    lqr_test: bool,
    lqr_gain_scale: float,
    lqr_k: np.ndarray,
    lqr_x0: np.ndarray,
    lqr_u0: np.ndarray,
    lqr_auto_design: bool,
    lqr_control_period_steps: int,
    lqr_x_reference: float,
    lqr_x_source: str,
    lqr_x_outer_kp: float,
    lqr_x_outer_max_v: float,
    lqr_wheel_sign: float,
    lqr_pitch_sign: float,
    lqr_t_limit: float,
    lqr_tp_limit: float,
    landing_hold_t_limit: float,
    lqr_output_rate_limit: float,
    lqr_output_lowpass_hz: float,
    wheel_ctrl_deadzone: float,
    history_sample_interval: int,
    initial_data: object | None = None,
    impact_level: str | None = None,
    speed_profile: str | None = None,
    turn_direction: str | None = None,
    turn_speed: str = "high",
    turn_test: bool = False,
    manual_drive_input: object | None = None,
    leg_sync_kp: float = 30.0,
    leg_sync_kd: float = 20.0,
    yaw_turn_kp: float = 1.8,
    yaw_turn_kd: float = 0.0,
    leg_height_test: bool = False,
    leg_height_levels: tuple[float, float, float] = (0.16, 0.23, 0.30),
    leg_length_sine_test: bool = False,
    leg_length_sine_period: float = 1,
    jump_test: bool = False,
    jump_time_s: float = 1.0,
    branch_guard_enabled: bool = True,
    minimum_leg_length: float = 0.16,
    maximum_leg_length: float = 0.30,
    length_schedule: LengthSchedule | None = None,
    startup_ramp_seconds: float = 1.0,
    flight_detection_enabled: bool = False,
    flight_airborne_force_threshold: float = 20.0,
    flight_airborne_confirm_seconds: float = 0.05,
    flight_airborne_rearm_seconds: float = 1.0,
    runtime_controls: RuntimeControlConfig | None = None,
    time_offset_s: float = 0.0,
    rollout_context: dict[str, object] | None = None,
    reuse_initial_data: bool = False,
    copy_final_data: bool = True,
    control_decimation_steps: int = 1,
    fast_result: bool = False,
    on_initialized: Callable[[object], Callable[..., bool] | None] | None = None,
) -> VirtualRodResult:
    if runtime_controls is None:
        raise ValueError("runtime_controls must be provided by the YAML-backed runner")
    control_decimation_steps = max(1, int(control_decimation_steps))
    if initial_data is not None:
        data = initial_data if reuse_initial_data else _copy_data(mujoco, model, initial_data)
    else:
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
    if initial_data is None and lqr_test and lqr_auto_design:
        data = _prepare_lqr_operating_point(
            mujoco,
            model,
            left_target,
            right_target,
            leg_branch,
            ik_search_radius,
            ik_search_samples,
        )
    if lock_base:
        _lock_base_to_initial(mujoco, model, data)
    left_wheel_body = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "left_wheel")
    right_wheel_body = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "right_wheel")
    track_width = abs(float(data.xpos[left_wheel_body, 1] - data.xpos[right_wheel_body, 1]))
    max_abs_ctrl = 0.0
    max_length_error = 0.0
    max_theta_error = 0.0
    max_left_branch_violation = 0.0
    max_right_branch_violation = 0.0
    saturated_steps = 0
    min_ctrl = np.full(model.nu, math.inf)
    max_ctrl = np.full(model.nu, -math.inf)
    negative_saturated_steps = np.zeros(model.nu, dtype=int)
    positive_saturated_steps = np.zeros(model.nu, dtype=int)
    final_lqr_state: LqrState | None = None
    max_abs_lqr_wheel_torque = 0.0
    max_abs_lqr_pitch_torque = 0.0
    max_abs_left_wheel_speed = 0.0
    max_abs_right_wheel_speed = 0.0
    previous_wheel_torque = 0.0
    previous_pitch_torque = 0.0
    previous_yaw_error = 0.0
    previous_raw_yaw_error = 0.0
    filtered_yaw_rate = 0.0
    filtered_yaw_error = 0.0
    filtered_yaw_error_rate = 0.0
    filtered_sync_error = 0.0
    filtered_sync_error_rate = 0.0
    filtered_left_theta = 0.0
    filtered_right_theta = 0.0
    raw_yaw_error_rate = 0.0
    yaw_error_rate = 0.0
    yaw_p_torque = 0.0
    yaw_d_torque = 0.0
    raw_sync_error = 0.0
    raw_sync_error_rate = 0.0
    sync_p_torque = 0.0
    sync_d_torque = 0.0
    yaw_rate_reference = 0.0
    turn_torque = 0.0
    left_wheel_torque = 0.0
    right_wheel_torque = 0.0
    length_force_delta = 0.0
    x_velocity_reference = 0.0
    history: list[LqrHistorySample] = []
    ik_target_cache: dict[tuple[str, float, float, str, float, int], tuple[float, float]] = {}
    vmc_memory: dict[str, VmcSideMemory] = {}
    vmc_diagnostics: dict[str, VmcDiagnostics] = {}
    odometry = SimulatedOdometry()
    executed_steps = 0
    max_base_height = float(data.qpos[2]) if model.nq >= 3 else math.nan
    min_left_contact_force = math.inf
    min_right_contact_force = math.inf
    airborne_steps = 0
    first_airborne_time: float | None = None
    last_airborne_time: float | None = None
    contact_detection_armed = False
    was_airborne = False
    landing_phase = "ground"
    landing_hold_active = False
    landing_hold_stable_time = 0.0
    landing_hold_required_time = 0.5
    raw_airborne_start_time: float | None = None
    airborne_rearm_time = 0.0
    suppress_speed_profile_after_airborne = False
    jump_started = False
    active_jump_time_s = float(jump_time_s)
    jump_extension_start_time: float | None = None
    jump_crouch_leg_length = max(0.18, minimum_leg_length)
    previous_step_virtual_rod_control = virtual_rod_control
    if rollout_context is not None:
        previous_wheel_torque = float(rollout_context.get("previous_wheel_torque", previous_wheel_torque))
        previous_pitch_torque = float(rollout_context.get("previous_pitch_torque", previous_pitch_torque))
        previous_yaw_error = float(rollout_context.get("previous_yaw_error", previous_yaw_error))
        previous_raw_yaw_error = float(rollout_context.get("previous_raw_yaw_error", previous_raw_yaw_error))
        filtered_yaw_rate = float(rollout_context.get("filtered_yaw_rate", filtered_yaw_rate))
        filtered_yaw_error = float(rollout_context.get("filtered_yaw_error", filtered_yaw_error))
        filtered_yaw_error_rate = float(rollout_context.get("filtered_yaw_error_rate", filtered_yaw_error_rate))
        filtered_sync_error = float(rollout_context.get("filtered_sync_error", filtered_sync_error))
        filtered_sync_error_rate = float(rollout_context.get("filtered_sync_error_rate", filtered_sync_error_rate))
        filtered_left_theta = float(rollout_context.get("filtered_left_theta", filtered_left_theta))
        filtered_right_theta = float(rollout_context.get("filtered_right_theta", filtered_right_theta))
        raw_yaw_error_rate = float(rollout_context.get("raw_yaw_error_rate", raw_yaw_error_rate))
        yaw_error_rate = float(rollout_context.get("yaw_error_rate", yaw_error_rate))
        yaw_p_torque = float(rollout_context.get("yaw_p_torque", yaw_p_torque))
        yaw_d_torque = float(rollout_context.get("yaw_d_torque", yaw_d_torque))
        raw_sync_error = float(rollout_context.get("raw_sync_error", raw_sync_error))
        raw_sync_error_rate = float(rollout_context.get("raw_sync_error_rate", raw_sync_error_rate))
        sync_p_torque = float(rollout_context.get("sync_p_torque", sync_p_torque))
        sync_d_torque = float(rollout_context.get("sync_d_torque", sync_d_torque))
        yaw_rate_reference = float(rollout_context.get("yaw_rate_reference", yaw_rate_reference))
        turn_torque = float(rollout_context.get("turn_torque", turn_torque))
        left_wheel_torque = float(rollout_context.get("left_wheel_torque", left_wheel_torque))
        right_wheel_torque = float(rollout_context.get("right_wheel_torque", right_wheel_torque))
        length_force_delta = float(rollout_context.get("length_force_delta", length_force_delta))
        x_velocity_reference = float(rollout_context.get("x_velocity_reference", x_velocity_reference))
        ik_target_cache = rollout_context.setdefault("ik_target_cache", ik_target_cache)  # type: ignore[assignment]
        vmc_memory = rollout_context.setdefault("vmc_memory", vmc_memory)  # type: ignore[assignment]
        vmc_diagnostics = rollout_context.setdefault("vmc_diagnostics", vmc_diagnostics)  # type: ignore[assignment]
        odometry = rollout_context.setdefault("odometry", odometry)  # type: ignore[assignment]
        contact_detection_armed = bool(rollout_context.get("contact_detection_armed", contact_detection_armed))
        was_airborne = bool(rollout_context.get("was_airborne", was_airborne))
        landing_phase = str(rollout_context.get("landing_phase", landing_phase))
        landing_hold_active = bool(rollout_context.get("landing_hold_active", landing_hold_active))
        landing_hold_stable_time = float(rollout_context.get("landing_hold_stable_time", landing_hold_stable_time))
        raw_airborne_start_time = rollout_context.get("raw_airborne_start_time", raw_airborne_start_time)  # type: ignore[assignment]
        airborne_rearm_time = float(rollout_context.get("airborne_rearm_time", airborne_rearm_time))
        suppress_speed_profile_after_airborne = bool(
            rollout_context.get("suppress_speed_profile_after_airborne", suppress_speed_profile_after_airborne)
        )
        jump_started = bool(rollout_context.get("jump_started", jump_started))
        active_jump_time_s = float(rollout_context.get("active_jump_time_s", active_jump_time_s))
        jump_extension_start_time = rollout_context.get("jump_extension_start_time", jump_extension_start_time)  # type: ignore[assignment]
        jump_crouch_leg_length = float(rollout_context.get("jump_crouch_leg_length", jump_crouch_leg_length))
        previous_step_virtual_rod_control = str(
            rollout_context.get("previous_step_virtual_rod_control", previous_step_virtual_rod_control)
        )
    startup_steps = int(math.ceil(startup_ramp_seconds / float(model.opt.timestep))) if startup_ramp_seconds > 0.0 else 0
    task_duration_s = max(0.0, time_offset_s + steps * float(model.opt.timestep) - startup_ramp_seconds)
    if jump_test:
        _prewarm_jump_position_ik(
            mujoco,
            model,
            data,
            ik_target_cache,
            left_target,
            right_target,
            leg_branch,
            ik_search_radius,
            ik_search_samples,
            maximum_leg_length,
        )
    step_observer = on_initialized(data) if on_initialized is not None else None
    for step in range(steps):
        time_s = time_offset_s + step * float(model.opt.timestep)
        if step % control_decimation_steps != 0:
            mujoco.mj_step(model, data)
            executed_steps = step + 1
            if model.nq >= 3:
                max_base_height = max(max_base_height, float(data.qpos[2]))
            if step_observer is not None and not step_observer(data, executed_steps, landing_phase):
                break
            continue
        left_contact_force = _contact_normal_force_for_wheel(mujoco, model, data, "left")
        right_contact_force = _contact_normal_force_for_wheel(mujoco, model, data, "right")
        if left_contact_force >= flight_airborne_force_threshold or right_contact_force >= flight_airborne_force_threshold:
            contact_detection_armed = True
        if contact_detection_armed:
            min_left_contact_force = min(min_left_contact_force, left_contact_force)
            min_right_contact_force = min(min_right_contact_force, right_contact_force)
        raw_airborne_candidate = (
            flight_detection_enabled
            and contact_detection_armed
            and time_s >= airborne_rearm_time
            and left_contact_force < flight_airborne_force_threshold
            and right_contact_force < flight_airborne_force_threshold
        )
        if raw_airborne_candidate:
            if raw_airborne_start_time is None:
                raw_airborne_start_time = time_s
            raw_airborne = time_s - raw_airborne_start_time >= flight_airborne_confirm_seconds
        else:
            raw_airborne_start_time = None
            raw_airborne = False
        airborne = raw_airborne and not landing_hold_active
        if airborne:
            was_airborne = True
            suppress_speed_profile_after_airborne = True
            airborne_steps += 1
            if first_airborne_time is None:
                first_airborne_time = time_s
            last_airborne_time = time_s
            landing_phase = "airborne"
        else:
            if landing_phase == "airborne":
                landing_hold_active = True
                landing_hold_stable_time = 0.0
                _reset_vmc_memory(vmc_memory)
            landing_phase = "landing_hold" if landing_hold_active else "ground"
        active_impact = impact_level if step >= startup_steps else None
        _apply_base_impact(mujoco, model, data, active_impact, max(0, step - startup_steps))
        _update_simulated_odometry(mujoco, model, data, odometry)
        if lock_base:
            _lock_base_to_initial(mujoco, model, data)
        wheel_torque = previous_wheel_torque
        pitch_torque = previous_pitch_torque
        active_lqr_k = lqr_k
        active_lqr_x0 = lqr_x0
        active_lqr_u0 = lqr_u0
        active_length_force_ff = length_force_ff
        if length_schedule is not None:
            left_schedule_leg = _compute_virtual_leg_state(mujoco, model, data, "left")
            right_schedule_leg = _compute_virtual_leg_state(mujoco, model, data, "right")
            scheduled = length_schedule.evaluate(0.5 * (left_schedule_leg.length + right_schedule_leg.length))
            active_lqr_k = scheduled.lqr_k
            active_lqr_x0 = scheduled.lqr_x0
            active_lqr_u0 = scheduled.lqr_u0
            active_length_force_ff = scheduled.force_ff
        if lqr_test and step % lqr_control_period_steps == 0:
            task_time_s = max(0.0, time_s - startup_ramp_seconds)
            control_dt = float(model.opt.timestep) * lqr_control_period_steps
            _, x_reference_rate = _trapezoid_speed_reference(speed_profile, task_time_s)
            if manual_drive_input is not None:
                manual_drive_input.update(control_dt)
                x_reference_rate = float(manual_drive_input.forward_speed_reference())
            if suppress_speed_profile_after_airborne:
                x_reference_rate = 0.0
            effective_x_reference = float(odometry.position) - float(active_lqr_x0[2])
            wheel_torque, pitch_torque, length_force_delta, final_lqr_state, x_velocity_reference = _lqr_middle_control(
                mujoco,
                model,
                data,
                effective_x_reference,
                lqr_x_source,
                lqr_gain_scale,
                active_lqr_k,
                active_lqr_x0,
                active_lqr_u0,
                lqr_wheel_sign,
                lqr_pitch_sign,
                lqr_t_limit,
                lqr_tp_limit,
                lqr_x_outer_kp,
                lqr_x_outer_max_v,
                x_reference_rate,
                odometry,
            )
            if airborne:
                wheel_torque = 0.0
                pitch_torque = _airborne_lqr_torque(
                    final_lqr_state,
                    lqr_gain_scale,
                    active_lqr_k,
                    active_lqr_x0,
                    active_lqr_u0,
                    lqr_pitch_sign,
                    lqr_tp_limit,
                )
                length_force_delta = 0.0
            if landing_hold_active:
                if _landing_state_is_balanced(final_lqr_state):
                    landing_hold_stable_time += float(model.opt.timestep) * lqr_control_period_steps
                    if landing_hold_stable_time >= landing_hold_required_time:
                        landing_hold_active = False
                        landing_phase = "ground"
                        airborne_rearm_time = time_s + flight_airborne_rearm_seconds
                        raw_airborne_start_time = None
                        _reset_vmc_memory(vmc_memory)
                else:
                    landing_hold_stable_time = 0.0
            wheel_torque = _lowpass_value(
                wheel_torque,
                previous_wheel_torque,
                lqr_output_lowpass_hz,
                control_dt,
            )
            pitch_torque = _lowpass_value(
                pitch_torque,
                previous_pitch_torque,
                lqr_output_lowpass_hz,
                control_dt,
            )
            if lqr_output_rate_limit > 0.0:
                max_delta = lqr_output_rate_limit * float(model.opt.timestep)
                wheel_torque = float(
                    np.clip(
                        wheel_torque,
                        previous_wheel_torque - max_delta,
                        previous_wheel_torque + max_delta,
                    )
                )
                pitch_torque = float(
                    np.clip(
                        pitch_torque,
                        previous_pitch_torque - max_delta,
                        previous_pitch_torque + max_delta,
                    )
                )
            if landing_hold_active:
                wheel_torque = float(np.clip(wheel_torque, -landing_hold_t_limit, landing_hold_t_limit))
            if airborne:
                wheel_torque = 0.0
            previous_wheel_torque = wheel_torque
            previous_pitch_torque = pitch_torque
            yaw_rate_reference = (
                _synchronized_turn_rate_reference(turn_speed, task_time_s, task_duration_s)
                if leg_length_sine_test
                else _turn_rate_reference(turn_direction, turn_speed, turn_test, task_time_s)
            )
            if manual_drive_input is not None:
                yaw_rate_reference = float(manual_drive_input.yaw_rate_reference())
            if airborne:
                yaw_rate_reference = 0.0
            raw_yaw_rate = float(data.qvel[5])
            filtered_yaw_rate = _lowpass_value(
                raw_yaw_rate,
                filtered_yaw_rate,
                runtime_controls.yaw_turn_input_lowpass_hz,
                control_dt,
            )
            raw_yaw_error = yaw_rate_reference - filtered_yaw_rate
            filtered_yaw_error = _lowpass_value(
                raw_yaw_error,
                filtered_yaw_error,
                runtime_controls.yaw_turn_error_lowpass_hz,
                control_dt,
            )
            if previous_yaw_error == 0.0 and step == 0:
                previous_yaw_error = raw_yaw_error
                previous_raw_yaw_error = raw_yaw_error
                yaw_error_rate = 0.0
            else:
                raw_yaw_error_rate = (raw_yaw_error - previous_raw_yaw_error) / control_dt
                yaw_error_rate = _lowpass_value(
                    raw_yaw_error_rate,
                    filtered_yaw_error_rate,
                    runtime_controls.yaw_turn_derivative_lowpass_hz,
                    control_dt,
                )
                filtered_yaw_error_rate = yaw_error_rate
            previous_raw_yaw_error = raw_yaw_error
            yaw_p_torque = yaw_turn_kp * filtered_yaw_error
            yaw_d_torque = yaw_turn_kd * yaw_error_rate
            turn_torque, previous_yaw_error = _yaw_turn_torque(
                yaw_rate_reference,
                filtered_yaw_rate,
                previous_yaw_error,
                control_dt,
                kp=yaw_turn_kp,
                kd=yaw_turn_kd,
                error_rate=yaw_error_rate,
                error=filtered_yaw_error,
            )
            if airborne:
                turn_torque = 0.0
                yaw_p_torque = 0.0
                yaw_d_torque = 0.0
            left_wheel_torque, right_wheel_torque = _split_wheel_torque(wheel_torque, turn_torque)
            max_abs_lqr_wheel_torque = max(max_abs_lqr_wheel_torque, abs(wheel_torque))
            max_abs_lqr_pitch_torque = max(max_abs_lqr_pitch_torque, abs(pitch_torque))
        elif lqr_test:
            max_abs_lqr_wheel_torque = max(max_abs_lqr_wheel_torque, abs(wheel_torque))
            max_abs_lqr_pitch_torque = max(max_abs_lqr_pitch_torque, abs(pitch_torque))
        if airborne:
            turn_torque = 0.0
            left_wheel_torque, right_wheel_torque = _split_wheel_torque(wheel_torque, turn_torque)
        pitch_for_theta_ff = final_lqr_state.pitch if final_lqr_state is not None else _base_pitch_from_qpos(data.qpos)
        task_time_s = max(0.0, time_s - startup_ramp_seconds)
        left_leg = _compute_virtual_leg_state(mujoco, model, data, "left")
        right_leg = _compute_virtual_leg_state(mujoco, model, data, "right")
        if jump_test and not jump_started:
            if task_time_s >= jump_time_s:
                jump_started = True
                active_jump_time_s = float(jump_time_s)
                jump_extension_start_time = None
        average_leg_length = 0.5 * (left_leg.length + right_leg.length)
        jump_crouch_ready = average_leg_length <= jump_crouch_leg_length + 0.003
        jump_crouch_timeout = jump_started and task_time_s >= active_jump_time_s + 0.10
        if (
            jump_test
            and jump_started
            and jump_extension_start_time is None
            and not was_airborne
            and landing_phase == "ground"
            and (jump_crouch_ready or jump_crouch_timeout)
        ):
            jump_extension_start_time = task_time_s
        jump_extension_active = (
            jump_extension_start_time is not None
            and jump_extension_start_time <= task_time_s < jump_extension_start_time + 0.50
            and not was_airborne
            and landing_phase == "ground"
        )
        scheduled_left_target = _startup_length_target(left_target, time_s, startup_ramp_seconds)
        scheduled_right_target = _startup_length_target(right_target, time_s, startup_ramp_seconds)
        scheduled_left_target = _leg_height_test_target(scheduled_left_target, task_time_s, leg_height_test, leg_height_levels)
        scheduled_right_target = _leg_height_test_target(scheduled_right_target, task_time_s, leg_height_test, leg_height_levels)
        scheduled_left_target = _jump_length_target(
            scheduled_left_target,
            task_time_s,
            jump_started,
            active_jump_time_s,
            jump_crouch_leg_length,
            maximum_leg_length,
        )
        scheduled_right_target = _jump_length_target(
            scheduled_right_target,
            task_time_s,
            jump_started,
            active_jump_time_s,
            jump_crouch_leg_length,
            maximum_leg_length,
        )
        if jump_extension_active:
            scheduled_left_target = (maximum_leg_length, scheduled_left_target[1])
            scheduled_right_target = (maximum_leg_length, scheduled_right_target[1])
        scheduled_left_target = _leg_length_sine_target(
            scheduled_left_target,
            task_time_s,
            leg_length_sine_test,
            minimum_leg_length,
            maximum_leg_length,
            leg_length_sine_period,
        )
        scheduled_right_target = _leg_length_sine_target(
            scheduled_right_target,
            task_time_s,
            leg_length_sine_test,
            minimum_leg_length,
            maximum_leg_length,
            leg_length_sine_period,
        )
        scheduled_left_target = (
            _clamp_leg_length(scheduled_left_target[0], minimum_leg_length, maximum_leg_length),
            scheduled_left_target[1],
        )
        scheduled_right_target = (
            _clamp_leg_length(scheduled_right_target[0], minimum_leg_length, maximum_leg_length),
            scheduled_right_target[1],
        )
        step_left_target = _apply_theta_pitch_feedforward(scheduled_left_target, pitch_for_theta_ff, theta_pitch_ff)
        step_right_target = _apply_theta_pitch_feedforward(scheduled_right_target, pitch_for_theta_ff, theta_pitch_ff)
        roll_targets = _roll_leg_length_targets(
            step_left_target[0],
            step_right_target[0],
            _base_roll_angle(data.qpos),
            runtime_controls.roll_reference,
            track_width,
            runtime_controls.roll_force_kp,
            minimum_leg_length,
            maximum_leg_length,
        )
        step_left_target = (roll_targets.left_length, step_left_target[1])
        step_right_target = (roll_targets.right_length, step_right_target[1])
        if landing_hold_active:
            step_left_target = (
                _clamp_leg_length(left_leg.length, minimum_leg_length, maximum_leg_length),
                step_left_target[1],
            )
            step_right_target = (
                _clamp_leg_length(right_leg.length, minimum_leg_length, maximum_leg_length),
                step_right_target[1],
            )
        effective_theta_kp = 0.0 if lqr_test else theta_kp
        effective_theta_kd = 0.0 if lqr_test else theta_kd
        raw_sync_error = right_leg.theta - left_leg.theta
        filtered_left_theta = _lowpass_value(
            left_leg.theta,
            filtered_left_theta,
            runtime_controls.leg_sync_input_lowpass_hz,
            float(model.opt.timestep),
        )
        filtered_right_theta = _lowpass_value(
            right_leg.theta,
            filtered_right_theta,
            runtime_controls.leg_sync_input_lowpass_hz,
            float(model.opt.timestep),
        )
        filtered_sync_input_error = filtered_right_theta - filtered_left_theta
        sync_error = _lowpass_value(
            filtered_sync_input_error,
            filtered_sync_error,
            runtime_controls.leg_sync_error_lowpass_hz,
            float(model.opt.timestep),
        )
        filtered_sync_error = sync_error
        raw_sync_error_rate = right_leg.theta_rate - left_leg.theta_rate
        sync_error_rate = _lowpass_value(
            raw_sync_error_rate,
            filtered_sync_error_rate,
            runtime_controls.leg_sync_derivative_lowpass_hz,
            float(model.opt.timestep),
        )
        filtered_sync_error_rate = sync_error_rate
        sync_p_torque = leg_sync_kp * sync_error
        sync_d_torque = leg_sync_kd * sync_error_rate
        sync_torque = sync_p_torque + sync_d_torque
        whole_body = _compose_whole_body_command(
            wheel_torque,
            pitch_torque,
            sync_torque,
            active_length_force_ff + length_force_delta,
            roll_targets.force,
        )
        side_length_force_ff = (
            whole_body.left_length_force_bias,
            whole_body.right_length_force_bias,
        )
        left_pitch_torque = whole_body.left_pitch_torque
        right_pitch_torque = whole_body.right_pitch_torque
        step_virtual_rod_control = virtual_rod_control
        step_length_kp = length_kp
        step_length_kd = length_kd
        step_length_ki = length_ki
        if landing_hold_active:
            step_length_kp = 0.0
            step_length_ki = 0.0
        if jump_extension_active:
            step_virtual_rod_control = "position"
            step_length_kp = 3000.0
            step_length_kd = 40.0
            step_length_ki = 0.0
        if previous_step_virtual_rod_control == "position" and step_virtual_rod_control != "position":
            _reset_vmc_memory(vmc_memory)
        previous_step_virtual_rod_control = step_virtual_rod_control
        ctrl_abs, saturated, length_error, theta_error = _virtual_rod_ik_ctrl(
            mujoco,
            model,
            data,
            step_left_target,
            step_right_target,
            step_virtual_rod_control,
            leg_branch,
            ik_search_radius,
            ik_search_samples,
            step_length_kp,
            step_length_kd,
            effective_theta_kp,
            effective_theta_kd,
            joint_kd,
            ik_target_cache,
            theta_force_offset=(left_pitch_torque, right_pitch_torque),
            length_force_ff=side_length_force_ff,
            length_ki=step_length_ki,
            length_integral_limit=length_integral_limit,
            length_force_rate_limit=length_force_rate_limit,
            branch_guard_enabled=branch_guard_enabled,
            vmc_memory=vmc_memory,
            vmc_diagnostics=vmc_diagnostics,
        )
        if lqr_test:
            saturated = (
                _apply_wheel_torque_pair_ctrl(
                    mujoco,
                    model,
                    data,
                    left_wheel_torque,
                    right_wheel_torque,
                    wheel_ctrl_deadzone,
                )
                or saturated
            )
            ctrl_abs = max(ctrl_abs, float(np.max(np.abs(data.ctrl))))
        max_abs_ctrl = max(max_abs_ctrl, ctrl_abs)
        max_length_error = max(max_length_error, length_error)
        max_theta_error = max(max_theta_error, theta_error)
        if not fast_result:
            left_branch = _compute_leg_branch_metrics(mujoco, model, data, "left")
            right_branch = _compute_leg_branch_metrics(mujoco, model, data, "right")
            max_left_branch_violation = max(max_left_branch_violation, left_branch.violation)
            max_right_branch_violation = max(max_right_branch_violation, right_branch.violation)
        if saturated:
            saturated_steps += 1
        min_ctrl = np.minimum(min_ctrl, data.ctrl)
        max_ctrl = np.maximum(max_ctrl, data.ctrl)
        for actuator_id in range(model.nu):
            low, high = model.actuator_ctrlrange[actuator_id]
            if abs(float(data.ctrl[actuator_id]) - low) < 1e-12:
                negative_saturated_steps[actuator_id] += 1
            if abs(float(data.ctrl[actuator_id]) - high) < 1e-12:
                positive_saturated_steps[actuator_id] += 1
        mujoco.mj_step(model, data)
        executed_steps = step + 1
        if model.nq >= 3:
            max_base_height = max(max_base_height, float(data.qpos[2]))
        if not fast_result:
            _assert_finite("qpos", data.qpos)
            _assert_finite("qvel", data.qvel)
            left_wheel_speed, right_wheel_speed = _wheel_speeds(mujoco, model, data)
            max_abs_left_wheel_speed = max(max_abs_left_wheel_speed, abs(left_wheel_speed))
            max_abs_right_wheel_speed = max(max_abs_right_wheel_speed, abs(right_wheel_speed))
        if not fast_result and lqr_test and (step % history_sample_interval == 0 or step == steps - 1):
            history.append(
                _collect_lqr_history_sample(
                    mujoco,
                    model,
                    data,
                    step + 1,
                    wheel_torque,
                    pitch_torque,
                    whole_body.left_pitch_torque,
                    whole_body.right_pitch_torque,
                    yaw_rate_reference,
                    raw_yaw_rate,
                    filtered_yaw_rate,
                    previous_yaw_error,
                    raw_yaw_error_rate,
                    yaw_error_rate,
                    yaw_p_torque,
                    yaw_d_torque,
                    turn_torque,
                    raw_sync_error,
                    sync_error,
                    raw_sync_error_rate,
                    sync_error_rate,
                    sync_p_torque,
                    sync_d_torque,
                    sync_torque,
                    effective_x_reference,
                    lqr_x_source,
                    step_left_target,
                    step_right_target,
                    vmc_diagnostics,
                    x_velocity_reference,
                    odometry,
                    runtime_controls.roll_reference,
                    roll_targets.geometric_offset,
                    roll_targets.force,
                    side_length_force_ff,
                    airborne,
                    landing_phase,
                )
            )
        if step_observer is not None and not step_observer(data, executed_steps, landing_phase):
            break

    if lock_base:
        _lock_base_to_initial(mujoco, model, data)
    left_state = _compute_virtual_leg_state(mujoco, model, data, "left")
    right_state = _compute_virtual_leg_state(mujoco, model, data, "right")
    final_left_wheel_speed, final_right_wheel_speed = _wheel_speeds(mujoco, model, data)
    if fast_result:
        left_branch = None
        right_branch = None
        final_operating_point = None
    else:
        left_branch = _compute_leg_branch_metrics(mujoco, model, data, "left")
        right_branch = _compute_leg_branch_metrics(mujoco, model, data, "right")
        final_operating_point = _collect_static_operating_point_sample(
            mujoco,
            model,
            data,
            "final_virtual_rod_state",
            previous_wheel_torque,
            previous_pitch_torque,
            x_source=lqr_x_source,
            left_diag=vmc_diagnostics.get("left"),
            right_diag=vmc_diagnostics.get("right"),
        )
    actuator_ctrl_stats = tuple(
        ActuatorCtrlStats(
            actuator=_name(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id),
            min_ctrl=float(min_ctrl[actuator_id]),
            max_ctrl=float(max_ctrl[actuator_id]),
            negative_saturated_steps=int(negative_saturated_steps[actuator_id]),
            positive_saturated_steps=int(positive_saturated_steps[actuator_id]),
        )
        for actuator_id in range(model.nu)
    )
    if rollout_context is not None:
        rollout_context.update(
            previous_wheel_torque=previous_wheel_torque,
            previous_pitch_torque=previous_pitch_torque,
            previous_yaw_error=previous_yaw_error,
            previous_raw_yaw_error=previous_raw_yaw_error,
            filtered_yaw_rate=filtered_yaw_rate,
            filtered_yaw_error=filtered_yaw_error,
            filtered_yaw_error_rate=filtered_yaw_error_rate,
            filtered_sync_error=filtered_sync_error,
            filtered_sync_error_rate=filtered_sync_error_rate,
            filtered_left_theta=filtered_left_theta,
            filtered_right_theta=filtered_right_theta,
            raw_yaw_error_rate=raw_yaw_error_rate,
            yaw_error_rate=yaw_error_rate,
            yaw_p_torque=yaw_p_torque,
            yaw_d_torque=yaw_d_torque,
            raw_sync_error=raw_sync_error,
            raw_sync_error_rate=raw_sync_error_rate,
            sync_p_torque=sync_p_torque,
            sync_d_torque=sync_d_torque,
            yaw_rate_reference=yaw_rate_reference,
            turn_torque=turn_torque,
            left_wheel_torque=left_wheel_torque,
            right_wheel_torque=right_wheel_torque,
            length_force_delta=length_force_delta,
            x_velocity_reference=x_velocity_reference,
            contact_detection_armed=contact_detection_armed,
            was_airborne=was_airborne,
            landing_phase=landing_phase,
            landing_hold_active=landing_hold_active,
            landing_hold_stable_time=landing_hold_stable_time,
            raw_airborne_start_time=raw_airborne_start_time,
            airborne_rearm_time=airborne_rearm_time,
            suppress_speed_profile_after_airborne=suppress_speed_profile_after_airborne,
            jump_started=jump_started,
            active_jump_time_s=active_jump_time_s,
            jump_extension_start_time=jump_extension_start_time,
            jump_crouch_leg_length=jump_crouch_leg_length,
            previous_step_virtual_rod_control=previous_step_virtual_rod_control,
        )
    return VirtualRodResult(
        steps=executed_steps,
        lock_base=lock_base,
        left_target_length=left_target[0],
        left_target_theta=left_target[1],
        right_target_length=right_target[0],
        right_target_theta=right_target[1],
        final_left_length=left_state.length,
        final_left_theta=left_state.theta,
        final_right_length=right_state.length,
        final_right_theta=right_state.theta,
        max_length_error=max_length_error,
        max_theta_error=max_theta_error,
        max_left_branch_violation=max_left_branch_violation,
        max_right_branch_violation=max_right_branch_violation,
        final_left_branch_violation=0.0 if left_branch is None else left_branch.violation,
        final_right_branch_violation=0.0 if right_branch is None else right_branch.violation,
        max_abs_ctrl=max_abs_ctrl,
        saturated_steps=saturated_steps,
        actuator_ctrl_stats=actuator_ctrl_stats,
        final_lqr_state=final_lqr_state,
        history=tuple(history),
        max_abs_lqr_wheel_torque=max_abs_lqr_wheel_torque,
        max_abs_lqr_pitch_torque=max_abs_lqr_pitch_torque,
        max_abs_left_wheel_speed=max_abs_left_wheel_speed,
        max_abs_right_wheel_speed=max_abs_right_wheel_speed,
        final_left_wheel_speed=final_left_wheel_speed,
        final_right_wheel_speed=final_right_wheel_speed,
        final_base_height=float(data.qpos[2]) if model.nq >= 3 else math.nan,
        max_base_height=max_base_height,
        min_left_contact_force=0.0 if math.isinf(min_left_contact_force) else min_left_contact_force,
        min_right_contact_force=0.0 if math.isinf(min_right_contact_force) else min_right_contact_force,
        airborne_steps=airborne_steps,
        first_airborne_time=first_airborne_time,
        last_airborne_time=last_airborne_time,
        final_operating_point=final_operating_point,
        final_data=_copy_data(mujoco, model, data) if copy_final_data else data,
    )


def _check_lower_loop_constraints(
    mujoco,
    model,
    steps: int,
    use_virtual_rod: bool,
    lock_base: bool,
    left_target: tuple[float, float],
    right_target: tuple[float, float],
    virtual_rod_control: str,
    leg_branch: str,
    ik_search_radius: float,
    ik_search_samples: int,
    motor_kp: float,
    motor_kd: float,
    motor_ki: float,
    motor_length_force_ff: float,
    motor_length_integral_limit: float,
    motor_length_force_rate_limit: float,
    theta_kp: float,
    theta_kd: float,
    joint_kd: float,
    theta_pitch_ff: float,
    lqr_test: bool,
    lqr_gain_scale: float,
    lqr_k: np.ndarray,
    lqr_x0: np.ndarray,
    lqr_u0: np.ndarray,
    lqr_auto_design: bool,
    lqr_control_period_steps: int,
    lqr_x_reference: float,
    lqr_x_source: str,
    lqr_x_outer_kp: float,
    lqr_x_outer_max_v: float,
    lqr_wheel_sign: float,
    lqr_pitch_sign: float,
    lqr_t_limit: float,
    lqr_tp_limit: float,
    lqr_output_rate_limit: float,
    lqr_output_lowpass_hz: float,
    wheel_ctrl_deadzone: float,
) -> ConstraintCheckResult:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    if lqr_test and lqr_auto_design:
        data = _prepare_lqr_operating_point(
            mujoco,
            model,
            left_target,
            right_target,
            leg_branch,
            ik_search_radius,
            ik_search_samples,
        )
    max_left_error = 0.0
    max_right_error = 0.0
    max_left_branch_violation = 0.0
    max_right_branch_violation = 0.0
    previous_wheel_torque = 0.0
    previous_pitch_torque = 0.0
    length_force_delta = 0.0
    ik_target_cache: dict[tuple[str, float, float, str, float, int], tuple[float, float]] = {}
    vmc_memory: dict[str, VmcSideMemory] = {}
    odometry = SimulatedOdometry()
    for step in range(steps):
        _update_simulated_odometry(mujoco, model, data, odometry)
        if lock_base:
            _lock_base_to_initial(mujoco, model, data)
        if use_virtual_rod:
            wheel_torque = previous_wheel_torque
            pitch_torque = previous_pitch_torque
            if lqr_test and step % lqr_control_period_steps == 0:
                effective_x_reference = float(odometry.position) - float(lqr_x0[2])
                wheel_torque, pitch_torque, length_force_delta, _, _ = _lqr_middle_control(
                    mujoco,
                    model,
                    data,
                    effective_x_reference,
                    lqr_x_source,
                    lqr_gain_scale,
                    lqr_k,
                    lqr_x0,
                    lqr_u0,
                    lqr_wheel_sign,
                    lqr_pitch_sign,
                    lqr_t_limit,
                    lqr_tp_limit,
                    lqr_x_outer_kp,
                    lqr_x_outer_max_v,
                    odometry=odometry,
                )
                control_dt = float(model.opt.timestep) * lqr_control_period_steps
                wheel_torque = _lowpass_value(
                    wheel_torque,
                    previous_wheel_torque,
                    lqr_output_lowpass_hz,
                    control_dt,
                )
                pitch_torque = _lowpass_value(
                    pitch_torque,
                    previous_pitch_torque,
                    lqr_output_lowpass_hz,
                    control_dt,
                )
                if lqr_output_rate_limit > 0.0:
                    max_delta = lqr_output_rate_limit * float(model.opt.timestep)
                    wheel_torque = float(
                        np.clip(
                            wheel_torque,
                            previous_wheel_torque - max_delta,
                            previous_wheel_torque + max_delta,
                        )
                    )
                    pitch_torque = float(
                        np.clip(
                            pitch_torque,
                            previous_pitch_torque - max_delta,
                            previous_pitch_torque + max_delta,
                        )
                    )
                previous_wheel_torque = wheel_torque
                previous_pitch_torque = pitch_torque
            pitch_for_theta_ff = _base_pitch_from_qpos(data.qpos)
            step_left_target = _apply_theta_pitch_feedforward(left_target, pitch_for_theta_ff, theta_pitch_ff)
            step_right_target = _apply_theta_pitch_feedforward(right_target, pitch_for_theta_ff, theta_pitch_ff)
            effective_theta_kp = 0.0 if lqr_test else theta_kp
            effective_theta_kd = 0.0 if lqr_test else theta_kd
            _virtual_rod_ik_ctrl(
                mujoco,
                model,
                data,
                step_left_target,
                step_right_target,
                virtual_rod_control,
                leg_branch,
                ik_search_radius,
                ik_search_samples,
                motor_kp,
                motor_kd,
                effective_theta_kp,
                effective_theta_kd,
                joint_kd,
                ik_target_cache,
                theta_force_offset=pitch_torque,
                length_force_ff=motor_length_force_ff + length_force_delta,
                length_ki=motor_ki,
                length_integral_limit=motor_length_integral_limit,
                length_force_rate_limit=motor_length_force_rate_limit,
                vmc_memory=vmc_memory,
            )
            if lqr_test:
                _apply_additive_wheel_torque_ctrl(
                    mujoco,
                    model,
                    data,
                    wheel_torque,
                    wheel_ctrl_deadzone,
                )
        else:
            data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        _assert_finite("qpos", data.qpos)
        _assert_finite("qvel", data.qvel)
        max_left_error = max(max_left_error, _lower_loop_error(mujoco, model, data, "left"))
        max_right_error = max(max_right_error, _lower_loop_error(mujoco, model, data, "right"))
        left_branch = _compute_leg_branch_metrics(mujoco, model, data, "left")
        right_branch = _compute_leg_branch_metrics(mujoco, model, data, "right")
        max_left_branch_violation = max(max_left_branch_violation, left_branch.violation)
        max_right_branch_violation = max(max_right_branch_violation, right_branch.violation)
    final_left_branch = _compute_leg_branch_metrics(mujoco, model, data, "left")
    final_right_branch = _compute_leg_branch_metrics(mujoco, model, data, "right")
    return ConstraintCheckResult(
        steps=steps,
        max_left_error=max_left_error,
        max_right_error=max_right_error,
        final_left_error=_lower_loop_error(mujoco, model, data, "left"),
        final_right_error=_lower_loop_error(mujoco, model, data, "right"),
        max_left_branch_violation=max_left_branch_violation,
        max_right_branch_violation=max_right_branch_violation,
        final_left_branch_violation=final_left_branch.violation,
        final_right_branch_violation=final_right_branch.violation,
        final_base_height=float(data.qpos[2]) if model.nq >= 3 else math.nan,
    )



