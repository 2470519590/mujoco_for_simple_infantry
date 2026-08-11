"""顶层 smoke 运行编排。"""

from __future__ import annotations

import math
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from .io.cli import build_parser
from .core.config import DEFAULT_CONFIG_PATH, RunConfig, load_yaml_defaults, runtime_control_config
from .core.constants import (
    DEFAULT_LQR_K,
    LOCKED_EQUILIBRIUM_EVAL_STEPS,
    LOCKED_EQUILIBRIUM_FL0,
    LOCKED_EQUILIBRIUM_QPOS,
    LOCKED_EQUILIBRIUM_FL0_SCALE,
    LOCKED_EQUILIBRIUM_L0,
    LOCKED_EQUILIBRIUM_LENGTH_KD,
    LOCKED_EQUILIBRIUM_LENGTH_KP,
    LOCKED_EQUILIBRIUM_PITCH_KD,
    LOCKED_EQUILIBRIUM_PITCH_KP,
    LOCKED_EQUILIBRIUM_STEPS,
    LOCKED_EQUILIBRIUM_THETA,
    LOCKED_EQUILIBRIUM_THETA_KD,
    LOCKED_EQUILIBRIUM_THETA_KP,
    LOCKED_EQUILIBRIUM_TP_BIAS,
    LOCKED_EQUILIBRIUM_WHEEL_COM_KP,
    LOCKED_EQUILIBRIUM_WHEEL_DAMPING,
    LOCKED_LQR_CONTROL_PERIOD_STEPS,
    LOCKED_LQR_DESIGN_STEPS,
    LOCKED_LQR_K,
    LOCKED_LQR_U0,
    LOCKED_LQR_X0,
    PROJECT_ROOT,
)
from .experiments.equilibrium import (
    _equilibrium_score,
    _print_static_operating_point_sample,
    _run_equilibrium_search,
    _run_equilibrium_static_pose_check,
)
from .experiments.fivebar_checks import _run_fivebar_jacobian_check, _run_fivebar_kinematics_check
from .experiments.fl_tests import _run_fl_channel_test, _run_fl_pulse_test
from .experiments.manual_drive import ManualDriveInput
from .control.lqr_design import _compute_lqr_state, _design_lqr_gain
from .control.lqr import lqr_state_vector as _lqr_state_vector
from .control.length_schedule import LengthSchedule, load_length_schedule
from .control.ik import _branch_aware_ik_targets
from .control.vmc import _drive_joint_position_ctrl
from .control.turning import turn_rate_magnitude
from .model.kinematics import compute_virtual_leg_state as _compute_virtual_leg_state
from .model_smoke import (
    _build_virtual_rod_targets,
    _estimated_support_force_per_leg,
    _format_matrix_rows,
    _print_virtual_leg_state,
    _probe_actuators,
    _run_pd_hold,
    _scan_virtual_rod_dynamic,
    _scan_virtual_rod_geometry,
    _visualize_pd_hold,
)
from .core.mujoco_utils import (
    iter_actuator_names as _iter_actuator_names,
    iter_joint_names as _iter_joint_names,
    joint_for_actuator as _joint_for_actuator,
    load_mujoco as _load_mujoco,
    lock_base_to_qpos as _lock_base_to_qpos,
    step_with_ctrl as _step_with_ctrl,
)
from .io.output import (
    plot_lqr_debug_history as _plot_lqr_debug_history,
    plot_roll_length_history as _plot_roll_length_history,
    plot_turning_history as _plot_turning_history,
    resolve_output_path as _resolve_output_path,
)
from .core.types import EquilibriumSearchResult, LqrDesignResult
from .experiments.virtual_rod import _check_lower_loop_constraints, _run_virtual_rod_test
from .experiments.viewer import MujocoViewerObserver
from .model.mechanics import support_force_scale_for_length


def _scheduled_initial_pose(
    mujoco,
    model,
    left_target: tuple[float, float],
    right_target: tuple[float, float],
    leg_branch: str,
    ik_search_radius: float,
    ik_search_samples: int,
) -> object:
    """Build an initial pose at the commanded scheduled leg length."""
    data = mujoco.MjData(model)
    data.qpos[:] = LOCKED_EQUILIBRIUM_QPOS
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    target_length = 0.5 * (float(left_target[0]) + float(right_target[0]))
    data.qpos[2] += target_length - LOCKED_EQUILIBRIUM_L0
    mujoco.mj_forward(model, data)
    base_qpos = data.qpos[:7].copy()

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
    for _ in range(160):
        _lock_base_to_qpos(mujoco, model, data, base_qpos)
        data.ctrl[:] = 0.0
        _drive_joint_position_ctrl(mujoco, model, data, "left", left_front_target, left_rear_target, 40.0, 1.4)
        _drive_joint_position_ctrl(mujoco, model, data, "right", right_front_target, right_rear_target, 40.0, 1.4)
        mujoco.mj_step(model, data)
    _lock_base_to_qpos(mujoco, model, data, base_qpos)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    return data


def _mesh_asset(name: str, vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]) -> str:
    vertex_text = " ".join(f"{x:.6g} {y:.6g} {z:.6g}" for x, y, z in vertices)
    face_text = " ".join(f"{a} {b} {c}" for a, b, c in faces)
    return f'    <mesh name="{name}" vertex="{vertex_text}" face="{face_text}"/>\n'


def _triangular_ramp_mesh(length: float, width: float, height: float) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    half_length = 0.5 * length
    half_width = 0.5 * width
    vertices = [
        (-half_length, -half_width, 0.0),
        (-half_length, half_width, 0.0),
        (half_length, -half_width, 0.0),
        (half_length, half_width, 0.0),
        (half_length, -half_width, height),
        (half_length, half_width, height),
    ]
    faces = [
        (0, 2, 3), (0, 3, 1),
        (2, 4, 5), (2, 5, 3),
        (0, 4, 2), (1, 3, 5),
        (0, 4, 5), (0, 5, 1),
    ]
    return vertices, faces


def _trapezoid_ramp_mesh(length: float, width: float, height: float, ramp_length: float) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    half_length = 0.5 * length
    half_width = 0.5 * width
    x0 = -half_length
    x1 = -half_length + ramp_length
    x2 = half_length - ramp_length
    x3 = half_length
    vertices = [
        (x0, -half_width, 0.0),
        (x0, half_width, 0.0),
        (x1, -half_width, height),
        (x1, half_width, height),
        (x2, -half_width, height),
        (x2, half_width, height),
        (x3, -half_width, 0.0),
        (x3, half_width, 0.0),
    ]
    faces = [
        (0, 6, 7), (0, 7, 1),
        (0, 2, 3), (0, 3, 1),
        (2, 4, 5), (2, 5, 3),
        (4, 6, 7), (4, 7, 5),
        (0, 4, 6), (0, 2, 4),
        (1, 7, 5), (1, 5, 3),
    ]
    return vertices, faces


def _roll_test_model_xml(model_path: Path) -> str:
    """Return a roll-test scene with low single-wheel ramps and one full-width ramp."""
    xml_text = model_path.read_text(encoding="utf-8")
    small_vertices, small_faces = _triangular_ramp_mesh(length=0.84, width=0.60, height=0.06)
    wide_vertices, wide_faces = _trapezoid_ramp_mesh(length=1.20, width=1.20, height=0.12, ramp_length=0.32)
    asset_xml = (
        '    <material name="roll_ramp_mat" rgba="0.72 0.52 0.20 1" specular="0.0" shininess="0.0"/>\n'
        + _mesh_asset("roll_small_tri_ramp_mesh", small_vertices, small_faces)
        + _mesh_asset("roll_wide_trap_ramp_mesh", wide_vertices, wide_faces)
    )
    ramp_geom_template = (
        '    <geom name="{name}" type="mesh" mesh="{mesh}" pos="{x:.6g} {y:.6g} 0" '
        'material="roll_ramp_mat" friction="1.2 0.02 0.001" conaffinity="3"/>\n'
    )
    world_xml = (
        ramp_geom_template.format(name="roll_left_small_tri_ramp", mesh="roll_small_tri_ramp_mesh", x=2.50, y=0.255)
        + ramp_geom_template.format(name="roll_right_small_tri_ramp", mesh="roll_small_tri_ramp_mesh", x=3.70, y=-0.255)
        + ramp_geom_template.format(name="roll_wide_trap_ramp", mesh="roll_wide_trap_ramp_mesh", x=5.10, y=0.0)
    )
    xml_text = xml_text.replace("  </asset>", asset_xml + "  </asset>", 1)
    xml_text = xml_text.replace(
        '    <geom name="floor" type="plane" size="20 20 0.1" material="floor_mat" conaffinity="3"/>\n',
        '    <geom name="floor" type="plane" size="20 20 0.1" material="floor_mat" conaffinity="3"/>\n' + world_xml,
        1,
    )
    return xml_text


def _flight_test_model_xml(model_path: Path) -> str:
    """Return a flight-test scene with one high full-width launch ramp."""
    xml_text = model_path.read_text(encoding="utf-8")
    ramp_vertices, ramp_faces = _triangular_ramp_mesh(length=1.80, width=1.40, height=0.32)
    asset_xml = (
        '    <material name="flight_ramp_mat" rgba="0.62 0.48 0.22 1" specular="0.0" shininess="0.0"/>\n'
        + _mesh_asset("flight_full_width_ramp_mesh", ramp_vertices, ramp_faces)
    )
    world_xml = (
        '    <geom name="flight_full_width_ramp" type="mesh" mesh="flight_full_width_ramp_mesh" '
        'pos="3.8 0 0" material="flight_ramp_mat" friction="1.2 0.02 0.001" conaffinity="3"/>\n'
    )
    xml_text = xml_text.replace("  </asset>", asset_xml + "  </asset>", 1)
    xml_text = xml_text.replace(
        '    <geom name="floor" type="plane" size="20 20 0.1" material="floor_mat" conaffinity="3"/>\n',
        '    <geom name="floor" type="plane" size="20 20 0.1" material="floor_mat" conaffinity="3"/>\n' + world_xml,
        1,
    )
    return xml_text


def _print_low_length_stage_diagnostics(history) -> None:
    """Print the competing VMC torque paths during the 4-7 s low-leg stage."""
    samples = [sample for sample in history if 4.0 <= sample.time_s < 7.0]
    if not samples:
        return
    print("  leg_height_low_stage_diagnostics:")
    for side in ("left", "right"):
        force = np.array([getattr(sample, f"{side}_length_force") for sample in samples])
        support = np.array(
            [
                [getattr(sample, f"{side}_front_tau"), getattr(sample, f"{side}_rear_tau")]
                for sample in samples
            ]
        )
        guard = np.array(
            [
                [
                    getattr(sample, f"{side}_guard_tau_front"),
                    getattr(sample, f"{side}_guard_tau_rear"),
                ]
                for sample in samples
            ]
        )
        theta = np.array(
            [
                [
                    getattr(sample, f"{side}_theta_tau_front"),
                    getattr(sample, f"{side}_theta_tau_rear"),
                ]
                for sample in samples
            ]
        )
        guard_error = max(getattr(sample, f"{side}_branch_guard_error") for sample in samples)
        print(
            f"    {side}: F_l=[{force.min():.6g}, {force.max():.6g}] N, "
            f"total_joint_tau_max=[{np.max(np.abs(support[:, 0])):.6g}, {np.max(np.abs(support[:, 1])):.6g}] N*m, "
            f"guard_tau_max=[{np.max(np.abs(guard[:, 0])):.6g}, {np.max(np.abs(guard[:, 1])):.6g}] N*m, "
            f"theta_tau_max=[{np.max(np.abs(theta[:, 0])):.6g}, {np.max(np.abs(theta[:, 1])):.6g}] N*m, "
            f"guard_error_max={guard_error:.6g}"
        )


def run_smoke(config: RunConfig) -> int:
    mujoco = _load_mujoco()
    runtime_controls = runtime_control_config(config)
    if config.flight_test:
        model = mujoco.MjModel.from_xml_string(_flight_test_model_xml(Path(config.model_path)))
    elif config.roll_test:
        model = mujoco.MjModel.from_xml_string(_roll_test_model_xml(Path(config.model_path)))
    else:
        model = mujoco.MjModel.from_xml_path(str(config.model_path))
    length_schedule: LengthSchedule | None = None
    if config.length_schedule:
        schedule_path = Path(config.length_schedule_path)
        if not schedule_path.is_absolute():
            schedule_path = PROJECT_ROOT / schedule_path
        length_schedule = load_length_schedule(schedule_path)
        print(
            f"length_schedule: enabled, path={schedule_path}, "
            f"range={length_schedule.min_length:.3f}..{length_schedule.max_length:.3f} m"
        )
    else:
        print("length_schedule: disabled, using locked/manual K and F_l0")
    behavior_passed = True
    estimated_support_force_per_leg = _estimated_support_force_per_leg(model)
    if config.virtual_rod_gravity_comp_scale is not None:
        config.virtual_rod_length_force_ff = config.virtual_rod_gravity_comp_scale * estimated_support_force_per_leg

    print("MuJoCo smoke check")
    print(f"config.model_path: {config.model_path}")
    print(f"mujoco_version: {mujoco.__version__}")
    print(f"nq: {model.nq}")
    print(f"nv: {model.nv}")
    print(f"nu: {model.nu}")
    print(f"njnt: {model.njnt}")
    print(f"timestep: {model.opt.timestep}")
    print(f"estimated_support_force_per_leg: {estimated_support_force_per_leg:.6g}")
    if config.impact_level is not None:
        force = 40.0 if config.impact_level == "small" else 80.0
        print(f"impact: {config.impact_level}, +X {force:.0f} N, t=3.0 s, duration=0.15 s")
    if config.speed_profile is not None:
        speed = {"low": 1.00, "medium": 2.00, "high": 3.00}[config.speed_profile]
        cruise = 3.6 if config.speed_profile == "high" else 6.0
        print(f"config.speed_profile: {config.speed_profile}, peak={speed:.2f} m/s, ramp=0.75 s, cruise={cruise:.1f} s")
    if config.turn_direction is not None:
        print(f"turn: {config.turn_direction}, speed={config.turn_speed}, desired_yaw_rate={turn_rate_magnitude(config.turn_speed):.6g} rad/s")
    if config.turn_test:
        print(f"config.turn_test: speed={config.turn_speed}, peak_yaw_rate={turn_rate_magnitude(config.turn_speed):.6g} rad/s")
    if config.leg_length_sine_test:
        print(
            f"leg_length_sine_test: range={float(config.minimum_leg_length):.3f}.."
            f"{float(config.maximum_leg_length):.3f} m, period={float(config.leg_length_sine_period):.3g} s"
        )
    if config.roll_test:
        print("roll_test: medium forward; left/right low single-wheel ramps x=2.5/3.7 m, full-width jump ramp x=5.1 m")
    if config.flight_test:
        print(
            f"flight_test: {config.flight_test_speed}-speed full-width launch ramp x=3.8 m, "
            "length=1.80 m, width=1.40 m, height=0.32 m"
        )
        print(
            "flight_detection: airborne when both wheel normal forces are below "
            f"{float(config.flight_airborne_force_threshold):.6g} N"
        )
    if config.jump_test:
        print(
            "jump_test: stand at nominal leg length, crouch to 0.18 m at 1.0 s through VMC, "
            "extend after measured crouch completion; flight detection remains enabled"
        )
    if config.manual_drive:
        print("manual_drive: medium speed; arrow up/down forward/backward, arrow left/right yaw; close viewer to exit")
    print(
        f"roll_control: reference={runtime_controls.roll_reference:.6g} rad, "
        f"force_kp={runtime_controls.roll_force_kp:.6g} N/rad"
    )
    if config.turn_test:
        print(f"yaw_turn_pd: kp={config.yaw_turn_kp:.6g}, kd={config.yaw_turn_kd:.6g}")
        print(f"leg_sync_pd: kp={config.leg_sync_kp:.6g}, kd={config.leg_sync_kd:.6g}")
    print()

    if config.print_virtual_leg or config.virtual_rod_test:
        _print_virtual_leg_state(mujoco, model)
        print()

    if config.scan_virtual_rod:
        _scan_virtual_rod_geometry(mujoco, model, config.scan_virtual_rod_sample)
        print()

    if config.scan_virtual_rod_dynamic:
        _scan_virtual_rod_dynamic(mujoco, model, config.scan_virtual_rod_sample, 250, 30.0, 1.2)
        print()

    if config.fl_channel_test:
        _run_fl_channel_test(mujoco, model, config.fl_channel_forces, config.fl_channel_steps)
        print()

    if config.fl_pulse_test:
        _run_fl_pulse_test(
            mujoco,
            model,
            config.fl_pulse_base_force,
            config.fl_pulse_delta_force,
            config.fl_pulse_settle_steps,
            config.fl_pulse_steps,
        )
        print()

    if config.fivebar_kinematics_check:
        _run_fivebar_kinematics_check(
            mujoco,
            model,
            config.fivebar_kinematics_l_slices,
            config.fivebar_kinematics_theta_refs,
        )
        print()

    if config.fivebar_jacobian_check:
        _run_fivebar_jacobian_check(
            mujoco,
            model,
            config.fivebar_jacobian_l_slices,
            config.fivebar_jacobian_theta_refs,
            config.fivebar_jacobian_force_scale,
            config.fivebar_jacobian_load_steps,
        )
        print()

    if config.equilibrium_static_pose_check:
        _run_equilibrium_static_pose_check(
            mujoco,
            model,
            config.equilibrium_l_slices,
            config.equilibrium_theta_refs,
            config.equilibrium_fl0_scales,
            config.leg_branch,
            config.ik_search_radius,
            config.ik_search_samples,
        )
        print()

    equilibrium_results: tuple[EquilibriumSearchResult, ...] = ()
    best_equilibrium: EquilibriumSearchResult | None = None
    if config.equilibrium_search:
        equilibrium_results = _run_equilibrium_search(
            mujoco,
            model,
            config.equilibrium_init_modes,
            config.equilibrium_l_slices,
            config.equilibrium_theta_refs,
            config.equilibrium_fl0_scales,
            config.equilibrium_tp_biases,
            config.equilibrium_wheel_com_kps,
            config.equilibrium_wheel_dampings,
            config.equilibrium_init_drop_steps,
            config.equilibrium_steps,
            config.equilibrium_eval_steps,
            config.equilibrium_length_kp,
            config.equilibrium_length_kd,
            config.equilibrium_theta_kp,
            config.equilibrium_theta_kd,
            config.equilibrium_pitch_kp,
            config.equilibrium_pitch_kd,
            config.lqr_t_limit,
            config.lqr_tp_limit,
            45.0,
            config.leg_branch,
            config.ik_search_radius,
            config.ik_search_samples,
        )
        best_equilibrium = min(equilibrium_results, key=_equilibrium_score) if equilibrium_results else None
        print()

    if config.diagnostics_only:
        return 0

    left_target, right_target = _build_virtual_rod_targets(
        mujoco,
        model,
        config.virtual_rod_length_delta,
        config.virtual_rod_theta_target,
        config.left_rod_length,
        config.right_rod_length,
        config.left_rod_theta,
        config.right_rod_theta,
    )
    lqr_design_result: LqrDesignResult | None = None
    lqr_operating_data = None
    lqr_operating_u0 = None
    if config.use_locked_equilibrium:
        if config.length_schedule:
            lqr_operating_data = _scheduled_initial_pose(
                mujoco,
                model,
                left_target,
                right_target,
                config.leg_branch,
                config.ik_search_radius,
                config.ik_search_samples,
            )
        else:
            lqr_operating_data = mujoco.MjData(model)
            lqr_operating_data.qpos[:] = LOCKED_EQUILIBRIUM_QPOS
            lqr_operating_data.qvel[:] = 0.0
            lqr_operating_data.ctrl[:] = 0.0
            mujoco.mj_forward(model, lqr_operating_data)
        lqr_operating_u0 = LOCKED_LQR_U0.copy()
        config.lqr_x0 = LOCKED_LQR_X0.copy()
        config.lqr_u0 = LOCKED_LQR_U0.copy()
        if not config.length_force_ff_is_explicit:
            config.virtual_rod_length_force_ff = LOCKED_EQUILIBRIUM_FL0
        if config.length_schedule:
            print("lqr_startup_pose: scheduled initial leg length; runtime K/F_l0 come from length_schedule")
        else:
            print("lqr_operating_point: locked 0.35 m equilibrium")
    if config.lqr_test and config.lqr_auto_design:
        if config.lqr_use_equilibrium_operating_point:
            raise RuntimeError("scheduled equilibrium LQR is disabled while restoring the locked balance baseline")
        else:
            lqr_design_result = _design_lqr_gain(
                mujoco,
                model,
                left_target,
                right_target,
                config.leg_branch,
                config.ik_search_radius,
                config.ik_search_samples,
                config.virtual_rod_length_kp,
                config.virtual_rod_length_kd,
                config.virtual_rod_length_ki,
                config.virtual_rod_length_force_ff,
                config.virtual_rod_length_integral_limit,
                config.virtual_rod_length_force_rate_limit,
                config.virtual_rod_joint_kd,
                config.lqr_q_diag,
                config.lqr_r_diag,
                config.lqr_state_eps,
                config.lqr_input_eps,
                config.lqr_design_steps,
                config.lqr_x_source,
                config.lqr_wheel_sign,
                config.lqr_pitch_sign,
                leg_control_enabled=config.virtual_rod_control != "off",
                branch_guard_enabled=bool(config.leg_branch_guard_enabled),
                operating_data=lqr_operating_data,
                operating_u0=lqr_operating_u0,
            )
            config.lqr_k = lqr_design_result.k_matrix
            config.lqr_x0 = _lqr_state_vector(lqr_design_result.operating_state)
            config.lqr_u0 = lqr_design_result.operating_u0.copy()

    print("joints:")
    for joint_name in _iter_joint_names(mujoco, model):
        print(f"  - {joint_name}")
    print()

    print("actuators:")
    for actuator_id, actuator_name in enumerate(_iter_actuator_names(mujoco, model)):
        _, joint_name = _joint_for_actuator(mujoco, model, actuator_id)
        low, high = model.actuator_ctrlrange[actuator_id]
        gear = model.actuator_gear[actuator_id, 0]
        print(
            f"  - {actuator_name}: joint={joint_name}, "
            f"ctrlrange=[{low:.3g}, {high:.3g}], gear={gear:.3g}"
        )
    print()

    zero_ctrl = np.zeros(model.nu)
    qpos, qvel = _step_with_ctrl(mujoco, model, config.zero_steps, zero_ctrl)
    print(f"zero_input_steps: {config.zero_steps}")
    print(f"zero_input_qpos_norm: {np.linalg.norm(qpos):.6g}")
    print(f"zero_input_qvel_norm: {np.linalg.norm(qvel):.6g}")
    print()

    probes = _probe_actuators(mujoco, model, config.probe_steps, config.ctrl)
    print(f"actuator_probe_steps: {config.probe_steps}")
    for probe in probes:
        sign_status = "opposite" if probe.opposite_sign else "not-confirmed"
        print(
            f"  - {probe.actuator}: joint={probe.joint}, config.ctrl=+/-{probe.ctrl:.3g}, "
            f"qvel(+)= {probe.positive_velocity:.6g}, "
            f"qvel(-)= {probe.negative_velocity:.6g}, sign={sign_status}"
        )

    not_confirmed = [probe.actuator for probe in probes if not probe.opposite_sign]
    if not_confirmed:
        print()
        print("warning: some actuator direction probes were finite but not sign-confirmed:")
        for actuator_name in not_confirmed:
            print(f"  - {actuator_name}")

    print()
    pd_result = _run_pd_hold(mujoco, model, config.pd_hold_steps, config.pd_kp, config.pd_kd)
    print("pd_hold:")
    print(f"  steps: {pd_result.steps}")
    print(f"  kp: {pd_result.kp:.6g}")
    print(f"  kd: {pd_result.kd:.6g}")
    print(f"  final_base_height: {pd_result.final_base_height:.6g}")
    print(f"  final_qpos_norm: {pd_result.final_qpos_norm:.6g}")
    print(f"  final_qvel_norm: {pd_result.final_qvel_norm:.6g}")
    print(f"  max_abs_ctrl: {pd_result.max_abs_ctrl:.6g}")
    print(f"  saturated_steps: {pd_result.saturated_steps}")
    print(f"  max_abs_joint_error: {pd_result.max_abs_joint_error:.6g}")
    print(f"  final_abs_joint_error: {pd_result.final_abs_joint_error:.6g}")
    print("  note: finite PD hold smoke only; not a stable-controller claim")

    if config.check_constraints:
        constraint_result = _check_lower_loop_constraints(
            mujoco,
            model,
            config.constraint_steps,
            config.virtual_rod_test,
            config.virtual_rod_lock_base,
            left_target,
            right_target,
            config.virtual_rod_control,
            config.leg_branch,
            config.ik_search_radius,
            config.ik_search_samples,
            config.virtual_rod_length_kp,
            config.virtual_rod_length_kd,
            config.virtual_rod_length_ki,
            config.virtual_rod_length_force_ff,
            config.virtual_rod_length_integral_limit,
            config.virtual_rod_length_force_rate_limit,
            config.virtual_rod_theta_kp,
            config.virtual_rod_theta_kd,
            config.virtual_rod_joint_kd,
            config.virtual_rod_theta_pitch_ff,
            config.lqr_test,
            config.lqr_gain_scale,
            config.lqr_k,
            config.lqr_x0,
            config.lqr_u0,
            config.lqr_auto_design,
            config.lqr_control_period_steps,
            config.lqr_x_reference,
            config.lqr_x_source,
            config.lqr_x_outer_kp,
            config.lqr_x_outer_max_v,
            config.lqr_wheel_sign,
            config.lqr_pitch_sign,
            config.lqr_t_limit,
            config.lqr_tp_limit,
            config.lqr_output_rate_limit,
            config.lqr_output_lowpass_hz,
            config.wheel_ctrl_deadzone,
        )
        print()
        print("lower_loop_constraint_check:")
        print(f"  steps: {constraint_result.steps}")
        print(f"  max_left_error_m: {constraint_result.max_left_error:.6g}")
        print(f"  max_right_error_m: {constraint_result.max_right_error:.6g}")
        print(f"  final_left_error_m: {constraint_result.final_left_error:.6g}")
        print(f"  final_right_error_m: {constraint_result.final_right_error:.6g}")
        print(f"  max_left_branch_violation: {constraint_result.max_left_branch_violation:.6g}")
        print(f"  max_right_branch_violation: {constraint_result.max_right_branch_violation:.6g}")
        print(f"  final_left_branch_violation: {constraint_result.final_left_branch_violation:.6g}")
        print(f"  final_right_branch_violation: {constraint_result.final_right_branch_violation:.6g}")
        print(f"  final_base_height: {constraint_result.final_base_height:.6g}")

    if config.virtual_rod_test:
        manual_drive_input = ManualDriveInput() if config.manual_drive else None
        viewer_observer = (
            MujocoViewerObserver(
                mujoco,
                model,
                config.realtime,
                key_callback=manual_drive_input.key_callback if manual_drive_input is not None else None,
                overlay_provider=manual_drive_input.overlay_text if manual_drive_input is not None else None,
            )
            if config.visualize
            else None
        )
        rollout_steps = config.virtual_rod_steps
        if viewer_observer is not None:
            print("config.visualize: opening MuJoCo viewer for the same virtual-rod rollout")
        try:
            virtual_result = _run_virtual_rod_test(
                mujoco,
                model,
                rollout_steps,
                config.virtual_rod_lock_base,
                left_target,
                right_target,
                config.virtual_rod_control,
                config.leg_branch,
                config.ik_search_radius,
                config.ik_search_samples,
                config.virtual_rod_length_kp,
                config.virtual_rod_length_kd,
                config.virtual_rod_length_ki,
                config.virtual_rod_length_force_ff,
                config.virtual_rod_length_integral_limit,
                config.virtual_rod_length_force_rate_limit,
                config.virtual_rod_theta_kp,
                config.virtual_rod_theta_kd,
                config.virtual_rod_joint_kd,
                config.virtual_rod_theta_pitch_ff,
                config.lqr_test,
                config.lqr_gain_scale,
                config.lqr_k,
                config.lqr_x0,
                config.lqr_u0,
                config.lqr_auto_design,
                config.lqr_control_period_steps,
                config.lqr_x_reference,
                config.lqr_x_source,
                config.lqr_x_outer_kp,
                config.lqr_x_outer_max_v,
                config.lqr_wheel_sign,
                config.lqr_pitch_sign,
                config.lqr_t_limit,
                config.lqr_tp_limit,
                config.landing_hold_t_limit,
                config.lqr_output_rate_limit,
                config.lqr_output_lowpass_hz,
                config.wheel_ctrl_deadzone,
                config.history_sample_interval,
                initial_data=lqr_operating_data,
                impact_level=config.impact_level,
                speed_profile=config.speed_profile,
                turn_direction=config.turn_direction,
                turn_speed=config.turn_speed,
                turn_test=config.turn_test,
                manual_drive_input=manual_drive_input,
                leg_sync_kp=config.leg_sync_kp,
                leg_sync_kd=config.leg_sync_kd,
                yaw_turn_kp=config.yaw_turn_kp,
                yaw_turn_kd=config.yaw_turn_kd,
                leg_height_test=config.leg_height_test,
                leg_height_levels=tuple(float(value) for value in config.leg_height_levels),
                leg_length_sine_test=bool(config.leg_length_sine_test),
                leg_length_sine_period=float(config.leg_length_sine_period),
                jump_test=bool(config.jump_test),
                branch_guard_enabled=bool(config.leg_branch_guard_enabled),
                minimum_leg_length=float(config.minimum_leg_length),
                maximum_leg_length=float(config.maximum_leg_length),
                length_schedule=length_schedule,
                startup_ramp_seconds=float(config.startup_ramp_seconds),
                flight_detection_enabled=bool(config.flight_detection_enabled),
                flight_airborne_force_threshold=float(config.flight_airborne_force_threshold),
                flight_airborne_confirm_seconds=float(config.flight_airborne_confirm_seconds),
                flight_airborne_rearm_seconds=float(config.flight_airborne_rearm_seconds),
                runtime_controls=runtime_controls,
                on_initialized=viewer_observer.start if viewer_observer is not None else None,
            )
        finally:
            if viewer_observer is not None:
                viewer_observer.close()
        print()
        print("config.virtual_rod_test:")
        print(f"  steps: {virtual_result.steps}")
        print(f"  lock_base: {virtual_result.lock_base}")
        print(f"  control: {config.virtual_rod_control}")
        print(f"  leg_branch_guard_enabled: {config.leg_branch_guard_enabled}")
        print(
            f"  left_target: l={virtual_result.left_target_length:.6g}, "
            f"theta={virtual_result.left_target_theta:.6g}"
        )
        print(
            f"  left_final: l={virtual_result.final_left_length:.6g}, "
            f"theta={virtual_result.final_left_theta:.6g}"
        )
        print(
            f"  right_target: l={virtual_result.right_target_length:.6g}, "
            f"theta={virtual_result.right_target_theta:.6g}"
        )
        print(
            f"  right_final: l={virtual_result.final_right_length:.6g}, "
            f"theta={virtual_result.final_right_theta:.6g}"
        )
        print(f"  max_length_error: {virtual_result.max_length_error:.6g}")
        print(f"  max_theta_error: {virtual_result.max_theta_error:.6g}")
        print(f"  max_left_branch_violation: {virtual_result.max_left_branch_violation:.6g}")
        print(f"  max_right_branch_violation: {virtual_result.max_right_branch_violation:.6g}")
        print(f"  final_left_branch_violation: {virtual_result.final_left_branch_violation:.6g}")
        print(f"  final_right_branch_violation: {virtual_result.final_right_branch_violation:.6g}")
        print(f"  max_abs_ctrl: {virtual_result.max_abs_ctrl:.6g}")
        print(f"  saturated_steps: {virtual_result.saturated_steps}")
        print(f"  max_abs_left_wheel_speed_rad_s: {virtual_result.max_abs_left_wheel_speed:.6g}")
        print(f"  max_abs_right_wheel_speed_rad_s: {virtual_result.max_abs_right_wheel_speed:.6g}")
        print(f"  final_left_wheel_speed_rad_s: {virtual_result.final_left_wheel_speed:.6g}")
        print(f"  final_right_wheel_speed_rad_s: {virtual_result.final_right_wheel_speed:.6g}")
        print(f"  config.lqr_test: {config.lqr_test}")
        if length_schedule is not None:
            print("  length_schedule_control: active; K/F_l0 are interpolated each step from YAML")
            print("  config.lqr_k: fallback only when schedule is disabled")
        if virtual_result.final_lqr_state is not None:
            lqr_state = virtual_result.final_lqr_state
            print(f"  config.lqr_gain_scale: {config.lqr_gain_scale:.6g}")
            print(f"  config.lqr_x_reference: {config.lqr_x_reference:.6g}")
            print(f"  config.lqr_x_source: {config.lqr_x_source}")
            print(f"  config.lqr_x_outer_kp: {config.lqr_x_outer_kp:.6g}")
            print(f"  config.lqr_x_outer_max_v: {config.lqr_x_outer_max_v:.6g}")
            print(f"  config.lqr_wheel_sign: {config.lqr_wheel_sign:.6g}")
            print(f"  config.lqr_pitch_sign: {config.lqr_pitch_sign:.6g}")
            print(f"  config.lqr_t_limit: {config.lqr_t_limit:.6g}")
            print(f"  config.lqr_tp_limit: {config.lqr_tp_limit:.6g}")
            print(f"  config.landing_hold_t_limit: {config.landing_hold_t_limit:.6g}")
            print(f"  config.lqr_output_rate_limit: {config.lqr_output_rate_limit:.6g}")
            print(f"  config.lqr_output_lowpass_hz: {config.lqr_output_lowpass_hz:.6g}")
            print(f"  config.wheel_ctrl_deadzone: {config.wheel_ctrl_deadzone:.6g}")
            print("  config.lqr_k:")
            for row in _format_matrix_rows(config.lqr_k):
                print(f"    {row}")
            print("  theta_pd_main_disabled: True")
            print("  lqr_state_order: theta_world, dtheta_world, x, dx, phi, dphi")
            print("  lqr_input_order: T, Tp")
            print("  lqr_law: [T, Tp]^T = U0 - config.lqr_gain_scale * K * (X - X0)")
            print("  config.lqr_x0: [" + ", ".join(f"{value:.6g}" for value in config.lqr_x0) + "]")
            print("  config.lqr_u0: [" + ", ".join(f"{value:.6g}" for value in config.lqr_u0) + "]")
            print("  lqr_T_limit: +/-config.lqr_t_limit")
            print("  lqr_Tp_limit: +/-config.lqr_tp_limit before VMC theta-force split")
            print("  vmc_length_force_limit: 250")
            print("  vmc_theta_force_limit: 80")
            print("  vmc_joint_tau_limit: 45")
            print("  vmc_length_law: F_l = F_l0 + k_l * e_l + k_i * integral(e_l) - d_l * dL")
            print(f"  vmc_length_kp: {config.virtual_rod_length_kp:.6g}")
            print(f"  vmc_length_kd: {config.virtual_rod_length_kd:.6g}")
            print(f"  vmc_length_ki: {config.virtual_rod_length_ki:.6g}")
            print(f"  vmc_length_integral_limit: {config.virtual_rod_length_integral_limit:.6g}")
            print(f"  vmc_length_force_ff_or_F_l0: {config.virtual_rod_length_force_ff:.6g}")
            print(
                "  vmc_gravity_comp_scale: "
                f"{config.virtual_rod_gravity_comp_scale if config.virtual_rod_gravity_comp_scale is not None else 'off'}"
            )
            print(f"  vmc_length_force_rate_limit: {config.virtual_rod_length_force_rate_limit:.6g}")
            if lqr_design_result is not None:
                print("  lqr_design: finite-difference A/B at current operating point + iterative DARE")
                print(f"  lqr_design_q_diag: {list(lqr_design_result.q_diag)}")
                print(f"  lqr_design_r_diag: {list(lqr_design_result.r_diag)}")
                print(f"  lqr_design_state_eps: {list(lqr_design_result.state_eps)}")
                print(f"  lqr_design_input_eps: {list(lqr_design_result.input_eps)}")
                print(f"  config.lqr_design_steps: {config.lqr_design_steps}")
                print(f"  config.lqr_control_period_steps: {config.lqr_control_period_steps}")
                print(
                    "  lqr_design_operating_state: "
                    f"theta={lqr_design_result.operating_state.theta:.6g}, "
                    f"theta_rate={lqr_design_result.operating_state.theta_rate:.6g}, "
                    f"x={lqr_design_result.operating_state.x:.6g}, "
                    f"x_rate={lqr_design_result.operating_state.x_rate:.6g}, "
                    f"pitch={lqr_design_result.operating_state.pitch:.6g}, "
                    f"pitch_rate={lqr_design_result.operating_state.pitch_rate:.6g}"
                )
                print(f"  lqr_design_riccati_iterations: {lqr_design_result.riccati_iterations}")
                print(f"  lqr_design_closed_loop_max_abs_eig: {lqr_design_result.closed_loop_max_abs_eig:.6g}")
                print(
                    "  lqr_design_affine_residual: ["
                    + ", ".join(f"{value:.6g}" for value in lqr_design_result.affine_residual)
                    + "]"
                )
                print("  lqr_design_A:")
                for row in _format_matrix_rows(lqr_design_result.a_matrix):
                    print(f"    {row}")
                print("  lqr_design_B:")
                for row in _format_matrix_rows(lqr_design_result.b_matrix):
                    print(f"    {row}")
            print(f"  max_abs_lqr_wheel_torque: {virtual_result.max_abs_lqr_wheel_torque:.6g}")
            print(f"  max_abs_lqr_pitch_torque: {virtual_result.max_abs_lqr_pitch_torque:.6g}")
            print(
                "  final_lqr_state: "
                f"theta={lqr_state.theta:.6g}, "
                f"theta_rate={lqr_state.theta_rate:.6g}, "
                f"x={lqr_state.x:.6g}, "
                f"x_rate={lqr_state.x_rate:.6g}, "
                f"pitch={lqr_state.pitch:.6g}, "
                f"pitch_rate={lqr_state.pitch_rate:.6g}"
            )
        print("  actuator_ctrl_limits:")
        for stats in virtual_result.actuator_ctrl_stats:
            print(
                f"    - {stats.actuator}: min={stats.min_ctrl:.6g}, "
                f"max={stats.max_ctrl:.6g}, "
                f"sat_neg_steps={stats.negative_saturated_steps}, "
                f"sat_pos_steps={stats.positive_saturated_steps}"
            )
        print(f"  final_base_height: {virtual_result.final_base_height:.6g}")
        print(f"  max_base_height: {virtual_result.max_base_height:.6g}")
        print(
            "  contact_normal_force_min: "
            f"left={virtual_result.min_left_contact_force:.6g}, "
            f"right={virtual_result.min_right_contact_force:.6g}"
        )
        print(
            "  airborne_detection: "
            f"enabled={config.flight_detection_enabled}, "
            f"threshold={float(config.flight_airborne_force_threshold):.6g} N, "
            f"steps={virtual_result.airborne_steps}, "
            f"first_time={virtual_result.first_airborne_time}, "
            f"last_time={virtual_result.last_airborne_time}"
        )
        if config.lqr_test and virtual_result.final_lqr_state is not None:
            final_state = virtual_result.final_lqr_state
            if (
                config.leg_length_sine_test
                or config.roll_test
                or config.flight_test
                or config.jump_test
            ):
                behavior_passed = True
                if config.jump_test:
                    diagnostic_name = "jump_test"
                elif config.flight_test:
                    diagnostic_name = "flight_test"
                elif config.leg_length_sine_test:
                    diagnostic_name = "dynamic_length_test"
                else:
                    diagnostic_name = "roll_terrain_test"
                print(f"  behavior_qualified: diagnostic_{diagnostic_name}")
            else:
                behavior_passed = (
                    abs(final_state.theta) < 0.15
                    and abs(final_state.pitch) < 0.10
                    and abs(virtual_result.final_left_wheel_speed) < 2.0
                    and abs(virtual_result.final_right_wheel_speed) < 2.0
                    and abs(virtual_result.final_left_length - virtual_result.left_target_length) < 0.01
                    and abs(virtual_result.final_right_length - virtual_result.right_target_length) < 0.01
                    and virtual_result.final_left_branch_violation < 1e-9
                    and virtual_result.final_right_branch_violation < 1e-9
                    and virtual_result.final_base_height > 0.30
                )
                print(f"  behavior_qualified: {behavior_passed}")
        if config.print_static_operating_point and virtual_result.final_operating_point is not None:
            _print_static_operating_point_sample(
                virtual_result.final_operating_point,
                force_ff=config.virtual_rod_length_force_ff,
            )
        if config.turn_pd_plot_path is not None:
            plot_path = _resolve_output_path(config.turn_pd_plot_path)
            _plot_turning_history(
                plot_path,
                virtual_result.history,
                config.yaw_turn_kp,
                config.yaw_turn_kd,
                config.leg_sync_kp,
                config.leg_sync_kd,
            )
            print(f"  turn_pd_plot: {plot_path}")
        if config.roll_length_plot_path is not None:
            plot_path = _resolve_output_path(config.roll_length_plot_path)
            _plot_roll_length_history(
                plot_path,
                virtual_result.history,
                config.virtual_rod_length_kp,
                config.virtual_rod_length_ki,
                config.virtual_rod_length_kd,
                config.roll_force_kp,
            )
            print(f"  roll_length_plot: {plot_path}")
        if config.lqr_debug_plot_path is not None:
            plot_path = _resolve_output_path(config.lqr_debug_plot_path)
            _plot_lqr_debug_history(
                plot_path,
                virtual_result.history,
                title=(
                    f"LQR debug | L0={config.left_rod_length:.3g} m, "
                    f"speed={config.speed_profile or 'none'}, "
                    f"K source={'schedule' if length_schedule is not None else 'locked/manual'}"
                ),
            )
            print(f"  lqr_debug_plot: {plot_path}")
        if config.leg_height_test:
            _print_low_length_stage_diagnostics(virtual_result.history)
        print("  note: experimental virtual-rod middle-layer smoke")

    if config.visualize:
        print()
        visualize_steps = (
            max(1, int(math.ceil(config.visualize_seconds / float(model.opt.timestep))))
            if config.visualize_seconds is not None
            else None
        )
        if not config.virtual_rod_test:
            print("config.visualize: opening MuJoCo viewer for PD hold")
            _visualize_pd_hold(
                mujoco,
                model,
                visualize_steps if visualize_steps is not None else config.pd_hold_steps,
                config.pd_kp,
                config.pd_kd,
                config.realtime,
            )

    print()
    if behavior_passed:
        print("result: PASS finite model/load/step smoke")
        return 0
    print("result: FAIL behavior recovery criteria")
    return 2


def main(argv: list[str] | None = None) -> int:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args(argv)
    yaml_defaults = load_yaml_defaults(config_args.config)
    parser = build_parser()
    parser.add_argument("--config", type=Path, default=config_args.config)
    parser.set_defaults(**yaml_defaults)
    args = parser.parse_args(argv)
    if args.flight_test:
        args.lqr_true_equilibrium = True
        args.speed_profile = args.flight_test_speed
        args.turn_test = False
        args.turn_direction = None
        args.impact_level = None
        args.flight_detection_enabled = True
        if args.visualize_seconds is None:
            args.visualize_seconds = 10.0
    if args.jump_test:
        args.lqr_true_equilibrium = True
        args.speed_profile = None
        args.turn_test = False
        args.turn_direction = None
        args.impact_level = None
        args.flight_detection_enabled = True
        if args.visualize_seconds is None:
            args.visualize_seconds = 10.0
    if args.roll_test:
        args.lqr_true_equilibrium = True
        args.speed_profile = "medium"
        args.turn_test = False
        args.turn_direction = None
        args.impact_level = None
    if args.turn_length_sine_test:
        args.lqr_true_equilibrium = True
        args.turn_test = True
        args.turn_speed = "high"
        args.speed_profile = None
        args.turn_direction = None
        args.impact_level = None
        args.leg_length_sine_test = True
        if args.visualize_seconds is None:
            args.visualize_seconds = 10.0
    if args.manual_drive:
        args.lqr_true_equilibrium = True
        args.visualize = True
        args.speed_profile = None
        args.turn_test = False
        args.turn_direction = None
        args.impact_level = None
    if args.initial_leg_length is None:
        args.initial_leg_length = args.leg_length
    if (
        args.lqr_true_equilibrium
        or args.leg_height_test
        or args.turn_test
        or args.turn_direction is not None
        or args.manual_drive
    ):
        args.virtual_rod_test = True
        target_leg_length = float(args.initial_leg_length)
        if args.left_rod_length is None:
            args.left_rod_length = target_leg_length
        if args.right_rod_length is None:
            args.right_rod_length = target_leg_length
        if args.length_schedule:
            min_length = float(args.minimum_leg_length)
            max_length = float(args.maximum_leg_length)
            if not min_length <= target_leg_length <= max_length:
                parser.error(
                    f"--initial-leg-length must be within the configured leg length range "
                    f"{min_length:.2f}..{max_length:.2f}"
                )
        elif abs(target_leg_length - LOCKED_EQUILIBRIUM_L0) > 1e-9:
            parser.error("--lqr-true-equilibrium is currently restored to locked --leg-length 0.35")
        args.equilibrium_search = False
        args.lqr_test = True
        args.lqr_auto_design = False
        args.lqr_use_equilibrium_operating_point = False
        args.use_locked_equilibrium = True
        args.lqr_k = LOCKED_LQR_K.copy()
        args.lqr_x0 = LOCKED_LQR_X0.copy()
        args.lqr_u0 = LOCKED_LQR_U0.copy()
        args.zero_steps = 1
        args.probe_steps = 1
        args.pd_hold_steps = 1
        args.equilibrium_init_modes = ("upright-ik",)
        args.equilibrium_l_slices = (args.leg_length,)
        args.equilibrium_theta_refs = (LOCKED_EQUILIBRIUM_THETA,)
        support_scale = support_force_scale_for_length(args.leg_length)
        args.equilibrium_fl0_scales = (support_scale,)
        args.equilibrium_steps = LOCKED_EQUILIBRIUM_STEPS
        args.equilibrium_eval_steps = LOCKED_EQUILIBRIUM_EVAL_STEPS
        args.equilibrium_length_kp = LOCKED_EQUILIBRIUM_LENGTH_KP
        args.equilibrium_length_kd = LOCKED_EQUILIBRIUM_LENGTH_KD
        args.equilibrium_theta_kp = LOCKED_EQUILIBRIUM_THETA_KP
        args.equilibrium_theta_kd = LOCKED_EQUILIBRIUM_THETA_KD
        args.equilibrium_pitch_kp = LOCKED_EQUILIBRIUM_PITCH_KP
        args.equilibrium_pitch_kd = LOCKED_EQUILIBRIUM_PITCH_KD
        args.equilibrium_tp_biases = (LOCKED_EQUILIBRIUM_TP_BIAS,)
        args.equilibrium_wheel_com_kps = (LOCKED_EQUILIBRIUM_WHEEL_COM_KP,)
        args.equilibrium_wheel_dampings = (LOCKED_EQUILIBRIUM_WHEEL_DAMPING,)
        args.equilibrium_init_drop_steps = 0
        args.virtual_rod_gravity_comp_scale = None
        args.virtual_rod_length_delta = None
        args.virtual_rod_theta_target = LOCKED_EQUILIBRIUM_THETA
        args.left_rod_length = target_leg_length
        args.right_rod_length = target_leg_length
        args.left_rod_theta = LOCKED_EQUILIBRIUM_THETA
        args.right_rod_theta = LOCKED_EQUILIBRIUM_THETA
        args.lqr_design_steps = LOCKED_LQR_DESIGN_STEPS
        args.lqr_control_period_steps = LOCKED_LQR_CONTROL_PERIOD_STEPS
    if args.turn_test or args.turn_direction is not None:
        args.virtual_rod_test = True
        args.lqr_test = True
        args.lqr_auto_design = False
        args.use_locked_equilibrium = True
        args.speed_profile = None
        args.impact_level = None
        if args.turn_test:
            args.turn_direction = None
    if args.turn_pd_plot:
        if not args.turn_test:
            parser.error("--turn-pd-plot requires --turn-test")
        args.turn_pd_plot_path = Path("output") / f"{datetime.now():%H%M%S}.png"
        if args.visualize_seconds is not None:
            args.virtual_rod_steps = int(math.ceil(args.visualize_seconds / 0.001))
    if args.roll_length_plot:
        if args.turn_pd_plot:
            parser.error("--roll-length-plot cannot be combined with --turn-pd-plot")
        if not args.lqr_test:
            parser.error("--roll-length-plot requires --lqr-true-equilibrium or a turn test")
        args.roll_length_plot_path = Path("output") / f"{datetime.now():%H%M%S}.png"
        if args.visualize_seconds is not None:
            args.virtual_rod_steps = int(math.ceil(args.visualize_seconds / 0.001))
    if args.lqr_debug_plot:
        if args.turn_pd_plot or args.roll_length_plot:
            parser.error("--lqr-debug-plot cannot be combined with --turn-pd-plot or --roll-length-plot")
        if not args.lqr_test:
            parser.error("--lqr-debug-plot requires --lqr-true-equilibrium or a turn test")
        args.lqr_debug_plot_path = Path("output") / f"{datetime.now():%H%M%S}.png"
        if args.visualize_seconds is not None:
            args.virtual_rod_steps = int(math.ceil(args.visualize_seconds / 0.001))
    if args.wheel_balance_only:
        args.virtual_rod_test = True
        args.lqr_test = True
        args.lqr_auto_design = True
        args.lqr_use_equilibrium_operating_point = True
        args.use_locked_equilibrium = False
        args.equilibrium_search = True
        args.virtual_rod_control = "off"
    if args.length_kd is not None:
        args.virtual_rod_length_kd = args.length_kd
    if args.length_ki is not None:
        args.virtual_rod_length_ki = args.length_ki
    if args.length_integral_limit is not None:
        args.virtual_rod_length_integral_limit = args.length_integral_limit
    if args.length_force_ff is not None:
        args.virtual_rod_length_force_ff = args.length_force_ff
    model_path = args.model
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        parser.error(f"model file not found: {model_path}")
    if args.zero_steps <= 0 or args.probe_steps <= 0:
        parser.error("step counts must be positive")
    if args.pd_hold_steps <= 0:
        parser.error("--pd-hold-steps must be positive")
    if args.ctrl <= 0.0:
        parser.error("--ctrl must be positive")
    if args.scan_virtual_rod_sample <= 0.0:
        parser.error("--scan-virtual-rod-sample must be positive")
    if args.pd_kp < 0.0 or args.pd_kd < 0.0:
        parser.error("--pd-kp and --pd-kd must be non-negative")
    if args.visualize_seconds is not None and args.visualize_seconds <= 0.0:
        parser.error("--visualize-seconds must be positive")
    if args.startup_ramp_seconds < 0.0:
        parser.error("--startup-ramp-seconds must be non-negative")
    if args.initial_leg_length <= 0.0:
        parser.error("--initial-leg-length must be positive")
    if args.virtual_rod_test or args.lqr_test:
        startup_steps = int(math.ceil(float(args.startup_ramp_seconds) / 0.001))
        if args.manual_drive and args.visualize_seconds is None:
            args.virtual_rod_steps = 1_000_000_000
        elif args.visualize_seconds is not None:
            args.virtual_rod_steps = int(math.ceil((args.visualize_seconds + args.startup_ramp_seconds) / 0.001))
        else:
            args.virtual_rod_steps += startup_steps
    if args.virtual_rod_steps <= 0:
        parser.error("--virtual-rod-steps must be positive")
    if args.leg_length <= 0.0:
        parser.error("--leg-length must be positive")
    if args.constraint_steps <= 0:
        parser.error("--constraint-steps must be positive")
    if args.ik_search_radius < 0.0:
        parser.error("--ik-search-radius must be non-negative")
    if args.ik_search_samples <= 0:
        parser.error("--ik-search-samples must be positive")
    lqr_auto_design = bool(args.lqr_auto_design or (args.lqr_test and args.lqr_k is None))
    leg_sync_kp = args.leg_sync_kp
    leg_sync_kd = args.leg_sync_kd
    if leg_sync_kp < 0.0 or leg_sync_kd < 0.0:
        parser.error("--leg-sync-kp/--leg-sync-kd must be non-negative")
    yaw_turn_kp = args.yaw_turn_kp
    yaw_turn_kd = args.yaw_turn_kd
    if yaw_turn_kp < 0.0 or yaw_turn_kd < 0.0:
        parser.error("--yaw-turn-kp/--yaw-turn-kd must be non-negative")
    lqr_wheel_sign = args.lqr_wheel_sign
    lqr_pitch_sign = args.lqr_pitch_sign
    lqr_output_rate_limit = args.lqr_output_rate_limit
    lqr_tp_limit = args.lqr_tp_limit
    if lqr_tp_limit is None:
        lqr_tp_limit = args.lqr_t_limit
    lqr_gain_scale = args.lqr_gain_scale
    if lqr_gain_scale is None:
        lqr_gain_scale = 1.0
    if lqr_gain_scale < 0.0:
        parser.error("--lqr-gain-scale must be non-negative")
    lqr_k = DEFAULT_LQR_K if args.lqr_k is None else np.array(args.lqr_k, dtype=float).reshape(2, 6)
    lqr_x0 = np.array(args.lqr_x0, dtype=float)
    lqr_u0 = np.array(args.lqr_u0, dtype=float)
    lqr_q_diag = np.array(args.lqr_q_diag, dtype=float)
    lqr_r_diag = np.array(args.lqr_r_diag, dtype=float)
    lqr_state_eps = np.array(args.lqr_state_eps, dtype=float)
    lqr_input_eps = np.array(args.lqr_input_eps, dtype=float)
    lqr_design_steps = args.lqr_design_steps
    if lqr_design_steps is None:
        lqr_design_steps = 1
    lqr_control_period_steps = args.lqr_control_period_steps
    if lqr_control_period_steps is None:
        lqr_control_period_steps = 1
    if lqr_design_steps <= 0 or lqr_control_period_steps <= 0:
        parser.error("--lqr-design-steps and --lqr-control-period-steps must be positive")
    lqr_x_source = args.lqr_x_source
    if lqr_x_source is None:
        lqr_x_source = "wheel"
    if np.any(lqr_q_diag <= 0.0) or np.any(lqr_r_diag <= 0.0):
        parser.error("--lqr-q-diag and --lqr-r-diag must be positive")
    if np.any(lqr_state_eps <= 0.0) or np.any(lqr_input_eps <= 0.0):
        parser.error("--lqr-state-eps and --lqr-input-eps must be positive")
    if not np.all(np.isfinite(lqr_x0)) or not np.all(np.isfinite(lqr_u0)):
        parser.error("--lqr-x0 and --lqr-u0 must be finite")
    if args.lqr_x_outer_kp < 0.0 or args.lqr_x_outer_max_v < 0.0:
        parser.error("--lqr-x-outer-kp and --lqr-x-outer-max-v must be non-negative")
    if args.lqr_output_lowpass_hz < 0.0:
        parser.error("--lqr-output-lowpass-hz must be non-negative")
    if args.wheel_ctrl_deadzone < 0.0:
        parser.error("--wheel-ctrl-deadzone must be non-negative")
    if min(
        args.lqr_t_limit,
        lqr_tp_limit,
        args.landing_hold_t_limit,
        lqr_output_rate_limit,
    ) < 0.0:
        parser.error("LQR torque limits and rate limit must be non-negative")
    if args.history_sample_interval <= 0:
        parser.error("--history-sample-interval must be positive")
    if args.flight_airborne_confirm_seconds < 0.0 or args.flight_airborne_rearm_seconds < 0.0:
        parser.error("flight airborne timing parameters must be non-negative")
    if args.fl_channel_steps <= 0 or args.fl_pulse_settle_steps <= 0 or args.fl_pulse_steps <= 0:
        parser.error("F_l channel/pulse step counts must be positive")
    if args.fl_pulse_delta_force <= 0.0:
        parser.error("--fl-pulse-delta-force must be positive")
    if args.diagnostics_only and not (
        args.fl_channel_test
        or args.fl_pulse_test
        or args.fivebar_kinematics_check
        or args.fivebar_jacobian_check
        or args.equilibrium_static_pose_check
        or args.equilibrium_search
    ):
        parser.error(
            "--diagnostics-only requires --fl-channel-test, --fl-pulse-test, "
            "--fivebar-kinematics-check, --fivebar-jacobian-check, "
            "--equilibrium-static-pose-check, or --equilibrium-search"
        )
    if args.fivebar_kinematics_check and any(value <= 0.0 for value in args.fivebar_kinematics_l_slices):
        parser.error("--fivebar-kinematics-l-slices must be positive")
    if args.fivebar_jacobian_check:
        if any(value <= 0.0 for value in args.fivebar_jacobian_l_slices):
            parser.error("--fivebar-jacobian-l-slices must be positive")
        if args.fivebar_jacobian_force_scale <= 0.0:
            parser.error("--fivebar-jacobian-force-scale must be positive")
        if args.fivebar_jacobian_load_steps < 0:
            parser.error("--fivebar-jacobian-load-steps must be non-negative")
    if args.equilibrium_static_pose_check or args.equilibrium_search:
        if args.equilibrium_steps <= 0 or args.equilibrium_eval_steps <= 0:
            parser.error("--equilibrium-steps and --equilibrium-eval-steps must be positive")
        if args.equilibrium_init_drop_steps < 0:
            parser.error("--equilibrium-init-drop-steps must be non-negative")
        if any(value <= 0.0 for value in args.equilibrium_l_slices):
            parser.error("--equilibrium-l-slices must be positive")
        if any(value <= 0.0 for value in args.equilibrium_fl0_scales):
            parser.error("--equilibrium-fl0-scales must be positive")
        if any(value < 0.0 for value in args.equilibrium_wheel_com_kps):
            parser.error("--equilibrium-wheel-com-kps must be non-negative")
        if any(value < 0.0 for value in args.equilibrium_wheel_dampings):
            parser.error("--equilibrium-wheel-dampings must be non-negative")
        if min(
            args.equilibrium_length_kp,
            args.equilibrium_length_kd,
            args.equilibrium_theta_kp,
            args.equilibrium_theta_kd,
            args.equilibrium_pitch_kp,
            args.equilibrium_pitch_kd,
        ) < 0.0:
            parser.error("equilibrium search gains and damping must be non-negative")
    virtual_rod_length_kp = args.virtual_rod_length_kp
    virtual_rod_length_kd = args.virtual_rod_length_kd
    motor_servo_kp = args.motor_servo_kp if args.motor_servo_kp is not None else virtual_rod_length_kp
    motor_servo_kd = args.motor_servo_kd if args.motor_servo_kd is not None else virtual_rod_length_kd
    if min(
        motor_servo_kp,
        motor_servo_kd,
        args.virtual_rod_length_ki,
        abs(args.virtual_rod_length_force_ff),
        0.0 if args.virtual_rod_gravity_comp_scale is None else args.virtual_rod_gravity_comp_scale,
        args.virtual_rod_length_integral_limit,
        args.virtual_rod_length_force_rate_limit,
        args.virtual_rod_theta_kp,
        args.virtual_rod_theta_kd,
        args.virtual_rod_joint_kd,
        abs(args.virtual_rod_theta_pitch_ff),
    ) < 0.0:
        parser.error("virtual rod gains must be non-negative")
    config = RunConfig.from_namespace(args)
    config.model_path = model_path
    config.virtual_rod_test = args.virtual_rod_test or args.lqr_test
    config.virtual_rod_lock_base = args.lock_base
    config.motor_servo_kp = motor_servo_kp
    config.motor_servo_kd = motor_servo_kd
    config.length_force_ff_is_explicit = (
        args.length_force_ff is not None
        or "virtual_rod_length_force_ff" in yaml_defaults
    )
    config.lqr_gain_scale = lqr_gain_scale
    config.lqr_k = lqr_k
    config.lqr_x0 = lqr_x0
    config.lqr_u0 = lqr_u0
    config.lqr_auto_design = lqr_auto_design
    config.lqr_q_diag = lqr_q_diag
    config.lqr_r_diag = lqr_r_diag
    config.lqr_state_eps = lqr_state_eps
    config.lqr_input_eps = lqr_input_eps
    config.lqr_design_steps = lqr_design_steps
    config.lqr_control_period_steps = lqr_control_period_steps
    config.lqr_x_source = lqr_x_source
    config.lqr_wheel_sign = lqr_wheel_sign
    config.lqr_pitch_sign = lqr_pitch_sign
    config.lqr_tp_limit = lqr_tp_limit
    config.lqr_output_rate_limit = lqr_output_rate_limit
    config.leg_sync_kp = leg_sync_kp
    config.leg_sync_kd = leg_sync_kd
    config.yaw_turn_kp = yaw_turn_kp
    config.yaw_turn_kd = yaw_turn_kd
    config.leg_height_levels = tuple(float(value) for value in config.leg_height_levels)
    config.realtime = not args.no_realtime
    return run_smoke(config)

