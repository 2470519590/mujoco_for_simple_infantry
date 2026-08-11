"""真实平衡点搜索辅助函数。"""

from __future__ import annotations

import math

import numpy as np

from ..model.actuators import apply_additive_wheel_torque_ctrl as _apply_additive_wheel_torque_ctrl
from ..model.fivebar import equilibrium_analytic_ik_targets as _equilibrium_analytic_ik_targets
from ..control.lqr import set_base_pitch as _set_base_pitch
from ..control.lqr import compute_lqr_state as _compute_lqr_state
from ..model.kinematics import (
    compute_virtual_leg_state as _compute_virtual_leg_state,
    wheel_radius as _wheel_radius,
    wheel_speeds as _wheel_speeds,
)
from ..control.ik import (
    _equilibrium_global_ik_targets,
    _virtual_rod_ik_ctrl,
)
from ..model.mechanics import (
    _average_wheel_x,
    _collect_static_operating_point_sample,
    _contact_normal_force_for_wheel,
    _estimated_support_force_per_leg,
    _mechanics_equilibrium_seed,
    _model_com_x,
    _print_static_operating_point_sample,
    _set_base_height_for_wheel_contact,
    _standing_base_qpos_for_virtual_leg_target,
)
from ..core.mujoco_utils import assert_finite as _assert_finite
from ..core.mujoco_utils import copy_data as _copy_data
from ..core.mujoco_utils import lock_base_to_qpos as _lock_base_to_qpos
from ..core.mujoco_utils import rms as _rms
from ..core.types import EquilibriumSearchResult, VmcDiagnostics, VmcSideMemory
from ..control.vmc import _drive_constant_length_force_ctrl, _drive_joint_position_ctrl

def _initialize_equilibrium_data(
    mujoco,
    model,
    l_slice: float,
    theta_ref: float,
    leg_branch: str,
    ik_search_radius: float,
    ik_search_samples: int,
    init_mode: str,
    drop_steps: int,
):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    if init_mode == "reset":
        return data
    if init_mode != "upright-ik":
        raise RuntimeError(f"unknown equilibrium init mode: {init_mode}")
    target = (l_slice, theta_ref)
    standing_base_qpos = _standing_base_qpos_for_virtual_leg_target(mujoco, model, target, target)
    data.qpos[:7] = standing_base_qpos
    _set_base_pitch(data.qpos, 0.0)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    left_analytic = _equilibrium_analytic_ik_targets(mujoco, model, "left", target)
    right_analytic = _equilibrium_analytic_ik_targets(mujoco, model, "right", target)
    if left_analytic is None:
        left_front_target, left_rear_target = _equilibrium_global_ik_targets(
            mujoco,
            model,
            data,
            "left",
            target,
            ik_search_samples,
        )
    else:
        left_front_target, left_rear_target = left_analytic
    if right_analytic is None:
        right_front_target, right_rear_target = _equilibrium_global_ik_targets(
            mujoco,
            model,
            data,
            "right",
            target,
            ik_search_samples,
        )
    else:
        right_front_target, right_rear_target = right_analytic
    for _ in range(180):
        _lock_base_to_qpos(mujoco, model, data, standing_base_qpos)
        data.ctrl[:] = 0.0
        _drive_joint_position_ctrl(
            mujoco,
            model,
            data,
            "left",
            left_front_target,
            left_rear_target,
            45.0,
            1.6,
        )
        _drive_joint_position_ctrl(
            mujoco,
            model,
            data,
            "right",
            right_front_target,
            right_rear_target,
            45.0,
            1.6,
        )
        mujoco.mj_step(model, data)
    _lock_base_to_qpos(mujoco, model, data, standing_base_qpos)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    _set_base_pitch(data.qpos, 0.0)
    mujoco.mj_forward(model, data)
    _set_base_height_for_wheel_contact(mujoco, model, data)
    if drop_steps > 0:
        ik_target_cache: dict[tuple[str, float, float, str, float, int], tuple[float, float]] = {}
        vmc_memory: dict[str, VmcSideMemory] = {}
        for _ in range(drop_steps):
            data.ctrl[:] = 0.0
            _set_base_pitch(data.qpos, 0.0)
            data.qvel[3:6] = 0.0
            _virtual_rod_ik_ctrl(
                mujoco,
                model,
                data,
                target,
                target,
                "vmc",
                leg_branch,
                ik_search_radius,
                ik_search_samples,
                300.0,
                35.0,
                0.0,
                0.0,
                35.0,
                ik_target_cache,
                theta_force_offset=0.0,
                length_force_ff=_estimated_support_force_per_leg(model),
                length_ki=0.0,
                length_integral_limit=0.0,
                length_force_rate_limit=0.0,
                vmc_memory=vmc_memory,
                vmc_diagnostics={},
            )
            mujoco.mj_step(model, data)
            _set_base_height_for_wheel_contact(mujoco, model, data)
        _set_base_pitch(data.qpos, 0.0)
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
    return data


def _run_single_equilibrium_candidate(
    mujoco,
    model,
    init_mode: str,
    l_slice: float,
    theta_ref: float,
    fl0_scale: float,
    steps: int,
    eval_steps: int,
    length_kp: float,
    length_kd: float,
    theta_kp: float,
    theta_kd: float,
    pitch_kp: float,
    pitch_kd: float,
    tp_bias: float,
    wheel_com_kp: float,
    wheel_damping: float,
    wheel_torque_limit: float,
    pitch_torque_limit: float,
    joint_tau_limit: float,
    leg_branch: str,
    ik_search_radius: float,
    ik_search_samples: int,
    thresholds: dict[str, float],
    init_drop_steps: int,
) -> EquilibriumSearchResult:
    data = _initialize_equilibrium_data(
        mujoco,
        model,
        l_slice,
        theta_ref,
        leg_branch,
        ik_search_radius,
        ik_search_samples,
        init_mode,
        init_drop_steps,
    )
    target = (l_slice, theta_ref)
    ik_target_cache: dict[tuple[str, float, float, str, float, int], tuple[float, float]] = {}
    vmc_memory: dict[str, VmcSideMemory] = {}
    vmc_diagnostics: dict[str, VmcDiagnostics] = {}
    mechanics_seed = _mechanics_equilibrium_seed(mujoco, model, data, theta_ref)
    fl0 = fl0_scale * mechanics_seed.length_force_per_leg
    pitch_torque_center = mechanics_seed.pitch_torque
    radius = _wheel_radius(mujoco, model)

    L_values: list[float] = []
    com_offset_values: list[float] = []
    phi_values: list[float] = []
    theta_values: list[float] = []
    theta_diff_values: list[float] = []
    dtheta_diff_values: list[float] = []
    length_diff_values: list[float] = []
    dx_values: list[float] = []
    dphi_values: list[float] = []
    dtheta_values: list[float] = []
    dL_values: list[float] = []
    T_values: list[float] = []
    Tp_values: list[float] = []
    contact_values: list[float] = []
    slip_values: list[float] = []
    T_sat_count = 0
    Tp_sat_count = 0
    joint_sat_count = 0

    sample_start = max(0, steps - eval_steps)
    wheel_torque = 0.0
    pitch_torque = 0.0
    for step in range(steps):
        data.ctrl[:] = 0.0
        state = _compute_lqr_state(mujoco, model, data, 0.0, "wheel")
        left_wheel_speed, right_wheel_speed = _wheel_speeds(mujoco, model, data)
        wheel_surface_speed = radius * 0.5 * (left_wheel_speed + right_wheel_speed)
        com_offset = _model_com_x(mujoco, model, data) - _average_wheel_x(mujoco, model, data)
        raw_wheel_torque = (
            wheel_com_kp * com_offset
            - wheel_damping * wheel_surface_speed
        )
        wheel_torque = float(np.clip(raw_wheel_torque, -wheel_torque_limit, wheel_torque_limit))
        raw_pitch_torque = pitch_torque_center - (
            theta_kp * state.theta
            + theta_kd * state.theta_rate
            + pitch_kp * state.pitch
            + pitch_kd * state.pitch_rate
        ) + tp_bias
        pitch_torque = float(np.clip(raw_pitch_torque, -pitch_torque_limit, pitch_torque_limit))
        if abs(raw_wheel_torque - wheel_torque) > 1e-12:
            T_sat_count += 1
        if abs(raw_pitch_torque - pitch_torque) > 1e-12:
            Tp_sat_count += 1
        left_leg = _compute_virtual_leg_state(mujoco, model, data, "left")
        right_leg = _compute_virtual_leg_state(mujoco, model, data, "right")
        sync_torque = 30.0 * (right_leg.theta - left_leg.theta) + 20.0 * (
            right_leg.theta_rate - left_leg.theta_rate
        )
        _, saturated, _, _ = _virtual_rod_ik_ctrl(
            mujoco,
            model,
            data,
            target,
            target,
            "vmc",
            leg_branch,
            ik_search_radius,
            ik_search_samples,
            length_kp,
            length_kd,
            0.0,
            0.0,
            max(length_kd, 1.0),
            ik_target_cache,
            theta_force_offset=(pitch_torque + sync_torque, pitch_torque - sync_torque),
            length_force_ff=fl0,
            length_ki=0.0,
            length_integral_limit=0.0,
            length_force_rate_limit=0.0,
            vmc_memory=vmc_memory,
            vmc_diagnostics=vmc_diagnostics,
        )
        wheel_saturated = _apply_additive_wheel_torque_ctrl(mujoco, model, data, wheel_torque)
        if saturated or wheel_saturated:
            joint_sat_count += 1
        mujoco.mj_step(model, data)
        _assert_finite("equilibrium qpos", data.qpos)
        _assert_finite("equilibrium qvel", data.qvel)
        if step >= sample_start:
            state = _compute_lqr_state(mujoco, model, data, 0.0, "wheel")
            left = _compute_virtual_leg_state(mujoco, model, data, "left")
            right = _compute_virtual_leg_state(mujoco, model, data, "right")
            left_wheel_speed, right_wheel_speed = _wheel_speeds(mujoco, model, data)
            contact_left = _contact_normal_force_for_wheel(mujoco, model, data, "left")
            contact_right = _contact_normal_force_for_wheel(mujoco, model, data, "right")
            contact_mean = 0.5 * (contact_left + contact_right)
            base_state = _compute_lqr_state(mujoco, model, data, 0.0, "base")
            wheel_surface_speed = radius * 0.5 * (left_wheel_speed + right_wheel_speed)
            com_offset = _model_com_x(mujoco, model, data) - _average_wheel_x(mujoco, model, data)
            slip_values.append(abs(base_state.x_rate - wheel_surface_speed))
            L_values.append(0.5 * (left.length + right.length))
            com_offset_values.append(com_offset)
            phi_values.append(state.pitch)
            theta_values.append(state.theta)
            theta_diff_values.append(left.theta - right.theta)
            dtheta_diff_values.append(left.theta_rate - right.theta_rate)
            length_diff_values.append(left.length - right.length)
            dx_values.append(state.x_rate)
            dphi_values.append(state.pitch_rate)
            dtheta_values.append(state.theta_rate)
            dL_values.append(0.5 * (left.length_rate + right.length_rate))
            T_values.append(wheel_torque)
            Tp_values.append(pitch_torque)
            contact_values.append(contact_mean)

    eval_count = max(1, len(L_values))
    L_mean = float(np.mean(L_values)) if L_values else math.nan
    length_target_error = L_mean - l_slice
    com_offset_mean = float(np.mean(com_offset_values)) if com_offset_values else math.nan
    phi_mean = float(np.mean(phi_values)) if phi_values else math.nan
    theta_mean = float(np.mean(theta_values)) if theta_values else math.nan
    contact_force_mean = float(np.mean(contact_values)) if contact_values else 0.0
    contact_force_min = float(np.min(contact_values)) if contact_values else 0.0
    dx_rms = _rms(dx_values)
    dphi_rms = _rms(dphi_values)
    dtheta_rms = _rms(dtheta_values)
    dL_rms = _rms(dL_values)
    theta_diff_rms = _rms(theta_diff_values)
    dtheta_diff_rms = _rms(dtheta_diff_values)
    length_diff_rms = _rms(length_diff_values)
    T_sat_ratio = T_sat_count / max(1, steps)
    Tp_sat_ratio = Tp_sat_count / max(1, steps)
    joint_sat_ratio = joint_sat_count / max(1, steps)
    slip_indicator = float(np.mean(slip_values)) if slip_values else math.nan
    qualified = (
        abs(phi_mean) < thresholds["angle"]
        and abs(length_target_error) < thresholds["length_target"]
        and abs(theta_mean) < thresholds["angle"]
        and abs(com_offset_mean) < thresholds["com_offset"]
        and dx_rms < thresholds["dx"]
        and dphi_rms < thresholds["dphi"]
        and dtheta_rms < thresholds["dtheta"]
        and dL_rms < thresholds["dL"]
        and theta_diff_rms < thresholds["theta_diff"]
        and dtheta_diff_rms < thresholds["dtheta_diff"]
        and length_diff_rms < thresholds["length_diff"]
        and T_sat_ratio <= thresholds["sat_ratio"]
        and Tp_sat_ratio <= thresholds["sat_ratio"]
        and joint_sat_ratio <= thresholds["sat_ratio"]
        and contact_force_min > thresholds["contact_min"]
        and slip_indicator < thresholds["slip"]
    )
    final_sample = _collect_static_operating_point_sample(
        mujoco,
        model,
        data,
        f"equilibrium_L0slice{l_slice:.6g}_FlScale{fl0_scale:.6g}",
        wheel_torque,
        pitch_torque,
        x_source="wheel",
        left_diag=vmc_diagnostics.get("left"),
        right_diag=vmc_diagnostics.get("right"),
    )
    return EquilibriumSearchResult(
        init_mode=init_mode,
        l_slice=l_slice,
        theta_ref=theta_ref,
        fl0_scale=fl0_scale,
        tp_bias=tp_bias,
        wheel_com_kp=wheel_com_kp,
        wheel_damping=wheel_damping,
        fl0_center=mechanics_seed.length_force_per_leg,
        tp_center=mechanics_seed.pitch_torque,
        fl0=fl0,
        steps=steps,
        eval_steps=eval_count,
        qualified=qualified,
        L_mean=L_mean,
        length_target_error=length_target_error,
        com_offset_mean=com_offset_mean,
        phi_mean=phi_mean,
        theta_mean=theta_mean,
        theta_diff_rms=theta_diff_rms,
        dtheta_diff_rms=dtheta_diff_rms,
        length_diff_rms=length_diff_rms,
        dx_rms=dx_rms,
        dphi_rms=dphi_rms,
        dtheta_rms=dtheta_rms,
        dL_rms=dL_rms,
        T_mean=float(np.mean(T_values)) if T_values else math.nan,
        Tp_mean=float(np.mean(Tp_values)) if Tp_values else math.nan,
        T_sat_ratio=T_sat_ratio,
        Tp_sat_ratio=Tp_sat_ratio,
        joint_sat_ratio=joint_sat_ratio,
        contact_force_mean=contact_force_mean,
        contact_force_min=contact_force_min,
        slip_indicator=slip_indicator,
        final_sample=final_sample,
        final_data=_copy_data(mujoco, model, data),
    )


def _equilibrium_score(result: EquilibriumSearchResult) -> float:
    return (
        15.0 * abs(result.com_offset_mean)
        + 20.0 * abs(result.length_target_error)
        + 8.0 * result.dx_rms
        + 2.0 * result.dphi_rms
        + 2.0 * result.dtheta_rms
        + 10.0 * result.dL_rms
        + abs(result.phi_mean)
        + abs(result.theta_mean)
        + 4.0 * result.theta_diff_rms
        + 2.0 * result.dtheta_diff_rms
        + 10.0 * result.length_diff_rms
        + 10.0 * (result.T_sat_ratio + result.Tp_sat_ratio + result.joint_sat_ratio)
        + max(0.0, 5.0 - result.contact_force_min)
        + result.slip_indicator
    )


def _run_equilibrium_search(
    mujoco,
    model,
    init_modes: tuple[str, ...],
    l_slices: tuple[float, ...],
    theta_refs: tuple[float, ...],
    fl0_scales: tuple[float, ...],
    tp_biases: tuple[float, ...],
    wheel_com_kps: tuple[float, ...],
    wheel_dampings: tuple[float, ...],
    init_drop_steps: int,
    steps: int,
    eval_steps: int,
    length_kp: float,
    length_kd: float,
    theta_kp: float,
    theta_kd: float,
    pitch_kp: float,
    pitch_kd: float,
    wheel_torque_limit: float,
    pitch_torque_limit: float,
    joint_tau_limit: float,
    leg_branch: str,
    ik_search_radius: float,
    ik_search_samples: int,
) -> tuple[EquilibriumSearchResult, ...]:
    support_force_per_leg = _estimated_support_force_per_leg(model)
    thresholds = {
        "angle": 0.08,
        "length_target": 0.003,
        "com_offset": 0.03,
        "dx": 0.005,
        "dphi": 0.08,
        "dtheta": 0.08,
        "dL": 0.015,
        "theta_diff": 0.03,
        "dtheta_diff": 0.08,
        "length_diff": 0.01,
        "sat_ratio": 0.01,
        "contact_min": 5.0,
        "slip": 0.15,
    }
    print("equilibrium_search:")
    print(
        "  note: article-style contact balance search; L0 is a frozen leg-length slice, "
        "not a searched balance state; no LQR position recovery"
    )
    print(f"  steps: {steps}")
    print(f"  eval_tail_steps: {eval_steps}")
    print(f"  support_force_per_leg: {support_force_per_leg:.6g}")
    print("  L0_slices: [" + ", ".join(f"{value:.6g}" for value in l_slices) + "]")
    print("  theta_refs: [" + ", ".join(f"{value:.6g}" for value in theta_refs) + "]")
    print(
        "  thresholds: "
        + ", ".join(f"{key}={value:.6g}" for key, value in thresholds.items())
    )
    results: list[EquilibriumSearchResult] = []
    for init_mode in init_modes:
        for l_slice in l_slices:
            for theta_ref in theta_refs:
                for fl0_scale in fl0_scales:
                    for tp_bias in tp_biases:
                        for wheel_com_kp in wheel_com_kps:
                            for wheel_damping in wheel_dampings:
                                result = _run_single_equilibrium_candidate(
                                    mujoco,
                                    model,
                                    init_mode,
                                    l_slice,
                                    theta_ref,
                                    fl0_scale,
                                    steps,
                                    min(eval_steps, steps),
                                    length_kp,
                                    length_kd,
                                    theta_kp,
                                    theta_kd,
                                    pitch_kp,
                                    pitch_kd,
                                    tp_bias,
                                    wheel_com_kp,
                                    wheel_damping,
                                    wheel_torque_limit,
                                    pitch_torque_limit,
                                    joint_tau_limit,
                                    leg_branch,
                                    ik_search_radius,
                                    ik_search_samples,
                                    thresholds,
                                    init_drop_steps,
                                )
                                results.append(result)
                                print(
                                    "  - "
                                    f"init={result.init_mode}, "
                                    f"L0_slice={result.l_slice:.6g}, "
                                    f"theta_ref={result.theta_ref:.6g}, "
                                    f"F_l0_scale={result.fl0_scale:.6g}, "
                                    f"Tp_bias={result.tp_bias:.6g}, "
                                    f"wheel_com_kp={result.wheel_com_kp:.6g}, "
                                    f"wheel_damping={result.wheel_damping:.6g}, "
                                    f"qualified={result.qualified}, "
                                    f"L_mean={result.L_mean:.6g}, "
                                    f"L_error={result.length_target_error:.6g}, "
                                    f"com_offset_mean={result.com_offset_mean:.6g}, "
                                    f"phi_mean={result.phi_mean:.6g}, "
                                    f"theta_mean={result.theta_mean:.6g}, "
                                    f"theta_diff_RMS={result.theta_diff_rms:.6g}, "
                                    f"dtheta_diff_RMS={result.dtheta_diff_rms:.6g}, "
                                    f"length_diff_RMS={result.length_diff_rms:.6g}, "
                                    f"dx_RMS={result.dx_rms:.6g}, "
                                    f"dphi_RMS={result.dphi_rms:.6g}, "
                                    f"dtheta_RMS={result.dtheta_rms:.6g}, "
                                    f"dL_RMS={result.dL_rms:.6g}, "
                                    f"T_mean={result.T_mean:.6g}, "
                                    f"Tp_mean={result.Tp_mean:.6g}, "
                                    f"F_l0_center={result.fl0_center:.6g}, "
                                    f"F_l0={result.fl0:.6g}, "
                                    f"Tp_center={result.tp_center:.6g}, "
                                    f"T_sat_ratio={result.T_sat_ratio:.6g}, "
                                    f"Tp_sat_ratio={result.Tp_sat_ratio:.6g}, "
                                    f"joint_sat_ratio={result.joint_sat_ratio:.6g}, "
                                    f"contact_force_mean={result.contact_force_mean:.6g}, "
                                    f"contact_force_min={result.contact_force_min:.6g}, "
                                    f"slip_indicator={result.slip_indicator:.6g}"
                                )
    if results:
        best = min(results, key=_equilibrium_score)
        print("  best_candidate_by_score:")
        print(
            f"    init={best.init_mode}, L0_slice={best.l_slice:.6g}, F_l0_scale={best.fl0_scale:.6g}, "
            f"theta_ref={best.theta_ref:.6g}, Tp_bias={best.tp_bias:.6g}, "
            f"F_l0_center={best.fl0_center:.6g}, Tp_center={best.tp_center:.6g}, "
            f"wheel_com_kp={best.wheel_com_kp:.6g}, "
            f"wheel_damping={best.wheel_damping:.6g}, "
            f"qualified={best.qualified}, score={_equilibrium_score(best):.6g}"
        )
        _print_static_operating_point_sample(best.final_sample, force_ff=best.fl0)
    return tuple(results)


def _run_equilibrium_static_pose_check(
    mujoco,
    model,
    l_slices: tuple[float, ...],
    theta_refs: tuple[float, ...],
    fl0_scales: tuple[float, ...],
    leg_branch: str,
    ik_search_radius: float,
    ik_search_samples: int,
) -> None:
    support_force_per_leg = _estimated_support_force_per_leg(model)
    print("equilibrium_static_pose_check:")
    print(
        "  note: constrained upright/contact pose construction for fixed L0 slices; "
        "not a true free equilibrium and not a search for unique leg length"
    )
    for l_slice in l_slices:
        for theta_ref in theta_refs:
            for fl0_scale in fl0_scales:
                data = _initialize_equilibrium_data(
                    mujoco,
                    model,
                    l_slice,
                    theta_ref,
                    leg_branch,
                    ik_search_radius,
                    ik_search_samples,
                    "upright-ik",
                    0,
                )
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                _set_base_height_for_wheel_contact(mujoco, model, data)
                mechanics_seed = _mechanics_equilibrium_seed(mujoco, model, data, theta_ref)
                fl0 = fl0_scale * mechanics_seed.length_force_per_leg
                _, left_diag = _drive_constant_length_force_ctrl(mujoco, model, data, "left", fl0)
                _, right_diag = _drive_constant_length_force_ctrl(mujoco, model, data, "right", fl0)
                mujoco.mj_forward(model, data)
                sample = _collect_static_operating_point_sample(
                    mujoco,
                    model,
                    data,
                    f"constrained_pose_L0slice{l_slice:.6g}_Theta{theta_ref:.6g}_FlScale{fl0_scale:.6g}",
                    0.0,
                    0.0,
                    x_source="wheel",
                    left_diag=left_diag,
                    right_diag=right_diag,
                )
                state = sample.lqr_state
                contact_min = min(sample.left_contact_normal_force, sample.right_contact_normal_force)
                com_offset = _model_com_x(mujoco, model, data) - _average_wheel_x(mujoco, model, data)
                gravity_moment_y = float(np.sum(model.body_mass) * 9.81 * com_offset)
                print(
                    "  - "
                    f"L0_slice={l_slice:.6g}, theta_ref={theta_ref:.6g}, F_l0_scale={fl0_scale:.6g}, "
                    f"L0={0.5 * (sample.left_length + sample.right_length):.6g}, "
                    f"theta={state.theta:.6g}, phi={state.pitch:.6g}, "
                    f"base_z={float(data.qpos[2]):.6g}, "
                    f"com_x_minus_wheel_x={com_offset:.6g}, "
                    f"gravity_moment_y={gravity_moment_y:.6g}, "
                    f"F_l0_center={mechanics_seed.length_force_per_leg:.6g}, "
                    f"Tp_center={mechanics_seed.pitch_torque:.6g}, "
                    f"contact_min={contact_min:.6g}, "
                    f"tau_support_left=[{sample.left_support_tau_front:.6g}, {sample.left_support_tau_rear:.6g}], "
                    f"tau_support_right=[{sample.right_support_tau_front:.6g}, {sample.right_support_tau_rear:.6g}]"
                )


