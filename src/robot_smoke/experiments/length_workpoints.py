"""离线腿长工作点诊断。

本模块在不改变运行时平衡控制器的前提下，生成固定腿长下的局部量表。
MuJoCo 只用于加载 XML 参数，并在解析 IK 后评估已建模的五连杆几何。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..control.lqr import solve_continuous_are_hamiltonian
from ..core.mujoco_utils import id_by_name
from ..model.fivebar import analytic_fivebar_kinematics_from_q, drive_joint_ids, equilibrium_analytic_ik_targets
from ..model.mechanics import support_force_scale_for_length


@dataclass(frozen=True)
class ReducedModelParameters:
    """Parameters used by the paper-style frozen-length sagittal model."""

    wheel_radius: float
    wheel_mass: float
    wheel_inertia: float
    leg_mass: float
    leg_com_ratio: float
    leg_inertia: float
    body_mass: float
    body_com_to_hip: float
    body_inertia: float
    gravity: float


@dataclass(frozen=True)
class LengthWorkpoint:
    """One frozen leg-length diagnostic row."""

    length: float
    reachable: bool
    q_left: tuple[float, float] | None
    q_right: tuple[float, float] | None
    jacobian_left: np.ndarray | None
    jacobian_right: np.ndarray | None
    jacobian_condition: float | None
    jacobian_fd_error: float | None
    support_force_seed: float | None
    support_force_scaled: float | None
    support_tau_left: np.ndarray | None
    support_tau_left_scaled: np.ndarray | None
    unit_tp_tau_left: np.ndarray | None
    controllability_rank: int | None
    k_matrix: np.ndarray | None
    note: str


def extract_reduced_model_parameters(mujoco, model) -> ReducedModelParameters:
    """Extract reduced-model masses/inertias from XML-expanded MuJoCo arrays."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base_id = id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "base")
    wheel_ids = (
        id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "left_wheel"),
        id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "right_wheel"),
    )
    leg_ids = tuple(body_id for body_id in range(1, model.nbody) if body_id not in {base_id, *wheel_ids})
    leg_mass = float(np.sum(model.body_mass[list(leg_ids)]))
    if leg_mass <= 0.0:
        raise RuntimeError("leg equivalent mass must be positive")

    leg_com = np.sum(model.body_mass[list(leg_ids), None] * data.xipos[list(leg_ids)], axis=0) / leg_mass
    hip_ids = tuple(
        id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("left_front_upper", "left_rear_upper", "right_front_upper", "right_rear_upper")
    )
    hip = np.mean(data.xpos[list(hip_ids)], axis=0)
    wheel = np.mean(data.xpos[list(wheel_ids)], axis=0)
    neutral_length = float(np.linalg.norm(hip[[0, 2]] - wheel[[0, 2]]))
    leg_com_ratio = float((leg_com[2] - wheel[2]) / neutral_length) if neutral_length > 1e-9 else 0.5

    axis_y = np.array([0.0, 1.0, 0.0])
    leg_inertia = 0.0
    for body_id in leg_ids:
        rotation = data.xmat[body_id].reshape(3, 3)
        world_inertia = rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T
        offset = data.xipos[body_id] - leg_com
        leg_inertia += float(axis_y @ world_inertia @ axis_y)
        leg_inertia += float(model.body_mass[body_id] * (offset[0] ** 2 + offset[2] ** 2))

    wheel_inertia = 0.0
    for body_id, joint_name in zip(wheel_ids, ("left_wheel_joint", "right_wheel_joint")):
        wheel_inertia += float(model.body_inertia[body_id, 1])
        joint_id = id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        wheel_inertia += float(model.dof_armature[int(model.jnt_dofadr[joint_id])])

    wheel_geom = id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_geom")
    return ReducedModelParameters(
        wheel_radius=float(model.geom_size[wheel_geom, 0]),
        wheel_mass=float(np.sum(model.body_mass[list(wheel_ids)])),
        wheel_inertia=wheel_inertia,
        leg_mass=leg_mass,
        leg_com_ratio=leg_com_ratio,
        leg_inertia=leg_inertia,
        body_mass=float(model.body_mass[base_id]),
        body_com_to_hip=float(data.xipos[base_id, 2] - hip[2]),
        body_inertia=float(model.body_inertia[base_id, 1]),
        gravity=abs(float(model.opt.gravity[2])),
    )


def linearize_frozen_length(length: float, params: ReducedModelParameters) -> tuple[np.ndarray, np.ndarray]:
    """Linearize the paper-style upright sagittal model at one frozen length."""
    radius = params.wheel_radius
    wheel_effective_mass = params.wheel_mass + params.wheel_inertia / (radius * radius)
    leg_com = params.leg_com_ratio * length
    hip_to_leg_com = length - leg_com
    mp = params.leg_mass
    body = params.body_mass
    body_com = params.body_com_to_hip
    mass_zero = np.array(
        [
            [wheel_effective_mass * length * length + mp * hip_to_leg_com**2, -(wheel_effective_mass * length + mp * hip_to_leg_com), 0.0],
            [-(wheel_effective_mass * length + mp * hip_to_leg_com), wheel_effective_mass + mp + body, -body * body_com],
            [0.0, -body * body_com, params.body_inertia + body * body_com * body_com],
        ],
        dtype=float,
    )
    gravity_state = np.array(
        [params.gravity * (leg_com * mp + length * body), 0.0, params.gravity * body * body_com],
        dtype=float,
    )
    input_map = np.array([[-length / radius, 1.0], [1.0 / radius, 0.0], [0.0, 1.0]], dtype=float)
    acceleration_by_state = np.linalg.solve(mass_zero, np.diag(gravity_state))
    acceleration_by_input = np.linalg.solve(mass_zero, input_map)
    a_matrix = np.zeros((6, 6), dtype=float)
    b_matrix = np.zeros((6, 2), dtype=float)
    a_matrix[0, 1] = 1.0
    a_matrix[2, 3] = 1.0
    a_matrix[4, 5] = 1.0
    a_matrix[[1, 3, 5], 0] = acceleration_by_state[:, 0]
    a_matrix[[1, 3, 5], 4] = acceleration_by_state[:, 2]
    b_matrix[[1, 3, 5], :] = acceleration_by_input
    return a_matrix, b_matrix


def controllability_rank(a_matrix: np.ndarray, b_matrix: np.ndarray) -> int:
    matrix = np.hstack([np.linalg.matrix_power(a_matrix, power) @ b_matrix for power in range(6)])
    return int(np.linalg.matrix_rank(matrix))


def lqr_gain(a_matrix: np.ndarray, b_matrix: np.ndarray, q_diag: np.ndarray, r_diag: np.ndarray) -> np.ndarray:
    q_matrix = np.diag(q_diag)
    r_matrix = np.diag(r_diag)
    p_matrix = solve_continuous_are_hamiltonian(a_matrix, b_matrix, q_matrix, r_matrix)
    return np.linalg.solve(r_matrix, b_matrix.T @ p_matrix)


def _set_leg_drive_qpos(mujoco, model, data, side: str, q_pair: tuple[float, float]) -> None:
    front_joint, rear_joint = drive_joint_ids(mujoco, model, side)
    data.qpos[int(model.jnt_qposadr[front_joint])] = q_pair[0]
    data.qpos[int(model.jnt_qposadr[rear_joint])] = q_pair[1]


def _finite_difference_jacobian_error(mujoco, model, side: str, q_pair: tuple[float, float], analytic_jacobian: np.ndarray) -> float:
    eps = 1e-6
    columns = []
    for index in range(2):
        q_plus = list(q_pair)
        q_minus = list(q_pair)
        q_plus[index] += eps
        q_minus[index] -= eps
        plus = analytic_fivebar_kinematics_from_q(mujoco, model, side, q_plus[0], q_plus[1])
        minus = analytic_fivebar_kinematics_from_q(mujoco, model, side, q_minus[0], q_minus[1])
        if plus is None or minus is None:
            return math.inf
        columns.append(np.array([(plus.length - minus.length) / (2.0 * eps), (plus.theta - minus.theta) / (2.0 * eps)]))
    numeric = np.column_stack(columns)
    return float(np.max(np.abs(numeric - analytic_jacobian)))


def evaluate_length_workpoint(
    mujoco,
    model,
    length: float,
    params: ReducedModelParameters,
    q_diag: np.ndarray,
    r_diag: np.ndarray,
) -> LengthWorkpoint:
    q_left = equilibrium_analytic_ik_targets(mujoco, model, "left", (length, 0.0))
    q_right = equilibrium_analytic_ik_targets(mujoco, model, "right", (length, 0.0))
    if q_left is None or q_right is None:
        return LengthWorkpoint(length, False, q_left, q_right, None, None, None, None, None, None, None, None, None, None, None, "IK unreachable")

    left_kin = analytic_fivebar_kinematics_from_q(mujoco, model, "left", *q_left)
    right_kin = analytic_fivebar_kinematics_from_q(mujoco, model, "right", *q_right)
    if left_kin is None or right_kin is None:
        return LengthWorkpoint(length, False, q_left, q_right, None, None, None, None, None, None, None, None, None, None, None, "forward kinematics failed")

    jacobian_condition = float(max(np.linalg.cond(left_kin.jacobian), np.linalg.cond(right_kin.jacobian)))
    jacobian_fd_error = max(
        _finite_difference_jacobian_error(mujoco, model, "left", q_left, left_kin.jacobian),
        _finite_difference_jacobian_error(mujoco, model, "right", q_right, right_kin.jacobian),
    )
    total_mass = float(np.sum(model.body_mass))
    support_force_seed = total_mass * params.gravity * 0.5
    support_force_scaled = support_force_scale_for_length(length) * support_force_seed
    support_tau_left = left_kin.jacobian.T @ np.array([support_force_seed, 0.0], dtype=float)
    support_tau_left_scaled = left_kin.jacobian.T @ np.array([support_force_scaled, 0.0], dtype=float)
    unit_tp_tau_left = left_kin.jacobian.T @ np.array([0.0, 1.0], dtype=float)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    _set_leg_drive_qpos(mujoco, model, data, "left", q_left)
    _set_leg_drive_qpos(mujoco, model, data, "right", q_right)
    mujoco.mj_forward(model, data)

    a_matrix, b_matrix = linearize_frozen_length(length, params)
    rank = controllability_rank(a_matrix, b_matrix)
    gain = lqr_gain(a_matrix, b_matrix, q_diag, r_diag) if rank == 6 else None
    note = "ok"
    if jacobian_condition > 100.0:
        note = "ill-conditioned J"
    return LengthWorkpoint(
        length=length,
        reachable=True,
        q_left=q_left,
        q_right=q_right,
        jacobian_left=left_kin.jacobian,
        jacobian_right=right_kin.jacobian,
        jacobian_condition=jacobian_condition,
        jacobian_fd_error=jacobian_fd_error,
        support_force_seed=support_force_seed,
        support_force_scaled=support_force_scaled,
        support_tau_left=support_tau_left,
        support_tau_left_scaled=support_tau_left_scaled,
        unit_tp_tau_left=unit_tp_tau_left,
        controllability_rank=rank,
        k_matrix=gain,
        note=note,
    )


def print_length_workpoint_report(
    points: list[LengthWorkpoint],
    params: ReducedModelParameters,
    *,
    detailed: bool = True,
) -> None:
    print("length_workpoint_diagnostics:")
    print(
        "  reduced_model: "
        f"R={params.wheel_radius:.6g}, mw={params.wheel_mass:.6g}, Iw={params.wheel_inertia:.6g}, "
        f"mp={params.leg_mass:.6g}, M={params.body_mass:.6g}, Ip={params.leg_inertia:.6g}, "
        f"IM={params.body_inertia:.6g}, l={params.body_com_to_hip:.6g}, "
        f"leg_com_ratio={params.leg_com_ratio:.6g}"
    )
    if not detailed:
        print("  compact_table:")
        print("    L0(m)  ok  cond(J)  J_fd_err  rank  F_theory  F_scaled  tau_scaled_front  tau_Tp_front  note")
        for point in points:
            if not point.reachable:
                print(f"    {point.length:5.3f}  no  -        -         -     -         -         -                 -             {point.note}")
                continue
            assert point.support_force_seed is not None
            assert point.support_force_scaled is not None
            assert point.support_tau_left_scaled is not None
            assert point.unit_tp_tau_left is not None
            print(
                f"    {point.length:5.3f}  yes {point.jacobian_condition:7.3f}  "
                f"{point.jacobian_fd_error:8.1e}  "
                f"{point.controllability_rank}/6  {point.support_force_seed:8.3f}  "
                f"{point.support_force_scaled:8.3f}  {point.support_tau_left_scaled[0]:16.4f}  "
                f"{point.unit_tp_tau_left[0]:12.5f}  {point.note}"
            )
        return

    for point in points:
        print(f"  - L0={point.length:.3f}, reachable={point.reachable}, note={point.note}")
        if not point.reachable:
            continue
        assert point.q_left is not None
        assert point.q_right is not None
        assert point.jacobian_left is not None
        assert point.support_force_seed is not None
        assert point.support_force_scaled is not None
        assert point.support_tau_left is not None
        assert point.support_tau_left_scaled is not None
        assert point.unit_tp_tau_left is not None
        print(f"    q_left=[{point.q_left[0]:.6g}, {point.q_left[1]:.6g}], q_right=[{point.q_right[0]:.6g}, {point.q_right[1]:.6g}]")
        print(
            f"    J_condition_max={point.jacobian_condition:.6g}, "
            f"J_fd_error_max={point.jacobian_fd_error:.6g}, controllability={point.controllability_rank}/6"
        )
        print(f"    F_l0_theory_per_leg={point.support_force_seed:.6g} N")
        print(f"    F_l0_scaled_per_leg={point.support_force_scaled:.6g} N")
        print(f"    tau_support_left=[{point.support_tau_left[0]:.6g}, {point.support_tau_left[1]:.6g}] N*m")
        print(f"    tau_support_scaled_left=[{point.support_tau_left_scaled[0]:.6g}, {point.support_tau_left_scaled[1]:.6g}] N*m")
        print(f"    tau_per_unit_Tp_left=[{point.unit_tp_tau_left[0]:.6g}, {point.unit_tp_tau_left[1]:.6g}] N*m/(N*m)")
        print("    J_left:")
        for row in point.jacobian_left:
            print("      [" + ", ".join(f"{value:.6g}" for value in row) + "]")
        if point.k_matrix is not None:
            print("    K:")
            for row in point.k_matrix:
                print("      [" + ", ".join(f"{value:.6g}" for value in row) + "]")
