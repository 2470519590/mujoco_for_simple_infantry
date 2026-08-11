"""五连杆腿部解析运动学辅助函数。"""

from __future__ import annotations

import math

import numpy as np

from ..core.mujoco_utils import body_pos as _body_pos
from ..core.mujoco_utils import id_by_name as _id_by_name
from ..core.mujoco_utils import site_pos as _site_pos
from ..core.types import AnalyticFivebarKinematics, LegBranchMetrics


def body_local_xz(mujoco, model, body_name: str) -> np.ndarray:
    body_id = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return np.array(
        [float(model.body_pos[body_id, 0]), float(model.body_pos[body_id, 2])],
        dtype=float,
    )


def circle_intersections_xz(
    center_a: np.ndarray,
    radius_a: float,
    center_b: np.ndarray,
    radius_b: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    delta = center_b - center_a
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-12:
        return None
    if distance > radius_a + radius_b or distance < abs(radius_a - radius_b):
        return None
    axis = delta / distance
    along = (radius_a * radius_a - radius_b * radius_b + distance * distance) / (2.0 * distance)
    height_sq = radius_a * radius_a - along * along
    if height_sq < -1e-10:
        return None
    height = math.sqrt(max(0.0, height_sq))
    midpoint = center_a + along * axis
    normal = np.array([-axis[1], axis[0]], dtype=float)
    return midpoint + height * normal, midpoint - height * normal


def drive_angle_from_upper_vector(reset_vector: np.ndarray, target_vector: np.ndarray) -> float:
    dot = float(np.dot(reset_vector, target_vector))
    cross_for_positive_y = float(reset_vector[1] * target_vector[0] - reset_vector[0] * target_vector[1])
    return math.atan2(cross_for_positive_y, dot)


def upper_vector_from_drive_angle(reset_vector: np.ndarray, q: float) -> np.ndarray:
    sin_q = math.sin(q)
    cos_q = math.cos(q)
    return np.array(
        [
            cos_q * reset_vector[0] + sin_q * reset_vector[1],
            -sin_q * reset_vector[0] + cos_q * reset_vector[1],
        ],
        dtype=float,
    )


def upper_vector_derivative_from_drive_angle(upper_vector: np.ndarray) -> np.ndarray:
    return np.array([upper_vector[1], -upper_vector[0]], dtype=float)


def drive_joint_ids(mujoco, model, side: str) -> tuple[int, int]:
    front = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_front_drive")
    rear = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_rear_drive")
    return front, rear


def drive_qpos_for_side(mujoco, model, data, side: str) -> tuple[float, float]:
    front_joint_id, rear_joint_id = drive_joint_ids(mujoco, model, side)
    return (
        float(data.qpos[int(model.jnt_qposadr[front_joint_id])]),
        float(data.qpos[int(model.jnt_qposadr[rear_joint_id])]),
    )


def compute_leg_branch_metrics(mujoco, model, data, side: str) -> LegBranchMetrics:
    base_id = _id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "base")
    base_pos = data.xpos[base_id].copy()
    base_rot = data.xmat[base_id].reshape(3, 3).copy()

    def to_base_frame(world_pos: np.ndarray) -> np.ndarray:
        return base_rot.T @ (world_pos - base_pos)

    front_elbow = to_base_frame(_body_pos(mujoco, model, data, f"{side}_front_elbow"))
    rear_elbow = to_base_frame(_body_pos(mujoco, model, data, f"{side}_rear_elbow"))
    carrier = to_base_frame(_site_pos(mujoco, model, data, f"{side}_carrier_site"))
    front_dx = float(front_elbow[0] - carrier[0])
    rear_dx = float(carrier[0] - rear_elbow[0])
    below_front = float(front_elbow[2] - carrier[2])
    below_rear = float(rear_elbow[2] - carrier[2])
    elbow_span = float(front_elbow[0] - rear_elbow[0])
    front_margin = 0.035
    rear_margin = 0.035
    below_margin = 0.055
    span_margin = 0.12
    violation = (
        max(0.0, front_margin - front_dx)
        + max(0.0, rear_margin - rear_dx)
        + max(0.0, below_margin - below_front)
        + max(0.0, below_margin - below_rear)
        + max(0.0, span_margin - elbow_span)
    )
    return LegBranchMetrics(
        name=side,
        front_dx=front_dx,
        rear_dx=rear_dx,
        below_front=below_front,
        below_rear=below_rear,
        elbow_span=elbow_span,
        violation=violation,
    )


def leg_branch_guard_error(metrics: LegBranchMetrics) -> float:
    front_guard_margin = 0.08
    rear_guard_margin = 0.08
    below_guard_margin = 0.12
    span_guard_margin = 0.20
    return (
        max(0.0, front_guard_margin - metrics.front_dx)
        + max(0.0, rear_guard_margin - metrics.rear_dx)
        + max(0.0, below_guard_margin - metrics.below_front)
        + max(0.0, below_guard_margin - metrics.below_rear)
        + max(0.0, span_guard_margin - metrics.elbow_span)
    )


def fivebar_link_lengths(mujoco, model, side: str) -> tuple[float, float, float, float]:
    front_upper = body_local_xz(mujoco, model, f"{side}_front_elbow")
    rear_upper = body_local_xz(mujoco, model, f"{side}_rear_elbow")
    front_lower = body_local_xz(mujoco, model, f"{side}_front_hub")
    rear_lower = body_local_xz(mujoco, model, f"{side}_rear_hub")
    return (
        float(np.linalg.norm(front_upper)),
        float(np.linalg.norm(rear_upper)),
        float(np.linalg.norm(front_lower)),
        float(np.linalg.norm(rear_lower)),
    )


def select_normal_carrier_from_elbows(
    intersections: tuple[np.ndarray, np.ndarray] | None,
    front_elbow: np.ndarray,
    rear_elbow: np.ndarray,
) -> np.ndarray | None:
    if intersections is None:
        return None

    def score(point: np.ndarray) -> float:
        below_loss = max(0.0, point[1] - front_elbow[1] + 0.02) + max(0.0, point[1] - rear_elbow[1] + 0.02)
        between_loss = max(0.0, rear_elbow[0] - point[0]) + max(0.0, point[0] - front_elbow[0])
        return 1000.0 * below_loss + between_loss + 0.001 * point[1]

    carrier = min(intersections, key=score)
    if carrier[1] > min(front_elbow[1], rear_elbow[1]) - 0.005:
        return None
    return carrier


def analytic_fivebar_kinematics_from_q(
    mujoco,
    model,
    side: str,
    q_front: float,
    q_rear: float,
) -> AnalyticFivebarKinematics | None:
    front_anchor = body_local_xz(mujoco, model, f"{side}_front_upper")
    rear_anchor = body_local_xz(mujoco, model, f"{side}_rear_upper")
    hip = 0.5 * (front_anchor + rear_anchor)
    front_upper_reset = body_local_xz(mujoco, model, f"{side}_front_elbow")
    rear_upper_reset = body_local_xz(mujoco, model, f"{side}_rear_elbow")
    front_upper_len, rear_upper_len, front_lower_len, rear_lower_len = fivebar_link_lengths(mujoco, model, side)
    front_upper = upper_vector_from_drive_angle(front_upper_reset, q_front)
    rear_upper = upper_vector_from_drive_angle(rear_upper_reset, q_rear)
    if abs(float(np.linalg.norm(front_upper)) - front_upper_len) > 1e-9:
        return None
    if abs(float(np.linalg.norm(rear_upper)) - rear_upper_len) > 1e-9:
        return None
    front_elbow = front_anchor + front_upper
    rear_elbow = rear_anchor + rear_upper
    carrier_intersections = circle_intersections_xz(front_elbow, front_lower_len, rear_elbow, rear_lower_len)
    carrier = select_normal_carrier_from_elbows(carrier_intersections, front_elbow, rear_elbow)
    if carrier is None:
        return None

    front_constraint = carrier - front_elbow
    rear_constraint = carrier - rear_elbow
    constraint_matrix = np.vstack([front_constraint, rear_constraint])
    if abs(float(np.linalg.det(constraint_matrix))) < 1e-10:
        return None

    front_elbow_dq = upper_vector_derivative_from_drive_angle(front_upper)
    rear_elbow_dq = upper_vector_derivative_from_drive_angle(rear_upper)
    carrier_dq_front = np.linalg.solve(
        constraint_matrix,
        np.array([float(np.dot(front_constraint, front_elbow_dq)), 0.0], dtype=float),
    )
    carrier_dq_rear = np.linalg.solve(
        constraint_matrix,
        np.array([0.0, float(np.dot(rear_constraint, rear_elbow_dq))], dtype=float),
    )
    rod = carrier - hip
    length = float(np.linalg.norm(rod))
    if length <= 1e-12:
        return None

    def shape_derivative(carrier_dq: np.ndarray) -> tuple[float, float]:
        d_length = float(np.dot(rod, carrier_dq) / length)
        d_theta = float((-rod[1] * carrier_dq[0] + rod[0] * carrier_dq[1]) / (length * length))
        return d_length, d_theta

    front_column = shape_derivative(carrier_dq_front)
    rear_column = shape_derivative(carrier_dq_rear)
    jacobian = np.array(
        [
            [front_column[0], rear_column[0]],
            [front_column[1], rear_column[1]],
        ],
        dtype=float,
    )
    return AnalyticFivebarKinematics(
        length=length,
        theta=math.atan2(float(rod[0]), float(-rod[1])),
        carrier_xz=carrier,
        front_elbow_xz=front_elbow,
        rear_elbow_xz=rear_elbow,
        jacobian=jacobian,
    )


def equilibrium_analytic_ik_targets(
    mujoco,
    model,
    side: str,
    target: tuple[float, float],
) -> tuple[float, float] | None:
    front_anchor = body_local_xz(mujoco, model, f"{side}_front_upper")
    rear_anchor = body_local_xz(mujoco, model, f"{side}_rear_upper")
    hip = 0.5 * (front_anchor + rear_anchor)
    carrier = np.array(
        [
            hip[0] + target[0] * math.sin(target[1]),
            hip[1] - target[0] * math.cos(target[1]),
        ],
        dtype=float,
    )
    front_upper_reset = body_local_xz(mujoco, model, f"{side}_front_elbow")
    rear_upper_reset = body_local_xz(mujoco, model, f"{side}_rear_elbow")
    front_hub_reset = body_local_xz(mujoco, model, f"{side}_front_hub")
    rear_hub_reset = body_local_xz(mujoco, model, f"{side}_rear_hub")
    front_upper_len = float(np.linalg.norm(front_upper_reset))
    rear_upper_len = float(np.linalg.norm(rear_upper_reset))
    front_lower_len = float(np.linalg.norm(front_hub_reset))
    rear_lower_len = float(np.linalg.norm(rear_hub_reset))

    front_intersections = circle_intersections_xz(front_anchor, front_upper_len, carrier, front_lower_len)
    rear_intersections = circle_intersections_xz(rear_anchor, rear_upper_len, carrier, rear_lower_len)
    if front_intersections is None or rear_intersections is None:
        return None

    def front_score(point: np.ndarray) -> float:
        return max(0.0, carrier[0] + 0.02 - point[0]) + max(0.0, carrier[1] + 0.02 - point[1])

    def rear_score(point: np.ndarray) -> float:
        return max(0.0, point[0] - (carrier[0] - 0.02)) + max(0.0, carrier[1] + 0.02 - point[1])

    front_elbow = min(front_intersections, key=front_score)
    rear_elbow = min(rear_intersections, key=rear_score)
    if front_score(front_elbow) > 1e-9 or rear_score(rear_elbow) > 1e-9:
        return None

    front_target_vector = front_elbow - front_anchor
    rear_target_vector = rear_elbow - rear_anchor
    front_q = drive_angle_from_upper_vector(front_upper_reset, front_target_vector)
    rear_q = drive_angle_from_upper_vector(rear_upper_reset, rear_target_vector)
    front_joint_id, rear_joint_id = drive_joint_ids(mujoco, model, side)
    low_front, high_front = model.jnt_range[front_joint_id]
    low_rear, high_rear = model.jnt_range[rear_joint_id]
    if not (low_front <= front_q <= high_front and low_rear <= rear_q <= high_rear):
        return None
    return float(front_q), float(rear_q)
