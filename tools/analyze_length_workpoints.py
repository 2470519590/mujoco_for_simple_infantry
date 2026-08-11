"""运行离线腿长工作点诊断。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.robot_smoke.core.constants import (
    LOCKED_EQUILIBRIUM_FL0,
    LOCKED_EQUILIBRIUM_L0,
    LOCKED_EQUILIBRIUM_QPOS,
    LOCKED_LQR_K,
    LOCKED_LQR_U0,
    LOCKED_LQR_X0,
    PROJECT_ROOT,
)
from src.robot_smoke.core.mujoco_utils import load_mujoco, lock_base_to_qpos
from src.robot_smoke.control.ik import _branch_aware_ik_targets
from src.robot_smoke.control.lqr_design import _design_lqr_gain
from src.robot_smoke.control.vmc import _drive_joint_position_ctrl
from src.robot_smoke.experiments.length_workpoints import (
    evaluate_length_workpoint,
    extract_reduced_model_parameters,
    print_length_workpoint_report,
)
from src.robot_smoke.model.kinematics import compute_virtual_leg_state


RUNTIME_Q_DIAG = "1,0.5,10,30,25,25"
RUNTIME_R_DIAG = "0.1,10"
RUNTIME_STATE_EPS = np.array([0.012, 0.25, 0.01, 0.2, 0.012, 0.25], dtype=float)
RUNTIME_INPUT_EPS = np.array([0.5, 0.5], dtype=float)
OPERATING_IK_SEARCH_RADIUS = 0.25
OPERATING_IK_SEARCH_SAMPLES = 5
LINEARIZATION_IK_SEARCH_RADIUS = 0.25
LINEARIZATION_IK_SEARCH_SAMPLES = 5
K_CONTINUATION_MAX_STEP_NORM = 12.0


def _float_list(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def _length_values(args: argparse.Namespace) -> tuple[float, ...]:
    if args.length_grid:
        count = int(round((args.length_stop - args.length_start) / args.length_step)) + 1
        return tuple(round(args.length_start + index * args.length_step, 10) for index in range(count))
    return _float_list(args.lengths)


def _runtime_operating_data(
    mujoco,
    model,
    length: float,
    leg_branch: str,
    ik_search_radius: float,
    ik_search_samples: int,
    seed_data: object | None = None,
    seed_length: float | None = None,
) -> object:
    target = (float(length), 0.0)
    data = mujoco.MjData(model)
    if seed_data is None:
        data.qpos[:] = LOCKED_EQUILIBRIUM_QPOS
        seed_length = LOCKED_EQUILIBRIUM_L0
    else:
        data.qpos[:] = seed_data.qpos
        seed_length = float(seed_length if seed_length is not None else LOCKED_EQUILIBRIUM_L0)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    if seed_data is None and abs(float(length) - LOCKED_EQUILIBRIUM_L0) < 1e-12:
        return data

    base_qpos = data.qpos[:7].copy()
    base_qpos[2] += float(length) - seed_length
    data.qpos[:7] = base_qpos
    mujoco.mj_forward(model, data)

    left_reset = compute_virtual_leg_state(mujoco, model, data, "left")
    right_reset = compute_virtual_leg_state(mujoco, model, data, "right")
    left_front_target, left_rear_target = _branch_aware_ik_targets(
        mujoco,
        model,
        data,
        "left",
        left_reset,
        target,
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
        target,
        leg_branch,
        ik_search_radius,
        ik_search_samples,
    )
    for _ in range(160):
        lock_base_to_qpos(mujoco, model, data, base_qpos)
        data.ctrl[:] = 0.0
        _drive_joint_position_ctrl(mujoco, model, data, "left", left_front_target, left_rear_target, 40.0, 1.4)
        _drive_joint_position_ctrl(mujoco, model, data, "right", right_front_target, right_rear_target, 40.0, 1.4)
        mujoco.mj_step(model, data)
    lock_base_to_qpos(mujoco, model, data, base_qpos)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    return data


def _runtime_schedule_entry(
    mujoco,
    model,
    point,
    q_diag: np.ndarray,
    r_diag: np.ndarray,
    *,
    anchor_locked_035: bool,
) -> tuple[dict[str, object], object] | None:
    if not point.reachable or point.support_force_scaled is None:
        return None
    length = float(point.length)
    if anchor_locked_035 and abs(length - LOCKED_EQUILIBRIUM_L0) < 1e-12:
        data = mujoco.MjData(model)
        data.qpos[:] = LOCKED_EQUILIBRIUM_QPOS
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        return {
            "L0": length,
            "F_l0": float(LOCKED_EQUILIBRIUM_FL0),
            "X0": LOCKED_LQR_X0.tolist(),
            "U0": LOCKED_LQR_U0.tolist(),
            "K": LOCKED_LQR_K.tolist(),
        }, data
    return _runtime_schedule_entry_from_seed(
        mujoco,
        model,
        point,
        q_diag,
        r_diag,
        seed_data=None,
        seed_length=None,
    )


def _runtime_schedule_entry_from_seed(
    mujoco,
    model,
    point,
    q_diag: np.ndarray,
    r_diag: np.ndarray,
    *,
    seed_data: object | None,
    seed_length: float | None,
) -> tuple[dict[str, object], object] | None:
    if not point.reachable or point.support_force_scaled is None:
        return None
    length = float(point.length)
    target = (length, 0.0)
    operating_data = _runtime_operating_data(
        mujoco,
        model,
        length,
        "normal",
        OPERATING_IK_SEARCH_RADIUS,
        OPERATING_IK_SEARCH_SAMPLES,
        seed_data=seed_data,
        seed_length=seed_length,
    )
    design = _design_lqr_gain(
        mujoco,
        model,
        target,
        target,
        "normal",
        LINEARIZATION_IK_SEARCH_RADIUS,
        LINEARIZATION_IK_SEARCH_SAMPLES,
        400.0,
        80.0,
        0.0,
        float(point.support_force_scaled),
        0.10,
        0.0,
        8.0,
        q_diag,
        r_diag,
        RUNTIME_STATE_EPS,
        RUNTIME_INPUT_EPS,
        5,
        "wheel",
        1.0,
        -1.0,
        leg_control_enabled=True,
        branch_guard_enabled=False,
        operating_data=operating_data,
        operating_u0=np.zeros(2, dtype=float),
    )
    state = design.operating_state
    return {
        "L0": length,
        "F_l0": float(point.support_force_scaled),
        "X0": [
            float(state.theta),
            float(state.theta_rate),
            float(state.x),
            float(state.x_rate),
            float(state.pitch),
            float(state.pitch_rate),
        ],
        "U0": design.operating_u0.tolist(),
        "K": design.k_matrix.tolist(),
        "closed_loop_max_abs_eig": float(design.closed_loop_max_abs_eig),
    }, operating_data


def _runtime_schedule_rows(
    mujoco,
    model,
    points,
    q_diag: np.ndarray,
    r_diag: np.ndarray,
    *,
    anchor_locked_035: bool,
) -> list[dict[str, object]]:
    ordered = sorted(points, key=lambda item: float(item.length))
    rows_by_length: dict[float, dict[str, object]] = {}
    anchor_index = next(
        (index for index, point in enumerate(ordered) if abs(float(point.length) - LOCKED_EQUILIBRIUM_L0) < 1e-12),
        None,
    )
    if anchor_index is None:
        seed_data = None
        seed_length = None
        for point in ordered:
            entry = _runtime_schedule_entry_from_seed(
                mujoco,
                model,
                point,
                q_diag,
                r_diag,
                seed_data=seed_data,
                seed_length=seed_length,
            )
            if entry is None:
                continue
            row, seed_data = entry
            seed_length = float(point.length)
            rows_by_length[seed_length] = row
        return [rows_by_length[float(point.length)] for point in ordered if float(point.length) in rows_by_length]

    anchor_point = ordered[anchor_index]
    anchor_entry = _runtime_schedule_entry(
        mujoco,
        model,
        anchor_point,
        q_diag,
        r_diag,
        anchor_locked_035=anchor_locked_035,
    )
    if anchor_entry is None:
        raise RuntimeError("failed to build locked L0=0.35 schedule anchor")
    anchor_row, anchor_data = anchor_entry
    rows_by_length[float(anchor_point.length)] = anchor_row

    seed_data = anchor_data
    seed_length = float(anchor_point.length)
    for point in reversed(ordered[:anchor_index]):
        entry = _runtime_schedule_entry_from_seed(
            mujoco,
            model,
            point,
            q_diag,
            r_diag,
            seed_data=seed_data,
            seed_length=seed_length,
        )
        if entry is None:
            continue
        row, seed_data = entry
        seed_length = float(point.length)
        rows_by_length[seed_length] = row

    seed_data = anchor_data
    seed_length = float(anchor_point.length)
    for point in ordered[anchor_index + 1:]:
        entry = _runtime_schedule_entry_from_seed(
            mujoco,
            model,
            point,
            q_diag,
            r_diag,
            seed_data=seed_data,
            seed_length=seed_length,
        )
        if entry is None:
            continue
        row, seed_data = entry
        seed_length = float(point.length)
        rows_by_length[seed_length] = row

    return [rows_by_length[float(point.length)] for point in ordered if float(point.length) in rows_by_length]


def _candidate_metadata(row: dict[str, object]) -> dict[str, object]:
    return {
        "candidate_K": row["K"],
        "candidate_X0": row["X0"],
        "candidate_U0": row["U0"],
        "candidate_closed_loop_max_abs_eig": row.get("closed_loop_max_abs_eig"),
    }


def _accepted_continuation_rows(
    runtime_rows: list[dict[str, object]],
    *,
    max_step_norm: float,
) -> list[dict[str, object]]:
    if not runtime_rows:
        return []
    rows = [dict(row) for row in sorted(runtime_rows, key=lambda item: float(item["L0"]))]
    anchor_index = next(
        (index for index, row in enumerate(rows) if abs(float(row["L0"]) - LOCKED_EQUILIBRIUM_L0) < 1e-12),
        None,
    )
    if anchor_index is None:
        anchor_index = 0

    def locked_row(source: dict[str, object], status: str, delta: float) -> dict[str, object]:
        row = dict(source)
        row.update(_candidate_metadata(source))
        row["K"] = LOCKED_LQR_K.tolist()
        row["X0"] = LOCKED_LQR_X0.tolist()
        row["U0"] = LOCKED_LQR_U0.tolist()
        row["accepted"] = status == "accepted"
        row["accept_status"] = status
        row["candidate_delta_from_previous"] = float(delta)
        return row

    accepted: dict[int, dict[str, object]] = {}
    anchor = locked_row(rows[anchor_index], "accepted", 0.0)
    accepted[anchor_index] = anchor

    previous_k = LOCKED_LQR_K.copy()
    previous_x0 = LOCKED_LQR_X0.copy()
    previous_u0 = LOCKED_LQR_U0.copy()
    for index in range(anchor_index - 1, -1, -1):
        candidate = rows[index]
        candidate_k = np.array(candidate["K"], dtype=float)
        delta = float(np.linalg.norm(candidate_k - previous_k))
        row = dict(candidate)
        row.update(_candidate_metadata(candidate))
        if delta <= max_step_norm:
            row["accepted"] = True
            row["accept_status"] = "accepted"
            row["candidate_delta_from_previous"] = delta
            previous_k = candidate_k
            previous_x0 = np.array(candidate["X0"], dtype=float)
            previous_u0 = np.array(candidate["U0"], dtype=float)
        else:
            row["K"] = previous_k.tolist()
            row["X0"] = previous_x0.tolist()
            row["U0"] = previous_u0.tolist()
            row["accepted"] = False
            row["accept_status"] = "rejected_large_K_jump"
            row["candidate_delta_from_previous"] = delta
        accepted[index] = row

    previous_k = LOCKED_LQR_K.copy()
    previous_x0 = LOCKED_LQR_X0.copy()
    previous_u0 = LOCKED_LQR_U0.copy()
    for index in range(anchor_index + 1, len(rows)):
        candidate = rows[index]
        candidate_k = np.array(candidate["K"], dtype=float)
        delta = float(np.linalg.norm(candidate_k - previous_k))
        row = dict(candidate)
        row.update(_candidate_metadata(candidate))
        if delta <= max_step_norm:
            row["accepted"] = True
            row["accept_status"] = "accepted"
            row["candidate_delta_from_previous"] = delta
            previous_k = candidate_k
            previous_x0 = np.array(candidate["X0"], dtype=float)
            previous_u0 = np.array(candidate["U0"], dtype=float)
        else:
            row["K"] = previous_k.tolist()
            row["X0"] = previous_x0.tolist()
            row["U0"] = previous_u0.tolist()
            row["accepted"] = False
            row["accept_status"] = "rejected_large_K_jump"
            row["candidate_delta_from_previous"] = delta
        accepted[index] = row

    return [accepted[index] for index in range(len(rows))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline frozen-length IK/J/LQR diagnostic table")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "assets" / "biped_wheel_leg.xml")
    parser.add_argument("--lengths", default="0.16,0.20,0.24,0.28,0.30")
    parser.add_argument("--length-grid", action="store_true", help="use --length-start:--length-step:--length-stop")
    parser.add_argument("--length-start", type=float, default=0.16)
    parser.add_argument("--length-stop", type=float, default=0.30)
    parser.add_argument("--length-step", type=float, default=0.01)
    parser.add_argument("--lqr-q-diag", default=RUNTIME_Q_DIAG)
    parser.add_argument("--lqr-r-diag", default=RUNTIME_R_DIAG)
    parser.add_argument(
        "--runtime-sign",
        action="store_true",
        help="export K with the current smoke controller sign convention",
    )
    parser.add_argument("--compact", action="store_true", help="print one-line rows without full J/K matrices")
    parser.add_argument("--write-schedule", type=Path, help="write a runtime length-schedule YAML table")
    parser.add_argument(
        "--schedule-source",
        choices=("runtime", "analytic", "locked-k", "continuation"),
        default="continuation",
        help="source for K/X0/U0 written to --write-schedule",
    )
    parser.add_argument(
        "--k-continuation-max-step",
        type=float,
        default=K_CONTINUATION_MAX_STEP_NORM,
        help="maximum allowed adjacent Frobenius norm jump for accepting a continued K row",
    )
    parser.add_argument(
        "--no-anchor-locked-035",
        action="store_true",
        help="do not replace the L0=0.35 schedule row with the verified locked MuJoCo controller",
    )
    args = parser.parse_args()

    model_path = args.model if args.model.is_absolute() else PROJECT_ROOT / args.model
    mujoco = load_mujoco()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    params = extract_reduced_model_parameters(mujoco, model)
    q_diag = np.array(_float_list(args.lqr_q_diag), dtype=float)
    r_diag = np.array(_float_list(args.lqr_r_diag), dtype=float)
    if q_diag.shape != (6,) or r_diag.shape != (2,):
        raise SystemExit("--lqr-q-diag must have 6 values and --lqr-r-diag must have 2 values")
    points = [
        evaluate_length_workpoint(mujoco, model, float(length), params, q_diag, r_diag)
        for length in _length_values(args)
    ]
    print_length_workpoint_report(points, params, detailed=not args.compact)
    if args.write_schedule is not None:
        rows = []
        anchor_locked = not args.no_anchor_locked_035
        if args.schedule_source == "locked-k":
            for point in points:
                if not point.reachable or point.support_force_scaled is None:
                    continue
                force_ff = (
                    float(LOCKED_EQUILIBRIUM_FL0)
                    if abs(float(point.length) - LOCKED_EQUILIBRIUM_L0) < 1e-12
                    else float(point.support_force_scaled)
                )
                rows.append(
                    {
                        "L0": float(point.length),
                        "F_l0": force_ff,
                        "X0": LOCKED_LQR_X0.tolist(),
                        "U0": LOCKED_LQR_U0.tolist(),
                        "K": LOCKED_LQR_K.tolist(),
                    }
                )
        elif args.schedule_source in ("runtime", "continuation"):
            runtime_rows = _runtime_schedule_rows(
                mujoco,
                model,
                points,
                q_diag,
                r_diag,
                anchor_locked_035=anchor_locked,
            )
            rows = (
                _accepted_continuation_rows(runtime_rows, max_step_norm=float(args.k_continuation_max_step))
                if args.schedule_source == "continuation"
                else runtime_rows
            )
        else:
            for point in points:
                if not point.reachable or point.k_matrix is None or point.support_force_scaled is None:
                    continue
                anchor_this_row = anchor_locked and abs(float(point.length) - LOCKED_EQUILIBRIUM_L0) < 1e-12
                k_matrix = -point.k_matrix if args.runtime_sign else point.k_matrix
                if anchor_this_row:
                    rows.append(
                        {
                            "L0": float(point.length),
                            "F_l0": float(LOCKED_EQUILIBRIUM_FL0),
                            "X0": LOCKED_LQR_X0.tolist(),
                            "U0": LOCKED_LQR_U0.tolist(),
                            "K": LOCKED_LQR_K.tolist(),
                        }
                    )
                else:
                    rows.append(
                        {
                            "L0": float(point.length),
                            "F_l0": float(point.support_force_scaled),
                            "X0": [0.0] * 6,
                            "U0": [0.0, 0.0],
                            "K": k_matrix.tolist(),
                        }
                    )
        if args.schedule_source == "locked-k":
            note = (
                "Generated with schedule_source=locked-k. Only F_l0 is scheduled by leg length; "
                "K/X0/U0 are kept at the verified locked L0=0.35 controller because raw multi-length "
                "finite-difference K is not yet reliable."
            )
        elif args.schedule_source == "continuation":
            note = (
                "Generated with schedule_source=continuation. Candidate K is built by locked-point "
                "continuation; formal K rows are accepted only when adjacent K jump is below "
                f"{float(args.k_continuation_max_step):.6g}. Rejected rows retain the previous accepted K."
            )
        else:
            anchor_note = (
                "L0=0.35 is anchored to locked constants."
                if anchor_locked
                else "L0=0.35 is recalculated; no locked row is used."
            )
            note = (
                f"Generated with schedule_source={args.schedule_source}. "
                f"{anchor_note}"
            )
        output = {
            "note": note,
            "interpolation": "linear",
            "points": rows,
        }
        schedule_path = args.write_schedule if args.write_schedule.is_absolute() else PROJECT_ROOT / args.write_schedule
        schedule_path.parent.mkdir(parents=True, exist_ok=True)
        with schedule_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(output, file, allow_unicode=True, sort_keys=False)
        print(f"length_schedule_written: {schedule_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
