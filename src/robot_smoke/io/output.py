"""转向控制器绘图输出。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.constants import PROJECT_ROOT
from ..core.types import LqrHistorySample


def resolve_output_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _series(samples: tuple[LqrHistorySample, ...], name: str, mask: np.ndarray) -> np.ndarray:
    return np.array([getattr(sample, name) for sample in samples], dtype=float)[mask]


def plot_turning_history(
    path: Path,
    samples: tuple[LqrHistorySample, ...],
    yaw_kp: float,
    yaw_kd: float,
    sync_kp: float,
    sync_kd: float,
) -> None:
    """Plot raw/filter signals and P/D contributions for the turn controller."""
    if not samples:
        return
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    path.parent.mkdir(parents=True, exist_ok=True)
    raw_time_s = np.array([sample.time_s for sample in samples], dtype=float)
    mask = raw_time_s <= 6.0
    time_s = raw_time_s[mask]
    yaw_ref = _series(samples, "yaw_rate_reference", mask)
    yaw_raw = _series(samples, "yaw_rate", mask)
    yaw_filtered = _series(samples, "yaw_rate_filtered", mask)
    yaw_d_raw = _series(samples, "yaw_error_rate_raw", mask)
    yaw_d_filtered = _series(samples, "yaw_error_rate", mask)
    yaw_p = _series(samples, "yaw_p_torque", mask)
    yaw_d = _series(samples, "yaw_d_torque", mask)
    yaw_total = _series(samples, "turn_torque", mask)
    theta_left = _series(samples, "left_theta", mask)
    theta_right = _series(samples, "right_theta", mask)
    sync_raw = _series(samples, "sync_error_raw", mask)
    sync_filtered = _series(samples, "sync_error", mask)
    sync_d_raw = _series(samples, "sync_error_rate_raw", mask)
    sync_d_filtered = _series(samples, "sync_error_rate", mask)
    sync_p = _series(samples, "sync_p_torque", mask)
    sync_d = _series(samples, "sync_d_torque", mask)
    sync_total = _series(samples, "sync_torque", mask)

    figure, axes = plt.subplots(6, 1, figsize=(10, 14), sharex=True)
    axes[0].plot(time_s, yaw_ref, label="yaw_rate ref")
    axes[0].plot(time_s, yaw_raw, label="yaw_rate raw")
    axes[0].plot(time_s, yaw_filtered, "--", label="yaw_rate input LPF")
    axes[0].set_ylabel("rad/s")
    axes[1].plot(time_s, yaw_d_raw, alpha=0.55, label="yaw D raw")
    axes[1].plot(time_s, yaw_d_filtered, label="yaw D LPF")
    axes[1].set_ylabel("rad/s^2")
    axes[2].plot(time_s, yaw_p, label="yaw P")
    axes[2].plot(time_s, yaw_d, label="yaw D")
    axes[2].plot(time_s, yaw_total, linewidth=1.5, label="tau_turn")
    axes[2].set_ylabel("N m")
    axes[3].plot(time_s, theta_left, label="theta_left")
    axes[3].plot(time_s, theta_right, label="theta_right")
    axes[3].plot(time_s, sync_raw, alpha=0.6, label="sync error raw")
    axes[3].plot(time_s, sync_filtered, "--", label="sync error LPF")
    axes[3].set_ylabel("rad")
    axes[4].plot(time_s, sync_d_raw, alpha=0.55, label="sync D raw")
    axes[4].plot(time_s, sync_d_filtered, label="sync D LPF")
    axes[4].set_ylabel("rad/s")
    axes[5].plot(time_s, sync_p, label="sync P")
    axes[5].plot(time_s, sync_d, label="sync D")
    axes[5].plot(time_s, sync_total, linewidth=1.5, label="Tp_sync")
    axes[5].set_ylabel("N m")
    axes[5].set_xlabel("time (s)")
    for axis in axes:
        axis.legend(loc="upper right")
        axis.grid(True)
    figure.suptitle(
        f"Yaw PD: Kp={yaw_kp:g}, Kd={yaw_kd:g} | "
        f"Leg-sync PD: Kp={sync_kp:g}, Kd={sync_kd:g} | 0-6 s"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_roll_length_history(
    path: Path,
    samples: tuple[LqrHistorySample, ...],
    length_kp: float,
    length_ki: float,
    length_kd: float,
    roll_force_kp: float,
) -> None:
    """Plot every measured and commanded quantity in the article 2.2 loop."""
    if not samples:
        return
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    path.parent.mkdir(parents=True, exist_ok=True)
    time_s = np.array([sample.time_s for sample in samples], dtype=float)
    left_length_ref = _series(samples, "left_length_reference", np.ones(len(samples), dtype=bool))
    right_length_ref = _series(samples, "right_length_reference", np.ones(len(samples), dtype=bool))
    left_length = _series(samples, "left_length", np.ones(len(samples), dtype=bool))
    right_length = _series(samples, "right_length", np.ones(len(samples), dtype=bool))
    left_length_rate = _series(samples, "left_length_rate", np.ones(len(samples), dtype=bool))
    right_length_rate = _series(samples, "right_length_rate", np.ones(len(samples), dtype=bool))
    roll = _series(samples, "roll", np.ones(len(samples), dtype=bool))
    roll_ref = _series(samples, "roll_reference", np.ones(len(samples), dtype=bool))
    left_theta_deg = np.degrees(_series(samples, "left_theta", np.ones(len(samples), dtype=bool)))
    right_theta_deg = np.degrees(_series(samples, "right_theta", np.ones(len(samples), dtype=bool)))
    left_output = _series(samples, "left_length_force", np.ones(len(samples), dtype=bool))
    right_output = _series(samples, "right_length_force", np.ones(len(samples), dtype=bool))
    left_force_bias = _series(samples, "left_length_force_ff", np.ones(len(samples), dtype=bool))
    right_force_bias = _series(samples, "right_length_force_ff", np.ones(len(samples), dtype=bool))
    roll_force = _series(samples, "roll_force", np.ones(len(samples), dtype=bool))
    left_contact = _series(samples, "left_contact_normal_force", np.ones(len(samples), dtype=bool))
    right_contact = _series(samples, "right_contact_normal_force", np.ones(len(samples), dtype=bool))

    figure, axes = plt.subplots(6, 1, figsize=(11, 15), sharex=True)
    axes[0].plot(time_s, left_length_ref, "--", label="left L_d")
    axes[0].plot(time_s, left_length, label="left L")
    axes[0].plot(time_s, right_length_ref, "--", label="right L_d")
    axes[0].plot(time_s, right_length, label="right L")
    axes[0].set_ylabel("m")
    axes[1].plot(time_s, left_length_rate, label="left dL")
    axes[1].plot(time_s, right_length_rate, label="right dL")
    axes[1].set_ylabel("m/s")
    axes[2].plot(time_s, roll_ref, "--", label="roll reference")
    axes[2].plot(time_s, roll, label="roll")
    axes[2].set_ylabel("rad")
    axes[3].plot(time_s, left_theta_deg, label="left theta")
    axes[3].plot(time_s, right_theta_deg, label="right theta")
    axes[3].axhline(0.0, color="black", linestyle="--", linewidth=1.0, label="theta reference = 0 deg")
    axes[3].set_ylabel("deg")
    axes[4].plot(time_s, left_output, linewidth=1.5, label="left F_l output")
    axes[4].plot(time_s, right_output, linewidth=1.5, label="right F_l output")
    axes[4].plot(time_s, left_force_bias, "--", label="left F_base + F_roll")
    axes[4].plot(time_s, right_force_bias, "--", label="right F_base - F_roll")
    axes[4].plot(time_s, roll_force, ":", label="F_roll")
    axes[4].set_ylabel("N")
    axes[5].plot(time_s, left_contact, label="left F_N")
    axes[5].plot(time_s, right_contact, label="right F_N")
    axes[5].axhline(20.0, color="black", linestyle="--", linewidth=1.0, label="airborne threshold 20 N")
    axes[5].set_ylabel("N")
    axes[5].set_xlabel("time (s)")
    for axis in axes:
        axis.legend(loc="upper right")
        axis.grid(True)
    figure.suptitle(
        "Article 2.2 leg length and roll control | "
        f"Kp,L={length_kp:g}, Ki,L={length_ki:g}, Kd,L={length_kd:g}, "
        f"K_gamma={roll_force_kp:g} N/rad"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_lqr_debug_history(
    path: Path,
    samples: tuple[LqrHistorySample, ...],
    *,
    title: str,
) -> None:
    """Plot the middle-layer balance quantities needed for LQR diagnosis."""
    if not samples:
        return
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.ones(len(samples), dtype=bool)
    time_s = np.array([sample.time_s for sample in samples], dtype=float)
    x = _series(samples, "x", mask)
    theta = _series(samples, "theta", mask)
    pitch = _series(samples, "pitch", mask)
    x_rate = _series(samples, "x_rate", mask)
    x_velocity_reference = _series(samples, "x_velocity_reference", mask)
    wheel_torque = _series(samples, "wheel_torque", mask)
    pitch_torque = _series(samples, "pitch_torque", mask)
    left_pitch_torque = _series(samples, "left_pitch_torque", mask)
    right_pitch_torque = _series(samples, "right_pitch_torque", mask)
    left_length = _series(samples, "left_length", mask)
    right_length = _series(samples, "right_length", mask)
    left_length_ref = _series(samples, "left_length_reference", mask)
    right_length_ref = _series(samples, "right_length_reference", mask)
    left_wheel_z = _series(samples, "left_wheel_z", mask)
    right_wheel_z = _series(samples, "right_wheel_z", mask)
    left_contact = _series(samples, "left_contact_normal_force", mask)
    right_contact = _series(samples, "right_contact_normal_force", mask)
    airborne = np.array([sample.airborne for sample in samples], dtype=bool)
    landing_phases = [sample.landing_phase for sample in samples]

    figure, axes = plt.subplots(9, 1, figsize=(12, 21), sharex=True)
    axes[0].plot(time_s, x, label="x - x_ref")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0, label="x_ref")
    axes[0].set_ylabel("m")

    axes[1].plot(time_s, x_velocity_reference, "--", label="dx_ref")
    axes[1].plot(time_s, x_rate, label="dx - dx_ref")
    axes[1].set_ylabel("m/s")

    axes[2].plot(time_s, np.degrees(pitch), label="pitch")
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("deg")

    axes[3].plot(time_s, np.degrees(theta), label="theta")
    axes[3].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[3].set_ylabel("deg")

    axes[4].plot(time_s, wheel_torque, label="T wheel")
    axes[4].set_ylabel("N m")

    axes[5].plot(time_s, pitch_torque, label="Tp LQR")
    axes[5].plot(time_s, left_pitch_torque, "--", label="Tp left")
    axes[5].plot(time_s, right_pitch_torque, "--", label="Tp right")
    axes[5].set_ylabel("N m")

    axes[6].plot(time_s, left_length_ref, "--", label="left L_ref")
    axes[6].plot(time_s, left_length, label="left L")
    axes[6].plot(time_s, right_length_ref, "--", label="right L_ref")
    axes[6].plot(time_s, right_length, label="right L")
    axes[6].set_ylabel("m")

    axes[7].plot(time_s, left_wheel_z, label="left wheel z")
    axes[7].plot(time_s, right_wheel_z, label="right wheel z")
    axes[7].axhline(0.085, color="black", linestyle="--", linewidth=1.0, label="wheel radius")
    axes[7].set_ylabel("m")

    axes[8].plot(time_s, left_contact, label="left F_N")
    axes[8].plot(time_s, right_contact, label="right F_N")
    axes[8].axhline(20.0, color="black", linestyle="--", linewidth=1.0, label="airborne threshold 20 N")
    phase_colors = {
        "airborne": "tab:red",
        "landing_hold": "tab:green",
        "ground": "0.35",
    }
    last_phase = landing_phases[0] if landing_phases else "ground"
    phase_labels_used: set[str] = set()
    for index, phase in enumerate(landing_phases[1:], start=1):
        if phase == last_phase:
            continue
        color = phase_colors.get(phase, "0.35")
        label = f"phase: {phase}" if phase not in phase_labels_used else None
        for axis in axes:
            axis.axvline(time_s[index], color=color, linestyle=":", linewidth=1.2, alpha=0.85, label=label)
            label = None
        phase_labels_used.add(phase)
        last_phase = phase
    previous_airborne = airborne[0] if len(airborne) else False
    start_label_used = False
    end_label_used = False
    for index, current_airborne in enumerate(airborne[1:], start=1):
        if current_airborne == previous_airborne:
            continue
        if current_airborne:
            label = "airborne start" if not start_label_used else None
            axes[8].axvline(time_s[index], color="tab:red", linestyle=":", linewidth=1.2, label=label)
            start_label_used = True
        else:
            label = "airborne end" if not end_label_used else None
            axes[8].axvline(time_s[index], color="tab:green", linestyle=":", linewidth=1.2, label=label)
            end_label_used = True
        previous_airborne = current_airborne
    axes[8].set_ylabel("N")
    axes[8].set_xlabel("time (s)")

    for axis in axes:
        axis.legend(loc="upper right")
        axis.grid(True)
    figure.suptitle(title)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(path, dpi=160)
    plt.close(figure)
