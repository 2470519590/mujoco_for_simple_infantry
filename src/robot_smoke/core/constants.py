"""MuJoCo smoke 脚本共享常量。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = PROJECT_ROOT / "assets" / "biped_wheel_leg.xml"
DEFAULT_VIRTUAL_ROD_LENGTH = 0.35

LOCKED_EQUILIBRIUM_L0 = 0.35
LOCKED_EQUILIBRIUM_THETA = 0.0
LOCKED_EQUILIBRIUM_FL0_SCALE = 0.68
LOCKED_EQUILIBRIUM_STEPS = 5000
LOCKED_EQUILIBRIUM_EVAL_STEPS = 2000
LOCKED_EQUILIBRIUM_LENGTH_KP = 400.0
LOCKED_EQUILIBRIUM_LENGTH_KD = 80.0
LOCKED_EQUILIBRIUM_THETA_KP = 3.0
LOCKED_EQUILIBRIUM_THETA_KD = 2.0
LOCKED_EQUILIBRIUM_PITCH_KP = 3.0
LOCKED_EQUILIBRIUM_PITCH_KD = 2.0
LOCKED_EQUILIBRIUM_WHEEL_COM_KP = 80.0
LOCKED_EQUILIBRIUM_WHEEL_DAMPING = 0.5
LOCKED_EQUILIBRIUM_TP_BIAS = 0.0
LOCKED_EQUILIBRIUM_FL0 = 34.5262
LOCKED_EQUILIBRIUM_QPOS = np.array(
    [
        -0.00691331, 1.51215e-07, 0.484628, 1.0, 2.74755e-11, -0.000647474,
        3.79926e-08, -0.217659, 0.437656, 0.00083341, -0.330512, 0.220818,
        -0.43906, -0.217659, 0.437656, 0.00083341, -0.330511, 0.220818,
        -0.43906,
    ],
    dtype=float,
)
LOCKED_LQR_U0 = np.array([-0.00319492, -0.000686043], dtype=float)
LOCKED_LQR_X0 = np.array([5.55636e-05, 0.0, -0.0068291, 0.0, -0.00129495, 0.0], dtype=float)
LOCKED_LQR_K = np.array(
    [
        [76.1803, 11.733, -8.25263, -19.1209, -13.3707, -7.18113],
        [-5.12679, -0.648237, 0.366758, 0.910707, -1.53501, -1.09064],
    ],
    dtype=float,
)
LOCKED_LQR_DESIGN_STEPS = 5
LOCKED_LQR_CONTROL_PERIOD_STEPS = 5

# Local physical input map measured at L=0.35 m, theta_world=0, free base.
# Rows are Xdot=[theta_dot, theta_ddot, x_dot, x_ddot, phi_dot, phi_ddot].
# Columns are the physical commands [T, Tp] before runtime sign conventions.
MEASURED_LQR_B_CONTINUOUS = np.array(
    [
        [0.0, 0.0],
        [5.7643581, 5.5843861],
        [0.0, 0.0],
        [2.0242316, 1.0726022],
        [0.0, 0.0],
        [-1.8469763, 7.6178206],
    ],
    dtype=float,
)

# The sagittal model assumes both virtual legs move in phase. The 3-D model
# needs a small differential damper to enforce that assumption.
LEG_THETA_SYNC_KP = 30
LEG_THETA_SYNC_KD = 20
LEG_SYNC_INPUT_LOWPASS_HZ = 500
LEG_SYNC_ERROR_LOWPASS_HZ = 10
LEG_SYNC_DERIVATIVE_LOWPASS_HZ = 10

YAW_TURN_KP = 3
YAW_TURN_KD = 0.1
YAW_TURN_INPUT_LOWPASS_HZ = 500
YAW_TURN_ERROR_LOWPASS_HZ = 10
YAW_TURN_DERIVATIVE_LOWPASS_HZ = 10

DEFAULT_LQR_K = np.array(
    [
        [-44.3788, -6.8496, -22.2828, -21.5569, 28.7706, 4.3751],
        [11.2006, 0.7339, 3.73, 3.2058, 151.73, 4.6387],
    ],
    dtype=float,
)
