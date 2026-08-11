"""由 YAML 支持的本地 smoke 实验运行配置。"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import yaml

from .constants import PROJECT_ROOT


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "smoke.yaml"


@dataclass(frozen=True)
class RuntimeControlConfig:
    """Runtime tuning values loaded from smoke.yaml."""

    leg_sync_input_lowpass_hz: float
    leg_sync_error_lowpass_hz: float
    leg_sync_derivative_lowpass_hz: float
    yaw_turn_input_lowpass_hz: float
    yaw_turn_error_lowpass_hz: float
    yaw_turn_derivative_lowpass_hz: float
    roll_reference: float
    roll_force_kp: float


class RunConfig:
    """Mutable named configuration passed through the smoke runner."""

    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)

    @classmethod
    def from_namespace(cls, namespace: Namespace) -> "RunConfig":
        return cls(**vars(namespace))


def runtime_control_config(config: RunConfig) -> RuntimeControlConfig:
    """Build the runtime control tuning bundle from the YAML-backed config."""
    return RuntimeControlConfig(
        leg_sync_input_lowpass_hz=float(config.leg_sync_input_lowpass_hz),
        leg_sync_error_lowpass_hz=float(config.leg_sync_error_lowpass_hz),
        leg_sync_derivative_lowpass_hz=float(config.leg_sync_derivative_lowpass_hz),
        yaw_turn_input_lowpass_hz=float(config.yaw_turn_input_lowpass_hz),
        yaw_turn_error_lowpass_hz=float(config.yaw_turn_error_lowpass_hz),
        yaw_turn_derivative_lowpass_hz=float(config.yaw_turn_derivative_lowpass_hz),
        roll_reference=float(config.roll_reference),
        roll_force_kp=float(config.roll_force_kp),
    )


def load_yaml_defaults(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    """Load flat runner defaults from YAML without changing CLI overrides."""
    with path.open(encoding="utf-8") as file:
        values = yaml.safe_load(file) or {}
    if not isinstance(values, dict):
        raise ValueError(f"smoke config must be a mapping: {path}")
    return values
