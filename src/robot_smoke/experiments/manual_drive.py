"""手动驾驶场景的键盘指令状态。"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from ..control.turning import turn_rate_magnitude


MANUAL_FORWARD_SPEED = 2.0
MANUAL_TURN_SPEED_TIER = "medium"
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28


def _windows_key_down(virtual_key: int) -> bool:
    if sys.platform != "win32":
        return False
    try:
        import ctypes  # pylint: disable=import-outside-toplevel

        return bool(ctypes.windll.user32.GetAsyncKeyState(int(virtual_key)) & 0x8000)
    except Exception:
        return False


@dataclass
class ManualDriveInput:
    """把 viewer 按键转换成速度和转向参考。"""

    speed_accel_limit: float = 1.2
    yaw_accel_limit: float = 4.0
    _forward_direction: int = 0
    _turn_direction: int = 0
    _speed_reference: float = 0.0
    _yaw_reference: float = 0.0

    def key_callback(self, key_code: int) -> None:
        return

    def update(self, dt: float) -> None:
        self._poll_keyboard()
        forward_direction = self._forward_direction
        turn_direction = self._turn_direction
        target_speed = forward_direction * MANUAL_FORWARD_SPEED
        target_yaw = turn_direction * turn_rate_magnitude(MANUAL_TURN_SPEED_TIER)
        self._speed_reference = _slew(self._speed_reference, target_speed, self.speed_accel_limit * dt)
        self._yaw_reference = _slew(self._yaw_reference, target_yaw, self.yaw_accel_limit * dt)

    def _poll_keyboard(self) -> None:
        up = _windows_key_down(VK_UP)
        down = _windows_key_down(VK_DOWN)
        left = _windows_key_down(VK_LEFT)
        right = _windows_key_down(VK_RIGHT)
        if up or down:
            self._forward_direction = (1 if up else 0) - (1 if down else 0)
        else:
            self._forward_direction = 0
        if left or right:
            self._turn_direction = (1 if left else 0) - (1 if right else 0)
        else:
            self._turn_direction = 0
    def forward_speed_reference(self) -> float:
        return self._speed_reference

    def yaw_rate_reference(self) -> float:
        return self._yaw_reference

    def overlay_text(self) -> tuple[str, str]:
        forward_text = {1: "forward", -1: "backward", 0: "stop"}[self._forward_direction]
        turn_text = {1: "left", -1: "right", 0: "stop"}[self._turn_direction]
        return (
            "speed\nv ref\nyaw ref\nmove\nturn\nkeys",
            (
                "medium\n"
                f"{self.forward_speed_reference():.2f} m/s\n"
                f"{self.yaw_rate_reference():.2f} rad/s\n"
                f"{forward_text}\n"
                f"{turn_text}\n"
                "Arrow keys"
            ),
        )


def _slew(current: float, target: float, max_delta: float) -> float:
    max_delta = max(float(max_delta), 0.0)
    if current < target:
        return min(current + max_delta, target)
    if current > target:
        return max(current - max_delta, target)
    return current
