"""用于现有 smoke rollout 的 MuJoCo viewer 观察器。"""

from __future__ import annotations

import time
from collections.abc import Callable


class MujocoViewerObserver:
    """Render the exact MjData stepped by an experiment, without a second loop."""

    def __init__(
        self,
        mujoco,
        model,
        realtime: bool,
        sync_hz: float = 60.0,
        key_callback: Callable[[int], None] | None = None,
        overlay_provider: Callable[[], tuple[str, str]] | None = None,
    ) -> None:
        self._mujoco = mujoco
        self._model = model
        self._realtime = realtime
        self._key_callback = key_callback
        self._overlay_provider = overlay_provider
        self._tracked_body_id = -1
        self._context = None
        self._viewer = None
        self._wall_start = 0.0
        self._sim_start = 0.0
        self._last_sync_sim = 0.0
        self._sync_interval = 1.0 / max(float(sync_hz), 1.0)
        self._phase = "ground"

    def start(self, data):
        import mujoco.viewer  # pylint: disable=import-outside-toplevel

        self._context = mujoco.viewer.launch_passive(
            self._model,
            data,
            key_callback=self._key_callback,
        )
        self._viewer = self._context.__enter__()
        self._sim_start = float(data.time)
        self._last_sync_sim = self._sim_start
        self._wall_start = time.perf_counter()
        self._configure_tracking_camera(data)
        self._add_overlay(data)
        self._viewer.sync()
        return self.step

    def _configure_tracking_camera(self, data) -> None:
        if self._viewer is None:
            return
        try:
            body_id = self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_BODY, "base")
            if body_id < 0:
                return
            self._tracked_body_id = int(body_id)
            cam = self._viewer.cam
            cam.type = self._mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = self._tracked_body_id
            cam.distance = 2.2
            cam.azimuth = 135.0
            cam.elevation = -18.0
            cam.lookat[:] = data.xpos[self._tracked_body_id]
        except Exception:  # pragma: no cover - viewer camera API differs across mujoco builds.
            self._tracked_body_id = -1

    def _update_camera_target(self, data) -> None:
        if self._viewer is None or self._tracked_body_id < 0:
            return
        try:
            self._viewer.cam.lookat[:] = data.xpos[self._tracked_body_id]
        except Exception:  # pragma: no cover - viewer camera API differs across mujoco builds.
            return

    def _add_overlay(self, data) -> None:
        if self._viewer is None:
            return
        label = "time\nphase"
        value = f"{float(data.time):.3f} s\n{self._phase}"
        if self._overlay_provider is not None:
            extra_label, extra_value = self._overlay_provider()
            label = f"{label}\n{extra_label}"
            value = f"{value}\n{extra_value}"
        labels = (label, value)
        try:
            if hasattr(self._viewer, "set_texts"):
                self._viewer.set_texts((None, self._mujoco.mjtGridPos.mjGRID_TOPLEFT, labels[0], labels[1]))
            elif hasattr(self._viewer, "add_overlay"):
                self._viewer.add_overlay(self._mujoco.mjtGridPos.mjGRID_TOPLEFT, labels[0], labels[1])
        except Exception:  # pragma: no cover - viewer overlay support depends on mujoco build.
            return

    def step(self, data, _step: int, phase: str | None = None) -> bool:
        if self._viewer is None or not self._viewer.is_running():
            return False
        if phase is not None:
            self._phase = phase
        if float(data.time) - self._last_sync_sim >= self._sync_interval:
            self._update_camera_target(data)
            self._add_overlay(data)
            self._viewer.sync()
            self._last_sync_sim = float(data.time)
        if self._realtime:
            target_elapsed = float(data.time) - self._sim_start
            delay = target_elapsed - (time.perf_counter() - self._wall_start)
            if delay > 0.0:
                time.sleep(delay)
        return True

    def close(self) -> None:
        if self._context is not None:
            self._context.__exit__(None, None, None)
            self._context = None
            self._viewer = None
