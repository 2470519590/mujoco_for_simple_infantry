"""smoke 模块共享的小型 MuJoCo 辅助函数。"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np

_NAME_ID_CACHE: dict[tuple[int, int, str], int] = {}
_JACOBIAN_BUFFER_CACHE: dict[tuple[int, str, int], np.ndarray] = {}


def prepare_mujoco_import() -> None:
    if os.name == "nt" and os.environ.get("MUJOCO_GL", "").lower() == "osmesa":
        os.environ.pop("MUJOCO_GL")


def load_mujoco():
    prepare_mujoco_import()
    import mujoco  # pylint: disable=import-outside-toplevel

    return mujoco


def name(mujoco, model, obj_type, index: int) -> str:
    value = mujoco.mj_id2name(model, obj_type, index)
    return value if value is not None else f"<unnamed:{index}>"


def iter_joint_names(mujoco, model) -> Iterable[str]:
    for joint_id in range(model.njnt):
        yield name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)


def iter_actuator_names(mujoco, model) -> Iterable[str]:
    for actuator_id in range(model.nu):
        yield name(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)


def assert_finite(label: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{label} contains NaN or Inf")


def rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def step_with_ctrl(mujoco, model, steps: int, ctrl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    data.ctrl[:] = ctrl
    for _ in range(steps):
        mujoco.mj_step(model, data)
        assert_finite("qpos", data.qpos)
        assert_finite("qvel", data.qvel)
    return data.qpos.copy(), data.qvel.copy()


def joint_for_actuator(mujoco, model, actuator_id: int) -> tuple[int, str]:
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    if joint_id < 0:
        return joint_id, "<no-joint>"
    return joint_id, name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)


def id_by_name(mujoco, model, obj_type, object_name: str) -> int:
    key = (id(model), int(obj_type), object_name)
    cached = _NAME_ID_CACHE.get(key)
    if cached is not None:
        return cached
    obj_id = mujoco.mj_name2id(model, obj_type, object_name)
    if obj_id < 0:
        raise RuntimeError(f"missing MuJoCo object: {object_name}")
    obj_id = int(obj_id)
    _NAME_ID_CACHE[key] = obj_id
    return obj_id


def _jacobian_buffer(model, label: str) -> np.ndarray:
    key = (id(model), label, int(model.nv))
    cached = _JACOBIAN_BUFFER_CACHE.get(key)
    if cached is None:
        cached = np.zeros((3, model.nv), dtype=float)
        _JACOBIAN_BUFFER_CACHE[key] = cached
    else:
        cached.fill(0.0)
    return cached


def body_linear_velocity(mujoco, model, data, body_id: int) -> np.ndarray:
    jacp = _jacobian_buffer(model, "body_jacp")
    mujoco.mj_jacBody(model, data, jacp, None, body_id)
    return jacp @ data.qvel


def body_pos_vel(mujoco, model, data, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    body_id = id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id].copy(), body_linear_velocity(mujoco, model, data, body_id)


def body_pos(mujoco, model, data, body_name: str) -> np.ndarray:
    body_id = id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id].copy()


def site_pos_vel(mujoco, model, data, site_name: str) -> tuple[np.ndarray, np.ndarray]:
    site_id = id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    jacp = _jacobian_buffer(model, "site_jacp")
    mujoco.mj_jacSite(model, data, jacp, None, site_id)
    return data.site_xpos[site_id].copy(), jacp @ data.qvel


def site_pos(mujoco, model, data, site_name: str) -> np.ndarray:
    site_id = id_by_name(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[site_id].copy()


def copy_data(mujoco, model, source_data):
    data = mujoco.MjData(model)
    data.qpos[:] = source_data.qpos
    data.qvel[:] = source_data.qvel
    data.act[:] = source_data.act
    data.ctrl[:] = source_data.ctrl
    mujoco.mj_forward(model, data)
    return data


def lock_base_to_initial(mujoco, model, data) -> None:
    data.qpos[:7] = model.qpos0[:7]
    data.qvel[:6] = 0.0
    mujoco.mj_forward(model, data)


def lock_base_to_qpos(mujoco, model, data, base_qpos: np.ndarray) -> None:
    data.qpos[:7] = base_qpos
    data.qvel[:6] = 0.0
    mujoco.mj_forward(model, data)
