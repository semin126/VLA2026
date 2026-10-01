"""SortingEnv — LIBERO 호환 관측/액션을 제공하는 MuJoCo 정리 환경.

관측 키는 LIBERO(robosuite) 관례를 따른다:
    agentview_image, sideview_image, robot0_eye_in_hand_image,
    robot0_joint_pos, robot0_gripper_qpos, robot0_eef_pos, robot0_eef_quat
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from . import config as C
from . import scene as scene_mod
from . import accessibility as acc
from .controller import OSCController, TOP_DOWN_MAT, mat_to_quat

# 렌더 백엔드: 화면이 있으면 glfw(윈도우/리눅스 데스크톱), 없으면 egl(Colab/서버).
if not os.environ.get("MUJOCO_GL"):
    import sys as _sys
    _has_display = _sys.platform.startswith("win") or _sys.platform == "darwin" or bool(os.environ.get("DISPLAY"))
    os.environ["MUJOCO_GL"] = "glfw" if _has_display else "egl"

import atexit  # noqa: E402

import mujoco  # noqa: E402

_OPEN_ENVS: list = []


@atexit.register
def _close_all_envs() -> None:
    """인터프리터 종료 시 렌더러를 먼저 닫는다.
    안 그러면 EGL 컨텍스트가 이미 사라진 뒤 소멸자가 돌면서
    무서워 보이는(하지만 무해한) EGLError 트레이스백이 찍힌다."""
    for env in list(_OPEN_ENVS):
        try:
            env.close()
        except Exception:
            pass


class SortingEnv:
    def __init__(
        self,
        layout: str = "tidy",
        seed: int = 0,
        image_size: int = C.IMAGE_SIZE,
        cameras: tuple[str, ...] = C.CAMERA_NAMES,
        scene_path: Path | None = None,
        build_scene: bool = True,
    ):
        self.image_size = image_size
        self.cameras = tuple(cameras)
        if build_scene or scene_path is None or not Path(scene_path).exists():
            self.scene_path = scene_mod.write_scene("tidy", path=scene_path)
        else:
            self.scene_path = Path(scene_path)   # 이미 있는 씬 XML 을 그대로 사용
        # 경로에 한글이 있으면 MuJoCo 가 파일을 못 연다 -> config.mujoco_path 참고
        self.model = mujoco.MjModel.from_xml_path(C.mujoco_path(self.scene_path))
        self.data = mujoco.MjData(self.model)

        self._jnt_adr = {
            n: self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_freejoint")
            ]
            for n in C.OBJECT_NAMES
        }
        self._arm_adr = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in C.ARM_JOINTS
        ])
        self._finger_adr = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in C.FINGER_JOINTS
        ])
        # 중력 보상 대상 DOF (팔 7 + 손가락 2). 실제 Franka 제어기도 중력 보상을 한다.
        # 이걸 안 하면 위치 제어 정상상태 오차가 6~16mm 생겨 파지가 어긋난다.
        self._gc_dof = np.array([
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in (C.ARM_JOINTS + C.FINGER_JOINTS)
        ])
        self.gravity_comp = True
        self.ctrl = OSCController(self.model, self.data)
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self.home_qpos = self._compute_home_qpos()
        self.step_count = 0
        _OPEN_ENVS.append(self)
        self.reset(layout=layout, seed=seed)

    # -- 초기화 ---------------------------------------------------------
    def _compute_home_qpos(self) -> np.ndarray:
        """작업 구역 가운데 위 25cm 를 내려다보는 자세를 IK 로 구한다."""
        # 카메라(top-down)를 최대한 덜 가리도록 로봇 쪽으로 당기고 높게 든 자세.
        target = np.array([0.0, C.WORK_Y[0] + 0.13, C.TABLE_TOP_Z + 0.34])
        q, ok = self.ctrl.ik.solve(target, mat_to_quat(TOP_DOWN_MAT), np.array(C.HOME_QPOS))
        if not ok:
            q = np.array(C.HOME_QPOS)
        return q

    def _sim_step(self, n: int = 1) -> None:
        for _ in range(n):
            if self.gravity_comp:
                self.data.qfrc_applied[self._gc_dof] = self.data.qfrc_bias[self._gc_dof]
            mujoco.mj_step(self.model, self.data)

    def reset(self, layout: str = "tidy", seed: int = 0, settle_steps: int = 260):
        mujoco.mj_resetData(self.model, self.data)
        fn = scene_mod.LAYOUTS[layout]
        lay = fn() if layout == "tidy" else fn(seed)
        self.layout_name, self.layout = layout, lay

        self.data.qpos[self._arm_adr] = self.home_qpos
        self.data.qpos[self._finger_adr] = C.GRIPPER_QPOS_OPEN
        for name, (x, y, yaw) in lay.items():
            self.set_object_pose(name, x, y, yaw, forward=False)

        for i, aid in enumerate(self.ctrl.act_ids):
            self.data.ctrl[aid] = self.home_qpos[i]
        self.data.ctrl[self.ctrl.grip_act_id] = C.GRIPPER_CTRL_OPEN
        mujoco.mj_forward(self.model, self.data)
        self._sim_step(settle_steps)
        self.ctrl.sync_target_to_current()
        self.ctrl.last_gripper = C.GRIPPER_OPEN_CMD
        self.step_count = 0
        return self.get_obs()

    # -- 물체 ------------------------------------------------------------
    def set_object_pose(self, name: str, x: float, y: float, yaw: float = 0.0,
                        forward: bool = True) -> None:
        a = self._jnt_adr[name]
        hz = C.half_size(C.OBJECT_BY_NAME[name])[2]
        self.data.qpos[a:a + 3] = [x, y, C.TABLE_TOP_Z + hz + 0.0015]
        self.data.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        self.data.qvel[self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_freejoint")]:][:6] = 0
        if forward:
            mujoco.mj_forward(self.model, self.data)

    def object_pos(self, name: str) -> np.ndarray:
        a = self._jnt_adr[name]
        return self.data.qpos[a:a + 3].copy()

    def object_yaw(self, name: str) -> float:
        a = self._jnt_adr[name]
        qw, qx, qy, qz = self.data.qpos[a + 3:a + 7]
        return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

    def object_positions(self) -> dict[str, list[float]]:
        return {n: self.object_pos(n).round(4).tolist() for n in C.OBJECT_NAMES}

    def zone_of(self, name: str) -> str | None:
        x, y, _ = self.object_pos(name)
        for zname, z in C.ZONES.items():
            if z["x"][0] <= x <= z["x"][1] and z["y"][0] <= y <= z["y"][1]:
                return zname
        return None

    def misplaced_objects(self) -> list[str]:
        return [o["name"] for o in C.OBJECTS if self.zone_of(o["name"]) != o["home_zone"]]

    def is_tidy(self) -> bool:
        return not self.misplaced_objects()

    # -- 관측 ------------------------------------------------------------
    def _renderer(self, h: int, w: int) -> "mujoco.Renderer":
        key = (h, w)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=h, width=w)
        return self._renderers[key]

    def render(self, camera: str = "agentview", height: int | None = None,
               width: int | None = None) -> np.ndarray:
        h = height or self.image_size
        w = width or self.image_size
        r = self._renderer(h, w)
        r.update_scene(self.data, camera=camera)
        return r.render().copy()

    def get_obs(self) -> dict:
        obs: dict = {}
        for cam in self.cameras:
            obs[f"{cam}_image"] = self.render(cam)
        obs["robot0_joint_pos"] = self.data.qpos[self._arm_adr].copy().astype(np.float32)
        obs["robot0_gripper_qpos"] = self.data.qpos[self._finger_adr].copy().astype(np.float32)
        obs["robot0_eef_pos"] = self.ctrl.eef_pos().astype(np.float32)
        obs["robot0_eef_quat"] = self.ctrl.eef_quat().astype(np.float32)
        return obs

    @staticmethod
    def state_vector(obs: dict) -> np.ndarray:
        return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel()
                               for k in C.STATE_KEYS]).astype(np.float32)

    # -- 진행 ------------------------------------------------------------
    def step(self, action, n_sim: int = C.SIM_STEPS_PER_ACTION):
        self.ctrl.apply_action(action)
        self._sim_step(n_sim)
        self.step_count += 1
        obs = self.get_obs()
        info = {"is_tidy": self.is_tidy(), "misplaced": self.misplaced_objects(),
                "step": self.step_count}
        return obs, 1.0 if info["is_tidy"] else 0.0, False, info

    def close(self) -> None:
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:
                pass
        self._renderers.clear()
        if self in _OPEN_ENVS:
            _OPEN_ENVS.remove(self)

    # -- 접근성 (제목의 '물체 접근성 및 공간 제약') -----------------------
    # 기하 판정은 vlapipe/accessibility.py 에 순수 함수로 두었다.
    # 플래너가 가상 상태로 같은 판정을 해야 하기 때문이다.

    def object_xy_yaw(self) -> tuple[dict, dict]:
        pos = {n: tuple(self.object_pos(n)[:2]) for n in C.OBJECT_NAMES}
        yaw = {n: self.object_yaw(n) for n in C.OBJECT_NAMES}
        return pos, yaw

    def blocking_objects(self, name: str) -> list[str]:
        pos, yaw = self.object_xy_yaw()
        return acc.blockers(name, pos, yaw)

    def is_reachable(self, name: str) -> bool:
        return acc.is_reachable(self.object_pos(name)[:2])

    def is_accessible(self, name: str) -> tuple[bool, str]:
        pos, yaw = self.object_xy_yaw()
        return acc.is_accessible(name, pos, yaw)

    def accessibility_report(self) -> dict:
        """씬 전체의 접근성 요약 — 씬 난이도 지표이자 실험 변수."""
        pos, yaw = self.object_xy_yaw()
        blocked, unreach, n_ok = {}, [], 0
        for n in C.OBJECT_NAMES:
            ok, why = acc.is_accessible(n, pos, yaw)
            if ok:
                n_ok += 1
            elif why == "out_of_reach":
                unreach.append(n)
            else:
                blocked[n] = acc.blockers(n, pos, yaw)
        todo = set(self.misplaced_objects())
        return {
            "n_accessible": n_ok, "n_blocked": len(blocked), "n_unreachable": len(unreach),
            "blocked": blocked, "unreachable": unreach,
            # 정리해야 하는데 지금은 못 집는 물체 = 순서를 강제하는 물체
            "blocked_and_misplaced": sorted(set(blocked) & todo),
            "occlusion_rate": round(len(blocked) / len(C.OBJECT_NAMES), 3),
        }

    # -- 편의 ------------------------------------------------------------
    def grasp_pose(self, name: str, height: float) -> tuple[np.ndarray, np.ndarray]:
        """물체 name 을 위에서 잡기 위한 (TCP 위치, TCP 자세)."""
        from .controller import yaw_to_grasp_quat
        p = self.object_pos(name)
        obj = C.OBJECT_BY_NAME[name]
        yaw = self.object_yaw(name)
        # 개폐축(hand +y)이 물체의 짧은 변과 나란하도록 90도 보정
        hx, hy, _ = C.half_size(obj)
        if hx > hy:
            yaw += math.pi / 2
        return np.array([p[0], p[1], height]), yaw_to_grasp_quat(yaw)
