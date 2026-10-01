"""VLA 정책 — 저수준 제어기.

모든 정책은 LeRobot/SmolVLA 와 같은 인터페이스를 갖는다:

    action = policy.select_action({
        "observation.image": ...,   # HxWx3 uint8
        "observation.state": ...,   # (STATE_DIM,)
        "task": "pick up the red block and place it in the right storage zone",
    })

- ScriptedSortingPolicy : 규칙 기반 teacher. 학습 전에도 파이프라인이 끝까지 돈다.
                          물체 위치를 시뮬레이터에서 직접 읽으므로 '특권 정보'를 쓴다.
                          => 데모 생성/베이스라인 용도이지 학습된 정책이 아니다.
- SmolVLAPolicy         : LeRobot 체크포인트를 불러오는 진짜 VLA. 파인튜닝 후 사용.
"""

from __future__ import annotations

import math
import re

import numpy as np

from . import config as C
from . import scene as scene_mod
from .controller import yaw_to_grasp_quat

# 접근/파지 높이
Z_HOVER = 0.16      # 물체 위 접근 높이 (테이블 기준)
Z_CARRY = 0.22      # 이송 높이
GRASP_DEPTH_FRAC = 0.55   # 물체 높이의 몇 % 지점을 TCP 로 잡을지
Z_RELEASE_MARGIN = 0.015

POS_TOL = 0.006
CARRY_TOL = 0.012
GRIP_HOLD_STEPS = 26

# 단계별 속도 배율. 1.0 이면 한 스텝에 MAX_DPOS(5cm) = 1 m/s 로 최대 속도.
# 물건을 든 채로 최대 속도로 움직이면 관성으로 손에서 빠져나간다.
# 사람 시연도 이송 구간은 천천히 하므로 학습 데이터 품질에도 이 편이 낫다.
PHASE_SPEED = {
    "hover": 1.0, "descend": 0.35, "close": 0.0,
    "lift": 0.30, "carry": 0.30, "place": 0.28, "open": 0.0, "retreat": 0.55,
    # 밀기 — 접촉을 유지해야 하므로 더 천천히
    "push_hover": 0.9, "push_descend": 0.30, "push_through": 0.22, "push_retreat": 0.55,
}

# 밀기 접근 여유 — 닫힌 그리퍼가 물체 옆에 설 거리
PUSH_STANDOFF = 0.075


def place_target(env, obj_name: str, zone: str) -> tuple[float, float]:
    """물체를 zone 안 어디에 놓을지 결정한다."""
    home = C.OBJECT_BY_NAME[obj_name]["home_zone"]
    if zone == home:
        x, y, _ = scene_mod.tidy_layout()[obj_name]
        return x, y
    z = C.ZONES[zone]
    others = [env.object_pos(n)[:2] for n in C.OBJECT_NAMES if n != obj_name]
    best, best_d = None, -1.0
    for gx in np.linspace(z["x"][0] + 0.06, z["x"][1] - 0.06, 4):
        for gy in np.linspace(max(z["y"][0] + 0.07, 0.12), z["y"][1] - 0.06, 4):
            if not (C.REACH_MIN <= math.hypot(gx - C.ROBOT_BASE_XY[0],
                                              gy - C.ROBOT_BASE_XY[1]) <= C.REACH_MAX):
                continue
            d = min((math.hypot(gx - o[0], gy - o[1]) for o in others), default=9.9)
            if d > best_d:
                best, best_d = (float(gx), float(gy)), d
    return best or C.zone_center(zone)


class ScriptedSortingPolicy:
    """pick-and-place 를 단계별 waypoint 로 실행하는 teacher 정책."""

    PHASES = ("hover", "descend", "close", "lift", "carry", "place", "open", "retreat",
              "push_hover", "push_descend", "push_through", "push_retreat", "done")

    def __init__(self, env):
        self.env = env
        self.subtask: dict | None = None
        self._enter("done")
        self._hold = 0
        self._grasp_quat = None
        self._pick_xy = None
        self._place_xy = None
        self._half_z = 0.02
        self._attempt = 0
        self.max_attempts = 2
        self.failure: str | None = None

    # -- 서브태스크 --------------------------------------------------------
    def set_subtask(self, subtask: dict) -> None:
        """subtask = {'object':…, 'target_zone':…, 'action': transport|clear|push, …}"""
        self.subtask = subtask
        self.failure = None
        name = subtask["object"]
        self.action = subtask.get("action", "transport")
        if self.action == "push":
            self._setup_push(subtask)
            return
        self._half_z = C.half_size(C.OBJECT_BY_NAME[name])[2]
        p = self.env.object_pos(name)
        self._pick_xy = (float(p[0]), float(p[1]))
        _, self._grasp_quat = self.env.grasp_pose(name, 0.0)
        self._place_xy = place_target(self.env, name, subtask["target_zone"])
        self._attempt = 0
        self._enter("hover")

    @property
    def done(self) -> bool:
        return self.phase == "done"

    # -- 제어 --------------------------------------------------------------
    STALL_PATIENCE = 45          # 이 스텝 수 동안 나아지지 않으면 정체로 본다
    STALL_IMPROVE = 5e-4         # '나아졌다'로 볼 최소 개선폭 [m]

    def _enter(self, phase: str) -> None:
        self.phase = phase
        self._hold = 0
        self._best = float("inf")
        self._stall = 0

    def _at(self, pos, tol) -> bool:
        """waypoint 도달 판정. 중력/접촉 때문에 영원히 수렴 못 하는 경우를 대비해
        일정 스텝 동안 개선이 없으면 정체로 보고 넘어간다(크게 빗나갔으면 실패 기록)."""
        d = float(np.linalg.norm(self.env.ctrl.eef_pos() - np.asarray(pos, float)))
        if d < tol:
            return True
        if d < self._best - self.STALL_IMPROVE:
            self._best, self._stall = d, 0
        else:
            self._stall += 1
        if self._stall >= self.STALL_PATIENCE:
            if d > tol * 4:
                self.failure = f"stalled_in_{self.phase}({d*1000:.0f}mm)"
            return True
        return False

    def _move(self, tgt, quat, grip) -> np.ndarray:
        """현재 phase 의 속도 제한을 적용한 액션."""
        a = self.env.ctrl.action_towards(tgt, quat, grip)
        a[:3] *= PHASE_SPEED.get(self.phase, 1.0)
        return a

    def _setup_push(self, subtask: dict) -> None:
        """밀기 준비 — 닫힌 그리퍼로 물체를 옆으로 민다 (잡지 않는 비파지 동작).

        상호 가림처럼 **양쪽 다 위에서 잡을 수 없는 상황**에서 필요하다.
        학생들이 1학기에 정의한 push 라벨이 바로 이 경우를 위한 것이었다.
        """
        name = subtask["object"]
        p = self.env.object_pos(name)
        self._half_z = C.half_size(C.OBJECT_BY_NAME[name])[2]
        dest = subtask.get("push_to")
        if dest is None:                       # 목적지가 없으면 로봇 반대쪽으로 민다
            bx, by = C.ROBOT_BASE_XY
            ux, uy = float(p[0]) - bx, float(p[1]) - by
            n = math.hypot(ux, uy) or 1.0
            dest = (float(p[0]) + 0.11 * ux / n, float(p[1]) + 0.11 * uy / n)
        self._push_to = (float(dest[0]), float(dest[1]))
        ux, uy = self._push_to[0] - float(p[0]), self._push_to[1] - float(p[1])
        n = math.hypot(ux, uy) or 1.0
        self._push_dir = (ux / n, uy / n)
        # 접근점: 물체를 기준으로 미는 방향의 반대편
        self._push_start = (float(p[0]) - PUSH_STANDOFF * self._push_dir[0],
                            float(p[1]) - PUSH_STANDOFF * self._push_dir[1])
        # 밀기 자세: 개폐축이 진행 방향과 수직이어야 접촉면이 넓다
        self._grasp_quat = yaw_to_grasp_quat(math.atan2(uy, ux) + math.pi / 2)
        self._attempt = 0
        self._enter("push_hover")

    def _grasp_z(self) -> float:
        """TCP 목표 높이 — 물체 높이의 GRASP_DEPTH_FRAC 지점.
        납작한 타일(높이 3cm)은 낮게, 정육면체(5cm)는 높게 잡아야 안정적이다."""
        full_h = 2.0 * self._half_z
        return C.TABLE_TOP_Z + max(0.012, GRASP_DEPTH_FRAC * full_h)

    def _refresh_pick(self) -> None:
        """하강 직전에 물체의 현재 위치/각도를 다시 읽는다 (접근 중 밀렸을 수 있음)."""
        name = self.subtask["object"]
        p = self.env.object_pos(name)
        self._pick_xy = (float(p[0]), float(p[1]))
        _, self._grasp_quat = self.env.grasp_pose(name, 0.0)

    def _holding(self) -> bool:
        """물체를 아직 쥐고 있는가 (이송 중 낙하 감지용)."""
        z = self.env.object_pos(self.subtask["object"])[2]
        return z > C.TABLE_TOP_Z + self._half_z + 0.04

    def select_action(self, observation: dict | None = None) -> np.ndarray:
        if self.subtask is None or self.phase == "done":
            return np.array([0, 0, 0, 0, 0, 0, self.env.ctrl.last_gripper], dtype=np.float32)

        env, q = self.env, self._grasp_quat
        tz = C.TABLE_TOP_Z
        px, py = self._pick_xy
        qx, qy = self._place_xy

        # ---- 밀기 경로 ----
        if self.phase.startswith("push_"):
            sx, sy = self._push_start
            gx2, gy2 = self._push_to
            zc = tz + max(0.012, 0.45 * 2 * self._half_z)
            if self.phase == "push_hover":
                t = [sx, sy, tz + Z_HOVER]
                if self._at(t, POS_TOL):
                    self._enter("push_descend")
                return self._move(t, q, C.GRIPPER_CLOSE_CMD)
            if self.phase == "push_descend":
                t = [sx, sy, zc]
                if self._at(t, 0.005):
                    self._enter("push_through")
                return self._move(t, q, C.GRIPPER_CLOSE_CMD)
            if self.phase == "push_through":
                t = [gx2 - PUSH_STANDOFF * self._push_dir[0] * 0.35,
                     gy2 - PUSH_STANDOFF * self._push_dir[1] * 0.35, zc]
                if self._at(t, 0.012):
                    self._enter("push_retreat")
                return self._move(t, q, C.GRIPPER_CLOSE_CMD)
            if self.phase == "push_retreat":
                t = [env.ctrl.target_pos[0], env.ctrl.target_pos[1], tz + Z_CARRY]
                if self._at(t, CARRY_TOL):
                    self._enter("done")
                return self._move(t, q, C.GRIPPER_CLOSE_CMD)

        if self.phase == "hover":
            tgt = [px, py, tz + Z_HOVER]
            if self._at(tgt, POS_TOL):
                self._refresh_pick()          # 접근 중 밀렸으면 위치 갱신
                px, py = self._pick_xy
                q = self._grasp_quat
                self._enter("descend")
            return self._move(tgt, q, C.GRIPPER_OPEN_CMD)

        if self.phase == "descend":
            tgt = [px, py, self._grasp_z()]
            if self._at(tgt, 0.004):
                self._enter("close")
            return self._move(tgt, q, C.GRIPPER_OPEN_CMD)

        if self.phase == "close":
            self._hold += 1
            if self._hold >= GRIP_HOLD_STEPS:
                self._enter("lift")
            return env.ctrl.action_towards(env.ctrl.target_pos, q, C.GRIPPER_CLOSE_CMD)

        if self.phase == "lift":
            tgt = [px, py, tz + Z_CARRY]
            if self._at(tgt, CARRY_TOL):
                # 실제로 들렸는지 확인 — 안 들렸으면 실패로 기록하고 종료
                if env.object_pos(self.subtask["object"])[2] < tz + self._half_z + 0.05:
                    self._attempt += 1
                    if self._attempt < self.max_attempts:
                        self._refresh_pick()        # 다시 조준해서 한 번 더
                        self._enter("hover")
                    else:
                        self.failure = "grasp_failed"
                        self._enter("retreat")
                else:
                    self._enter("carry")
            return self._move(tgt, q, C.GRIPPER_CLOSE_CMD)

        if self.phase == "carry":
            if not self._holding():
                self.failure = "dropped_during_carry"
                self._enter("retreat")
                return self._move(env.ctrl.target_pos, q, C.GRIPPER_OPEN_CMD)
            tgt = [qx, qy, tz + Z_CARRY]
            if self._at(tgt, CARRY_TOL):
                self._enter("place")
            return self._move(tgt, q, C.GRIPPER_CLOSE_CMD)

        if self.phase == "place":
            tgt = [qx, qy, tz + self._half_z + Z_RELEASE_MARGIN]
            if self._at(tgt, POS_TOL):
                self._enter("open")
            return self._move(tgt, q, C.GRIPPER_CLOSE_CMD)

        if self.phase == "open":
            self._hold += 1
            if self._hold >= 18:
                self._enter("retreat")
            return env.ctrl.action_towards(env.ctrl.target_pos, q, C.GRIPPER_OPEN_CMD)

        if self.phase == "retreat":
            tgt = [qx, qy, tz + Z_CARRY]
            if self._at(tgt, CARRY_TOL):
                self._enter("done")
            return self._move(tgt, q, C.GRIPPER_OPEN_CMD)

        return np.zeros(C.ACTION_DIM, dtype=np.float32)


class SmolVLAPolicy:
    """LeRobot 로 파인튜닝한 SmolVLA 체크포인트를 감싼다.

    12회차 파인튜닝 이후에 사용. lerobot 이 설치돼 있어야 한다.
    """

    def __init__(self, checkpoint: str, device: str = C.DEFAULT_DEVICE):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as _P  # noqa
        import torch

        device = C.pick_device(device)   # GPU 없으면 CPU 로 (planner.VLMPlanner 와 동일)
        self.torch = torch
        self.device = device
        self.policy = _P.from_pretrained(checkpoint).to(device).eval()

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def select_action(self, observation: dict) -> np.ndarray:
        torch = self.torch
        img = observation["observation.image"]
        state = observation["observation.state"]
        batch = {
            "observation.image": torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(self.device),
            "observation.state": torch.from_numpy(np.asarray(state, np.float32))[None].to(self.device),
            "task": [observation["task"]],
        }
        with torch.no_grad():
            action = self.policy.select_action(batch)
        return action.squeeze(0).float().cpu().numpy()[: C.ACTION_DIM]
