"""OSC 유사 제어기 — LIBERO 의 7차원 delta action 을 Panda 관절 목표로 바꾼다.

action = [dx, dy, dz, drx, dry, drz, gripper]
  - 앞 3개: 월드 좌표계 위치 델타, [-1,1] -> [-MAX_DPOS, MAX_DPOS] m
  - 중간 3개: 월드 좌표계 회전 델타(축각), [-1,1] -> [-MAX_DROT, MAX_DROT] rad
  - 마지막 1개: 그리퍼. -1 = 열기, +1 = 닫기 (LIBERO/robosuite 관례)

실제 IK 는 감쇠 최소자승(damped least squares)으로 푼다. 시뮬레이션 상태를
건드리지 않도록 별도의 MjData 를 써서 계산한다.
"""

from __future__ import annotations

import numpy as np

from . import config as C

try:
    import mujoco
except ImportError:  # 문서/테스트 목적의 임포트 허용
    mujoco = None  # type: ignore


# 위에서 내려다보는 기본 그리퍼 자세: hand +z(접근축) = 월드 -z, hand +y(개폐축) = 월드 +x
TOP_DOWN_MAT = np.array([[0.0, 1.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0]], dtype=np.float64)


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(mat, dtype=np.float64).reshape(9))
    return q


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.ascontiguousarray(quat, dtype=np.float64))
    return m.reshape(3, 3)


def quat_error(target_quat: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
    """current -> target 로 가는 회전 오차를 **월드 프레임** 3차원 회전벡터로.

    주의: mju_subQuat(res, qa, qb) 는 qb 의 *로컬* 프레임 기준 회전벡터를 준다
    (qb * quat(res) = qa). 반면 mj_jacSite 의 회전 자코비안은 월드 프레임이다.
    그래서 현재 자세의 회전행렬로 한 번 돌려 줘야 두 좌표계가 맞는다.
    """
    local = np.zeros(3)
    mujoco.mju_subQuat(local, np.ascontiguousarray(target_quat, dtype=np.float64),
                       np.ascontiguousarray(current_quat, dtype=np.float64))
    return quat_to_mat(current_quat) @ local


def axisangle_to_quat(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    angle = float(np.linalg.norm(vec))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    if angle < 1e-9:
        return q
    mujoco.mju_axisAngle2Quat(q, vec / angle, angle)
    return q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = np.zeros(4)
    mujoco.mju_mulQuat(r, np.ascontiguousarray(a, dtype=np.float64),
                       np.ascontiguousarray(b, dtype=np.float64))
    return r


def yaw_to_grasp_quat(yaw: float) -> np.ndarray:
    """위에서 내려다보되 개폐축을 yaw 만큼 돌린 그리퍼 자세."""
    base = mat_to_quat(TOP_DOWN_MAT)
    spin = axisangle_to_quat(np.array([0.0, 0.0, yaw]))
    return quat_mul(spin, base)


class IKSolver:
    """tcp 사이트에 대한 감쇠 최소자승 IK."""

    def __init__(self, model, damping: float = 0.08, max_iter: int = 60, tol_pos: float = 1e-3,
                 tol_rot: float = 8e-3):
        self.model = model
        self.data = mujoco.MjData(model)
        self.damping = damping
        self.max_iter = max_iter
        self.tol_pos = tol_pos
        self.tol_rot = tol_rot
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, C.EEF_SITE)
        if self.site_id < 0:
            raise RuntimeError(f"site '{C.EEF_SITE}' 를 찾지 못했습니다. setup_assets.py 를 실행하세요.")
        self.jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in C.ARM_JOINTS]
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in self.jnt_ids])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in self.jnt_ids])
        self.lo = np.array([model.jnt_range[j][0] for j in self.jnt_ids])
        self.hi = np.array([model.jnt_range[j][1] for j in self.jnt_ids])

    def solve(self, target_pos, target_quat, q_init) -> tuple[np.ndarray, bool]:
        q = np.clip(np.asarray(q_init, dtype=np.float64).copy(), self.lo, self.hi)
        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_quat = np.asarray(target_quat, dtype=np.float64)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        ok = False
        for _ in range(self.max_iter):
            self.data.qpos[self.qpos_adr] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            cur_pos = self.data.site_xpos[self.site_id]
            cur_quat = mat_to_quat(self.data.site_xmat[self.site_id].reshape(3, 3))
            e_pos = target_pos - cur_pos
            e_rot = quat_error(target_quat, cur_quat)
            if np.linalg.norm(e_pos) < self.tol_pos and np.linalg.norm(e_rot) < self.tol_rot:
                ok = True
                break
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            J = np.vstack([jacp[:, self.dof_adr], jacr[:, self.dof_adr]])   # 6 x 7
            err = np.concatenate([e_pos, e_rot])
            JJt = J @ J.T + (self.damping ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            step = float(np.linalg.norm(dq))
            if step > 0.25:
                dq *= 0.25 / step
            q = np.clip(q + dq, self.lo, self.hi)
        return q, ok


class OSCController:
    """delta action 을 받아 관절 위치 목표(data.ctrl)를 갱신한다."""

    # 안전 작업 공간 (월드 좌표) — 여기서 벗어나는 목표는 잘라낸다.
    BOUNDS_LO = np.array([C.WORK_X[0] - 0.06, C.WORK_Y[0] - 0.10, C.TABLE_TOP_Z + 0.005])
    BOUNDS_HI = np.array([C.WORK_X[1] + 0.06, C.WORK_Y[1] + 0.08, C.TABLE_TOP_Z + 0.42])

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.ik = IKSolver(model)
        self.site_id = self.ik.site_id
        self.qpos_adr = self.ik.qpos_adr
        self.act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                        for a in C.ARM_ACTUATORS]
        self.grip_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, C.GRIPPER_ACTUATOR)
        self.target_pos = np.zeros(3)
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_gripper = C.GRIPPER_OPEN_CMD

    # -- 상태 조회 -------------------------------------------------------
    def eef_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.site_id].copy()

    def eef_quat(self) -> np.ndarray:
        return mat_to_quat(self.data.site_xmat[self.site_id].reshape(3, 3))

    def arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    # -- 제어 -----------------------------------------------------------
    def sync_target_to_current(self) -> None:
        """현재 실제 자세를 목표로 맞춘다 (리셋 직후 튐 방지)."""
        self.target_pos = self.eef_pos()
        self.target_quat = self.eef_quat()

    def set_target(self, pos, quat) -> None:
        self.target_pos = np.clip(np.asarray(pos, dtype=np.float64), self.BOUNDS_LO, self.BOUNDS_HI)
        self.target_quat = np.asarray(quat, dtype=np.float64).copy()

    def gripper_ctrl(self, cmd: float) -> float:
        """[-1,1] 그리퍼 명령 -> actuator8 ctrl (0=닫힘, 255=열림)."""
        cmd = float(np.clip(cmd, -1.0, 1.0))
        t = (cmd - C.GRIPPER_OPEN_CMD) / (C.GRIPPER_CLOSE_CMD - C.GRIPPER_OPEN_CMD)  # 0=열기,1=닫기
        return float(C.GRIPPER_CTRL_OPEN + t * (C.GRIPPER_CTRL_CLOSE - C.GRIPPER_CTRL_OPEN))

    def apply_action(self, action) -> None:
        a = np.asarray(action, dtype=np.float64).reshape(C.ACTION_DIM)
        dpos = np.clip(a[:3], -1, 1) * C.MAX_DPOS
        drot = np.clip(a[3:6], -1, 1) * C.MAX_DROT
        self.target_pos = np.clip(self.target_pos + dpos, self.BOUNDS_LO, self.BOUNDS_HI)
        if np.linalg.norm(drot) > 1e-8:
            self.target_quat = quat_mul(axisangle_to_quat(drot), self.target_quat)
        q, _ = self.ik.solve(self.target_pos, self.target_quat, self.arm_qpos())
        for i, aid in enumerate(self.act_ids):
            self.data.ctrl[aid] = q[i]
        self.last_gripper = float(np.clip(a[6], -1, 1))
        self.data.ctrl[self.grip_act_id] = self.gripper_ctrl(self.last_gripper)

    def action_towards(self, pos, quat, gripper: float) -> np.ndarray:
        """현재 목표에서 (pos, quat) 로 가기 위한 정규화된 delta action 을 만든다.

        scripted policy 와 텔레옵이 같은 액션 규격을 쓰도록 하는 핵심 함수.
        """
        dpos = np.asarray(pos, dtype=np.float64) - self.target_pos
        e_rot = quat_error(np.asarray(quat, dtype=np.float64), self.target_quat)
        a = np.zeros(C.ACTION_DIM)
        a[:3] = np.clip(dpos / C.MAX_DPOS, -1.0, 1.0)
        a[3:6] = np.clip(e_rot / C.MAX_DROT, -1.0, 1.0)
        a[6] = gripper
        return a
