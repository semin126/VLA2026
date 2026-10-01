"""역재생 — '정돈된 씬을 어지르는' 녹화를 '어지러운 씬을 정돈하는' 데모로 뒤집는다.

왜 이렇게 하나?
  텔레오퍼레이션으로 '정확히 구역 안에 내려놓기'는 어렵지만
  '구역에서 집어 아무 데나 놓기'는 쉽다. 어지르는 걸 녹화해서 뒤집으면
  정밀 배치 데모가 공짜로 생긴다.

수학적으로 뒤집기:
  위치/회전 델타 액션은 부호만 뒤집으면 된다.
    forward : target[t+1] = target[t] + a[t]        =>  a'[k] = -a[T-1-k]
    회전    : q[t+1] = dq_t * q[t]                  =>  dq'_k = dq_{T-1-k}^{-1}  (축각 부호 반전)
  그리퍼는 델타가 아니라 절대 명령이라 부호 반전이 안 된다.
  대신 '다음 프레임에서 손가락이 닫혀 있어야 하는가'로 다시 만든다.

주의: 물체를 던지거나 떨어뜨린 구간은 역재생하면 물체가 저절로 날아오르는
      물리적으로 불가능한 장면이 된다. check_reversibility() 로 걸러낸다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as C
from .episode import Episode

# 손가락 한쪽 열림이 이 값보다 작으면 '쥐고 있다'로 본다 (완전 열림 = 0.04)
GRASP_QPOS_THRESH = 0.035
# 역재생 불가 판정: 그리퍼가 열린 상태에서 물체가 이 속도[m/s] 이상으로 움직이면 낙하/투척
FREE_MOTION_SPEED = 0.28


def _object_xyz(qpos_row: np.ndarray, adr: int) -> np.ndarray:
    return qpos_row[adr:adr + 3]


def check_reversibility(ep: Episode, object_adr: dict[str, int]) -> dict:
    """물체가 잡히지 않은 채로 빠르게 움직인 프레임을 찾는다."""
    fps = ep.fps
    grip = ep.obs["robot0_gripper_qpos"]          # (T+1, 2)
    closed = grip.min(axis=1) < GRASP_QPOS_THRESH
    flagged: list[dict] = []
    for name, adr in object_adr.items():
        xyz = ep.qpos[:, adr:adr + 3]
        speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * fps      # (T,)
        for t in np.where((speed > FREE_MOTION_SPEED) & (~closed[:-1]))[0]:
            flagged.append({"frame": int(t), "object": name, "speed": float(speed[t])})
    flagged.sort(key=lambda d: d["frame"])
    return {
        "ok": not flagged,
        "n_flagged": len(flagged),
        "flagged": flagged[:40],
        "hint": ("물체를 던지거나 떨어뜨린 구간이 있습니다. 역재생하면 물체가 저절로 "
                 "날아오르는 장면이 됩니다. 항상 '집어서-옮겨서-내려놓기'로 다시 녹화하세요."),
    }


def reverse_episode(ep: Episode, task: str | None = None) -> Episode:
    """어지르기 에피소드 -> 정돈하기 에피소드."""
    T = len(ep)
    rev = object.__new__(Episode)
    rev.path = ep.path
    rev.fps = ep.fps
    rev.cameras = ep.cameras
    rev.direction = "tidy"
    rev.task = task or ep.meta.get("reverse_task") or "Tidy up the table."
    rev.meta = {**ep.meta, "reversed_from": str(ep.path.name), "original_task": str(ep.task)}

    # 관측/qpos/타깃은 그대로 시간 역순
    rev.obs = {k: v[::-1].copy() for k, v in ep.obs.items()}
    rev.qpos = ep.qpos[::-1].copy()
    rev.target_pos = ep.target_pos[::-1].copy()
    rev.target_quat = ep.target_quat[::-1].copy()

    # 액션: 위치/회전은 부호 반전 + 시간 역순
    actions = np.zeros((T, C.ACTION_DIM), dtype=np.float32)
    actions[:, :6] = -ep.actions[::-1, :6]

    # 그리퍼: '다음(역재생 기준) 프레임에서 손가락이 닫혀 있어야 하는가'
    grip_rev = rev.obs["robot0_gripper_qpos"]     # (T+1, 2), 이미 역순
    next_closed = grip_rev[1:].min(axis=1) < GRASP_QPOS_THRESH
    actions[:, 6] = np.where(next_closed, C.GRIPPER_CLOSE_CMD, C.GRIPPER_OPEN_CMD)

    rev.actions = actions
    return rev


def reverse_file(src: Path, dst_dir: Path, object_adr: dict[str, int] | None = None,
                 task: str | None = None, strict: bool = True) -> tuple[Path | None, dict]:
    ep = Episode(src)
    report: dict = {"source": str(src), "frames": len(ep)}
    if object_adr:
        report["reversibility"] = check_reversibility(ep, object_adr)
        if strict and not report["reversibility"]["ok"]:
            report["written"] = None
            return None, report
    rev = reverse_episode(ep, task=task)
    dst = Path(dst_dir) / (Path(src).stem + "_reversed.hdf5")
    rev.save(dst)
    report["written"] = str(dst)
    return dst, report
