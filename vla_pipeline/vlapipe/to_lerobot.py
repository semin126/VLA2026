"""HDF5 에피소드 -> LeRobot 데이터셋 변환 (SmolVLA 파인튜닝 입력).

LeRobot 의 on-disk 포맷은 버전마다 조금씩 다르다. 직접 parquet/mp4 를 쓰지 않고
설치된 lerobot 의 LeRobotDataset API 를 그대로 쓴다 -> 버전이 올라가도 안 깨진다.

lerobot 이 없는 환경(윈도우 로컬)에서는 --dry-run 으로 데이터 검증만 한다.
실제 변환과 학습은 Colab 에서 한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as C
from .episode import Episode

# LeRobot / SmolVLA 가 기대하는 키 이름
KEY_IMAGE = "observation.images.agentview"
KEY_WRIST = "observation.images.wrist"
KEY_STATE = "observation.state"
KEY_ACTION = "action"

STATE_NAMES = ([f"joint_{i}" for i in range(1, 8)] + ["gripper_1", "gripper_2"]
               + ["eef_x", "eef_y", "eef_z"] + ["eef_qw", "eef_qx", "eef_qy", "eef_qz"])
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]


def episode_frames(ep: Episode) -> list[dict]:
    """에피소드를 LeRobot 프레임 리스트로. 액션 T개에 맞춰 관측도 앞 T개만 쓴다."""
    T = len(ep)
    states = ep.state_vectors()[:T]
    agent = ep.obs.get("agentview_image")
    wrist = ep.obs.get("robot0_eye_in_hand_image")
    frames = []
    for t in range(T):
        f = {KEY_STATE: states[t].astype(np.float32),
             KEY_ACTION: ep.actions[t].astype(np.float32),
             "task": str(ep.task)}
        if agent is not None:
            f[KEY_IMAGE] = agent[t]
        if wrist is not None:
            f[KEY_WRIST] = wrist[t]
        frames.append(f)
    return frames


def build_features(ep: Episode, use_videos: bool = True) -> dict:
    dtype = "video" if use_videos else "image"
    feats: dict = {
        KEY_STATE: {"dtype": "float32", "shape": (C.STATE_DIM,), "names": STATE_NAMES},
        KEY_ACTION: {"dtype": "float32", "shape": (C.ACTION_DIM,), "names": ACTION_NAMES},
    }
    if "agentview_image" in ep.obs:
        h, w = ep.obs["agentview_image"].shape[1:3]
        feats[KEY_IMAGE] = {"dtype": dtype, "shape": (h, w, 3), "names": ["height", "width", "channel"]}
    if "robot0_eye_in_hand_image" in ep.obs:
        h, w = ep.obs["robot0_eye_in_hand_image"].shape[1:3]
        feats[KEY_WRIST] = {"dtype": dtype, "shape": (h, w, 3), "names": ["height", "width", "channel"]}
    return feats


def validate(paths: list[Path]) -> dict:
    """lerobot 없이도 돌아가는 사전 검증."""
    report = {"n_episodes": 0, "n_frames": 0, "tasks": {}, "problems": []}
    shapes = set()
    for p in paths:
        ep = Episode(p)
        T = len(ep)
        if T < 5:
            report["problems"].append(f"{p.name}: 프레임이 {T}개뿐입니다.")
        if ep.direction != "tidy":
            report["problems"].append(
                f"{p.name}: direction='{ep.direction}' — 학습에는 역재생된 'tidy' 만 쓰세요.")
        if len(ep.obs["robot0_joint_pos"]) != T + 1:
            report["problems"].append(f"{p.name}: 관측 {len(ep.obs['robot0_joint_pos'])}개 / 액션 {T}개 불일치")
        a = ep.actions
        if np.abs(a[:, :6]).max() > 1.0001:
            report["problems"].append(f"{p.name}: 액션이 [-1,1] 범위를 벗어났습니다.")
        if not set(np.unique(a[:, 6])) <= {-1.0, 1.0}:
            report["problems"].append(f"{p.name}: 그리퍼 액션이 -1/+1 이 아닙니다.")
        if "agentview_image" in ep.obs:
            shapes.add(ep.obs["agentview_image"].shape[1:])
        report["n_episodes"] += 1
        report["n_frames"] += T
        report["tasks"][str(ep.task)] = report["tasks"].get(str(ep.task), 0) + 1
    if len(shapes) > 1:
        report["problems"].append(f"이미지 해상도가 섞여 있습니다: {shapes}")
    report["image_shapes"] = [tuple(s) for s in shapes]
    report["ok"] = not report["problems"]
    return report


def convert(paths: list[Path], repo_id: str, root: Path, fps: int = C.CONTROL_HZ,
            use_videos: bool = True, robot_type: str = "panda_vla_sorting"):
    """실제 LeRobot 데이터셋 생성. lerobot 이 설치된 환경(Colab)에서 실행."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # 구버전

    eps = [Episode(p) for p in paths]
    if not eps:
        raise ValueError("변환할 에피소드가 없습니다.")

    root = Path(root)
    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, root=root, robot_type=robot_type,
        features=build_features(eps[0], use_videos=use_videos), use_videos=use_videos,
    )
    for ep in eps:
        for frame in episode_frames(ep):
            task = frame.pop("task")
            try:
                ds.add_frame(frame, task=task)
            except TypeError:            # task 를 dict 안에 받는 버전
                frame["task"] = task
                ds.add_frame(frame)
        ds.save_episode()
    if hasattr(ds, "consolidate"):       # 구버전에만 있는 마무리 단계
        ds.consolidate()
    return ds
