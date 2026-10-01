"""에피소드 녹화 포맷 (HDF5).

한 에피소드 = 관측 T+1개 + 액션 T개.
  obs[t]    : action[t] 를 넣기 **직전**의 관측
  action[t] : 그때 넣은 7차원 액션
  qpos[t]   : 그 순간의 전체 MuJoCo qpos (나중에 다른 해상도/카메라로 재렌더 가능)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from . import config as C


class EpisodeWriter:
    def __init__(self, env, task: str, direction: str = "mess", cameras=None, meta: dict | None = None):
        self.env = env
        self.cameras = tuple(cameras or env.cameras)
        self.task = task
        self.direction = direction          # 'mess' = 정돈->어지르기(원본), 'tidy' = 정돈하기
        self.meta = dict(meta or {})
        self.obs: list[dict] = []
        self.actions: list[np.ndarray] = []
        self.qpos: list[np.ndarray] = []
        self.target_pos: list[np.ndarray] = []
        self.target_quat: list[np.ndarray] = []

    def record_obs(self) -> None:
        env = self.env
        o = {f"{c}_image": env.render(c) for c in self.cameras}
        o["robot0_joint_pos"] = env.data.qpos[env._arm_adr].copy().astype(np.float32)
        o["robot0_gripper_qpos"] = env.data.qpos[env._finger_adr].copy().astype(np.float32)
        o["robot0_eef_pos"] = env.ctrl.eef_pos().astype(np.float32)
        o["robot0_eef_quat"] = env.ctrl.eef_quat().astype(np.float32)
        self.obs.append(o)
        self.qpos.append(env.data.qpos.copy().astype(np.float32))
        self.target_pos.append(env.ctrl.target_pos.copy().astype(np.float32))
        self.target_quat.append(env.ctrl.target_quat.copy().astype(np.float32))

    def record_action(self, action) -> None:
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(C.ACTION_DIM))

    def __len__(self) -> int:
        return len(self.actions)

    def save(self, path: Path) -> Path:
        assert len(self.obs) == len(self.actions) + 1, (
            f"관측 {len(self.obs)}개 / 액션 {len(self.actions)}개 — 관측이 하나 더 많아야 합니다.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            g = f.create_group("obs")
            for k in self.obs[0]:
                arr = np.stack([o[k] for o in self.obs])
                g.create_dataset(k, data=arr, compression="gzip", compression_opts=4)
            f.create_dataset("actions", data=np.stack(self.actions))
            f.create_dataset("qpos", data=np.stack(self.qpos), compression="gzip", compression_opts=4)
            f.create_dataset("target_pos", data=np.stack(self.target_pos))
            f.create_dataset("target_quat", data=np.stack(self.target_quat))
            f.attrs["task"] = self.task
            f.attrs["direction"] = self.direction
            f.attrs["cameras"] = json.dumps(list(self.cameras))
            f.attrs["fps"] = C.CONTROL_HZ
            f.attrs["created"] = datetime.now().isoformat(timespec="seconds")
            f.attrs["meta"] = json.dumps(self.meta, ensure_ascii=False)
        return path


class Episode:
    """읽기용."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            self.obs = {k: f["obs"][k][:] for k in f["obs"]}
            self.actions = f["actions"][:]
            self.qpos = f["qpos"][:]
            self.target_pos = f["target_pos"][:]
            self.target_quat = f["target_quat"][:]
            self.task = f.attrs["task"]
            self.direction = f.attrs["direction"]
            self.cameras = tuple(json.loads(f.attrs["cameras"]))
            self.fps = int(f.attrs["fps"])
            self.meta = json.loads(f.attrs.get("meta", "{}"))

    def __len__(self) -> int:
        return len(self.actions)

    def state_vectors(self) -> np.ndarray:
        return np.concatenate([self.obs[k].reshape(len(self.obs[k]), -1) for k in C.STATE_KEYS],
                              axis=1).astype(np.float32)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            g = f.create_group("obs")
            for k, v in self.obs.items():
                g.create_dataset(k, data=v, compression="gzip", compression_opts=4)
            f.create_dataset("actions", data=self.actions)
            f.create_dataset("qpos", data=self.qpos, compression="gzip", compression_opts=4)
            f.create_dataset("target_pos", data=self.target_pos)
            f.create_dataset("target_quat", data=self.target_quat)
            f.attrs["task"] = self.task
            f.attrs["direction"] = self.direction
            f.attrs["cameras"] = json.dumps(list(self.cameras))
            f.attrs["fps"] = self.fps
            f.attrs["created"] = datetime.now().isoformat(timespec="seconds")
            f.attrs["meta"] = json.dumps(self.meta, ensure_ascii=False)
        return path
