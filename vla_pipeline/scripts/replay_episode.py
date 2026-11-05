"""Diagnostic physical action replay, not movie playback or training evaluation.

Restore qpos/controller target ONCE, then advance only via recorded actions.
Original format lacks qvel, actuator and solver state: this is an approximate
replay, so divergence is evidence to investigate, not an exact reversibility theorem.
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from vlapipe import config as C, numbered
from vlapipe.env import SortingEnv
from vlapipe.episode import Episode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('episode', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    ep = Episode(args.episode)
    if ep.meta.get('object_set') == 'number':
        numbered.activate(count=int(ep.meta['object_count']))
    elif ep.meta.get('object_set') == 'color':
        numbered.deactivate()
    else:
        raise ValueError('Replay requires object_set/object_count metadata')
    env = SortingEnv(layout='tidy', image_size=64)
    try:
        if ep.qpos.shape[1] != env.model.nq:
            raise ValueError('Episode/environment qpos dimensions differ')
        env.data.qpos[:] = ep.qpos[0]
        env.data.qvel[:] = ep.qvel[0] if ep.qvel is not None else 0
        if ep.actuator_ctrl is not None:
            env.data.ctrl[:] = ep.actuator_ctrl[0]
        mujoco.mj_forward(env.model, env.data)
        env.ctrl.target_pos = ep.target_pos[0].astype(float).copy()
        env.ctrl.target_quat = ep.target_quat[0].astype(float).copy()
        env.ctrl.last_gripper = float(ep.actions[0, 6])
        errs = []
        for t, action in enumerate(ep.actions):
            env.step(action)
            errs.append(float(np.linalg.norm(env.ctrl.eef_pos() - ep.obs['robot0_eef_pos'][t + 1])))
        objects = {}
        for name, adr in env._jnt_adr.items():
            delta = float(np.linalg.norm(env.object_pos(name) - ep.qpos[-1, adr:adr + 3]))
            objects[name] = {'endpoint_error_m': delta, 'final_zone': env.zone_of(name)}
        report = {'source': str(args.episode), 'direction': ep.direction, 'steps': len(ep),
                  'initialization': 'qpos + controller targets; recorded qvel/ctrl if available; no per-frame state forcing',
                  'recorded_qvel': ep.qvel is not None, 'recorded_actuator_ctrl': ep.actuator_ctrl is not None,
                  'limitation': 'solver warm-start and complete simulator state are not restored; approximate replay only',
                  'eef_mean_error_m': float(np.mean(errs)), 'eef_max_error_m': float(np.max(errs)),
                  'objects': objects, 'final_is_tidy': env.is_tidy()}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
