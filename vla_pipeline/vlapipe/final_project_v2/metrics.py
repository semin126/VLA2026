"""Final physical state, never planner text, determines the task metrics."""
from __future__ import annotations

import numpy as np

from .. import config as C


def scores(gold, predicted):
    if not gold or set(gold) != set(predicted):
        raise ValueError('Metrics require identical nonempty object IDs')
    labels = sorted(set(gold.values()))  # fixed true destination classes; invalid is always incorrect
    rows = {}
    for label in labels:
        tp = sum(gold[n] == label and predicted[n] == label for n in gold)
        fp = sum(gold[n] != label and predicted[n] == label for n in gold)
        fn = sum(gold[n] == label and predicted[n] != label for n in gold)
        rows[label] = {'tp': tp, 'fp': fp, 'fn': fn,
                       'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}
    acc = sum(gold[n] == predicted[n] for n in gold) / len(gold)
    return {'acc': acc, 'macro_f1': sum(r['f1'] for r in rows.values()) / len(rows),
            'scene_success': acc == 1.0, 'per_zone': rows, 'n_objects': len(gold)}


def observe_outcome(env):
    gold = {o['name']: o['home_zone'] for o in C.OBJECTS}
    predicted, detail = {}, {}
    released = float(np.min(env.data.qpos[env._finger_adr])) > .035
    for n in C.OBJECT_NAMES:
        pos = env.object_pos(n)
        expected_z = C.TABLE_TOP_Z + C.half_size(C.OBJECT_BY_NAME[n])[2]
        # Whole object footprint must fit inside the zone, not only its center.
        from .scenes import vertices
        poly = vertices([*pos[:2], env.object_yaw(n)], C.OBJECT_BY_NAME[n]['full'])
        zone = None
        for z, bounds in C.ZONES.items():
            if (poly[:, 0].min() >= bounds['x'][0] and poly[:, 0].max() <= bounds['x'][1]
                    and poly[:, 1].min() >= bounds['y'][0] and poly[:, 1].max() <= bounds['y'][1]):
                zone = z
                break
        import mujoco
        joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, n + '_freejoint')
        adr = env.model.jnt_dofadr[joint]
        velocity = env.data.qvel[adr:adr+6]
        quat = env.data.qpos[env._jnt_adr[n]+3:env._jnt_adr[n]+7]
        # Every face of a numbered cube is a valid support face. Requiring its
        # original local z axis to point upward falsely rejects 90-degree rolls.
        qw, qx, qy, qz = quat
        up_components = [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]
        upright = max(abs(v) for v in up_components) > .95
        stable = (abs(pos[2] - expected_z) < .012 and np.linalg.norm(velocity[:3]) < .04
                  and np.linalg.norm(velocity[3:]) < .5 and upright)
        held = not released and np.linalg.norm(pos - env.ctrl.eef_pos()) < .075
        predicted[n] = zone if stable and not held and zone is not None else 'invalid'
        detail[n] = {'zone': zone, 'stable': bool(stable), 'held': bool(held),
                     'correct': predicted[n] == gold[n], 'position': pos.tolist()}
    return {'gold': gold, 'predicted': predicted, 'objects': detail, **scores(gold, predicted)}
