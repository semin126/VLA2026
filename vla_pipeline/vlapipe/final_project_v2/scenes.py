"""Portable numbered scene files and frozen MuJoCo initial states.

The geometric access checker is a diagnostic, never input to A/B/C.
Its labels are not claims of physically impossible grasping.
"""
from __future__ import annotations

import json
import math
import random
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .. import accessibility as access, config as C, numbered, scene
from .common import digest, dump

SCHEMA = 'final_project_scene_v2'
DEFAULT_COMMAND = 'Tidy up the table: put every numbered object in its designated storage zone.'


def activate(spec):
    if spec.get('schema') != SCHEMA or spec.get('object_set') != 'number':
        raise ValueError('v2 currently requires a numbered v2 scene')
    numbered.activate(count=int(spec['count']))


def read_scene(path):
    spec = json.loads(Path(path).read_text(encoding='utf-8'))
    activate(spec)
    problems = validate_layout(spec)
    if problems['errors']:
        raise ValueError(problems)
    return spec


def vertices(pose, size):
    x, y, yaw = pose
    cs, sn = math.cos(yaw), math.sin(yaw)
    return np.array([[x + cs * dx - sn * dy, y + sn * dx + cs * dy]
                     for dx, dy in [(-size[0]/2, -size[1]/2), (size[0]/2, -size[1]/2),
                                    (size[0]/2, size[1]/2), (-size[0]/2, size[1]/2)]])


def overlaps(a, b):
    # Separating-axis test, tolerating touching faces but not initial penetration.
    for polygon in (a, b):
        for edge in (polygon[1] - polygon[0], polygon[2] - polygon[1]):
            axis = np.array([-edge[1], edge[0]])
            aa, bb = a @ axis, b @ axis
            if min(aa.max(), bb.max()) - max(aa.min(), bb.min()) <= 1e-7:
                return False
    return True


def validate_layout(spec):
    activate(spec)
    layout = spec['layout']
    errors = []
    if set(layout) != set(C.OBJECT_NAMES):
        return {'errors': ['layout object IDs do not match count'], 'warnings': []}
    polygons = {}
    for n, p in layout.items():
        if len(p) != 3 or not np.isfinite(p).all():
            errors.append(n + ': expected finite x/y/yaw')
            continue
        poly = vertices(p, C.OBJECT_BY_NAME[n]['full'])
        polygons[n] = poly
        if (poly[:, 0].min() < C.WORK_X[0] or poly[:, 0].max() > C.WORK_X[1]
                or poly[:, 1].min() < C.WORK_Y[0] or poly[:, 1].max() > C.WORK_Y[1]):
            errors.append(n + ': outside work area')
    names = list(polygons)
    for i, n in enumerate(names):
        for other in names[i+1:]:
            if overlaps(polygons[n], polygons[other]):
                errors.append(f'initial penetration: {n}/{other}')
    pos = {n: p[:2] for n, p in layout.items()}
    yaw = {n: p[2] for n, p in layout.items()}
    access_rows = {n: {'accessible': access.is_accessible(n, pos, yaw)[0],
                       'blockers': access.blockers(n, pos, yaw)} for n in pos}
    targets = spec.get('research_targets', [])
    if not targets or any(n not in pos for n in targets):
        errors.append('research_targets must be a nonempty subset of objects')
    reachable = [n for n in pos if not access.is_reachable(pos[n])]
    if reachable:
        errors.append('out of radial reach: ' + ','.join(reachable))
    ready = sum(access_rows[n]['accessible'] for n in targets if n in access_rows)
    actual = 'open' if ready == len(targets) else 'blocked' if ready == 0 else 'mixed'
    warnings = []
    if actual != spec.get('category'):
        warnings.append(f'claimed {spec.get("category")}; geometric diagnostic is {actual}')
    # Static capacity/reach sanity check of the fixed destinations.
    for n, p in scene.tidy_layout().items():
        if not access.is_reachable(p[:2]):
            errors.append(n + ': fixed destination outside radial reach')
    return {'errors': errors, 'warnings': warnings, 'diagnostic_category': actual,
            'accessibility': access_rows,
            'physical_solvability': 'unverified; inspect and execute D-state/human trials'}


def template(category, seed=0, split='development', count=10):
    if category not in ('open', 'blocked', 'mixed') or count != 10:
        raise ValueError('Starter templates use open/blocked/mixed and 10 cubes; editor can modify them.')
    numbered.activate(count=count)
    rng = random.Random(seed)
    layout = {}
    if category == 'open':
        targets = ['object_1', 'object_2', 'object_6', 'object_7']
        dy = rng.uniform(-0.012, 0.012)
        for n, (x, y) in zip(targets, [(-0.06, .22), (.06, .22), (-.06, .41), (.06, .41)]):
            layout[n] = [x, y + dy, 0.0]
    else:
        # Targets are blocked; their blockers are already in their own home zones
        # with perpendicular grasp axes. Transporting a blocker to its fixed slot
        # can clear the target without requiring an untrained pushing skill.
        targets = ['object_6', 'object_2']
        y1, y2 = .25 + rng.uniform(-.016, .016), .40 + rng.uniform(-.016, .016)
        dx = rng.uniform(-.008, .008)
        layout.update({'object_6': [.105 + dx, y1, 0.0], 'object_1': [.180 + dx, y1, math.pi/2],
                       'object_2': [-.105 + dx, y2, 0.0], 'object_7': [-.180 + dx, y2, math.pi/2]})
        if category == 'mixed':
            targets += ['object_3', 'object_8']
            layout['object_3'] = [rng.uniform(-.015, .015), .195, math.pi/2]
            layout['object_8'] = [rng.uniform(-.015, .015), .495, math.pi/2]
    # Place remaining home objects without invalid overlap. Shuffle identities
    # within destination groups so mixed tests differ beyond a translated pair.
    remaining = [n for n in C.OBJECT_NAMES if n not in layout]
    rng.shuffle(remaining)
    tidy = scene.tidy_layout()
    for n in remaining:
        sign = 1 if C.OBJECT_BY_NAME[n]['home_zone'] == 'right_zone' else -1
        candidates = [(sign*x, y) for y in (.14, .28, .42, .49) for x in (.21, .34)
                      if access.is_reachable((sign*x, y))]
        rng.shuffle(candidates)
        candidates.insert(0, tidy[n][:2])
        for x, y in candidates:
            if (all(math.hypot(x-p[0], y-p[1]) >= .113 for p in layout.values())
                    and all(other == n or math.hypot(x-p[0], y-p[1]) >= .075 for other, p in tidy.items())):
                layout[n] = [x, y, 0.0]
                break
        else:
            # Some identities occupy pair locations; alternate tightly spaced
            # home slots are legal but will be visible in access diagnostics.
            for x, y in candidates:
                if all(math.hypot(x-p[0], y-p[1]) >= .085 for p in layout.values()):
                    layout[n] = [x, y, math.pi/2]
                    break
            else:
                raise ValueError('No free home slot for starter template')
    spec = {'schema': SCHEMA, 'id': f'{split}_{category}_{seed}', 'seed': seed,
            'family': f'{split}_{category}_{seed}', 'split': split, 'category': category,
            'object_set': 'number', 'count': count, 'command': DEFAULT_COMMAND,
            'research_targets': targets, 'layout': layout,
            'note': 'Geometric diagnostic, not a proof that a physical grasp is impossible.'}
    report = validate_layout(spec)
    if report['errors'] or report['diagnostic_category'] != category:
        raise ValueError(report)
    return spec


def state_of(env):
    return {k: np.asarray(getattr(env.data, k)).tolist()
            for k in ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart')} | {
                'time': float(env.data.time), 'target_pos': env.ctrl.target_pos.tolist(),
                'target_quat': env.ctrl.target_quat.tolist(), 'last_gripper': float(env.ctrl.last_gripper)}


def restore(env, spec):
    import mujoco
    if spec.get('initial_state'):
        s = spec['initial_state']
        for key in ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart'):
            dest = getattr(env.data, key)
            val = np.asarray(s[key], dtype=float)
            if dest.shape != val.shape or not np.isfinite(val).all():
                raise ValueError('Frozen state incompatible with environment: ' + key)
            dest[:] = val
        env.data.time = s['time']
        mujoco.mj_forward(env.model, env.data)
        # mj_forward may modify solver warmstart; restore the recorded input.
        env.data.qacc_warmstart[:] = s['qacc_warmstart']
        env.ctrl.target_pos = np.array(s['target_pos'])
        env.ctrl.target_quat = np.array(s['target_quat'])
        env.ctrl.last_gripper = s['last_gripper']
    else:
        for n, p in spec['layout'].items():
            env.set_object_pose(n, *p)
        env._sim_step(120)
        env.ctrl.sync_target_to_current()
    env.step_count = 0


@contextmanager
def environment(spec, image_size=128):
    from ..env import SortingEnv
    activate(spec)
    xml = C.PANDA_DIR / ('fpv2_' + uuid.uuid4().hex + '.xml')
    env = None
    try:
        env = SortingEnv(layout='tidy', image_size=image_size, scene_path=xml)
        restore(env, spec)
        yield env
    finally:
        if env is not None:
            env.close()
        xml.unlink(missing_ok=True)


def freeze(spec):
    report = validate_layout(spec)
    if report['errors']:
        raise ValueError(report)
    # Re-freezing after a layout edit MUST discard the previous initial state.
    editable = {k: v for k, v in spec.items() if k not in ('initial_state', 'state_sha256')}
    with environment(editable) as env:
        editable['initial_state'] = state_of(env)
        settled = {n: [*map(float, env.object_pos(n)[:2]), env.object_yaw(n)] for n in C.OBJECT_NAMES}
        editable['settled_diagnostics'] = validate_layout({**editable, 'layout': settled})
        for n in C.OBJECT_NAMES:
            z = C.TABLE_TOP_Z + C.half_size(C.OBJECT_BY_NAME[n])[2]
            if abs(env.object_pos(n)[2] - z) > .01:
                raise ValueError(n + ': unstable settled height')
        if editable['settled_diagnostics']['errors']:
            raise ValueError(editable['settled_diagnostics'])
    editable['state_sha256'] = digest(editable['initial_state'])
    return editable


def preview(spec, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with environment(spec) as env:
        panels = []
        for name in ('agentview', 'frontview', 'sideview', 'robot0_eye_in_hand'):
            im = Image.fromarray(env.render(name, 512, 512))
            ImageDraw.Draw(im).text((8, 8), name, fill='red')
            im.save(out / (name + '.png'))
            panels.append(im)
        sheet = Image.new('RGB', (1024, 1024))
        for i, im in enumerate(panels):
            sheet.paste(im, ((i % 2)*512, (i // 2)*512))
        sheet.save(out / 'scene.png')
    dump(out / 'diagnostics.json', validate_layout(spec))
    return out / 'scene.png'


def make_suite(out, tests=10, train=4, validation=2):
    if tests < 10 or min(train, validation) < 1:
        raise ValueError('At least 10 test and positive train/validation scenes required')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    specs = [template(cat, 100+i) for i, cat in enumerate(('open', 'blocked', 'mixed'))]
    for split, base, count in [('train', 1000, train), ('validation', 2000, validation),
                               ('sft_test', 2500, 2), ('test', 3000, tests)]:
        specs.extend(template('mixed', base+i, split) for i in range(count))
    manifest = []
    for s in specs:
        frozen = freeze(s)
        dest = out / s['split'] / (s['id'] + '.json')
        dump(dest, frozen)
        image = preview(frozen, dest.with_suffix(''))
        manifest.append({'path': dest.relative_to(out).as_posix(), 'family': s['family'],
                         'split': s['split'], 'state_sha256': frozen['state_sha256'],
                         'preview': image.relative_to(out).as_posix()})
    dump(out / 'suite.json', {'schema': SCHEMA, 'scenes': manifest,
                              'note': 'Starter generator uses one pair-based layout family. Broaden student-designed tests.'})
    links = ['<h1>Final project ver2.0 — Scenes</h1>']
    for row in manifest:
        links.append(f'<h3>{row["path"]}</h3><img width="600" src="{row["preview"]}">')
    (out / 'index.html').write_text('\n'.join(links), encoding='utf-8')
    return manifest


def register_scenes(paths, out):
    """Register student-authored frozen scene files without editing the originals."""
    specs = [read_scene(p) for p in paths]
    ids, owners = set(), {'family': {}, 'seed': {}, 'state_sha256': {}}
    for s in specs:
        if s['id'] in ids or not s.get('initial_state') or digest(s['initial_state']) != s.get('state_sha256'):
            raise ValueError('Duplicate ID or unfrozen/invalid state')
        ids.add(s['id'])
        if s['split'] not in ('development', 'train', 'validation', 'sft_test', 'test'):
            raise ValueError('Unknown split')
        for key in owners:
            value = s[key]
            if value in owners[key] and owners[key][value] != s['split']:
                raise ValueError('Scene split leakage: '+key)
            owners[key][value] = s['split']
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = []
    for i, s in enumerate(specs):
        # Use numeric filenames, so user-authored IDs cannot become filesystem paths.
        dest = out/s['split']/f'scene_{i:04d}.json'
        dump(dest, s)
        image = preview(s, dest.with_suffix(''))
        manifest.append({'path': dest.relative_to(out).as_posix(), 'family': s['family'],
                         'split': s['split'], 'state_sha256': s['state_sha256'],
                         'preview': image.relative_to(out).as_posix()})
    dump(out/'suite.json', {'schema': SCHEMA, 'scenes': manifest, 'source': 'student_authored'})
    return manifest
