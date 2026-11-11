from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

from .. import config as C
from ..planner import PUSH_DIRECTIONS, GRASP_AXES


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_strategy(path):
    path = Path(path).resolve(strict=True)
    spec = importlib.util.spec_from_file_location('strategy_' + file_hash(path)[:12], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('build_prompt', 'loop_decision'):
        if not callable(getattr(module, name, None)):
            raise ValueError(f'Strategy must define {name}')
    return module


def public_task(command):
    """Static task definition; no live positions, visibility or completion labels."""
    from ..scene import tidy_layout
    slots = tidy_layout()
    return {'command': command,
            'sorting_rule': f'Objects 1-{C.NUMBER_SPLIT}: right storage zone; '
                            f'objects {C.NUMBER_SPLIT+1}-{len(C.OBJECTS)}: left storage zone.',
            'objects': [{'object': o['name'], 'description': o['ref_en'],
                         'target_zone': o['home_zone'], 'fixed_release_xy': list(slots[o['name']][:2])} for o in C.OBJECTS],
            'zones': {n: {'label': z['label_en'], 'x': list(z['x']), 'y': list(z['y'])} for n, z in C.ZONES.items()},
            'available_actions': ['transport', 'push'],
            'grasp_axes': {'left-right': 'fingers open along world x / image left-right',
                           'front-back': 'fingers open along world y / robot front-back'},
            'push_distance_m': C.PUSH_DISTANCE,
            'push_directions': {key: list(xy) for key, xy in PUSH_DIRECTIONS.items()},
            'push_attempt_limit_per_object': C.MAX_PUSH_ATTEMPTS,
            'note': 'Release coordinates are fixed public goal definitions, NOT current object positions. '
                    'The low-level skill uses each object\'s fixed slot when transporting to its home zone. '
                    'Transport can also move a blocker to a storage zone. '
                    'Push requests use a fixed directional instruction; successful pushing depends on the installed VLA. '
                    'Pushes have a separate per-object limit and do not consume transport attempts.'}


def task_text(task):
    # VLA tokenizers may LEFT-truncate long text. Give A a compact complete
    # scene instruction, never a JSON dump whose final notes replace the task.
    return task['command'] + '\n' + task['sorting_rule']


OUTPUT_CONTRACT = '''Return ONE JSON object, without markdown:
{"assessments": [{"object": "object_1", "accessible": true, "blockers": [], "reason": "visual evidence"}],
 "plan": [{"object": "object_1", "action": "transport", "target_zone": "right_zone", "reason": "strategy"}],
 "done": false}
Use only listed object/zone identifiers. Plan entries are executed in order.
Transport: {"object": "object_1", "action": "transport", "target_zone": "right_zone", "reason": "strategy"}.
Push: {"object": "object_1", "action": "push", "direction": "toward_robot", "reason": "strategy"}.
Push direction must be toward_robot (-y), away (+y), left (-x/image left), or right (+x/image right).
Push distance is fixed at 0.07 m. A push does not require target_zone. Do not output coordinates.
Transport may optionally include "grasp_axis": "left-right" or "front-back".
The teacher uses the cube face axis nearest the requested world axis. Omit grasp_axis for legacy grasping.
Assessments may optionally include "free_axes": ["left-right", "front-back"] (or a subset/empty list).
Judge each axis from the image: one blocked axis does not prove both axes are blocked.
For legacy transport-only plans, an omitted action still means transport.
You may transport a blocker to a storage zone before the obstructed object.
Accessibility and ordering must be your own judgments from the supplied image and context.
'''


def compose_prompt(task, strategy, context):
    text = strategy.build_prompt(task, context)
    if not isinstance(text, str) or not text.strip():
        raise ValueError('build_prompt must return nonempty strategy text')
    # Match this exact wrapper during inference AND extra-data generation.
    return task_text(task) + '\nPublic task specification:\n' + json.dumps(task, ensure_ascii=False) + '\nStudent strategy:\n' + text + '\nOutput format:\n' + OUTPUT_CONTRACT


def parse_response(raw):
    """No regex repairs, default zones, oracle filtering or silent plan changes."""
    text = raw.strip()
    if text.startswith('```') and text.endswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict) or not isinstance(value.get('plan'), list):
        raise ValueError('Response requires plan:list and assessments:list')
    if not isinstance(value.get('assessments'), list) or type(value.get('done')) is not bool:
        raise ValueError('Response requires assessments:list and done:bool')
    for item in value['assessments']:
        if (not isinstance(item, dict) or item.get('object') not in C.OBJECT_NAMES
                or type(item.get('accessible')) is not bool or not isinstance(item.get('reason'), str)
                or not isinstance(item.get('blockers'), list)
                or any(n not in C.OBJECT_NAMES for n in item['blockers'])):
            raise ValueError('Invalid accessibility assessment')
        if 'free_axes' in item:
            axes = item['free_axes']
            if (not isinstance(axes, list)
                    or any(not isinstance(a, str) or a not in GRASP_AXES for a in axes)
                    or len(set(axes)) != len(axes)):
                raise ValueError('Invalid assessment free_axes')
    for item in value['plan']:
        if not isinstance(item, dict) or item.get('object') not in C.OBJECT_NAMES:
            raise ValueError('Invalid plan object')
        action = item.get('action', 'transport')
        if action == 'transport':
            if not isinstance(item.get('target_zone'), str) or item['target_zone'] not in C.ZONES:
                raise ValueError('Transport requires a listed target_zone')
            if 'grasp_axis' in item and (not isinstance(item['grasp_axis'], str)
                                        or item['grasp_axis'] not in GRASP_AXES):
                raise ValueError('Invalid transport grasp_axis')
        elif action == 'push':
            if 'grasp_axis' in item:
                raise ValueError('grasp_axis is only supported for transport')
            if not isinstance(item.get('direction'), str) or item['direction'] not in PUSH_DIRECTIONS:
                raise ValueError('Push requires a listed direction')
            if 'push_to' in item:
                raise ValueError('Push coordinates are not accepted; use direction')
        else:
            raise ValueError('Invalid plan action')
    if value['done'] and value['plan']:
        raise ValueError('done:true must have an empty plan')
    return value


def finite_positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value
