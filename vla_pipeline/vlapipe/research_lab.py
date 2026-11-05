"""Research infrastructure; policy decisions live in an explicitly chosen design.

Mode A is a runnable fixed-order baseline. B adds accessibility, C replanning,
D both. Shared observations, validation, retry/action budgets and outcome
measurements are infrastructure, not student blanks. Scene state is privileged.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import time
import uuid
from pathlib import Path

import numpy as np

from . import config as C, numbered, scene as scene_module, accessibility as access_module
from .env import SortingEnv
from .planner import VLMPlanner, to_subtask
from .policy import ScriptedSortingPolicy, SmolVLAPolicy


def load_design(path: Path):
    path = Path(path).resolve(strict=True)
    spec = importlib.util.spec_from_file_location('chosen_student_design', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('assess_accessibility', 'choose_next', 'should_replan'):
        if not callable(getattr(module, name, None)):
            raise ValueError(f'Design must define {name}')
    return module, {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def snapshot(env):
    pos, yaw = env.object_xy_yaw()
    return {'source': 'simulator_state_oracle', 'positions': {n: list(p) for n, p in pos.items()},
            'yaws': yaw, 'home_zones': {o['name']: o['home_zone'] for o in C.OBJECTS},
            'sizes': {o['name']: list(o['full']) for o in C.OBJECTS},
            'zones': C.ZONES, 'robot_base': list(C.ROBOT_BASE_XY),
            'home_positions': {n: list(p[:2]) for n, p in scene_module.tidy_layout().items()},
            'reach_limits': [C.REACH_MIN, C.REACH_MAX],
            'gripper_geometry': {'half_open': access_module.GRIPPER_HALF_OPEN,
                                 'finger_radius': access_module.FINGER_RADIUS,
                                 'clear_margin': access_module.CLEAR_MARGIN},
            'misplaced': env.misplaced_objects(),
            'unfinished': [n for n in C.OBJECT_NAMES if env.zone_of(n) != C.OBJECT_BY_NAME[n]['home_zone']
                           or not supported(env, n)], 'n_objects': len(C.OBJECTS)}


def supported(env, name):
    expected = C.TABLE_TOP_Z + C.half_size(C.OBJECT_BY_NAME[name])[2]
    return abs(float(env.object_pos(name)[2]) - expected) < 0.02


def scene_done(env):
    return (env.is_tidy() and all(supported(env, n) for n in C.OBJECT_NAMES)
            and float(np.min(env.data.qpos[env._finger_adr])) > 0.035)


def valid_subtask(st):
    return subtask_error(st) is None


def subtask_error(st):
    if not isinstance(st, dict):
        return 'proposal_is_not_a_dict'
    if st.get('object') not in C.OBJECT_BY_NAME:
        return 'unknown_object'
    if st.get('target_zone') not in C.ZONES:
        return 'unknown_zone'
    if st.get('action', 'transport') != 'transport':
        return 'unsupported_action'
    # This teaching task has public fixed sorting rules and home slots. Reject
    # wrong destinations equally in A/B/C/D; never silently repair the VLM.
    if st['target_zone'] != C.OBJECT_BY_NAME[st['object']]['home_zone']:
        return 'wrong_destination_for_public_sorting_rule'
    return None


def normalize_subtask(st):
    """Keep model/strategy text when valid; supply mechanical fields explicitly.

    Object/zone validation always precedes this function. Missing instruction
    text is a serialization omission, not an excuse to change a chosen target.
    """
    if not valid_subtask(st):
        raise ValueError('Cannot normalize an unknown object/zone/action')
    result = {**to_subtask(st['object'], st['target_zone'], 1), **st}
    if not isinstance(result.get('instruction'), str) or not result['instruction'].strip():
        result['instruction'] = to_subtask(st['object'], st['target_zone'], 1)['instruction']
    return result


def setup_case(env, case, seed):
    """Controlled fixtures modify positions only BEFORE any experiment actions."""
    if case in ('messy', 'cluttered'):
        env.reset(layout=case, seed=seed)
        return
    env.reset(layout='tidy', seed=seed)
    y = 0.26 + 0.008 * (seed % 3)
    env.set_object_pose('object_1', 0.0, y, 0.0)
    if case == 'blocked':
        # Target fingers open along x; blocker fingers along y. Non-overlapping
        # cubes may therefore block target while leaving blocker accessible.
        env.set_object_pose('object_2', 0.075, y, math.pi / 2)
    elif case == 'open':
        env.set_object_pose('object_2', 0.0, y + 0.14, 0.0)
    elif case == 'unreachable':
        env.set_object_pose('object_1', 0.0, 0.06, 0.0)
    elif case == 'destination_blocked':
        x, target_y, yaw = scene_module.tidy_layout()['object_1']
        env.set_object_pose('object_2', x, target_y, yaw)
    else:
        raise ValueError(case)
    env._sim_step(60)
    env.ctrl.sync_target_to_current()


def run_trial(*, mode='A', case='open', seed=0, out: Path, design_path=None,
              planner='oracle_transport', vlm_model=None, prompt_style='number', policy='scripted', checkpoint=None,
              total_budget=1200, max_steps=400, max_attempts=2, max_plan_calls=8,
              fault='none', count=4, image_size=128):
    if mode not in 'ABCD' or len(mode) != 1:
        raise ValueError('mode must be A/B/C/D')
    if min(total_budget, max_steps, max_attempts, max_plan_calls) < 1:
        raise ValueError('Budgets and caps must be positive')
    if count < 4:
        raise ValueError('Teaching fixtures require at least four cubes')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    design, design_info = load_design(design_path) if design_path else (None, None)
    access_enabled, replan_enabled = mode in 'BD', mode in 'CD'
    if mode != 'A' and design is None:
        raise ValueError('Modes B/C/D require --design student or an explicit --design-file')
    numbered.activate(count=count)
    # Unique XML path avoids scene-generation races between independent runs.
    # Menagerie mesh paths are relative to the Panda directory. A unique file
    # there keeps those paths valid without sharing the mutable vla_scene.xml.
    scene_xml = C.PANDA_DIR / ('lab_' + uuid.uuid4().hex + '.xml')
    env = SortingEnv(layout='tidy', image_size=image_size, scene_path=scene_xml)
    model = None
    report = {'protocol': 'teaching_lab_v1', 'mode': mode, 'case': case, 'seed': seed,
              'design': design_info, 'planner': planner, 'policy': policy,
              'prompt_style': prompt_style, 'rule_destination_assignment': planner == 'vlm' and prompt_style == 'number_read',
              'observation_source': 'simulator_state_oracle_for_all_conditions',
              'injected_fault': fault, 'fault_is_natural_model_failure': False,
              'budgets': {'total_steps': total_budget, 'per_action': max_steps,
                          'per_object_attempts': max_attempts, 'planner_calls': max_plan_calls},
              'planning': [], 'decisions': [], 'feedback_events': [], 'history': [],
              'invalid_proposals': [], 'stop_reason': None}
    start = time.perf_counter()
    try:
        setup_case(env, case, seed)
        report['initial_scene'] = snapshot(env)
        geometry = {
            'positions': {n: [round(float(v), 6) for v in p]
                          for n, p in report['initial_scene']['positions'].items()},
            'yaws': {n: round(float(v), 6) for n, v in report['initial_scene']['yaws'].items()},
            'sizes': report['initial_scene']['sizes'],
        }
        report['initial_geometry_sha256'] = hashlib.sha256(
            json.dumps(geometry, sort_keys=True).encode()).hexdigest()
        report['fixture_note'] = (
            'Controlled diagnostic fixtures have only 3 y offsets (seed modulo 3); '
            'different seed names do not imply independent layouts. Check geometry hashes.'
            if case not in ('messy', 'cluttered') else 'Seeded randomized layout; verify actual scene diversity.')
        load_start = time.perf_counter()
        if planner == 'vlm':
            model = VLMPlanner(model_id=vlm_model, prompt_style=prompt_style) if vlm_model else VLMPlanner(prompt_style=prompt_style)
        learned = SmolVLAPolicy(checkpoint) if policy == 'smolvla' else None
        report['model_load_seconds'] = time.perf_counter() - load_start
        from PIL import Image
        Image.fromarray(env.render('agentview', 512, 512)).save(out / 'before.png')
        execution_start = time.perf_counter()
        attempts, steps, calls = {}, 0, 0
        queue = []
        need_plan = True
        while steps < total_budget and calls <= max_plan_calls:
            if scene_done(env):
                report['stop_reason'] = 'scene_completed'
                break
            scene = snapshot(env)
            scene['attempts'] = dict(attempts)
            scene['max_attempts'] = max_attempts
            if need_plan:
                if calls >= max_plan_calls:
                    report['stop_reason'] = 'planner_call_cap'
                    break
                calls += 1
                # Same geometric input for every mode. Do not provide misplaced
                # lists or the checker's verdict: those would leak the answer.
                shared_context = {key: scene[key] for key in (
                    'positions', 'yaws', 'sizes', 'home_zones', 'home_positions',
                    'zones', 'robot_base', 'reach_limits', 'gripper_geometry')}
                shared_context['execution_history'] = [
                    {key: event[key] for key in ('object', 'success', 'failure', 'timed_out', 'steps')}
                    for event in report['history'][-5:]]
                if planner == 'vlm':
                    command = ('Tidy up the table. Shared simulator-state information '
                               '(same in all four conditions; no accessibility verdict is supplied):\n'
                               + json.dumps(shared_context))
                    raw = model.raw_response(env.render('agentview', 512, 512), command)
                    from .planner import parse_plan
                    proposals = parse_plan(raw, allow_default_zone=prompt_style == 'number_read')
                else:
                    raw = None
                    # Correct perception only. No accessibility ordering is supplied.
                    proposals = [to_subtask(n, C.OBJECT_BY_NAME[n]['home_zone'], i + 1)
                                 for i, n in enumerate(snapshot(env)['unfinished'])]
                if calls == 1 and fault == 'empty_first_plan':
                    proposals = []
                elif calls == 1 and fault == 'invalid_first_plan':
                    proposals = [{'object': 'not_in_scene', 'target_zone': 'left_zone'}] + proposals
                elif calls == 1 and fault == 'stale_first_plan':
                    proposals = [to_subtask('object_4', C.OBJECT_BY_NAME['object_4']['home_zone'], 1)] + proposals
                report['planning'].append({'call': calls, 'raw_response': raw, 'proposals': proposals,
                                            'shared_context': shared_context})
                queue = []
                for proposal in proposals:
                    if valid_subtask(proposal):
                        queue.append(normalize_subtask(proposal))
                    else:
                        report['invalid_proposals'].append({'proposal': proposal, 'reason': subtask_error(proposal)})
                need_plan = False
            # Common infrastructure excludes already satisfied targets and capped
            # retries; these are not counted as newly completed actions.
            queue = [st for st in queue if attempts.get(st['object'], 0) < max_attempts
                     and not (env.zone_of(st['object']) == st['target_zone'] and supported(env, st['object']))]
            if not queue:
                feedback = {'event': 'empty_or_exhausted_plan', 'object': None,
                            'success': False, 'steps': 0, 'failure': 'no_executable_proposal',
                            'timed_out': False, 'lifted': False, 'already_satisfied': False,
                            'fault_injected': calls == 1 and fault == 'empty_first_plan'}
                report['feedback_events'].append(feedback)
                if replan_enabled and design.should_replan(scene, feedback, report['history']):
                    if all(attempts.get(n, 0) >= max_attempts for n in scene['unfinished']):
                        report['stop_reason'] = 'attempt_cap'
                        break
                    need_plan = True
                    continue
                report['stop_reason'] = 'empty_or_exhausted_plan'
                break
            assessments = ([design.assess_accessibility(scene, st) for st in queue] if access_enabled
                           else [{'accessible': True, 'reason': 'checker_disabled', 'blockers': []} for _ in queue])
            chosen = design.choose_next(scene, queue, assessments, report['history']) if access_enabled else queue[0]
            report['decisions'].append({'scene': scene, 'proposals': list(queue),
                                        'assessments': assessments,
                                        'checker_enabled': access_enabled, 'chosen': chosen})
            if chosen is None:
                report['stop_reason'] = 'no_admissible_action'
                break
            if not valid_subtask(chosen):
                raise ValueError('Design returned an invalid action')
            chosen = normalize_subtask(chosen)
            name = chosen['object']
            if attempts.get(name, 0) >= max_attempts:
                report['stop_reason'] = 'design_selected_capped_object'
                break
            attempts[name] = attempts.get(name, 0) + 1
            already_satisfied = env.zone_of(name) == chosen['target_zone'] and supported(env, name)
            queue = [st for st in queue if st['object'] != name]
            teacher = ScriptedSortingPolicy(env)
            if learned:
                learned.reset()
            else:
                teacher.set_subtask(chosen)
            limit = min(max_steps, total_budget - steps)
            injected = fault == 'timeout_every_action' or (fault == 'timeout_first_action' and not report['history'])
            if injected:
                limit = min(limit, 3)
            peak = float(env.object_pos(name)[2])
            n, stable = 0, 0
            while n < limit:
                if learned:
                    learned.set_controller_target(env.ctrl.target_pos)
                    action = learned.select_action({**env.get_obs(), 'task': chosen['instruction']})
                else:
                    action = teacher.select_action()
                obs, _, _, _ = env.step(action)
                n += 1
                peak = max(peak, float(env.object_pos(name)[2]))
                placed = env.zone_of(name) == chosen['target_zone'] and supported(env, name)
                released = float(np.min(obs['robot0_gripper_qpos'])) > 0.035
                stable = stable + 1 if placed and released else 0
                if (stable >= 5 if learned else teacher.done):
                    break
            steps += n
            lifted = peak > C.TABLE_TOP_Z + C.half_size(C.OBJECT_BY_NAME[name])[2] + 0.04
            feedback = {'event': 'execution_complete', 'object': name, 'target_zone': chosen['target_zone'], 'steps': n,
                        'success': bool(env.zone_of(name) == chosen['target_zone'] and supported(env, name)
                                        and float(np.min(env.data.qpos[env._finger_adr])) > 0.035 and lifted),
                        'lifted': bool(lifted), 'timed_out': n == limit and not (stable >= 5 if learned else teacher.done),
                        'failure': None if learned else teacher.failure,
                        'attempt': attempts[name], 'fault_injected': injected,
                        'already_satisfied': bool(already_satisfied),
                        'scene_after': snapshot(env)}
            report['history'].append(feedback)
            report['feedback_events'].append(feedback)
            if replan_enabled and design.should_replan(snapshot(env), feedback, report['history']):
                need_plan = True
        if report['stop_reason'] is None:
            report['stop_reason'] = 'total_step_budget'
        report['summary'] = {'scene_completed': scene_done(env), 'total_steps': steps,
                             'planner_calls': calls, 'attempts': attempts,
                             'new_transport_successes': sum(h['success'] and not h['already_satisfied'] for h in report['history']),
                             'actions': len(report['history']), 'final_misplaced': env.misplaced_objects(),
                             'execution_wall_seconds': time.perf_counter() - execution_start,
                             'wall_seconds': time.perf_counter() - start}
        Image.fromarray(env.render('agentview', 512, 512)).save(out / 'after.png')
    except Exception as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        env.close()
        scene_xml.unlink(missing_ok=True)
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report
