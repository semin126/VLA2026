"""A VLA only; B one VLM plan; C student feedback loop; D-state; E teleop.

All policies run at most 20 Hz wall time. Scoring is privileged, but only the
common final-success termination is visible to learned/human agents. B/C never
see live oracle positions, target completion or geometric access judgments.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import json
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .. import config as C
from .. import accessibility as access
from ..planner import instruction_text
from ..policy import ScriptedSortingPolicy, SmolVLAPolicy
from . import PROTOCOL
from .common import compose_prompt, digest, dump, file_hash, finite_positive, load_strategy, parse_response, public_task, task_text
from .metrics import observe_outcome
from .models import ImagePlanner, synchronize
from .scenes import environment, read_scene, state_of


@dataclass
class TrialConfig:
    condition: str = 'D'
    seconds: float = C.FINAL_PROJECT_SECONDS
    max_steps: int = C.FINAL_PROJECT_MAX_STEPS
    action_steps: int = 240
    observer_steps: int = 60
    max_calls: int = 8
    max_attempts: int = 3
    image_size: int = 128
    planner_image_size: int = 512
    planner_tokens: int = 1024
    seed: int = 42
    checkpoint: str | None = None
    vlm_model: str | None = None
    model_source: str = 'auto'
    vlm_source: str | None = None
    vla_source: str | None = None
    models_dir: str | None = None
    vla_hub_id: str | None = None
    vlm_hub_id: str | None = None
    strategy: str | None = None
    operator: str = ''
    input_kind: str = 'gamepad'
    gamepad_map: str = 'gamepad_map.json'
    video: bool = True
    device: str = 'auto'
    final_test: bool = False
    command: str | None = None
    realtime: bool = True

    def validate(self):
        if not self.realtime and self.condition != 'D':
            raise ValueError('Accelerated execution is only for D-state demonstration collection')
        if self.condition not in ('A', 'B', 'C', 'D', 'E'):
            raise ValueError('condition must be A/B/C/D/E')
        finite_positive(self.seconds, 'seconds')
        if min(self.max_steps, self.action_steps, self.observer_steps, self.max_calls,
               self.max_attempts, self.image_size, self.planner_image_size, self.planner_tokens) < 1:
            raise ValueError('All budgets/sizes must be positive')
        if self.model_source not in ('auto','local','hub'):
            raise ValueError('model_source must be auto/local/hub')
        if any(s is not None and s not in ('auto','local','hub') for s in (self.vlm_source,self.vla_source)):
            raise ValueError('Per-model source must be auto/local/hub')
        if self.condition == 'B' and not self.strategy:
            self.strategy = str(Path(__file__).resolve().parents[2]/'final_project_v2/baseline_strategy.py')
        if self.condition == 'C' and not self.strategy:
            raise ValueError('C requires an explicitly selected student/teacher strategy file')
        if self.command is not None and (not isinstance(self.command, str) or not self.command.strip()):
            raise ValueError('command must be a nonempty string when supplied')
        if self.condition == 'E' and (not self.operator or self.input_kind not in ('gamepad', 'keyboard', 'keyboard-gamepad')):
            raise ValueError('E requires operator and supported input_kind')


def hardware():
    import importlib.metadata
    result = {'system': platform.platform(), 'python': sys.version, 'machine': platform.node(),
              'packages': {}}
    for name in ('mujoco', 'numpy', 'torch', 'transformers', 'lerobot'):
        try:
            result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        result['cuda'] = torch.cuda.is_available()
        if result['cuda']:
            result['gpu'] = torch.cuda.get_device_name()
    return result


def classical_next(env, attempts, cap):
    """Deterministic transport-only strategy; uses oracle perception, explicitly D-state."""
    outcome = observe_outcome(env)
    todo = [n for n in C.OBJECT_NAMES if not outcome['objects'][n]['correct'] and attempts.get(n, 0) < cap]
    pos, yaw = env.object_xy_yaw()
    from ..scene import tidy_layout
    slots = tidy_layout()
    def destination_blockers(n):
        return [other for other, p in pos.items() if other != n
                and float(np.linalg.norm(np.asarray(p)-np.asarray(slots[n][:2]))) < .075]
    key = lambda n: (float(np.linalg.norm(np.asarray(pos[n]) - np.asarray(C.ROBOT_BASE_XY))), n)
    ready = [n for n in todo if access.is_accessible(n, pos, yaw)[0] and not destination_blockers(n)]
    if ready:
        chosen = min(ready, key=key)
    else:
        blockers = {b for n in todo for b in access.blockers(n, pos, yaw) + destination_blockers(n)
                    if attempts.get(b, 0) < cap and access.is_accessible(b, pos, yaw)[0]
                    and not destination_blockers(b)}
        if not blockers:
            return None
        chosen = min(blockers, key=key)
    return {'object': chosen, 'target_zone': C.OBJECT_BY_NAME[chosen]['home_zone']}


def make_human(env, config):
    from ..teleop import GamepadInput, KeyboardInput, KeyboardGamepadInput, load_gamepad_map
    from ..camview import CameraPanel
    mapping = load_gamepad_map(config.gamepad_map) if Path(config.gamepad_map).is_file() else None
    cls = {'gamepad': GamepadInput, 'keyboard': KeyboardInput, 'keyboard-gamepad': KeyboardGamepadInput}[config.input_kind]
    kwargs = {'speed': .55}
    if config.input_kind != 'keyboard':
        kwargs['mapping'] = mapping
    source = cls(**kwargs)
    panel = CameraPanel(env, cameras=('agentview', 'robot0_eye_in_hand'), cols=2,
                        title='Final project ver2.0 E — Enter/record: start; Esc: stop')
    print('E ready: inspect the task, then press Enter / record button in camera panel to START. Resets are disabled.')
    while not source.quit:
        panel.pump_input(source)
        source.poll()
        if 'record' in source.events:
            source.events.clear()
            return source, panel
        if panel.quit:
            break
        source.events.clear()
        time.sleep(.02)
    panel.close()
    raise RuntimeError('Human cancelled before trial start')


def render_replay(spec, trace, report, out):
    """Offline state rendering, not physical replay or action verification.

MP4 runs on simulation time; inference pauses are represented by wall timestamps
in events.json/trace.npz, not silently described as task completion time.
"""
    import imageio.v2 as imageio
    import mujoco
    out = Path(out)
    stride = 4
    with environment(spec) as env:
        with imageio.get_writer(out / 'process.mp4', fps=C.CONTROL_HZ/stride, codec='libx264', quality=7) as writer:
            indices = list(range(0, len(trace['qpos']), stride))
            if indices[-1] != len(trace['qpos'])-1:
                indices.append(len(trace['qpos'])-1)
            for i in indices:
                env.data.qpos[:] = trace['qpos'][i]
                mujoco.mj_forward(env.model, env.data)
                panels = [Image.fromarray(env.render(cam, 320, 320))
                          for cam in ('agentview', 'robot0_eye_in_hand')]
                frame = Image.new('RGB', (640, 368))
                for j, im in enumerate(panels):
                    frame.paste(im, (j*320, 0))
                # Last action/plan event at this sample, with escaped strings in HTML separately.
                past = [e for e in report['events'] if e.get('step', 0) <= i]
                event = past[-1]['event'] if past else 'initial'
                ImageDraw.Draw(frame).text((5, 325), f'{report["condition"]} step={i} wall={trace["wall"][i]:.2f}s {event}', fill='white')
                writer.append_data(np.asarray(frame))
    return 'process.mp4'


def run_trial(scene_path, out, config: TrialConfig, *, planner_adapter=None, policy_adapter=None):
    """Adapters are a test seam; injected trials are labelled diagnostic automatically."""
    config.validate()
    lifecycle_start = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    spec = read_scene(scene_path)
    input_scene_sha256 = digest(spec)
    if config.command is not None:
        spec = {**spec, 'command': config.command.strip()}
    if spec['split'] == 'test' and not config.final_test:
        raise ValueError('Final benchmark requires final_test=True after freezing strategy/model selection')
    if not spec.get('initial_state') or digest(spec['initial_state']) != spec.get('state_sha256'):
        raise ValueError('Freeze the scene after editing; initial state/hash is missing or invalid')
    dump(out / 'scene.json', spec)
    strategy = load_strategy(config.strategy) if config.condition in 'BC' else None
    task = public_task(spec['command'])
    strategy_info = None
    if strategy:
        shutil.copyfile(config.strategy, out / 'strategy_used.py')
        strategy_info = {'sha256': file_hash(config.strategy), 'path': str(Path(config.strategy).resolve())}
    diagnostic = planner_adapter is not None or policy_adapter is not None
    implementation = {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')}
    for name in ('policy.py','model_loading.py','action_targets.py','controller.py','config.py','env.py','numbered.py'):
        implementation['core/'+name]=file_hash(Path(__file__).parent.parent/name)
    report = {'protocol': PROTOCOL if config.realtime else 'final_project_v2_collection', 'condition': config.condition, 'variant': 'D-state' if config.condition == 'D' else config.condition,
              'diagnostic': diagnostic, 'scene_id': spec['id'], 'family': spec['family'], 'split': spec['split'],
              'scene_sha256': digest(spec), 'input_scene_sha256': input_scene_sha256,
              'initial_state_sha256': spec['state_sha256'], 'task_sha256': digest(task),
              'configuration': asdict(config), 'strategy': strategy_info, 'events': [], 'status': 'running',
              'feedback_source': 'images_and_nonprivileged_execution_metadata',
              'perception': 'oracle_state' if config.condition == 'D' else 'camera_images_and_robot_proprioception',
              'timing_note': ('20 Hz maximum; inference counted; synchronous calls may overrun but late actions are discarded'
                              if config.realtime else 'Accelerated physical demonstrations; not a wall-time benchmark'),
              'hardware': hardware(), 'implementation_sha256': digest(implementation)}
    dump(out / 'task.json', task)
    steps = calls = 0
    history, attempts, queue = [], {}, []
    start = None
    source = panel = None
    trace = {'qpos': [], 'qvel': [], 'actions': [], 'wall': [], 'tasks': [],
             'actuator_ctrl': [], 'target_pos': [], 'target_quat': []}
    planning_seconds = action_inference_seconds = 0.0
    with environment(spec, config.image_size) as env:
        try:
            actual = digest(state_of(env))
            if actual != spec['state_sha256']:
                raise ValueError('Restored initial state does not match frozen scene')
            report['restored_state_sha256'] = actual
            report['initial'] = observe_outcome(env)
            Image.fromarray(env.render('agentview', 512, 512)).save(out / 'before.png')
            report['initial_image_sha256'] = file_hash(out / 'before.png')
            load_start = time.perf_counter()
            learned = policy_adapter
            model = planner_adapter
            if config.condition in 'ABC' and learned is None:
                import torch
                from .model_sources import resolve_model
                choice = resolve_model('vla_single' if config.condition == 'A' else 'vla_hier',
                    source=config.vla_source or config.model_source, reference=config.checkpoint, models_dir=config.models_dir, hub_id=config.vla_hub_id)
                config.checkpoint = choice.location
                report.setdefault('models', {})['vla'] = choice.to_dict()
                torch.manual_seed(config.seed)
                learned = SmolVLAPolicy(config.checkpoint, device=config.device)
                report['models']['vla']['action_representation']=learned.action_representation['name']
            if config.condition in 'BC' and model is None:
                model = ImagePlanner(config.vlm_model, device=config.device, max_new_tokens=config.planner_tokens,
                                     source=config.vlm_source or config.model_source, models_dir=config.models_dir, hub_id=config.vlm_hub_id)
                config.vlm_model = model.choice.location
                report.setdefault('models', {})['vlm'] = model.choice.to_dict()
            report['configuration'] = asdict(config)
            if learned is not None:
                learned.reset()
            random.seed(config.seed)
            np.random.seed(config.seed)
            if config.condition in 'ABC' and not diagnostic:
                import torch
                torch.manual_seed(config.seed)
                checkpoint = Path(getattr(learned, 'loaded_checkpoint', config.checkpoint))
                if checkpoint.is_dir():
                    report['checkpoint_manifest'] = {p.name: file_hash(p) for p in sorted(checkpoint.iterdir())
                                                     if p.is_file() and p.suffix in ('.json', '.safetensors')}
                report['vlm_revision'] = getattr(getattr(getattr(model, 'backend', None), 'model', None), 'config', None)
                report['vlm_revision'] = getattr(report['vlm_revision'], '_commit_hash', None)
                if config.condition in 'BC':
                    vlm_directory = Path(config.vlm_model)
                    report['vlm_checkpoint_manifest'] = ({p.name: file_hash(p) for p in sorted(vlm_directory.iterdir())
                        if p.is_file() and p.suffix in ('.json','.safetensors')} if vlm_directory.is_dir()
                        else {'hub_id':config.vlm_model,'revision':report['vlm_revision']})
            if config.condition == 'E':
                print(task_text(task))
                report['model_load_seconds'] = time.perf_counter()-load_start
                human_prepare_start = time.perf_counter()
                source, panel = make_human(env, config)
                report['human_prepare_seconds'] = time.perf_counter()-human_prepare_start
            synchronize()
            if config.condition != 'E':
                report['model_load_seconds'] = time.perf_counter() - load_start
            report['hardware'] = hardware()
            start = time.perf_counter()
            report['setup_wall_seconds'] = start-lifecycle_start
            deadline = start + config.seconds
            trace['qpos'].append(env.data.qpos.copy())
            trace['qvel'].append(env.data.qvel.copy())
            trace['wall'].append(0.0)
            trace['actuator_ctrl'].append(env.data.ctrl.copy())
            trace['target_pos'].append(env.ctrl.target_pos.copy())
            trace['target_quat'].append(env.ctrl.target_quat.copy())
            feedback = {'event': 'initial'}
            last_model_response = None

            def event(name, **kw):
                # Queues and feedback keep changing after emission. Preserve the
                # same snapshot in report.json and the durable events.jsonl.
                row = deepcopy({'event': name, 'step': steps, 'wall': time.perf_counter()-start, **kw})
                report['events'].append(row)
                # Durable per-event progress during a long run.
                with (out / 'events.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')

            def context(name):
                # Student hooks receive JSON snapshots, not aliases to mutable
                # executor actions, history or the last parsed model response.
                return deepcopy({'event': name, 'feedback': feedback, 'history': list(history[-20:]),
                        'remaining_seconds': max(0, deadline-time.perf_counter()),
                        'remaining_calls': config.max_calls-calls, 'queued_actions': list(queue),
                        'last_model_response': last_model_response})

            def decide(name):
                observed = context(name)
                observed['observation'] = {'agentview_image': env.render('agentview', config.planner_image_size, config.planner_image_size),
                                           'robot0_eye_in_hand_image': env.render('robot0_eye_in_hand')}
                answer = strategy.loop_decision(observed)
                decision_feedback = {}
                if isinstance(answer, dict):
                    custom_feedback = answer.get('feedback')
                    # Student-chosen image analysis can be passed into the next VLM
                    # prompt, but it must remain ordinary, bounded JSON metadata.
                    if len(json.dumps(custom_feedback, allow_nan=False)) > 12000:
                        raise ValueError('Student feedback exceeds 12000 characters')
                    feedback['student_feedback'] = deepcopy(custom_feedback)
                    decision_feedback['feedback'] = custom_feedback
                    answer = answer.get('decision')
                if answer not in ('continue', 'replan', 'stop'):
                    raise ValueError('loop_decision must return continue/replan/stop')
                event('student_decision', trigger=name, decision=answer, **decision_feedback)
                return answer

            need_plan = config.condition in 'BC'
            stop = None
            current = None
            current_steps = 0
            teacher = None
            task_label = task_text(task)
            outcome = report['initial']
            stable_scene_steps = 5 if outcome['scene_success'] else 0
            while steps < config.max_steps:
                if time.perf_counter() >= deadline:
                    stop = 'time_limit'
                    break
                if outcome['scene_success'] and stable_scene_steps >= 5:
                    stop = 'scene_completed'
                    break
                if need_plan:
                    limit = 1 if config.condition == 'B' else config.max_calls
                    if calls >= limit:
                        stop = 'planner_call_limit'
                        break
                    image = env.render('agentview', config.planner_image_size, config.planner_image_size)
                    initial = calls == 0
                    # Initial B/C context is deliberately identical, independent of
                    # condition name, loading latency or permitted future calls.
                    ctx = {'event': 'initial', 'history': [], 'feedback': None} if initial else context('replan')
                    prompt = compose_prompt(task, strategy, ctx)
                    if initial:
                        report['initial_prompt_sha256'] = digest(prompt)
                        # Snapshot the actual prompt resource used, in addition to
                        # the module. plan_001.json preserves the full assembled input.
                        from .prompting import PROMPT_FILE
                        if PROMPT_FILE.is_file():
                            shutil.copyfile(PROMPT_FILE, out/'starter_strategy_used.txt')
                    call_start = time.perf_counter()
                    calls += 1
                    Image.fromarray(image).save(out / f'plan_{calls:03d}.png')
                    raw = model.generate(image, prompt, deadline)
                    synchronize()
                    duration = time.perf_counter()-call_start
                    planning_seconds += duration
                    row = {'prompt': prompt, 'raw': raw, 'call': calls, 'seconds': duration,
                           'image': f'plan_{calls:03d}.png', 'context': ctx}
                    if time.perf_counter() >= deadline:
                        dump(out / f'plan_{calls:03d}.json', {**row, 'discarded': 'deadline'})
                        event('late_plan_discarded', call=calls)
                        stop = 'time_limit'
                        break
                    try:
                        parsed = parse_response(raw)
                        last_model_response = parsed
                        queue = parsed['plan'].copy()
                        row['response'] = parsed
                        feedback = {'event': 'plan_received', 'actions': len(queue), 'model_done': parsed['done']}
                        event('plan_received', call=calls, plan=queue, assessments=parsed['assessments'])
                    except (ValueError, TypeError, json.JSONDecodeError) as exc:
                        queue = []
                        row['parse_error'] = str(exc)
                        feedback = {'event': 'invalid_plan', 'error': str(exc)}
                        event('invalid_plan', call=calls, error=str(exc))
                    dump(out / f'plan_{calls:03d}.json', row)
                    need_plan = False
                    current = None
                    if not queue:
                        decision = decide('invalid_plan' if 'parse_error' in row else 'plan_exhausted') if config.condition == 'C' else 'stop'
                        if decision == 'replan':
                            need_plan = True
                            continue
                        stop = 'empty_or_invalid_plan'
                        break
                if current is None and config.condition in 'BCD':
                    if config.condition == 'D':
                        chosen = classical_next(env, attempts, config.max_attempts)
                    else:
                        chosen = queue.pop(0) if queue else None
                    if chosen is None:
                        feedback = {'event': 'plan_exhausted'}
                        decision = decide('plan_exhausted') if config.condition == 'C' else 'stop'
                        if decision == 'replan':
                            need_plan = True
                            continue
                        stop = 'plan_exhausted'
                        break
                    n = chosen['object']
                    if attempts.get(n, 0) >= config.max_attempts:
                        stop = 'object_attempt_limit'
                        break
                    attempts[n] = attempts.get(n, 0)+1
                    current = chosen
                    task_label = instruction_text(n, chosen['target_zone'])[0]
                    current_steps = 0
                    if learned is not None:
                        learned.reset()
                    else:
                        teacher = ScriptedSortingPolicy(env)
                        teacher.max_attempts = 1
                        teacher.set_subtask(chosen)
                    event('action_started', instruction=task_label, action=chosen)
                tick = time.perf_counter()
                if config.condition == 'E':
                    panel.pump_input(source)
                    action = source.poll()
                    source.events.clear()  # Reset/discard cannot alter an evaluation trial.
                    if source.quit or panel.quit:
                        stop = 'operator_stopped'
                        break
                elif config.condition == 'D':
                    action = teacher.select_action()
                else:
                    t = time.perf_counter()
                    env.image_size = max(config.image_size, getattr(learned, 'image_size', config.image_size))
                    if hasattr(learned,'set_controller_target'):
                        learned.set_controller_target(env.ctrl.target_pos)
                    action = learned.select_action({**env.get_obs(), 'task': task_label})
                    synchronize()
                    action_inference_seconds += time.perf_counter()-t
                if time.perf_counter() >= deadline:
                    event('late_action_discarded')
                    stop = 'time_limit'
                    break
                action = np.asarray(action, dtype=np.float32)
                if action.shape != (C.ACTION_DIM,) or not np.isfinite(action).all() or np.max(np.abs(action)) > 1.00001:
                    raise ValueError('Policy must return finite bounded action7')
                state_before_step = state_of(env)
                env.step(action)
                if time.perf_counter() >= deadline:
                    # Keep the last completely observed state within the budget.
                    # A control integration that finishes after the deadline is
                    # not credited as a within-time placement.
                    from .scenes import restore
                    restore(env, {**spec, 'initial_state': state_before_step})
                    event('late_step_discarded')
                    stop = 'time_limit'
                    break
                steps += 1
                current_steps += 1
                trace['actions'].append(action.copy())
                trace['tasks'].append(task_label)
                trace['qpos'].append(env.data.qpos.copy())
                trace['qvel'].append(env.data.qvel.copy())
                trace['wall'].append(time.perf_counter()-start)
                trace['actuator_ctrl'].append(env.data.ctrl.copy())
                trace['target_pos'].append(env.ctrl.target_pos.copy())
                trace['target_quat'].append(env.ctrl.target_quat.copy())
                outcome = observe_outcome(env)
                stable_scene_steps = stable_scene_steps+1 if outcome['scene_success'] else 0
                if panel:
                    panel.update()
                action_done = current is not None and (current_steps >= config.action_steps or (teacher is not None and teacher.done))
                if action_done:
                    feedback = {'event': 'after_action', 'instruction': task_label, 'action': current,
                                'steps': current_steps, 'termination': 'fixed_action_horizon' if learned else 'classical_controller_finished'}
                    history.append(feedback.copy())
                    event('after_action', feedback=feedback)
                    current = None
                    if config.condition == 'C':
                        decision = decide('after_action')
                        if decision == 'stop':
                            stop = 'student_stopped'
                            break
                        need_plan = decision == 'replan'
                elif config.condition == 'C' and current_steps % config.observer_steps == 0:
                    feedback = {'event': 'observation_tick', 'instruction': task_label, 'steps': current_steps}
                    decision = decide('observation_tick')
                    if decision == 'stop':
                        stop = 'student_stopped'
                        break
                    if decision == 'replan':
                        history.append({'event': 'interrupted', 'instruction': task_label, 'steps': current_steps})
                        need_plan = True
                # Common wall-time pacing avoids giving D accelerated physics and
                # human E real time. Inference slower than 20 Hz is not hidden.
                pause = min(tick + 1/C.CONTROL_HZ, deadline) - time.perf_counter() if config.realtime else 0
                if pause > 0:
                    time.sleep(pause)
            report['stop_reason'] = stop or 'step_limit'
            report['status'] = 'completed'
            report['final'] = outcome
        except Exception as exc:
            report['status'] = 'error'
            report['stop_reason'] = 'exception'
            report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            report['final'] = observe_outcome(env)
        finally:
            synchronize()
            wall = time.perf_counter()-start if start is not None else 0.0
            report['summary'] = {'total_wall_seconds': wall, 'deadline_seconds': config.seconds,
                                 'deadline_overrun_seconds': max(0, wall-config.seconds),
                                 'simulation_seconds': steps/C.CONTROL_HZ, 'steps': steps, 'planner_calls': calls,
                                 'planner_seconds': planning_seconds, 'vla_inference_seconds': action_inference_seconds,
                                 'attempts': attempts}
            report['history'] = history
            Image.fromarray(env.render('agentview', 512, 512)).save(out / 'after.png')
            if not trace['qpos']:
                trace['qpos'].append(env.data.qpos.copy())
                trace['qvel'].append(env.data.qvel.copy())
                trace['wall'].append(0.)
                trace['actuator_ctrl'].append(env.data.ctrl.copy())
                trace['target_pos'].append(env.ctrl.target_pos.copy())
                trace['target_quat'].append(env.ctrl.target_quat.copy())
            np.savez_compressed(out / 'trace.npz', **{k: np.asarray(v) for k, v in trace.items()})
            if panel:
                panel.close()
                import pygame
                pygame.quit()
            dump(out / 'report.json', report)
    if config.video:
        render_start = time.perf_counter()
        try:
            report['video'] = render_replay(spec, trace, report, out)
        except Exception as exc:
            report['video_error'] = str(exc)
        report['artifact_render_seconds'] = time.perf_counter()-render_start
    report['end_to_end_wall_seconds'] = time.perf_counter()-lifecycle_start
    dump(out / 'report.json', report)
    return report
