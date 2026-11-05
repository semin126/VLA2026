"""Generate candidates, explicitly review VLM labels, and collect real action demos.

Benchmark test scenes are never accepted for training data. sft_test is a separate
training-data holdout; it is not the final A–E benchmark.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .common import compose_prompt, digest, dump, file_hash, load_strategy, parse_response, public_task, task_text
from .models import ImagePlanner
from .scenes import environment, read_scene

DATA_SPLITS = {'train', 'validation', 'sft_test'}


def require_training_scene(spec):
    if spec['split'] not in DATA_SPLITS:
        raise ValueError('Extra data require train/validation/sft_test scenes; never benchmark test or development scenes')


def generate_candidates(scene_paths, strategy_path, model_id, out, device='auto', seconds_per_call=60, command=None):
    from .common import finite_positive
    finite_positive(seconds_per_call, 'seconds_per_call')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    strategy = load_strategy(strategy_path)
    model = ImagePlanner(model_id, device=device)
    records = []
    for path in scene_paths:
        spec = read_scene(path)
        if command is not None:
            if not isinstance(command, str) or not command.strip():
                raise ValueError('command must be a nonempty string')
            spec = {**spec, 'command': command.strip()}
        require_training_scene(spec)
        with environment(spec) as env:
            image = env.render('agentview', 512, 512)
        prompt = compose_prompt(public_task(spec['command']), strategy,
                                {'event': 'initial', 'history': [], 'feedback': None})
        start = time.perf_counter()
        raw = model.generate(image, prompt, start+seconds_per_call)
        ident = digest({'scene': spec, 'prompt': prompt})[:20]
        Image.fromarray(image).save(out / (ident + '.png'))
        row = {'id': ident, 'scene_id': spec['id'], 'family': spec['family'], 'seed': spec['seed'],
               'scene_sha256': digest(spec), 'state_sha256': spec['state_sha256'], 'split': spec['split'],
               'prompt': prompt, 'raw': raw, 'image': ident + '.png', 'seconds': time.perf_counter()-start,
               'source': 'model_generated_candidate', 'review_status': 'pending', 'physical_execution_verified': False,
               'model': model_id, 'strategy_sha256': file_hash(strategy_path)}
        try:
            row['response'] = parse_response(raw)
        except (ValueError, TypeError) as exc:
            row['parse_error'] = str(exc)
        records.append(row)
        dump(out / 'candidates.json', records)
    dump(out / 'review.json', {r['id']: {'decision': 'pending', 'reviewer': '',
                                        'candidate_sha256': digest(r), 'response': r.get('response'), 'note': ''}
                               for r in records})
    return records


def import_feedback_candidates(trial_root, out):
    """Preserve the exact image and prompt used by C, including feedback context."""
    import shutil
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for report_path in Path(trial_root).rglob('report.json'):
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report.get('protocol') != 'final_project_v2.0' or report['condition'] != 'C' or report['diagnostic']:
            continue
        spec = read_scene(report_path.parent / 'scene.json')
        require_training_scene(spec)
        for plan in sorted(report_path.parent.glob('plan_*.json')):
            p = json.loads(plan.read_text(encoding='utf-8'))
            ident = digest({'scene': spec, 'prompt': p['prompt'], 'raw': p['raw']})[:20]
            shutil.copyfile(report_path.parent / p['image'], out / (ident + '.png'))
            rows.append({'id': ident, 'scene_id': spec['id'], 'family': spec['family'], 'seed': spec['seed'],
                         'scene_sha256': digest(spec), 'state_sha256': spec['state_sha256'], 'split': spec['split'],
                         'prompt': p['prompt'], 'raw': p['raw'], 'image': ident+'.png', 'response': p.get('response'),
                         'review_status': 'pending', 'physical_execution_verified': False,
                         'source': 'feedback_rollout_candidate', 'source_trial': str(report_path),
                         'strategy_sha256': report['strategy']['sha256'], 'model': report['configuration']['vlm_model']})
    if not rows:
        raise ValueError('No eligible non-diagnostic C feedback trials')
    dump(out / 'candidates.json', rows)
    dump(out / 'review.json', {r['id']: {'decision': 'pending', 'reviewer': '', 'candidate_sha256': digest(r),
                                        'response': r.get('response'), 'note': ''} for r in rows})
    return rows


def export_reviewed(candidate_dirs, out):
    """Labels are accepted only by an explicit named human review, with lineage checks."""
    import shutil
    out = Path(out)
    # Validate fully before creating the export, so rejected input leaves no partial dataset.
    records = []
    ownership = {'family': {}, 'seed': {}, 'state_sha256': {}}
    seen = set()
    for directory in map(Path, candidate_dirs):
        candidates = json.loads((directory / 'candidates.json').read_text(encoding='utf-8'))
        reviews = json.loads((directory / 'review.json').read_text(encoding='utf-8'))
        for row in candidates:
            if row['split'] not in DATA_SPLITS:
                raise ValueError('Benchmark/development leakage')
            review = reviews[row['id']]
            if review['decision'] != 'accepted':
                continue
            if not review.get('reviewer', '').strip() or review.get('candidate_sha256') != digest(row):
                raise ValueError('Review requires a named reviewer and matching candidate hash')
            from .. import numbered
            # Starter suite is 10 numbered cubes. Record count explicitly in a future object-set extension.
            numbered.activate(count=10)
            completion = json.dumps(review['response'], ensure_ascii=False)
            parse_response(completion)
            for key, owners in ownership.items():
                v = row[key]
                if v in owners and owners[v] != row['split']:
                    raise ValueError('Split leakage in ' + key)
                owners[v] = row['split']
            unique = digest({'image': file_hash(directory / row['image']), 'prompt': row['prompt'], 'completion': completion})
            if unique in seen:
                continue
            seen.add(unique)
            records.append((row, review, completion, directory / row['image']))
    for split in DATA_SPLITS:
        if not any(r['split'] == split for r, *_ in records):
            raise ValueError('Need accepted records in train, validation and sft_test before export')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()
    emitted = []
    for split, filename in [('train', 'train'), ('validation', 'validation'), ('sft_test', 'test')]:
        with (out / (filename+'.jsonl')).open('w', encoding='utf-8') as f:
            for row, review, completion, image in records:
                if row['split'] != split:
                    continue
                dst = out / 'images' / (file_hash(image)+'.png')
                shutil.copyfile(image, dst)
                record = {**row, 'image': dst.relative_to(out).as_posix(), 'completion': completion,
                          'review_status': 'accepted', 'reviewer': review['reviewer'], 'review_note': review.get('note', '')}
                f.write(json.dumps(record, ensure_ascii=False)+'\n')
                emitted.append({'id': row['id'], 'family': row['family'], 'split': split, 'seed': row['seed']})
    dump(out / 'manifest.json', {'status': 'complete', 'protocol': 'final_project_v2.0', 'records': emitted,
                                 'label_source': 'diagnostic_fixture' if any(row.get('diagnostic', False) for row, *_ in records)
                                                 else 'human_reviewed_model_candidates',
                                 'diagnostic': any(row.get('diagnostic', False) for row, *_ in records),
                                 'test_note': 'test.jsonl comes from sft_test, never the final A–E benchmark'})
    return emitted


def collect_action_data(scene_path, out, seconds=120, max_steps=2400, command=None, *, image_size=128, accelerated=False):
    """D-state physical trajectories, exported with both scene and subtask labels.

Images are rendered from recorded states offline. The control trajectory was
actually stepped during the trial. This is not a reversed or imagined demo.
"""
    import mujoco
    from ..episode import EpisodeWriter
    from .runner import TrialConfig, run_trial
    from .metrics import observe_outcome
    spec = read_scene(scene_path)
    require_training_scene(spec)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    report = run_trial(scene_path, out / 'trial', TrialConfig(condition='D', seconds=seconds, max_steps=max_steps,
                       command=command, realtime=not accelerated, video=not accelerated))
    spec = read_scene(out/'trial/scene.json')  # The effective command is used for scene-level VLA labels too.
    trace = np.load(out / 'trial' / 'trace.npz')
    meta = {'source': 'scripted_v2_physical_rollout', 'seed': spec['seed'], 'family': spec['family'],
            'split': spec['split'], 'scene_sha256': digest(spec), 'task_level': 'scene',
            'success': report['final']['scene_success'], 'state_sha256': spec['state_sha256']}
    manifest = []
    with environment(spec, image_size=image_size) as env:
        def pose(i):
            env.data.qpos[:] = trace['qpos'][i]
            env.data.qvel[:] = trace['qvel'][i]
            env.data.ctrl[:] = trace['actuator_ctrl'][i]
            env.ctrl.target_pos = trace['target_pos'][i].copy()
            env.ctrl.target_quat = trace['target_quat'][i].copy()
            mujoco.mj_forward(env.model, env.data)

        def save(start, end, task, metadata, dest):
            writer = EpisodeWriter(env, task, direction='tidy', meta=metadata)
            for i in range(start, end+1):
                pose(i)
                writer.record_obs()
                if i < end:
                    writer.record_action(trace['actions'][i])
            writer.save(dest)
            manifest.append({'path': str(dest.relative_to(out)), 'success': metadata['success'],
                             'task_level': metadata['task_level'], 'steps': end-start})

        n = len(trace['actions'])
        if n:
            destination = out / 'scene' / ('accepted' if meta['success'] else 'rejected') / 'scene.hdf5'
            save(0, n, task_text(public_task(spec['command'])), meta, destination)
        starts = [e for e in report['events'] if e['event'] == 'action_started']
        for j, event in enumerate(starts):
            start = event['step']
            end = starts[j+1]['step'] if j+1 < len(starts) else n
            if end <= start:
                continue
            name = event['action']['object']
            pose(end)
            final = observe_outcome(env)
            height = trace['qpos'][start:end+1, env._jnt_adr[name]+2]
            expected_z = trace['qpos'][start, env._jnt_adr[name]+2]
            success = bool(final['objects'][name]['correct'] and height.max() > expected_z+.04)
            metadata = {**meta, 'task_level': 'subtask', 'object': name,
                        'target_zone': event['action']['target_zone'], 'success': success}
            dest = out / 'subtask' / ('accepted' if success else 'rejected') / f'transfer_{j:03d}.hdf5'
            save(start, end, event['instruction'], metadata, dest)
    dump(out / 'collection.json', {'scene': spec['id'], 'split': spec['split'], 'episodes': manifest,
                                  'source_trial': 'trial/report.json', 'failures_retained': True})
    return manifest


def prepare_action_data(collections, system, out, *, convert=False, repo_id='dshs2026/final_project_v2',action_representation='delta'):
    """Enforce v2 family/state/split provenance before the existing converter runs."""
    import subprocess
    import sys
    from ..episode import Episode
    from ..demo_validation import require_training_demo
    label = {'single': 'scene', 'hier': 'subtask'}[system]
    owners = {'family': {}, 'state_sha256': {}, 'seed': {}}
    rows, accepted_dirs, holdout = [], set(), set()
    for path in Path(collections).glob('*/'+label+'/accepted/*.hdf5'):
        ep = Episode(path,load_images=False)
        require_training_demo(ep, task_level=label)
        split = ep.meta.get('split')
        if split not in DATA_SPLITS:
            raise ValueError('Only v2 train/validation/sft_test episodes can be converted')
        for key in owners:
            val = ep.meta.get(key)
            if val is None:
                raise ValueError('Missing v2 lineage: '+key)
            if val in owners[key] and owners[key][val] != split:
                raise ValueError('VLA split leakage: '+key)
            owners[key][val] = split
        if split != 'train':
            holdout.add(int(ep.meta['seed']))
        accepted_dirs.add(path.parent.resolve())
        rows.append({'path': str(path.resolve()), 'split': split, 'family': ep.meta['family'], 'seed': ep.meta['seed']})
    if not rows or not any(r['split'] == 'train' for r in rows) or not holdout:
        raise ValueError('Need successful training and independent holdout episodes')
    script = Path(__file__).resolve().parents[2] / 'scripts' / 'prepare_vla_training.py'
    command = [sys.executable, str(script), '--system', system, '--holdout-seeds', ','.join(map(str, sorted(holdout))),
               '--out', str(out), '--repo-id', repo_id,'--action-representation',action_representation]
    for folder in sorted(accepted_dirs):
        command += ['--input-dir', str(folder)]
    if convert:
        command += ['--convert', '--images']
    subprocess.run(command, check=True)
    dump(Path(out)/'v2_lineage.json', {'episodes': rows, 'note': 'validation and sft_test remain labelled separately in this lineage; converter trains on train only'})
    return rows


def evaluate_vlm(dataset, split, model_id, out, device='auto'):
    """Planning label evaluation, distinct from the physical A–E task metrics."""
    dataset, out = Path(dataset), Path(out)
    if split not in ('validation', 'test'):
        raise ValueError('Select validation or the separate SFT test split')
    rows = [json.loads(line) for line in (dataset/(split+'.jsonl')).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty evaluation set')
    out.mkdir(parents=True, exist_ok=False)
    model = ImagePlanner(model_id, device=device)
    results = []
    for row in rows:
        from .. import numbered
        numbered.activate(count=10)
        image = np.asarray(Image.open(dataset/row['image']).convert('RGB'))
        raw = model.generate(image, row['prompt'], time.perf_counter()+120)
        gold = parse_response(row['completion'])
        item = {'id': row['id'], 'raw': raw, 'format_ok': False, 'exact_plan': False, 'f1': 0.0,
                'accessibility_acc': 0.0, 'done_correct': False}
        try:
            predicted = parse_response(raw)
            predicted_access={a['object']:a['accessible'] for a in predicted['assessments']}
            item['accessibility_acc']=sum(predicted_access.get(a['object']) == a['accessible']
                                           for a in gold['assessments'])/max(1,len(gold['assessments']))
            item['done_correct']=predicted['done']==gold['done']
            p = {(x['object'], x['target_zone']) for x in predicted['plan']}
            g = {(x['object'], x['target_zone']) for x in gold['plan']}
            item.update(format_ok=True, exact_plan=[(x['object'], x['target_zone']) for x in predicted['plan']]
                        == [(x['object'], x['target_zone']) for x in gold['plan']],
                        f1=2*len(p & g)/(len(p)+len(g)) if p or g else 1.0)
        except (ValueError, TypeError) as exc:
            item['parse_error'] = str(exc)
        results.append(item)
        dump(out/'predictions.json', results)
    summary = {'task': 'VLM planning labels, not robot success', 'n': len(results),
               'format_accuracy': float(np.mean([r['format_ok'] for r in results])),
               'plan_pair_f1': float(np.mean([r['f1'] for r in results])),
               'geometric_accessibility_acc': float(np.mean([r['accessibility_acc'] for r in results])),
               'done_accuracy': float(np.mean([r['done_correct'] for r in results])),
               'exact_plan_accuracy': float(np.mean([r['exact_plan'] for r in results]))}
    dump(out/'summary.json', summary)
    return summary
