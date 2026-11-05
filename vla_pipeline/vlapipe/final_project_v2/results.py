"""Lossless trial table, coverage, configuration groups and paired differences."""
from __future__ import annotations

import csv
import html
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import PROTOCOL
from .common import digest, dump


def aggregate(root, out=None, include_diagnostic=False):
    root = Path(root).resolve()
    out = Path(out or root / 'comparison').resolve()
    out.mkdir(parents=True, exist_ok=True)
    reports, skipped = [], []
    for path in sorted(root.rglob('report.json')):
        r = json.loads(path.read_text(encoding='utf-8'))
        if r.get('protocol') != PROTOCOL:
            continue
        if r.get('diagnostic') and not include_diagnostic:
            skipped.append(str(path))
            continue
        r['_path'] = path
        reports.append(r)
    if not reports:
        raise ValueError('No eligible v2 trials; diagnostic trials are excluded by default')
    rows, grouped, paired = [], defaultdict(list), defaultdict(dict)
    seen = set()
    for r in reports:
        cfg = r['configuration']
        cohort = digest({'budgets': {k: cfg[k] for k in ('seconds', 'max_steps', 'action_steps', 'observer_steps',
                                                       'max_calls', 'max_attempts', 'image_size', 'planner_image_size')},
                         'planner_tokens': cfg.get('planner_tokens', 512),
                         'hardware': r['hardware'], 'split': r['split'], 'task': r['task_sha256'],
                         'implementation': r.get('implementation_sha256', 'pre_fingerprint')})[:12]
        identity = (cohort, r['scene_sha256'], cfg['seed'], r['condition'])
        if identity in seen:
            raise ValueError('Duplicate scene/seed/condition in one cohort; aggregate separate experiment folders')
        seen.add(identity)
        row = {'cohort': cohort, 'condition': r['variant'], 'scene': r['scene_id'], 'split': r['split'],
               'seed': cfg['seed'], 'status': r['status'], 'stop_reason': r['stop_reason'],
               'acc': r['final']['acc'], 'macro_f1': r['final']['macro_f1'],
               'scene_success': int(r['final']['scene_success']),
               'initial_acc': r.get('initial', r['final'])['acc'],
               'new_correct_objects': sum(r['final']['objects'][n]['correct'] and not r.get('initial', r['final'])['objects'][n]['correct']
                                          for n in r['final']['objects']),
               **r['summary'], 'diagnostic': r['diagnostic'],
               'checkpoint': cfg.get('checkpoint'), 'vlm_model': cfg.get('vlm_model') if r['condition'] in 'BC' else '',
               'vla_source': (r.get('models',{}).get('vla',{}).get('source','injected' if r['diagnostic'] else 'legacy_explicit') if r['condition'] in 'ABC' else ''),
               'vlm_source': (r.get('models',{}).get('vlm',{}).get('source','injected' if r['diagnostic'] else 'legacy_explicit') if r['condition'] in 'BC' else ''),
               'action_representation':r.get('models',{}).get('vla',{}).get('action_representation','delta') if r['condition'] in 'ABC' else '',
               'strategy_sha256': (r.get('strategy') or {}).get('sha256', ''),
               'initial_prompt_sha256': r.get('initial_prompt_sha256', ''),
               'model_load_seconds': r.get('model_load_seconds', 0),
               'setup_wall_seconds': r.get('setup_wall_seconds', 0),
               'end_to_end_wall_seconds': r.get('end_to_end_wall_seconds', 0),
               'report': os.path.relpath(r['_path'], out).replace(os.sep, '/')}
        row['attempts'] = json.dumps(row['attempts'], sort_keys=True)
        row['acc_gain']=row['acc']-row['initial_acc']
        rows.append(row)
        grouped[(cohort, r['variant'])].append(row)
        paired[(cohort, r['scene_sha256'], cfg['seed'])][r['condition']] = r
    with (out / 'results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for (cohort, condition), group in grouped.items():
        summaries.append({'cohort': cohort, 'condition': condition, 'n': len(group),
                          'errors': sum(r['status'] == 'error' for r in group),
                          **{k: float(np.mean([r[k] for r in group])) for k in
                             ('initial_acc','acc','acc_gain','new_correct_objects','macro_f1', 'scene_success', 'total_wall_seconds', 'planner_calls')},
                          'completed_only': {'n':sum(r['status']=='completed' for r in group),
                              **{k:float(np.mean([r[k] for r in group if r['status']=='completed']))
                                 if any(r['status']=='completed' for r in group) else None
                                 for k in ('acc','acc_gain','macro_f1','scene_success')}},
                          'model_sources':sorted({(r['vla_source'],r['vlm_source']) for r in group}),
                          'note': 'Initial correct objects are included in ACC; compare ACC gain as well. Error trials retained; completed-only means also shown.'})
    deltas, warnings = [], []
    for left, right in [('A', 'B'), ('B', 'C'), ('A', 'C')]:
        by_cohort = defaultdict(list)
        for (cohort, _, _), group in paired.items():
            if left not in group or right not in group:
                continue
            l, r = group[left], group[right]
            left_rep=l.get('models',{}).get('vla',{}).get('action_representation','delta')
            right_rep=r.get('models',{}).get('vla',{}).get('action_representation','delta')
            if left_rep!=right_rep:
                warnings.append(f'{cohort}: neural action representations differ; pair omitted')
                continue
            def initial_prompt_hash(report):
                if report.get('initial_prompt_sha256'):
                    return report['initial_prompt_sha256']
                plan = report['_path'].parent/'plan_001.json'
                return digest(json.loads(plan.read_text(encoding='utf-8'))['prompt']) if plan.is_file() else None
            if left == 'B' and (not initial_prompt_hash(l) or initial_prompt_hash(l) != initial_prompt_hash(r)
                                or l.get('checkpoint_manifest', l['configuration']['checkpoint']) != r.get('checkpoint_manifest', r['configuration']['checkpoint'])
                                or l.get('vlm_checkpoint_manifest', l['configuration']['vlm_model']) != r.get('vlm_checkpoint_manifest', r['configuration']['vlm_model'])):
                warnings.append(f'{cohort}: B/C initial prompt or models differ; pair omitted')
                continue
            by_cohort[cohort].append((l, r))
        for cohort, group in by_cohort.items():
            row = {'cohort': cohort, 'comparison': f'{right} minus {left}', 'paired_n': len(group)}
            for metric in ('acc', 'macro_f1'):
                values = np.array([r['final'][metric]-l['final'][metric] for l, r in group])
                row[metric + '_difference'] = float(values.mean())
                if len(values) >= 2:
                    rng = np.random.default_rng(314159)
                    samples = rng.choice(values, size=(2000, len(values))).mean(axis=1)
                    row[metric + '_bootstrap_95'] = np.quantile(samples, [.025, .975]).tolist()
            deltas.append(row)
    schedules = list(root.glob('schedule.json'))
    coverage = []
    for schedule in schedules:
        for item in json.loads(schedule.read_text(encoding='utf-8'))['trials']:
            coverage.append({**item, 'report_exists': (root / item['output'] / 'report.json').is_file()})
    summary = {'protocol': PROTOCOL, 'groups': summaries, 'paired': deltas,
               'warnings': sorted(set(warnings)), 'scheduled_coverage': coverage,
               'excluded_diagnostic_reports': skipped,
               'note': 'Compare within cohorts. E is usually a different hardware cohort. Ten scenes give exploratory evidence.'}
    dump(out / 'summary.json', summary)
    columns = ('cohort', 'condition', 'scene', 'status','initial_acc','acc','acc_gain','new_correct_objects',
               'macro_f1','vla_source','vlm_source','action_representation','total_wall_seconds', 'planner_calls')
    parts = ['<!doctype html><meta charset="utf-8"><title>Final project ver2.0</title>',
             '<style>body{font:16px sans-serif;max-width:1400px;margin:30px auto}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}video{max-width:640px}</style>',
             '<h1>Final project ver2.0</h1><p>MP4: simulation-time playback. Task timing: CSV wall seconds. Inference pauses: events and trace.</p>',
             '<p>Groups and paired differences: <a href="summary.json">summary.json</a> · <a href="results.csv">results.csv</a></p>',
             '<table><tr>' + ''.join('<th>'+html.escape(c)+'</th>' for c in columns) + '<th>Artifacts</th></tr>']
    for row in rows:
        base = str(Path(row['report']).parent).replace('\\', '/')
        cells = ''.join('<td>'+html.escape(str(row[k]))+'</td>' for k in columns)
        links = ' · '.join(f'<a href="{html.escape(base+"/"+file, quote=True)}">{label}</a>' for file, label in
                           [('report.json', 'report'), ('before.png', 'before'), ('after.png', 'after'), ('process.mp4', 'video')])
        parts.append('<tr>'+cells+'<td>'+links+'</td></tr>')
    parts.append('</table>')
    (out / 'index.html').write_text('\n'.join(parts), encoding='utf-8')
    return summary
