"""Meaningful v2 contract checks with actual MuJoCo and labelled diagnostic adapters."""
import _bootstrap  # noqa: F401
import argparse
import json
import time
from pathlib import Path

import numpy as np

from vlapipe.final_project_v2.common import digest, dump, parse_response
from vlapipe.final_project_v2.metrics import scores
from vlapipe.final_project_v2.scenes import environment, freeze, state_of, template, validate_layout
from vlapipe.final_project_v2.runner import TrialConfig, run_trial


class IdlePolicy:
    def __init__(self):
        self.tasks = []

    def reset(self):
        pass

    def select_action(self, observation):
        self.tasks.append(observation['task'])
        assert set(observation) <= {'task', 'agentview_image', 'robot0_eye_in_hand_image',
                                   'robot0_joint_pos', 'robot0_gripper_qpos', 'robot0_eef_pos', 'robot0_eef_quat'}
        return np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


class Plans:
    def __init__(self, raw=None, delay=0):
        self.calls = []
        self.raw = raw or json.dumps({'assessments': [], 'plan': [
            {'object': 'object_1', 'target_zone': 'right_zone'},
            {'object': 'object_2', 'target_zone': 'right_zone'}], 'done': False})
        self.delay = delay

    def generate(self, image, prompt, deadline):
        self.calls.append(prompt)
        assert image.shape == (512, 512, 3)
        time.sleep(self.delay)
        return self.raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    checks = []
    for category in ('open', 'blocked', 'mixed'):
        for seed in range(5):
            s = template(category, seed)
            assert validate_layout(s)['diagnostic_category'] == category
    checks.append('three categories: 15 valid, nonpenetrating, reachable layouts')
    s = freeze(template('mixed', 100))
    with environment(s) as env:
        assert digest(state_of(env)) == s['state_sha256']
    checks.append('frozen state restored exactly')
    path = args.out/'scene.json'
    dump(path, s)
    a = scores({'a': 'L', 'b': 'L', 'c': 'R', 'd': 'R'}, {'a': 'L', 'b': 'invalid', 'c': 'L', 'd': 'R'})
    assert a['acc'] == .5 and abs(a['macro_f1'] - (.5+2/3)/2) < 1e-10
    assert scores({'a': 'L'}, {'a': 'invalid'})['macro_f1'] == 0
    checks.append('ACC and macro-F1 against hand-computed errors and invalid placements')
    try:
        parse_response('{"plan":[{"object":"object_99","target_zone":"right_zone"}],"assessments":[],"done":false}')
        raise AssertionError('invalid plan accepted')
    except ValueError:
        pass
    # Wrong destinations are valid model decisions, not silently repaired by an oracle.
    assert parse_response('{"plan":[{"object":"object_1","target_zone":"left_zone"}],"assessments":[],"done":false}')['plan'][0]['target_zone'] == 'left_zone'
    strategy = args.out/'strategy.py'
    strategy.write_text('import json\ndef build_prompt(task,context):\n return "TEST " + json.dumps(context,sort_keys=True)\n'
                        'def loop_decision(context):\n return "replan" if context["event"] in ("after_action","invalid_plan","plan_exhausted") else "continue"\n', encoding='utf-8')
    p = IdlePolicy()
    r = run_trial(path, args.out/'A', TrialConfig(condition='A', seconds=2, max_steps=3, checkpoint='diagnostic'), policy_adapter=p)
    assert r['status'] == 'completed' and r['summary']['planner_calls'] == 0 and len(p.tasks) == 3
    assert len(set(p.tasks)) == 1 and 'Objects 1-5' in p.tasks[0] and '6-10' in p.tasks[0]
    b, c = Plans(), Plans()
    common = dict(seconds=3, max_steps=4, action_steps=2, observer_steps=1, max_calls=3,
                  strategy=str(strategy), checkpoint='diagnostic')
    rb = run_trial(path, args.out/'B', TrialConfig(condition='B', **common), planner_adapter=b, policy_adapter=IdlePolicy())
    rc = run_trial(path, args.out/'C', TrialConfig(condition='C', **common), planner_adapter=c, policy_adapter=IdlePolicy())
    assert rb['status'] == rc['status'] == 'completed', (rb.get('error'), rc.get('error'))
    assert len(b.calls) == 1 and len(c.calls) == 2 and b.calls[0] == c.calls[0]
    assert all('"positions":' not in text and '"success":' not in text for text in c.calls)
    assert rb['restored_state_sha256'] == rc['restored_state_sha256']
    checks.append('A has no planner/object selection; B one call; C feedback; identical B/C initial prompt/state; no oracle feedback')
    project = Path(__file__).resolve().parent.parent
    provided = dict(seconds=3, max_steps=2, action_steps=1, observer_steps=1, max_calls=2,
                    checkpoint='diagnostic', command='Tidy up these numbered objects.')
    bp, cp = Plans(), Plans()
    pb = run_trial(path, args.out/'provided_B', TrialConfig(condition='B', **provided),
                   planner_adapter=bp, policy_adapter=IdlePolicy())
    pc = run_trial(path, args.out/'provided_C', TrialConfig(condition='C', strategy=str(project/'final_project_v2/loop_example.py'),
                   **provided), planner_adapter=cp, policy_adapter=IdlePolicy())
    assert pb['status'] == pc['status'] == 'completed', (pb.get('error'), pc.get('error'))
    assert len(bp.calls) == 1 and len(cp.calls) == 2
    assert pb['initial_prompt_sha256'] == pc['initial_prompt_sha256']
    assert pb['strategy']['sha256'] != pc['strategy']['sha256']  # Different loop modules are intentional.
    assert pb['restored_state_sha256'] == s['state_sha256']
    assert json.loads((args.out/'provided_B/task.json').read_text(encoding='utf-8'))['command'] == provided['command']
    assert Path(pb['strategy']['path']).name == 'baseline_strategy.py'
    checks.append('complete B needs no student module; supplied C example shares initial prompt; command override preserves state')
    import runpy
    cli = runpy.run_path(str(project/'scripts/final_project_v2.py'), run_name='cli_contract')
    parsed = cli['parser']().parse_args(['suite', '--scenes', str(args.out), '--conditions', 'A,B,D',
                '--out', str(args.out/'unused'), '--checkpoint', 'diagnostic', '--task-command', 'Tidy up the table.'])
    cfg = cli['trial_config'](parsed, 'B')
    assert parsed.command == 'suite' and cfg.command == 'Tidy up the table.'
    assert Path(cfg.strategy).name == 'baseline_strategy.py'
    checks.append('CLI A/B/D suite selects complete B and keeps task command separate from subcommand')
    late = Plans(delay=.12)
    r = run_trial(path, args.out/'deadline', TrialConfig(condition='B', seconds=.04, max_steps=2,
                  strategy=str(strategy), checkpoint='diagnostic'), planner_adapter=late, policy_adapter=IdlePolicy())
    assert r['summary']['steps'] == 0 and r['stop_reason'] == 'time_limit' and r['summary']['deadline_overrun_seconds'] > 0
    checks.append('late inference discarded; no action after deadline; overrun measured')
    invalid = Plans(raw='not JSON')
    r = run_trial(path, args.out/'invalid', TrialConfig(condition='C', seconds=2, max_steps=2,
                  strategy=str(strategy), checkpoint='diagnostic', max_calls=2), planner_adapter=invalid, policy_adapter=IdlePolicy())
    assert r['summary']['planner_calls'] == 2 and r['summary']['steps'] == 0 and r['stop_reason'] == 'planner_call_limit'
    checks.append('invalid plans preserved, call cap terminates zero-action loop')
    from vlapipe.final_project_v2.data import require_training_scene
    try:
        require_training_scene({**s, 'split': 'test'})
        raise AssertionError('benchmark leakage')
    except ValueError:
        pass
    checks.append('benchmark test leakage rejected')
    from vlapipe.final_project_v2.results import aggregate
    result = aggregate(args.out, include_diagnostic=True)
    assert len(result['groups']) >= 3
    assert not result['warnings']
    assert sum(row['comparison'] == 'C minus B' for row in result['paired']) >= 2
    checks.append('CSV/HTML grouping and paired summary generated from actual diagnostic runs')
    dump(args.out/'checks.json', {'checks': checks, 'passed': len(checks),
                                 'note': 'Injected idle policies test contracts only, not model performance.'})
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
