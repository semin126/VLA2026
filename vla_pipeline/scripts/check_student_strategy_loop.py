"""방법 2: 가짜(diagnostic) VLM/VLA 어댑터로 전체 C 루프를 실제 씬에서 검증.

학습된 체크포인트나 모델 다운로드 없이, run_trial 의 planner_adapter/policy_adapter
테스트 시드를 써서 우리가 만든 validation 씬 3개에 condition C(=student_strategy.py)를
끝까지 돌린다. MuJoCo 가 필요하므로 Colab 등 시뮬레이터 환경에서 실행한다.

확인 항목(씬마다):
  - 상태 completed (우리 코드가 끝까지 동작)
  - planner 호출 2회 = 재계획이 실제로 트리거됨 (B 는 1회)
  - REPLAN_GUIDANCE 가 2번째 프롬프트에만 등장, 1번째엔 없음
  - student_feedback 이 2번째 프롬프트에 전달됨
  - oracle 누수 없음 (프롬프트에 좌표/성공 플래그 없음)
  - (scene0) B 와 C 의 최초 프롬프트가 글자까지 동일

    python scripts/check_student_strategy_loop.py --scenes <suite_dir> --out <out_dir>
노트북에서는:  run.run('scripts/check_student_strategy_loop.py', '--scenes', SCENES, '--out', run.root/'check_c_loop')
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import numpy as np

from vlapipe.final_project_v2.common import load_strategy
from vlapipe.final_project_v2.runner import TrialConfig, run_trial

PROJECT = Path(__file__).resolve().parent.parent
STUDENT = PROJECT / 'final_project_v2/student_strategy.py'
COMMAND = 'Tidy up the table: put every numbered object in its designated storage zone.'


class IdlePolicy:
    """가짜 VLA: 아무 것도 하지 않는 유효한 action7 을 돌려준다."""
    def reset(self):
        pass

    def select_action(self, observation):
        assert set(observation) <= {'task', 'agentview_image', 'robot0_eye_in_hand_image',
                                    'robot0_joint_pos', 'robot0_gripper_qpos',
                                    'robot0_eef_pos', 'robot0_eef_quat'}
        return np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


class FakePlanner:
    """가짜 VLM: 매 호출 같은 2-동작 계획을 돌려주고, 받은 프롬프트를 기록한다."""
    def __init__(self):
        self.calls = []
        self.raw = json.dumps({'assessments': [], 'done': False, 'plan': [
            {'object': 'object_1', 'target_zone': 'right_zone', 'reason': 'diagnostic'},
            {'object': 'object_2', 'target_zone': 'right_zone', 'reason': 'diagnostic'}]})

    def generate(self, image, prompt, deadline):
        self.calls.append(prompt)
        assert image.shape == (512, 512, 3)
        return self.raw


def run_one(scene_path, out, condition, strategy):
    cfg = TrialConfig(condition=condition, seconds=30, max_steps=40, action_steps=2,
                      observer_steps=2, max_calls=2, checkpoint='diagnostic',
                      strategy=str(strategy), command=COMMAND, video=False)
    planner = FakePlanner()
    report = run_trial(scene_path, out, cfg, planner_adapter=planner, policy_adapter=IdlePolicy())
    return report, planner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', type=Path, required=True, help='suite.json 이 있는 디렉터리')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    guidance = load_strategy(STUDENT).REPLAN_GUIDANCE
    registry = json.loads((args.scenes / 'suite.json').read_text(encoding='utf-8'))
    scenes = [(r['family'], args.scenes / r['path']) for r in registry['scenes'] if r['split'] == 'validation']
    if not scenes:
        raise SystemExit('No validation scenes in suite.json')

    results = []
    for i, (family, scene_path) in enumerate(scenes):
        rc, cplan = run_one(scene_path, args.out / f'C_{i:02d}_{family}', 'C', STUDENT)
        assert rc['status'] == 'completed', (family, rc.get('error'))
        durable_events = [json.loads(line) for line in
                          (args.out / f'C_{i:02d}_{family}' / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        assert rc['events'] == durable_events, 'Final events changed after their durable snapshot'
        assert all(len(e['plan']) == 2 for e in rc['events'] if e['event'] == 'plan_received'), 'Consumed queue erased the original plan'
        assert len(cplan.calls) == 2, (family, 'expected 2 planner calls, got', len(cplan.calls))
        assert guidance not in cplan.calls[0], (family, 'guidance leaked into initial prompt')
        assert guidance in cplan.calls[1], (family, 'guidance missing from replan prompt')
        assert 'student_feedback' in cplan.calls[1], (family, 'student_feedback not passed to replan')
        for text in cplan.calls:
            assert '"positions":' not in text and '"success":' not in text, (family, 'oracle leak in prompt')
        results.append({'scene': family, 'status': rc['status'], 'planner_calls': len(cplan.calls),
                        'stop_reason': rc['stop_reason'], 'acc': rc['final']['acc'],
                        'guidance_initial': guidance in cplan.calls[0],
                        'guidance_replan': guidance in cplan.calls[1],
                        'student_feedback_in_replan': 'student_feedback' in cplan.calls[1]})

    # scene0 에서 B 와 C 의 최초 프롬프트가 동일한지 (런타임에서 직접) 확인
    _, bplan = run_one(scenes[0][1], args.out / 'B_00', 'B', PROJECT / 'final_project_v2/baseline_strategy.py')
    _, cplan0 = run_one(scenes[0][1], args.out / 'C_00_identity', 'C', STUDENT)
    assert len(bplan.calls) == 1, ('B should call once', len(bplan.calls))
    assert bplan.calls[0] == cplan0.calls[0], 'B and C initial prompts differ'
    identical_initial = bplan.calls[0] == cplan0.calls[0]

    # Regression: a full ten-action initial plan must leave room to replan.
    from vlapipe.final_project_v2.notebook_runtime import check_research_budget
    check_research_budget(['C'], 30, 12, 1, 2)
    long_plan = FakePlanner()
    long_plan.raw = json.dumps({'assessments': [], 'done': False, 'plan': [
        {'object': f'object_{i}', 'target_zone': 'right_zone' if i <= 5 else 'left_zone'}
        for i in range(1, 11)]})
    long_report = run_trial(scenes[0][1], args.out/'C_ten_action_plan',
        TrialConfig(condition='C', seconds=30, max_steps=12, action_steps=1,
                    observer_steps=1, max_calls=2, checkpoint='diagnostic',
                    strategy=str(STUDENT), video=False),
        planner_adapter=long_plan, policy_adapter=IdlePolicy())
    assert long_report['status'] == 'completed', long_report.get('error')
    assert len(long_plan.calls) == 2, 'Ten-action plan never reached replan'
    assert long_report['summary']['steps'] == 12

    # Hooks may edit their input for analysis; those edits must not alter live
    # actions, past events or the next context's original VLM response.
    mutation_strategy = args.out / 'mutation_strategy.py'
    mutation_strategy.write_text('''from vlapipe.final_project_v2.prompting import build_strategy_prompt
build_prompt = build_strategy_prompt
def loop_decision(context):
    assert len(context['last_model_response']['plan']) == 2
    assert all(h['termination'] == 'fixed_action_horizon' for h in context['history'])
    if context['queued_actions']:
        context['queued_actions'][0]['object'] = 'object_10'
    context['last_model_response']['plan'].clear()
    context['history'][0]['termination'] = 'context_mutation'
    context['feedback']['termination'] = 'context_mutation'
    return {'decision': 'continue', 'feedback': {'context_edit_test': True}}
''', encoding='utf-8')
    isolated = run_trial(scenes[0][1], args.out/'C_metadata_isolation',
        TrialConfig(condition='C', seconds=30, max_steps=4, action_steps=2,
                    observer_steps=2, max_calls=1, strategy=str(mutation_strategy), video=False),
        planner_adapter=FakePlanner(), policy_adapter=IdlePolicy())
    assert isolated['status'] == 'completed', isolated.get('error')
    assert [e['action']['object'] for e in isolated['events'] if e['event'] == 'action_started'] == ['object_1','object_2']
    assert all(h['termination'] == 'fixed_action_horizon' for h in isolated['history'])
    durable = [json.loads(line) for line in (args.out/'C_metadata_isolation/events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert isolated['events'] == durable
    assert all(e['feedback'] == {'context_edit_test': True} for e in isolated['events'] if e['event']=='student_decision')

    summary = {'method': 2, 'scenes_checked': len(scenes), 'per_scene': results,
               'B_calls': len(bplan.calls), 'B_C_initial_prompt_identical': identical_initial,
               'ten_action_plan_replanned': len(long_plan.calls) == 2,
               'event_snapshots_preserved': True, 'student_metadata_isolated': True,
               'note': 'Fake idle VLA / fixed-plan VLM; validates C-loop wiring, not model performance.'}
    (args.out / 'check_c_loop.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
