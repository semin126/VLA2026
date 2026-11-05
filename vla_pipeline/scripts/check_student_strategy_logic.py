"""방법 1: student_strategy.py 순수 함수 검증 (로컬, GPU/MuJoCo/체크포인트 불필요).

build_prompt 공정성(초기=B 동일, replan=보강 추가), loop_decision 규칙,
_student_feedback 경계(JSON 직렬화 + 12000자 미만)를 합성 입력으로 확인한다.
모델을 전혀 로드하지 않으므로 표준 라이브러리 환경에서 바로 실행된다.

    python scripts/check_student_strategy_logic.py
"""
import _bootstrap  # noqa: F401  (vla_pipeline 를 sys.path 에 추가)
import json
from pathlib import Path

from vlapipe.final_project_v2.common import compose_prompt, load_strategy, public_task

PROJECT = Path(__file__).resolve().parent.parent
COMMAND = 'Tidy up the table: put every numbered object in its designated storage zone.'


def main():
    student = load_strategy(PROJECT / 'final_project_v2/student_strategy.py')
    baseline = load_strategy(PROJECT / 'final_project_v2/baseline_strategy.py')
    task = public_task(COMMAND)
    checks = []

    # --- build_prompt 공정성: 초기 호출은 B와 완전히 동일해야 한다 ---
    initial_ctx = {'event': 'initial', 'history': [], 'feedback': None}
    assert student.build_prompt(task, initial_ctx) == baseline.build_prompt(task, initial_ctx), \
        'initial build_prompt differs from baseline'
    assert compose_prompt(task, student, initial_ctx) == compose_prompt(task, baseline, initial_ctx), \
        'initial composed prompt differs from baseline'
    checks.append('initial prompt identical to B (build_prompt and composed)')

    # --- replan 호출에만 C 전용 보강이 붙는다 ---
    replan_ctx = {'event': 'replan', 'history': [], 'feedback': {'event': 'after_action'},
                  'remaining_seconds': 50, 'remaining_calls': 3, 'queued_actions': [],
                  'last_model_response': None}
    base_replan = baseline.build_prompt(task, replan_ctx)
    stud_replan = student.build_prompt(task, replan_ctx)
    assert stud_replan != base_replan, 'replan prompt should differ from B'
    assert student.REPLAN_GUIDANCE in stud_replan, 'REPLAN_GUIDANCE missing on replan'
    assert student.REPLAN_GUIDANCE not in student.build_prompt(task, initial_ctx), \
        'REPLAN_GUIDANCE must NOT appear on initial'
    checks.append('replan prompt appends REPLAN_GUIDANCE; initial does not')

    # --- loop_decision 규칙 ---
    def decide(**kw):
        ctx = {'event': 'after_action', 'remaining_seconds': 50, 'remaining_calls': 3,
               'feedback': {}, 'last_model_response': {}, 'queued_actions': []}
        ctx.update(kw)
        return student.loop_decision(ctx)

    assert decide(remaining_seconds=0) == 'stop', 'time budget exhausted -> stop'
    assert decide(remaining_seconds=-1) == 'stop'
    assert decide(feedback={'model_done': True}) == 'stop', 'model_done -> stop'
    assert decide(last_model_response={'done': True}) == 'stop', 'last done -> stop'
    assert decide(event='observation_tick') == 'continue', 'tick -> continue'
    assert decide(event='after_action',
                  queued_actions=[{'object': 'object_1', 'target_zone': 'right_zone'}]) == 'continue', \
        'after_action with queue -> continue'

    r = decide(event='after_action', queued_actions=[], remaining_calls=2)
    assert isinstance(r, dict) and r.get('decision') == 'replan' and 'feedback' in r, \
        'after_action empty queue + budget -> replan dict with feedback'
    assert decide(event='after_action', queued_actions=[], remaining_calls=0) == 'stop', \
        'after_action empty queue, no calls -> stop'

    for ev in ('plan_exhausted', 'invalid_plan'):
        r = decide(event=ev, remaining_calls=2)
        assert isinstance(r, dict) and r.get('decision') == 'replan', f'{ev} + budget -> replan'
        assert decide(event=ev, remaining_calls=0) == 'stop', f'{ev}, no calls -> stop'
    checks.append('loop_decision rules: budget / done / tick / after_action(queue) / replan / stop')

    # --- 반환값은 항상 실행기가 받아들이는 형태여야 한다 ---
    for r in (decide(event='after_action', queued_actions=[], remaining_calls=2),
              decide(event='plan_exhausted', remaining_calls=2),
              decide(event='invalid_plan', remaining_calls=2)):
        decision = r['decision'] if isinstance(r, dict) else r
        assert decision in ('continue', 'replan', 'stop')
    checks.append('every decision is a valid continue/replan/stop (or dict)')

    # --- _student_feedback: 직렬화 가능 + 12000자 미만 (최악의 긴 history 에서도) ---
    long_history = [{'event': 'after_action', 'instruction': 'move object_%d to right_zone' % (i % 10 + 1),
                     'steps': 240, 'termination': 'fixed_action_horizon'} for i in range(100)]
    for ev in ('after_action', 'plan_exhausted', 'invalid_plan'):
        fb = student._student_feedback({'event': ev, 'history': long_history})
        text = json.dumps(fb, allow_nan=False)
        assert len(text) < 12000, f'feedback too large for {ev}: {len(text)}'
        assert len(fb.get('executed_actions', [])) <= 10, 'feedback should cap executed actions'
    checks.append('student feedback JSON-serializable and < 12000 chars (history-bounded)')

    print(json.dumps({'method': 1, 'checks': checks, 'passed': len(checks),
                      'note': 'Pure-function validation; no model loaded.'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
