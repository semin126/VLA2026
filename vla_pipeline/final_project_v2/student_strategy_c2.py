"""C2 (eager / 매 시행마다 재계획) — 전략 v2.

전략 명명(문제 조사.txt): C0=전체계획 후 재계획, C1=실패검사 후 재계획(reactive),
C2=이 파일. 매 운반 직후 '무조건' 새 이미지로 전 물체를 다시 계획한다.
= 사실상 '매 운반 실패를 즉시 복구'. 실패 추정이 필요 없는 대신 VLM 호출·시간을 많이 쓴다.

반드시 max_calls 를 넉넉히(예: 물체 수 이상) 주어야 재계획 기회가 보장된다.
호출을 C0/C1과 똑같이 묶으면 C2가 불리해지므로, 비교 시 호출당 효율도 함께 본다
(scripts/check_final_project_c2.py 가 성공률과 비용을 같이 보고한다).

build_prompt는 C0/C1과 동일하게 초기 호출은 B와 글자까지 같게 둔다.
"""

from vlapipe.final_project_v2.prompting import build_strategy_prompt

# 재계획 호출에만 덧붙이는 지침. 초기 호출(event=='initial')에는 붙이지 않는다.
REPLAN_GUIDANCE = (
    "\nEager re-planning step (replan after every transport):\n"
    "- Exactly one transport was just executed; it may have failed.\n"
    "- Look at the NEW image FIRST; re-verify every object's current zone.\n"
    "- Plan transports ONLY for objects not yet in their correct storage zone,\n"
    "  in execution order; clear an accessible blocker to its home zone before an\n"
    "  obstructed target.\n"
    "- Prefer the single most useful next transport; the loop will re-observe again.\n"
    "- If the image shows everything sorted, return done=true with an empty plan.\n"
)


def build_prompt(public_task: dict, context: dict) -> str:
    base = build_strategy_prompt(public_task, context)
    if context.get('event') == 'initial':
        return base  # B와 동일해야 한다(공정 비교).
    return base + REPLAN_GUIDANCE


def _student_feedback(context: dict) -> dict:
    """가볍고 경계 있는 실행 메타데이터만 전달(성공여부·oracle 미포함, 한도 12000자)."""
    history = context.get('history') or []
    executed = [
        {'instruction': h.get('instruction'), 'steps': h.get('steps'),
         'termination': h.get('termination'),
         'object': (h.get('action') or {}).get('object')}
        for h in history if h.get('event') == 'after_action'
    ][-10:]
    event = context.get('event')
    if event == 'invalid_plan':
        return {'event': 'invalid_plan',
                'directive': ('Previous response was not valid JSON. Return STRICT JSON '
                              'per the output contract, without markdown or extra text.'),
                'executed_actions': executed}
    return {
        'event': event,
        'directive': ('Re-verify every object from the new image after this single transport. '
                      'Plan only objects not yet correctly sorted; clear blockers first; '
                      'return done=true if all sorted.'),
        'executed_actions': executed,
        'executed_count': len(executed),
    }


def loop_decision(context: dict) -> str | dict:
    remaining_seconds = context.get('remaining_seconds', 0)
    remaining_calls = context.get('remaining_calls', 0)
    event = context.get('event')
    feedback = context.get('feedback') or {}
    last_response = context.get('last_model_response') or {}

    # (1) 시간 소진 -> 종료.
    if remaining_seconds <= 0:
        return 'stop'
    # (2) 모델이 완료를 선언 -> 종료.
    if feedback.get('model_done') or last_response.get('done'):
        return 'stop'
    # (3) 동작 진행 중 관측 틱 -> 계속(재계획은 동작 '종료' 경계에서 한다).
    if event == 'observation_tick':
        return 'continue'
    # (4) 한 동작이 끝났다 -> C2의 핵심: 큐가 남아 있어도 '무조건' 재계획.
    if event == 'after_action':
        if remaining_calls > 0:
            return {'decision': 'replan', 'feedback': _student_feedback(context)}
        return 'stop'
    # (5) 계획 소진/파싱 실패 -> 예산 있으면 재계획.
    if event in ('plan_exhausted', 'invalid_plan'):
        if remaining_calls > 0:
            return {'decision': 'replan', 'feedback': _student_feedback(context)}
        return 'stop'
    return 'continue'
