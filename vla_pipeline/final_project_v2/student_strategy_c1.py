"""C1 (reactive / 실패검사 후 재계획) — 전략 v1.

문제 조사.txt의 전략 명명: C0=전체계획 후 재계획(student_strategy.py), C1=이 파일, C2=eager.

C0와의 유일한 차이는 loop_decision 한 곳이다.
  - C0: 한 계획의 큐를 '끝까지' 실행한 뒤에만 재계획한다(실행 도중 실패를 즉시 복구 못 함).
  - C1: 매 운반 직후 '실패로 의심되면' 큐가 남아 있어도 바로 재계획한다(reactive).

핵심 제약(공정성): 런타임에는 oracle(실제 위치/성공/접근성)을 볼 수 없다.
따라서 '실패 의심'은 oracle 없이 얻을 수 있는 메타데이터로만 추정한다(아래 _suspect_failure).
이 추정이 틀릴 수 있다는 점(민감도 한계)이 C1의 본질적 약점이며,
그 정확도 자체를 scripts/check_final_project_c1.py가 사후 로그로 측정한다.

build_prompt는 C0와 동일하게, 초기 호출은 B와 글자까지 같게 두고 재계획에만 지침을 덧붙인다.
입출력 계약은 student_strategy.py / docs/FINAL_PROJECT_V2_STUDENT_GUIDE.md 참조.
"""

from vlapipe.final_project_v2.prompting import build_strategy_prompt

# 재계획(replan) 호출에만 덧붙이는 C 전용 지침. 초기 호출(event=='initial')에는 붙이지 않는다.
REPLAN_GUIDANCE = (
    "\nReactive re-planning step (verify-then-replan feedback loop):\n"
    "- A just-finished transport may have FAILED (bad grasp, drop, wrong zone).\n"
    "- Look at the NEW image FIRST and re-verify every object's current zone.\n"
    "- feedback.student_feedback lists objects this loop suspects are NOT placed;\n"
    "  check those first, but trust the image over the suspicion.\n"
    "- Re-plan transports ONLY for objects not yet in their correct storage zone.\n"
    "- If a target is still blocked, transport the accessible blocker to its own\n"
    "  home storage zone first, then the freed target.\n"
    "- If the image shows everything sorted, return done=true with an empty plan.\n"
)


def build_prompt(public_task: dict, context: dict) -> str:
    base = build_strategy_prompt(public_task, context)
    if context.get('event') == 'initial':
        return base  # B와 동일해야 한다(공정 비교).
    return base + REPLAN_GUIDANCE


def _suspect_failure(context: dict) -> list[str]:
    """oracle 없이 '방금 끝난 운반이 실패했을 가능성'을 추정한다(메타데이터만 사용).

    사용 가능한 블라인드 신호(이미지 분석·정답 없이):
      (가) 재시도 신호: 방금 옮긴 물체를 '이전에도' 옮긴 적이 있다
           -> 지난 재계획이 그 물체를 다시 집어넣었다는 뜻 = 지난번 실패 가능성.
      (나) 모델 자신의 불확실성: 직전 계획에서 그 물체를 accessible=false 로 봤는데도
           계획에 넣었다면, 가림·접근성 때문에 파지가 어긋났을 위험이 높다.
    둘 중 하나라도 걸리면 '의심'으로 본다(재현율 우선, 정밀도는 check_c1이 측정).
    """
    feedback = context.get('feedback') or {}
    action = feedback.get('action') or {}
    obj = action.get('object')
    if not obj:
        return []
    history = context.get('history') or []
    # (가) 과거에 같은 물체를 이미 운반했는가(이번 after_action 이전 기록).
    prior = [h for h in history
             if h.get('event') == 'after_action' and (h.get('action') or {}).get('object') == obj]
    repeated = len(prior) >= 1
    # (나) 직전 모델 응답에서 이 물체를 '접근 불가'로 평가했는가.
    last = context.get('last_model_response') or {}
    risky = any(a.get('object') == obj and a.get('accessible') is False
                for a in (last.get('assessments') or []))
    return [obj] if (repeated or risky) else []


def _student_feedback(context: dict, suspected: list[str]) -> dict:
    """다음 VLM 프롬프트에 넘길 가볍고 경계 있는 분석(실행기 한도 12000자, 성공여부 미포함)."""
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
        'directive': ('Re-verify every object from the new image. Plan transports only for '
                      'objects not yet correctly sorted; clear a blocker to its home zone before '
                      'an obstructed target; return done=true if the image shows all sorted.'),
        'suspected_unplaced': suspected,  # check_c1 이 이 추정을 oracle과 비교한다.
        'executed_actions': executed,
        'executed_count': len(executed),
    }


def loop_decision(context: dict) -> str | dict:
    remaining_seconds = context.get('remaining_seconds', 0)
    remaining_calls = context.get('remaining_calls', 0)
    event = context.get('event')
    feedback = context.get('feedback') or {}
    last_response = context.get('last_model_response') or {}
    queue = context.get('queued_actions') or []

    # (1) 시간 소진 -> 종료.
    if remaining_seconds <= 0:
        return 'stop'
    # (2) 모델이 완료를 선언 -> 신뢰하고 종료.
    if feedback.get('model_done') or last_response.get('done'):
        return 'stop'
    # (3) 동작 진행 중 관측 틱 -> 계속(즉시 재계획은 after_action 경계에서만).
    if event == 'observation_tick':
        return 'continue'
    # (4) 한 동작이 끝났다 — C1의 핵심: 실패가 의심되면 큐가 남아도 즉시 재계획.
    if event == 'after_action':
        suspected = _suspect_failure(context)
        if suspected and remaining_calls > 0:
            return {'decision': 'replan', 'feedback': _student_feedback(context, suspected)}
        if queue:
            return 'continue'  # 의심 없음 + 남은 계획 있음 -> C0처럼 이어 실행.
        if remaining_calls > 0:
            return {'decision': 'replan', 'feedback': _student_feedback(context, suspected)}
        return 'stop'
    # (5) 계획 소진/파싱 실패 -> 예산 있으면 재계획.
    if event in ('plan_exhausted', 'invalid_plan'):
        if remaining_calls > 0:
            return {'decision': 'replan', 'feedback': _student_feedback(context, _suspect_failure(context))}
        return 'stop'
    return 'continue'
