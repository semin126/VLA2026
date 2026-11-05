"""학생 C의 설계 파일. B 기준선은 baseline_strategy.py로 독립 실행됩니다.

기본 내부 프롬프트는 prompts/starter_strategy.txt에서 편집합니다.
build_prompt의 기본 연결은 제공하며, 필요하면 학생이 함수도 바꿀 수 있습니다.
loop_decision은 학생이 구현합니다. 자세한 입출력: docs/FINAL_PROJECT_V2_STUDENT_GUIDE.md.

build_prompt: public_task와 이미지 기반 실행 문맥을 보고 전략 프롬프트를 작성합니다.
이미지 자체는 기반 코드가 VLM에 전달합니다. 초기 호출은 B/C에서 같습니다.
context = {event, history, feedback, remaining_seconds, remaining_calls,
           queued_actions, last_model_response}

loop_decision: C의 재관측/재계획 정책입니다. 현재 observation의 두 RGB 이미지도 받습니다.
반환은 'continue', 'replan', 'stop' 또는 {'decision': ..., 'feedback': JSON값}.
feedback으로 학생의 영상 분석·실패 해석을 다음 VLM 프롬프트에 전달할 수 있습니다.
after_action / plan_exhausted / invalid_plan / observation_tick에서 호출됩니다.
관측 주기(observer_steps)는 실험 설정으로 정합니다. 추가 이미지 분석이 필요하면
'replan'으로 새 VLM 호출을 요청할 수 있습니다. continue는 현재 지시/계획을 이어갑니다.
feedback에는 정답 성공 여부 대신 실행 스텝·시간·지시·종료 이유가 제공됩니다.
공통 실행기는 시간·스텝·VLM 호출 한도를 항상 강제합니다.

학생 연구 2 구현 요약
---------------------
핵심 설계 원칙: 역할을 세 곳으로 분리한다.
  1) 접근성 판정·타겟/목적지/순서 선택  -> VLM (프롬프트가 지시, plan/assessments로 출력)
  2) 재계획/중단 제어                     -> loop_decision (아래)
  3) 완료·예산 종료                       -> 공통 실행기가 자동 강제
지원 액션은 transport 하나뿐이다(밀기 없음). 방해물 처리는 그 방해물을
자기 home 저장 구역으로 transport 하는 것으로 표현한다.

공정성: B/C의 최초 VLM 프롬프트는 동일해야 한다(A는 상위 VLM을 호출하지 않는다).
현재 정책은 계획 전체를 소진한 뒤 재계획하며, 개별 동작의 즉시 실패 판정은 하지 않는다.
따라서 C 전용 프롬프트 보강은
replan 호출에만 덧붙이고, event=='initial' 에는 절대 붙이지 않는다. C의 이점은
초기 1회 계획(=B) 이후의 재관측→재계획 루프에서만 나온다.
"""

from vlapipe.final_project_v2.prompting import build_strategy_prompt

# replan 호출에만 덧붙이는 C 전용 재계획 지침(자연어). 초기 호출에는 붙이지 않는다.
# 실행기가 context(= history, feedback.student_feedback 포함)를 프롬프트에 JSON으로
# 함께 넣으므로, 여기서는 "그 정보를 어떻게 쓸지"만 지시한다.
REPLAN_GUIDANCE = (
    "\nRe-planning step (hierarchical VLM->VLA feedback loop):\n"
    "- Earlier transports were already executed. Look at the NEW image FIRST.\n"
    "- The execution context includes your history and feedback.student_feedback\n"
    "  describing what was already attempted; use it, do not rely on memory.\n"
    "- A fixed action horizon ending does NOT prove the object was placed; verify\n"
    "  each object's current zone from the image before planning.\n"
    "- Plan transports ONLY for objects that are not yet in their correct zone.\n"
    "  Do not re-transport an object the image shows already correctly sorted.\n"
    "- If a target is still blocked, transport the accessible blocker to its own\n"
    "  home storage zone first, then the freed target.\n"
    "- Keep the plan short and in execution order. If the image shows everything\n"
    "  sorted, return done=true with an empty plan.\n"
)


def build_prompt(public_task: dict, context: dict) -> str:
    base = build_strategy_prompt(public_task, context)
    # 초기 호출은 B와 글자까지 동일해야 한다(공정 비교). 보강하지 않는다.
    if context.get('event') == 'initial':
        return base
    # 재계획 호출에서만 C 전용 재계획 지침을 덧붙인다.
    return base + REPLAN_GUIDANCE


def _student_feedback(context: dict) -> dict:
    """다음 VLM 프롬프트에 넘길 가볍고 경계가 있는 분석.

    이미지 분석은 하지 않는다(연산·메모리 절약). history의 실행 메타데이터만 간결히
    추려 수백 바이트 수준으로 유지한다(실행기 한도 12000자). 성공 여부(정답)는
    제공되지 않으므로 포함하지 않는다.
    """
    event = context.get('event')
    history = context.get('history') or []
    executed = [
        {'instruction': h.get('instruction'),
         'steps': h.get('steps'),
         'termination': h.get('termination')}
        for h in history if h.get('event') == 'after_action'
    ][-10:]
    if event == 'invalid_plan':
        return {
            'event': 'invalid_plan',
            'directive': ('Previous response was not valid JSON. Return STRICT JSON '
                          'per the output contract, without markdown or extra text.'),
            'executed_actions': executed,
        }
    return {
        'event': event,
        'directive': ('Re-verify every object from the new image. Plan transports only '
                      'for objects not yet in their correct storage zone; clear a blocker '
                      'to its home zone before an obstructed target; return done=true with '
                      'an empty plan if the image shows all objects sorted.'),
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

    # (1) 시간 예산 소진 -> 종료(실행기도 강제하지만 방어적으로 먼저 멈춘다).
    if remaining_seconds <= 0:
        return 'stop'

    # (2) 모델이 완료를 선언했으면 신뢰하고 종료한다.
    #     feedback.model_done 은 직후에 덮일 수 있으므로 last_model_response.done 도 본다.
    if feedback.get('model_done') or last_response.get('done'):
        return 'stop'

    # (3) 행동 진행 중 관측 틱 -> 현재 행동을 계속한다(조기 이미지 분석 없음).
    if event == 'observation_tick':
        return 'continue'

    # (4) 한 행동이 끝났다.
    if event == 'after_action':
        # 같은 계획에 남은 transport가 있으면 VLM 호출 없이 이어서 실행한다(호출 절약).
        if queue:
            return 'continue'
        # 계획을 모두 실행했다 -> 예산이 있으면 새 이미지로 재계획(피드백 포함).
        if remaining_calls > 0:
            return {'decision': 'replan', 'feedback': _student_feedback(context)}
        return 'stop'

    # (5) 계획 소진/파싱 실패 -> 예산이 있으면 재계획, 없으면 종료.
    if event in ('plan_exhausted', 'invalid_plan'):
        if remaining_calls > 0:
            return {'decision': 'replan', 'feedback': _student_feedback(context)}
        return 'stop'

    return 'continue'
