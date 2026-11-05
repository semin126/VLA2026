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
"""

from vlapipe.final_project_v2.prompting import build_strategy_prompt


def build_prompt(public_task: dict, context: dict) -> str:
    # 기본은 B와 같은 내부 프롬프트입니다. TXT를 바꾸면 B와 C에 함께 적용됩니다.
    # 과제·이미지·JSON 출력 계약은 공통 코드가 붙이므로 여기서 복제하지 않습니다.
    return build_strategy_prompt(public_task, context)


def loop_decision(context: dict) -> str | dict:
    # TODO: 언제 현재 실행을 이어가고, 새 관측으로 재계획하고, 중단할지 정하세요.
    raise NotImplementedError('학생 연구 2: VLM–VLA 피드백 루프를 설계하세요.')
