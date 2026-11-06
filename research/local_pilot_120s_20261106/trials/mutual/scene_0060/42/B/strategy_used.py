"""제공하는 완성 B 기준선. 학생 C의 빈칸과 독립적으로 실행됩니다.

내부 프롬프트는 prompts/starter_strategy.txt에서 B와 C가 함께 읽습니다.
이 파일에 루프 함수가 있는 것은 공통 인터페이스 때문이며 B는 호출하지 않습니다.
"""
from vlapipe.final_project_v2.prompting import build_strategy_prompt


def build_prompt(public_task: dict, context: dict) -> str:
    return build_strategy_prompt(public_task, context)


def loop_decision(context: dict) -> str:
    return 'stop'  # B는 재계획하지 않습니다.
