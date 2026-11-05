"""선택해서 실행하는 C 연결 예제. 최적 전략 또는 학생 연구의 정답이 아닙니다.

노트북 C_STRATEGY를 이 파일로 명시했을 때만 사용됩니다.
기본 student_strategy.py의 빈칸을 자동으로 대신 채우지 않습니다.
"""
from vlapipe.final_project_v2.prompting import build_strategy_prompt


def build_prompt(public_task: dict, context: dict) -> str:
    return build_strategy_prompt(public_task, context)


def loop_decision(context: dict) -> str:
    if context['remaining_seconds'] <= 0 or context['remaining_calls'] <= 0:
        return 'stop'
    if (context.get('feedback') or {}).get('model_done'):
        return 'stop'
    if context['event'] in ('after_action', 'invalid_plan', 'plan_exhausted'):
        return 'replan'
    return 'continue'
