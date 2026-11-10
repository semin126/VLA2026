"""Directional push interface example; not a validated research strategy.

Select explicitly with --strategy final_project_v2/student_strategy_push_example.py.
Use a validated push-capable checkpoint only after instructor approval.
The default transport strategies/prompts are preserved for comparison.
Students decide when to push, which direction is clear and when to replan.
"""
import json


def build_prompt(public_task, context):
    return (
        'Use the current image and execution history to choose transport or push. '
        'Judge grasp clearance and destination occupancy from visual evidence. '
        'If choosing a push, explain which direction is clear and why moving that object helps. '
        'Use only toward_robot, away, left or right; the push distance is fixed at 7 cm. '
        'A push is not proof of placement: inspect the new image before choosing the next action. '
        'Avoid exhausting the independent per-object limits: transport 3, push 2. '
        'Do not infer success from a completed action horizon. '
        'Return the framework JSON contract.\nExecution context:\n'
        + json.dumps(context, ensure_ascii=False, allow_nan=False)
    )


def loop_decision(context):
    # Simple wiring example: replan after each action, not a tested optimum.
    if context.get('event') == 'observation_tick':
        return 'continue'
    if context.get('remaining_seconds', 0) <= 0 or context.get('remaining_calls', 0) <= 0:
        return 'continue' if context.get('queued_actions') else 'stop'
    if (context.get('last_model_response') or {}).get('done') is True:
        return 'stop'
    return 'replan'
