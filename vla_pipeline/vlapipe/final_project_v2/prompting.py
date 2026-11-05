"""Shared, editable internal strategy for the complete B baseline and student C."""
import json
from pathlib import Path

PROMPT_FILE = Path(__file__).resolve().parents[2]/'final_project_v2/prompts/starter_strategy.txt'


def build_strategy_prompt(public_task, context):
    text = PROMPT_FILE.read_text(encoding='utf-8').strip()
    if not text:
        raise ValueError('starter_strategy.txt must contain the VLM strategy')
    # The framework supplies the task and current image separately. No image
    # arrays or privileged scene/evaluation state belong in this JSON context.
    return text + '\nExecution context:\n' + json.dumps(context, ensure_ascii=False, allow_nan=False)
