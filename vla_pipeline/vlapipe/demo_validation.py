"""Shared provenance checks before action or planner training data preparation."""


def require_training_demo(ep, *, task_level=None, allow_unverified_reversed=False):
    source = str(ep.meta.get('source', ''))
    if ep.meta.get('synthetic_input') or 'fixture' in source:
        raise ValueError(f'{ep.path}: automated input fixtures verify software only; do not train on these as human demonstrations')
    if ep.direction != 'tidy' or ep.meta.get('success') is not True:
        raise ValueError(f'{ep.path}: only verified successful tidy demonstrations are accepted')
    if source == 'human_teleoperation' and ep.meta.get('review_status') != 'accepted':
        raise ValueError(f'{ep.path}: teleoperation needs human review; use review_teleop.py after inspection')
    if ep.meta.get('reversed_from') and not ep.meta.get('physical_replay_verified') and not allow_unverified_reversed:
        raise ValueError(f'{ep.path}: physically unverified reversal; use direct tidy data or explicitly allow exploratory reversal')
    if task_level is not None and ep.meta.get('task_level') != task_level:
        raise ValueError(f'{ep.path}: needs task_level={task_level}, not {ep.meta.get("task_level")}')
    if 'seed' not in ep.meta:
        raise ValueError(f'{ep.path}: original scene seed missing; cannot prevent leakage')
    return int(ep.meta['seed'])
