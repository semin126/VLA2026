"""Run the teaching baseline or your explicitly selected research design.

Default A needs no student answers. B/C/D require completed design hooks.
Use a fresh --out. This uses simulator state equally in all conditions.
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from vlapipe.research_lab import run_trial


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=list('ABCD'), default='A')
    ap.add_argument('--case', choices=['open', 'blocked', 'destination_blocked', 'unreachable', 'messy', 'cluttered'], default='open')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--design', choices=['student'])
    group.add_argument('--design-file', type=Path)
    ap.add_argument('--planner', choices=['oracle_transport', 'vlm'], default='oracle_transport')
    ap.add_argument('--vlm-model')
    ap.add_argument('--prompt-style', choices=['number', 'number_read'], default='number')
    ap.add_argument('--policy', choices=['scripted', 'smolvla'], default='scripted')
    ap.add_argument('--checkpoint')
    ap.add_argument('--count', type=int, default=4)
    ap.add_argument('--total-budget', type=int, default=1200)
    ap.add_argument('--max-steps', type=int, default=400)
    ap.add_argument('--max-attempts', type=int, default=2)
    ap.add_argument('--max-plan-calls', type=int, default=8)
    ap.add_argument('--fault', choices=['none', 'timeout_first_action', 'timeout_every_action',
                                      'empty_first_plan', 'invalid_first_plan', 'stale_first_plan'], default='none')
    args = ap.parse_args()
    if args.count < 4:
        ap.error('controlled fixtures require at least 4 number cubes')
    if args.policy == 'smolvla' and not args.checkpoint:
        ap.error('--checkpoint is required for smolvla')
    design = args.design_file
    if args.design == 'student':
        design = Path(__file__).resolve().parents[1] / 'vlapipe' / 'student_design.py'
    report = run_trial(mode=args.mode, case=args.case, seed=args.seed, out=args.out,
                       design_path=design, planner=args.planner, vlm_model=args.vlm_model,
                       prompt_style=args.prompt_style,
                       policy=args.policy, checkpoint=args.checkpoint, count=args.count,
                       total_budget=args.total_budget, max_steps=args.max_steps,
                       max_attempts=args.max_attempts, max_plan_calls=args.max_plan_calls, fault=args.fault)
    import json
    print(json.dumps(report['summary'], indent=2))
    print('Stop:', report['stop_reason'])


if __name__ == '__main__':
    main()
