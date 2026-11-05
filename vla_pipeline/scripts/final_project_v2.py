"""Final project ver2.0 CLI; the three notebooks call these same modules."""
import _bootstrap  # noqa: F401
import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from vlapipe import config as C
from vlapipe.final_project_v2.common import dump


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    scenes = sub.add_parser('scenes')
    scenes.add_argument('--out', type=Path, required=True)
    scenes.add_argument('--tests', type=int, default=10)
    scenes.add_argument('--train', type=int, default=4)
    scenes.add_argument('--validation', type=int, default=2)
    p = sub.add_parser('register-scenes')
    p.add_argument('--scene', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    for name in ('preview', 'freeze', 'edit'):
        p = sub.add_parser(name)
        p.add_argument('--scene', type=Path, required=True)
        p.add_argument('--out', type=Path, required=True)
        if name == 'edit':
            p.add_argument('--share', action='store_true')
    for name in ('trial', 'suite'):
        p = sub.add_parser(name)
        p.add_argument('--out', type=Path, required=True)
        if name == 'trial':
            p.add_argument('--scene', type=Path, required=True)
            p.add_argument('--condition', choices=list('ABCDE'), default='D')
        else:
            p.add_argument('--scenes', type=Path, required=True, help='Suite directory containing suite.json')
            p.add_argument('--split', choices=['development', 'train', 'validation', 'test'], default='development')
            p.add_argument('--conditions', default='D', help='Comma-separated A,B,C,D (E individually, local)')
            p.add_argument('--a-checkpoint', help='Scene-command VLA for A; B/C share --checkpoint')
            p.add_argument('--b-strategy', type=Path, help='Complete B strategy; independent of the student C module')
            p.add_argument('--repeat-seeds', default='42', help='Policy seeds; same for all paired conditions')
        p.add_argument('--seconds', type=float, default=C.FINAL_PROJECT_SECONDS)
        p.add_argument('--max-steps', type=int, default=C.FINAL_PROJECT_MAX_STEPS)
        p.add_argument('--action-steps', type=int, default=240)
        p.add_argument('--observer-steps', type=int, default=60)
        p.add_argument('--max-calls', type=int, default=8)
        p.add_argument('--planner-tokens', type=int, default=1024)
        p.add_argument('--max-attempts', type=int, default=3)
        p.add_argument('--checkpoint')
        p.add_argument('--task-command', help='Shared task command; omitted uses scene JSON command')
        p.add_argument('--vlm-model', default='HuggingFaceTB/SmolVLM2-500M-Video-Instruct')
        p.add_argument('--strategy', type=Path)
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--operator', default='')
        p.add_argument('--input-kind', default='gamepad', choices=['gamepad', 'keyboard', 'keyboard-gamepad'])
        p.add_argument('--gamepad-map', default='gamepad_map.json')
        p.add_argument('--device', default='auto')
        p.add_argument('--no-video', action='store_true')
        p.add_argument('--final-test', action='store_true')
    p = sub.add_parser('aggregate')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--include-diagnostic', action='store_true')
    p = sub.add_parser('candidates')
    p.add_argument('--scenes', type=Path, required=True)
    p.add_argument('--strategy', type=Path, required=True)
    p.add_argument('--model', default='HuggingFaceTB/SmolVLM2-500M-Video-Instruct')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--task-command')
    p = sub.add_parser('feedback-candidates')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('export-reviewed')
    p.add_argument('--candidates', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('collect-actions')
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=120)
    p.add_argument('--max-steps', type=int, default=2400)
    p.add_argument('--task-command')
    p = sub.add_parser('prepare-actions')
    p.add_argument('--collections', type=Path, required=True)
    p.add_argument('--system', choices=['single', 'hier'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repo-id', default='dshs2026/final_project_v2')
    p.add_argument('--convert', action='store_true')
    p = sub.add_parser('evaluate-vlm')
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--model', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='auto')
    return ap


def trial_config(args, condition):
    from vlapipe.final_project_v2.runner import TrialConfig
    cfg = TrialConfig(**{k: getattr(args, k) for k in TrialConfig.__dataclass_fields__ if k != 'command' and hasattr(args, k)})
    cfg.command = args.task_command
    cfg.condition = condition
    cfg.video = not args.no_video
    cfg.strategy = str(args.strategy.resolve()) if args.strategy else None
    if condition == 'B' and args.command == 'suite':
        cfg.strategy = str(args.b_strategy.resolve()) if args.b_strategy else None
    if condition == 'A' and getattr(args, 'a_checkpoint', None):
        cfg.checkpoint = args.a_checkpoint
    cfg.validate()
    return cfg


def main():
    args = parser().parse_args()
    from vlapipe.final_project_v2 import scenes
    if args.command == 'scenes':
        print(json.dumps(scenes.make_suite(args.out, args.tests, args.train, args.validation), indent=2))
    elif args.command == 'register-scenes':
        print(json.dumps(scenes.register_scenes(args.scene, args.out), indent=2))
    elif args.command == 'preview':
        print(scenes.preview(scenes.read_scene(args.scene), args.out))
    elif args.command == 'freeze':
        if args.out.exists():
            raise FileExistsError(args.out)
        dump(args.out, scenes.freeze(scenes.read_scene(args.scene)))
    elif args.command == 'edit':
        from vlapipe.final_project_v2.editor import launch
        launch(args.scene, args.out, share=args.share)
    elif args.command == 'trial':
        from vlapipe.final_project_v2.runner import run_trial
        r = run_trial(args.scene, args.out, trial_config(args, args.condition))
        print(json.dumps({'status': r['status'], 'summary': r['summary'], 'acc': r['final']['acc'],
                          'macro_f1': r['final']['macro_f1'], 'error': r.get('error')}, indent=2))
        return 1 if r['status'] == 'error' else 0
    elif args.command == 'suite':
        from vlapipe.final_project_v2.runner import run_trial
        from vlapipe.final_project_v2.results import aggregate
        conditions = args.conditions.split(',')
        if not conditions or any(c not in 'ABCD' or len(c) != 1 for c in conditions) or len(set(conditions)) != len(conditions):
            raise ValueError('suite supports unique A,B,C,D; run E locally with trial')
        configs = {c: trial_config(args, c) for c in conditions}
        registry = json.loads((args.scenes / 'suite.json').read_text(encoding='utf-8'))
        selected = [r for r in registry['scenes'] if r['split'] == args.split]
        if not selected:
            raise ValueError('No scenes in requested split')
        if args.split == 'test' and not args.final_test:
            raise ValueError('Freeze strategy/models first, then explicitly pass --final-test')
        schedule = []
        for seed in map(int, args.repeat_seeds.split(',')):
            for row in selected:
                order = conditions.copy()
                random.Random(str(seed)+row['path']).shuffle(order)
                for c in order:
                    schedule.append({'scene': row['path'], 'condition': c, 'seed': seed,
                                     'output': f'{Path(row["path"]).stem}/{seed}/{c}'})
        args.out.mkdir(parents=True, exist_ok=False)
        dump(args.out / 'schedule.json', {'trials': schedule, 'configuration': {c: asdict(cfg) for c, cfg in configs.items()}})
        errors = []
        for row in schedule:
            cfg = configs[row['condition']]
            cfg.seed = row['seed']
            try:
                report = run_trial(args.scenes / row['scene'], args.out / row['output'], cfg)
                if report['status'] == 'error':
                    errors.append(row)
            except Exception as exc:
                errors.append({**row, 'error': str(exc)})
                dump(args.out / row['output'] / 'launch_error.json', {'error': str(exc)})
            dump(args.out / 'launch_errors.json', errors)
        summary = aggregate(args.out)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return int(bool(errors))
    elif args.command == 'aggregate':
        from vlapipe.final_project_v2.results import aggregate
        print(json.dumps(aggregate(args.root, args.out, args.include_diagnostic), ensure_ascii=False, indent=2))
    elif args.command == 'candidates':
        from vlapipe.final_project_v2.data import generate_candidates, DATA_SPLITS
        registry = json.loads((args.scenes / 'suite.json').read_text(encoding='utf-8'))
        paths = [args.scenes/r['path'] for r in registry['scenes'] if r['split'] in DATA_SPLITS]
        generate_candidates(paths, args.strategy, args.model, args.out, device=args.device, command=args.task_command)
    elif args.command == 'feedback-candidates':
        from vlapipe.final_project_v2.data import import_feedback_candidates
        import_feedback_candidates(args.root, args.out)
    elif args.command == 'export-reviewed':
        from vlapipe.final_project_v2.data import export_reviewed
        export_reviewed(args.candidates, args.out)
    elif args.command == 'collect-actions':
        from vlapipe.final_project_v2.data import collect_action_data
        print(json.dumps(collect_action_data(args.scene, args.out, args.seconds, args.max_steps, command=args.task_command), indent=2))
    elif args.command == 'prepare-actions':
        from vlapipe.final_project_v2.data import prepare_action_data
        prepare_action_data(args.collections, args.system, args.out, convert=args.convert, repo_id=args.repo_id)
    elif args.command == 'evaluate-vlm':
        from vlapipe.final_project_v2.data import evaluate_vlm
        print(json.dumps(evaluate_vlm(args.dataset, args.split, args.model, args.out, device=args.device), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
