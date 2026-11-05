"""Render simulator-labelled VLM examples with scene-level train/val/test splits.

This produces visual planning supervision, not robot action demonstrations.
Example: python scripts/collect_planner_dataset.py --objects number --count 10 \
  --out outputs/planner_sft_demo --train-seeds 0:10 --val-seeds 100:103 --test-seeds 200:203
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from PIL import Image

from vlapipe import config as C, numbered
from vlapipe.env import SortingEnv
from vlapipe.planner import PROMPTS, RulePlanner


def seeds(text):
    """Comma-separated integers and start:stop ranges (stop excluded)."""
    result = []
    for part in text.split(','):
        if ':' in part:
            start, stop = map(int, part.split(':'))
            result.extend(range(start, stop))
        else:
            result.append(int(part))
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError('Use nonempty, unique seeds; ranges exclude stop.')
    return result


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, type=Path, help='New directory; never overwrite.')
    ap.add_argument('--objects', choices=['color', 'number'], default='number')
    ap.add_argument('--count', type=int, default=10, help='Number condition only.')
    ap.add_argument('--layouts', nargs='+', choices=['messy', 'cluttered', 'random', 'tidy'],
                    default=['messy', 'cluttered'])
    ap.add_argument('--prompt-style', choices=['number', 'number_read', 'json'], default=None)
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--train-seeds', type=seeds, default=seeds('0:30'))
    ap.add_argument('--val-seeds', type=seeds, default=seeds('100:105'))
    ap.add_argument('--test-seeds', type=seeds, default=seeds('200:210'))
    ap.add_argument('--command', default='Tidy up the table.')
    return ap.parse_args()


def main():
    args = parse_args()
    style = args.prompt_style or ('number' if args.objects == 'number' else 'json')
    if (args.objects == 'color') != (style == 'json'):
        raise SystemExit('color requires json; number requires number or number_read.')
    if args.image_size < 64:
        raise SystemExit('--image-size must be at least 64.')
    if len(set(args.layouts)) != len(args.layouts):
        raise SystemExit('Duplicate layouts are not allowed.')
    splits = {'train': args.train_seeds, 'validation': args.val_seeds, 'test': args.test_seeds}
    assigned = [seed for group in splits.values() for seed in group]
    if len(set(assigned)) != len(assigned):
        raise SystemExit('Seed splits overlap. All layouts of one seed must stay in one split.')
    if style == 'number_read' and 'random' in args.layouts:
        raise SystemExit('number_read asks only about center objects; random can misplace objects in storage zones.')
    if args.objects == 'number':
        numbered.activate(count=args.count)
    else:
        numbered.deactivate()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'images').mkdir()
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest = {
        'schema_version': 1, 'status': 'generating', 'settings': settings,
        'prompt_style': style, 'source': 'simulator ground truth; oracle supervision',
        'split_unit': 'seed (all layouts of a seed stay together)',
        'warning': 'Transport/recognition labels do not teach clear/push sequencing or physical control. '
                   'No SFT benefit has been established. Test split must never be used for training or prompt selection.',
        'files': {},
    }
    manifest_path = root / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    env = None
    try:
        env = SortingEnv(layout='tidy', image_size=args.image_size, cameras=('agentview',))
        for split, group in splits.items():
            count = 0
            path = root / f'{split}.jsonl'
            with path.open('x', encoding='utf-8') as stream:
                for seed in group:
                    for layout in args.layouts:
                        env.reset(layout=layout, seed=seed)
                        scene_id = f'{args.objects}_{len(C.OBJECT_NAMES)}_{layout}_{seed}'
                        image_path = root / 'images' / f'{scene_id}.png'
                        image = env.render('agentview', args.image_size, args.image_size)
                        Image.fromarray(image).save(image_path)
                        misplaced = env.misplaced_objects()
                        if style == 'number_read':
                            names = [n for n in C.OBJECT_NAMES if env.zone_of(n) == 'center_zone']
                            label = sorted(C.OBJECT_BY_NAME[n]['number'] for n in names)
                        else:
                            label = [{'object': C.OBJECT_BY_NAME[n]['ref_en'],
                                      'target_zone': C.ZONES[C.OBJECT_BY_NAME[n]['home_zone']]['label_en']}
                                     for n in misplaced]
                        answer = json.dumps(label, ensure_ascii=False)
                        prompt = PROMPTS[style].format(command=args.command)
                        record = {
                            'id': scene_id, 'seed': seed, 'layout': layout, 'split': split,
                            'object_set': args.objects, 'object_count': len(C.OBJECT_NAMES),
                            'label_source': 'simulator oracle; not VLM prediction',
                            'image': image_path.relative_to(root).as_posix(),
                            'images': [image_path.relative_to(root).as_posix()],
                            'prompt': prompt, 'completion': answer,
                            'messages': [
                                {'role': 'user', 'content': [
                                    {'type': 'image', 'image': image_path.relative_to(root).as_posix()},
                                    {'type': 'text', 'text': prompt}]},
                                {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]},
                            ],
                            'misplaced_objects': misplaced,
                            'object_positions': env.object_positions(),
                            'oracle_accessibility_plan': RulePlanner(env).plan(args.command),
                        }
                        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                        count += 1
            manifest['files'][split] = {'path': path.name, 'n_examples': count, 'seeds': group}
            print(f'{split}: {count} examples -> {path}')
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if env is not None:
            env.close()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Manifest: {manifest_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
