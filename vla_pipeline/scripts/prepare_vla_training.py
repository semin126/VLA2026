"""Split successful demonstrations by original scene seed BEFORE LeRobot conversion.

Use the same --holdout-seeds for single (scene labels) and hierarchical (subtask
labels) collections. Reversed and forward demonstrations of the same seed stay
in the same partition. Nothing is moved, relabelled, or uploaded.
"""
import _bootstrap  # noqa: F401
import argparse
import hashlib
import json
from pathlib import Path

from vlapipe.episode import Episode
from vlapipe.to_lerobot import convert, validate
from vlapipe.demo_validation import require_training_demo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', type=Path, action='append', default=[])
    ap.add_argument('--input-dirs-file', type=Path,
                    help='UTF-8 JSON list of directories; relative entries use this file\'s parent. Avoids Windows command-line limits.')
    ap.add_argument('--system', choices=['single', 'hier'], required=True)
    ap.add_argument('--holdout-seeds', required=True, help='Comma-separated original scene seeds, e.g. 5,6')
    ap.add_argument('--out', type=Path, required=True, help='Fresh output directory')
    ap.add_argument('--repo-id', default='dshs2026/vla_training')
    ap.add_argument('--convert', action='store_true')
    ap.add_argument('--images', action='store_true', help='Image dataset; avoids video codec dependency')
    ap.add_argument('--action-representation',choices=['delta','absolute_position'],default='delta')
    ap.add_argument('--allow-unverified-reversed', action='store_true', help='Exploratory reversal experiment only')
    args = ap.parse_args()
    directories = list(args.input_dir)
    if args.input_dirs_file:
        names = json.loads(args.input_dirs_file.read_text(encoding='utf-8-sig'))
        if not isinstance(names, list) or not all(isinstance(name, str) and name.strip() for name in names):
            ap.error('--input-dirs-file must contain a JSON list of nonempty directory names')
        for name in names:
            directory = Path(name).expanduser()
            directories.append(directory if directory.is_absolute() else args.input_dirs_file.parent/directory)
    if not directories:
        ap.error('Provide --input-dir or --input-dirs-file')
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError('Demonstration directory does not exist: '+str(directory))
    holdout_seeds = {int(s) for s in args.holdout_seeds.split(',')}
    paths = sorted({p.resolve() for d in directories for p in d.rglob('*.hdf5')})
    expected = 'scene' if args.system == 'single' else 'subtask'
    rows = []
    fingerprints = set()
    for path in paths:
        ep = Episode(path,load_images=False)
        seed = require_training_demo(ep, task_level=expected,
                                    allow_unverified_reversed=args.allow_unverified_reversed)
        # Byte-identical duplicates must not receive extra weight in the training set.
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        sha = digest.hexdigest()
        if sha in fingerprints:
            continue
        fingerprints.add(sha)
        rows.append({'path': str(path), 'sha256': sha, 'seed': seed,
                     'partition': 'holdout' if seed in holdout_seeds else 'train',
                     'frames': len(ep), 'task': ep.task, 'meta': ep.meta})
    train = [Path(r['path']) for r in rows if r['partition'] == 'train']
    holdout = [Path(r['path']) for r in rows if r['partition'] == 'holdout']
    if not train or not holdout:
        raise ValueError('Both train and holdout require successful episodes. Collect more original scene seeds.')
    for group in (train, holdout):
        report = validate(group)
        if not report['ok']:
            raise ValueError(report['problems'])
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {'system': args.system, 'holdout_seeds': sorted(holdout_seeds),
                'train_seeds': sorted({r['seed'] for r in rows if r['partition'] == 'train'}),
                'train_episodes': len(train), 'holdout_episodes': len(holdout), 'episodes': rows}
    (args.out / 'split.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.out / 'train_manifest.txt').write_text('\n'.join(map(str, train)) + '\n', encoding='utf-8')
    (args.out / f'holdout_{args.system}.txt').write_text('\n'.join(map(str, holdout)) + '\n', encoding='utf-8')
    if args.convert:
        convert(train, repo_id=args.repo_id, root=args.out / 'lerobot', use_videos=not args.images,
                action_representation=args.action_representation)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'episodes'}, indent=2))
    print('Split manifest:', args.out.resolve() / 'split.json')


if __name__ == '__main__':
    main()
