"""Prepare direct tidy or reversed mess demonstrations without overwriting data.
Reversal screening is a heuristic, not proof of physically valid execution.
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path
from vlapipe import config as C, to_lerobot
from vlapipe.env import SortingEnv
from vlapipe.episode import Episode
from vlapipe.reverse import reverse_file


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-dir', default=str(C.EPISODE_DIR))
    ap.add_argument('--out-dir', default=str(C.REVERSED_DIR))
    ap.add_argument('--input-kind', choices=['mess', 'tidy'], default='mess')
    ap.add_argument('--objects', choices=['color', 'number'], help='Required for legacy data lacking metadata')
    ap.add_argument('--count', type=int, default=10)
    ap.add_argument('--task', default=None, help='Override; default preserves per-subtask reverse_task')
    ap.add_argument('--keep-flagged', action='store_true', help='Exploratory only: retain reverse screening failures')
    ap.add_argument('--allow-unverified-reversed', action='store_true',
                    help='Explicitly permit exploratory training on physically unverified reversed data')
    ap.add_argument('--lerobot', action='store_true')
    ap.add_argument('--repo-id', default='dshs2026/vla_sorting')
    ap.add_argument('--lerobot-root', default=str(C.LEROBOT_DIR))
    return ap.parse_args()


def object_addresses(ep, args):
    supplied = ep.meta.get('object_qpos_adr')
    if supplied:
        addresses = {n: int(a) for n, a in supplied.items()}
        if not addresses or any(a < 0 or a + 7 > ep.qpos.shape[1] for a in addresses.values()):
            raise ValueError('Recorded object_qpos_adr is inconsistent with qpos')
        return addresses
    object_set = ep.meta.get('object_set') or args.objects
    if object_set is None:
        raise ValueError('Legacy recording has no object_set metadata: pass --objects color|number explicitly')
    from vlapipe import numbered
    if object_set == 'number':
        numbered.activate(count=int(ep.meta.get('object_count', args.count)))
    else:
        numbered.deactivate()
    env = SortingEnv(layout='tidy', image_size=64)
    try:
        names = ep.meta.get('object_names')
        if ep.qpos.shape[1] != env.model.nq or (names and set(names) != set(env._jnt_adr)):
            raise ValueError('Recording schema and requested object set/count do not match')
        return {n: int(a) for n, a in env._jnt_adr.items()}
    finally:
        env.close()


def main():
    args = parse_args()
    raw = sorted(Path(args.raw_dir).glob('*.hdf5'))
    if not raw:
        print(f'No HDF5 episodes in {args.raw_dir}. Collect demonstrations first.')
        return 1
    out = Path(args.out_dir)
    targets = [out / (p.name if args.input_kind == 'tidy' else p.stem + '_reversed.hdf5') for p in raw]
    collisions = [str(p) for p in targets + [out / 'build_report.json'] if p.exists()]
    if collisions:
        raise FileExistsError('Choose a NEW --out-dir; refusing to overwrite: ' + ', '.join(collisions))
    if args.lerobot and Path(args.lerobot_root).exists():
        raise FileExistsError('Choose a NEW --lerobot-root')
    if args.lerobot and args.input_kind == 'mess' and not args.allow_unverified_reversed:
        raise ValueError('Reversed actions are not physically verified. Use direct tidy data, or explicitly '
                         'pass --allow-unverified-reversed for the reversal experiment and report this limitation.')
    out.mkdir(parents=True, exist_ok=True)
    written, reports = [], []
    for p, target in zip(raw, targets):
        ep = Episode(p)
        if ep.direction != args.input_kind:
            raise ValueError(f'{p.name}: direction={ep.direction}, expected {args.input_kind}')
        if ep.meta.get('success') is False:
            reports.append({'source': str(p), 'written': None, 'reason': 'failed collection attempt'})
            continue
        if args.input_kind == 'tidy':
            if args.task:
                ep.task = args.task
            ep.meta['prepared_from'] = str(p.resolve())
            ep.save(target)
            dst, rep = target, {'source': str(p), 'written': str(target), 'direction': 'tidy'}
        else:
            dst, rep = reverse_file(p, out, object_adr=object_addresses(ep, args), task=args.task,
                                    strict=not args.keep_flagged)
        reports.append(rep)
        if dst:
            written.append(dst)
        print(json.dumps(rep, ensure_ascii=False), flush=True)
    val = to_lerobot.validate(written)
    if not written:
        val['ok'] = False
        val['problems'].append('No accepted episodes')
    (out / 'build_report.json').write_text(json.dumps({'prepare': reports, 'validate': val},
            ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(val, ensure_ascii=False), flush=True)
    if not val['ok']:
        return 1
    if args.lerobot:
        to_lerobot.convert(written, repo_id=args.repo_id, root=Path(args.lerobot_root))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
