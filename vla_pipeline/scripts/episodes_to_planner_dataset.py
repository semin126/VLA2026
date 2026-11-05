"""Successful direct-tidy HDF5 -> portable image/JSON planner supervision.

One initial observation per episode; all related episode frames retain the
original scene-seed split. Images are re-rendered at 512px from recorded qpos,
not upscaled from 128px observations. Labels are simulator-derived global sorting
rules, not the operator's free annotation and not accessibility/replanning labels.
"""
import _bootstrap  # noqa: F401
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
from PIL import Image
from vlapipe import config as C, numbered
from vlapipe.env import SortingEnv
from vlapipe.episode import Episode
from vlapipe.demo_validation import require_training_demo
from vlapipe.planner import PROMPTS, RulePlanner
from vlapipe.to_lerobot import validate


def seed_set(text):
    values = [int(x) for x in text.split(',')]
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('Nonempty comma-separated unique integer seeds required')
    return set(values)


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', action='append', required=True, type=Path)
    ap.add_argument('--train-seeds', required=True, type=seed_set)
    ap.add_argument('--val-seeds', required=True, type=seed_set)
    ap.add_argument('--test-seeds', required=True, type=seed_set)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--prompt-style', choices=['number', 'number_read', 'json'], default=None)
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--command', default='Tidy up the table.')
    args = ap.parse_args()
    if args.image_size < 64:
        ap.error('image-size must be >=64')
    groups = {'train': args.train_seeds, 'validation': args.val_seeds, 'test': args.test_seeds}
    if len(set.union(*groups.values())) != sum(map(len, groups.values())):
        ap.error('Original scene-seed splits overlap')
    paths = sorted({p.resolve() for d in args.input_dir for p in d.rglob('*.hdf5')})
    if not paths:
        ap.error('No HDF5 files; point to accepted or reviewed files, not all attempts')
    rows = []
    digests = set()
    for path in paths:
        ep = Episode(path)
        seed = require_training_demo(ep)
        integrity = validate([path])
        if not integrity['ok']:
            raise ValueError(f'{path}: {integrity["problems"]}')
        split = next((key for key, values in groups.items() if seed in values), None)
        if split is None:
            raise ValueError(f'{path}: seed {seed} has no declared split; do not silently discard sources')
        sha = file_sha(path)
        if sha in digests:
            continue
        digests.add(sha)
        object_set = ep.meta.get('object_set')
        if object_set not in ('color', 'number'):
            raise ValueError(f'{path}: object_set metadata is required to reproduce geometry')
        style = args.prompt_style or ('number' if object_set == 'number' else 'json')
        if (style == 'json') != (object_set == 'color'):
            raise ValueError('Use homogeneous color/json or number/number(_read) sources per dataset')
        rows.append({'path': path, 'seed': seed, 'split': split, 'sha': sha,
                     'object_set': object_set, 'count': int(ep.meta['object_count']), 'style': style})
    if any(not any(row['split'] == key for row in rows) for key in groups):
        ap.error('Each split needs at least one source episode')
    if len({(r['object_set'], r['count'], r['style']) for r in rows}) != 1:
        ap.error('One dataset must use one object set/count/prompt style for reproducible comparison')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()
    manifest = {'schema_version': 1, 'status': 'generating',
                'prompt_style': rows[0]['style'], 'files': {},
                'source': 'successful direct tidy episodes; simulator-derived planner labels',
                'sampling': 'one first observation per episode; no future frames',
                'split_unit': 'original scene seed; related episodes and frames stay together',
                'warning': 'Labels do not teach clear/push, accessibility, or replanning. '
                           'Current scene renderer is reconstructed from schema and recorded qpos; '
                           'custom scene geometry/camera edits require a dedicated reconstruction.',
                'sources': []}
    env = None
    try:
        schema = rows[0]
        if schema['object_set'] == 'number':
            numbered.activate(count=schema['count'])
        else:
            numbered.deactivate()
        env = SortingEnv(layout='tidy', image_size=args.image_size, cameras=('agentview',))
        for split in groups:
            n = 0
            with (out / f'{split}.jsonl').open('x', encoding='utf-8') as stream:
                for row in (r for r in rows if r['split'] == split):
                    ep = Episode(row['path'])
                    if ep.qpos.shape[1] != env.model.nq or set(ep.meta.get('object_names', [])) != set(C.OBJECT_NAMES):
                        raise ValueError(f'{ep.path}: recording schema does not match current environment')
                    recorded_adr = {k: int(v) for k, v in ep.meta.get('object_qpos_adr', {}).items()}
                    if recorded_adr != {k: int(v) for k, v in env._jnt_adr.items()}:
                        raise ValueError(f'{ep.path}: recorded object qpos addresses do not match renderer')
                    env.data.qpos[:] = ep.qpos[0]
                    env.data.qvel[:] = ep.qvel[0] if ep.qvel is not None else 0
                    if ep.actuator_ctrl is not None:
                        env.data.ctrl[:] = ep.actuator_ctrl[0]
                    # No env.step here: preserve the recorded instant rather than letting it settle.
                    mujoco.mj_forward(env.model, env.data)
                    env.ctrl.target_pos = ep.target_pos[0].copy()
                    env.ctrl.target_quat = ep.target_quat[0].copy()
                    scene_id = f'episode_{row["seed"]}_{row["sha"][:16]}'
                    image_rel = f'images/{scene_id}.png'
                    Image.fromarray(env.render('agentview', args.image_size, args.image_size)).save(out / image_rel)
                    misplaced = env.misplaced_objects()
                    if row['style'] == 'number_read':
                        center = [name for name in C.OBJECT_NAMES if env.zone_of(name) == 'center_zone']
                        if set(center) != set(misplaced):
                            raise ValueError(f'{ep.path}: number_read center-only question omits misplaced storage-zone objects')
                        label = sorted(C.OBJECT_BY_NAME[name]['number'] for name in center)
                    else:
                        label = [{'object': C.OBJECT_BY_NAME[name]['ref_en'],
                                  'target_zone': C.ZONES[C.OBJECT_BY_NAME[name]['home_zone']]['label_en']}
                                 for name in misplaced]
                    prompt = PROMPTS[row['style']].format(command=args.command)
                    answer = json.dumps(label, ensure_ascii=False)
                    record = {'id': scene_id, 'seed': row['seed'], 'split': split,
                              'layout': 'recorded_episode_start', 'object_set': row['object_set'],
                              'object_count': row['count'], 'image': image_rel, 'images': [image_rel],
                              'prompt': prompt, 'completion': answer,
                              'messages': [{'role': 'user', 'content': [{'type': 'image', 'image': image_rel},
                                            {'type': 'text', 'text': prompt}]},
                                           {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}],
                              'source_episode': str(ep.path), 'source_sha256': row['sha'],
                              'source_collection': ep.meta.get('source'), 'source_task': str(ep.task),
                              'source_observation_index': 0, 'label_source': 'simulator state + fixed sorting rule',
                              'render_source': 'recorded qpos; current same-schema renderer; no action stepping',
                              'misplaced_objects': misplaced, 'object_positions': env.object_positions(),
                              'oracle_accessibility_plan': RulePlanner(env).plan(args.command)}
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                    manifest['sources'].append({'path': str(ep.path), 'sha256': row['sha'],
                                                'seed': row['seed'], 'split': split, 'observation': 0})
                    n += 1
            manifest['files'][split] = {'path': f'{split}.jsonl', 'n_examples': n,
                                        'seeds': sorted(groups[split])}
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['error'] = str(exc)
        raise
    finally:
        if env is not None:
            env.close()
        (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest['files'], indent=2))
    print(out / 'manifest.json')


if __name__ == '__main__':
    main()
