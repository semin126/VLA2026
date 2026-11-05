"""Collect privileged-state, physical pick-and-place demonstrations.

Example: python scripts/collect_scripted.py --objects number --count 10 --direction tidy --seed 3
This teacher reads simulator positions. It is NOT a learned VLA or a VLM result.
Only reset initializes object poses; recorded motion uses env.step(actions), without teleporting.
Every invocation creates a fresh directory. Failed attempts are retained separately.
"""
import _bootstrap  # noqa: F401
import argparse
from datetime import datetime
import json
from pathlib import Path
import uuid

import numpy as np

from vlapipe import config as C
from vlapipe.env import SortingEnv
from vlapipe.episode import EpisodeWriter
from vlapipe.planner import RulePlanner, instruction_text, to_subtask
from vlapipe.policy import ScriptedSortingPolicy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--objects', choices=['color', 'number'], default='number')
    ap.add_argument('--count', type=int, default=10, help='Number condition only')
    ap.add_argument('--direction', choices=['tidy', 'mess'], default='tidy')
    ap.add_argument('--task-level', choices=['subtask', 'scene'], default='subtask')
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--episodes', type=int, default=1, help='Consecutive scene seeds')
    ap.add_argument('--max-transfers', type=int, default=0, help='0 means all; scene labels require all')
    ap.add_argument('--max-steps', type=int, default=500)
    ap.add_argument('--image-size', type=int, default=C.IMAGE_SIZE)
    ap.add_argument('--passes', type=int, default=2)
    ap.add_argument('--out-dir', type=Path, default=C.OUTPUT_DIR / 'scripted_collection')
    args = ap.parse_args()
    if args.episodes < 1 or args.max_steps < 1 or args.max_transfers < 0:
        ap.error('episodes/max-steps must be positive and max-transfers nonnegative')
    if args.task_level == 'scene' and args.max_transfers:
        ap.error('scene demonstrations must finish the entire scene; omit --max-transfers')
    if args.objects == 'number':
        from vlapipe import numbered
        numbered.activate(count=args.count)
    elif args.count != 10:
        ap.error('--count applies only to number objects')
    run = args.out_dir / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    manifest = {'teacher': 'privileged_state_scripted_pick_place', 'args': vars(args).copy(),
                'run_dir': str(run), 'trials': [], 'episodes': []}
    manifest['args']['out_dir'] = str(args.out_dir)
    try:
        for seed in range(args.seed, args.seed + args.episodes):
            env = SortingEnv(layout='messy' if args.direction == 'tidy' else 'tidy',
                             seed=seed, image_size=args.image_size)
            try:
                initial_positions = env.object_positions()
                base_meta = {'source': manifest['teacher'], 'seed': seed,
                             'object_set': C.ACTIVE_OBJECT_SET, 'object_count': len(C.OBJECT_NAMES),
                             'object_names': list(C.OBJECT_NAMES),
                             'object_qpos_adr': {k: int(v) for k, v in env._jnt_adr.items()},
                             'initial_positions': initial_positions, 'task_level': args.task_level,
                             'recorded_pose_teleportation': False}
                scene_task = ('Tidy up the table: put every object back into its storage zone.'
                              if args.direction == 'tidy' else 'Move the objects into the center workspace.')
                scene_writer = (EpisodeWriter(env, scene_task, args.direction, meta={**base_meta,
                                'reverse_task': 'Tidy up the table: put every object back into its storage zone.'})
                                if args.task_level == 'scene' else None)
                if scene_writer:
                    scene_writer.record_obs()
                log, transfers, all_ok = [], 0, True
                mess_order = list(C.OBJECT_NAMES)
                np.random.default_rng(seed).shuffle(mess_order)
                # Messing uses collision-aware empty center destinations. Tidying replans each pass.
                for pass_id in range(args.passes if args.direction == 'tidy' else 1):
                    plan = (RulePlanner(env).plan() if args.direction == 'tidy' else
                            [to_subtask(n, 'center_zone', i + 1) for i, n in enumerate(mess_order)])
                    for st in plan:
                        if args.max_transfers and transfers >= args.max_transfers:
                            break
                        # Keep pick-and-place data separate from non-grasp push demonstrations.
                        if st.get('action', 'transport') != 'transport':
                            log.append({'skipped': st, 'reason': 'collector supports grasp transport only'})
                            continue
                        pol = ScriptedSortingPolicy(env)
                        pol.set_subtask(st)
                        reverse_task, _ = instruction_text(st['object'], C.OBJECT_BY_NAME[st['object']]['home_zone'])
                        writer = scene_writer or EpisodeWriter(env, st['instruction'], args.direction,
                            meta={**base_meta, 'subtask': st, 'reverse_task': reverse_task,
                                  'pass': pass_id, 'transfer': transfers})
                        if not scene_writer:
                            writer.record_obs()
                        n, peak = 0, float(env.object_pos(st['object'])[2])
                        while not pol.done and n < args.max_steps:
                            action = pol.select_action()
                            writer.record_action(action)
                            env.step(action)
                            writer.record_obs()
                            peak = max(peak, float(env.object_pos(st['object'])[2]))
                            n += 1
                        # Zone membership alone can count an accidentally pushed object as a grasp.
                        lifted = peak > C.TABLE_TOP_Z + C.half_size(C.OBJECT_BY_NAME[st['object']])[2] + 0.04
                        ok = bool(pol.done and pol.failure is None and lifted and
                                  env.zone_of(st['object']) == st['target_zone'])
                        row = {'seed': seed, 'pass': pass_id, 'transfer': transfers,
                               'object': st['object'], 'target_zone': st['target_zone'],
                               'steps': n, 'success': ok, 'lifted': bool(lifted),
                               'failure': pol.failure, 'timed_out': not pol.done}
                        log.append(row)
                        all_ok = all_ok and ok
                        if not scene_writer:
                            writer.meta.update({'success': ok, 'outcome': row})
                            dest = run / ('accepted' if ok else 'rejected') / f'seed_{seed}_transfer_{transfers:03d}.hdf5'
                            writer.save(dest)
                            row['path'] = str(dest)
                            manifest['episodes'].append(row.copy())
                        print(json.dumps(row, ensure_ascii=False), flush=True)
                        transfers += 1
                    if env.is_tidy() or (args.max_transfers and transfers >= args.max_transfers):
                        break
                final_ok = (env.is_tidy() if args.direction == 'tidy' else
                            all(env.zone_of(n) == 'center_zone' for n in C.OBJECT_NAMES))
                if scene_writer and len(scene_writer):
                    ok = bool(all_ok and final_ok)
                    scene_writer.meta.update({'success': ok, 'outcomes': log})
                    dest = run / ('accepted' if ok else 'rejected') / f'seed_{seed}_scene.hdf5'
                    scene_writer.save(dest)
                    manifest['episodes'].append({'seed': seed, 'success': ok, 'path': str(dest),
                                                 'steps': len(scene_writer)})
                manifest['trials'].append({'seed': seed, 'transfers': transfers,
                    'is_tidy': env.is_tidy(), 'misplaced': env.misplaced_objects(), 'outcomes': log})
            finally:
                env.close()
    finally:
        manifest['accepted'] = sum(bool(r['success']) for r in manifest['episodes'])
        manifest['rejected'] = len(manifest['episodes']) - manifest['accepted']
        (run / 'collection_report.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Report: {run / "collection_report.json"}', flush=True)
    return 0 if manifest['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
