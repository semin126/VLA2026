"""막힌 큐브가 반드시 들어 있는 테스트 씬 묶음(instructor_test_v2)을 만든다. GPU·렌더링을 쓰지 않는다.

모든 씬에 손가락 자리가 막힌 큐브가 1개 이상 있다. 막는 방식에 따라 두 계열로 나눈다.
  oneway  막힌 큐브 옆에 90도 돌아간 큐브가 있다. 돌아간 큐브는 집을 수 있으므로 그것을 먼저 치우면 풀린다.
          D 제어기가 끝까지 정리하는 배치만 쓴다.
  mutual  같은 방향의 두 큐브가 나란히 붙어 서로의 손가락 자리를 막는다(학생 검증 씬 valid_blocked 와 같은 방식).
          D 제어기는 이 쌍을 집지 않는다. 쌍이 아닌 큐브만 D가 모두 옮기고 멈추는 배치만 쓴다.

층: 계열 2 x 어질러진 큐브 수 k (2, 4, 6, 8, 10) x 10개 = 100개. 막힌 쌍의 수는 1~min(3, k/2)에서 무작위.
공통: 나머지 큐브는 정리된 자리, split=test, seed 710200~716099, 기존 train/validation/development 씬과 3 cm 안으로 같은 배치는 제외.

make_test_scenes.py(v1)의 setup, d_rollout 을 그대로 쓴다. 같은 폴더에 두고 실행한다.
  python -B -I -X utf8 make_test_scenes_v2.py verify --project <vla_pipeline> --work <작업 폴더> [--shard i/n]
  python -B -I -X utf8 make_test_scenes_v2.py bundle --project <vla_pipeline> --work <작업 폴더> --out <새 폴더> --existing <inventory json>
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import make_test_scenes as V1

VERSION = 'instructor_test_v2'
SEED_BASE = 710000
FAMILIES = ('oneway', 'mutual')
KS = (2, 4, 6, 8, 10)
PER_STRATUM = 10
CANDIDATES_PER_STRATUM = 16
MAX_PAIRS = 3
AREA_X = (-0.085, 0.085)
AREA_Y = (0.195, 0.515)
MIN_DISTANCE = 0.07                 # 쌍이 아닌 큐브 사이의 중심 간 거리
PAIR_DISTANCE = {'oneway': (0.070, 0.082), 'mutual': (0.056, 0.072)}
PAIR_OFFSET_Y = 0.008
RESTARTS = 5000
NOVELTY_MIN_DISTANCE = 0.03


def candidate_seed(family, k, index):
    return SEED_BASE + (0 if family == 'oneway' else 5000) + k * 100 + index


def blocked_by(layout, names):
    """어질러진 큐브마다, 지금 손가락 자리를 막고 있는 큐브 목록."""
    from vlapipe import accessibility as access
    pos = {name: pose[:2] for name, pose in layout.items()}
    yaw = {name: pose[2] for name, pose in layout.items()}
    return {name: sorted(access.blockers(name, pos, yaw)) for name in names}


def make_spec(family, k, index):
    from vlapipe import config as C, scene
    from vlapipe.final_project_v2.scenes import SCHEMA, DEFAULT_COMMAND, validate_layout
    seed = candidate_seed(family, k, index)
    rng = random.Random(seed)
    names = rng.sample(C.OBJECT_NAMES, k)
    n_pairs = rng.randint(1, min(MAX_PAIRS, k // 2))
    pairs = [(names[2 * i], names[2 * i + 1]) for i in range(n_pairs)]   # (막힌 큐브, 막는 큐브)
    singles = names[2 * n_pairs:]
    tidy = {name: [float(v) for v in pose] for name, pose in scene.tidy_layout().items()}
    partner_yaw = math.pi / 2 if family == 'oneway' else 0.0

    def expected(placed):
        want = {}
        for a, b in pairs:
            if a in placed:
                want[a] = [b]
                want[b] = [] if family == 'oneway' else [a]
        for name in singles:
            if name in placed:
                want[name] = []
        return want

    def acceptable(placed):
        trial = {**{n: p for n, p in tidy.items() if n not in names}, **placed}
        return blocked_by(trial, list(placed)) == expected(placed)

    def far_enough(x, y, placed, skip=()):
        return all(math.hypot(x - p[0], y - p[1]) >= MIN_DISTANCE for n, p in placed.items() if n not in skip)

    for _ in range(RESTARTS):
        placed = {}
        for a, b in pairs:
            for _try in range(400):
                x, y = round(rng.uniform(*AREA_X), 4), round(rng.uniform(*AREA_Y), 4)
                side = rng.choice((-1, 1))
                bx = round(x + side * rng.uniform(*PAIR_DISTANCE[family]), 4)
                by = round(y + rng.uniform(-PAIR_OFFSET_Y, PAIR_OFFSET_Y), 4)
                if not (AREA_X[0] <= bx <= AREA_X[1] and AREA_Y[0] <= by <= AREA_Y[1]):
                    continue
                if not (far_enough(x, y, placed) and far_enough(bx, by, placed)):
                    continue
                trial = {**placed, a: [x, y, 0.0], b: [bx, by, partner_yaw]}
                if acceptable(trial):
                    placed = trial
                    break
            else:
                break
        if len(placed) != 2 * n_pairs:
            continue
        for name in singles:
            for _try in range(400):
                x, y = round(rng.uniform(*AREA_X), 4), round(rng.uniform(*AREA_Y), 4)
                if far_enough(x, y, placed) and acceptable({**placed, name: [x, y, 0.0]}):
                    placed[name] = [x, y, 0.0]
                    break
            else:
                break
        if len(placed) == k:
            break
    else:
        raise ValueError('배치를 만들지 못했다')
    layout = {**tidy, **placed}
    ident = f'{VERSION}_{family}_k{k:02d}_p{n_pairs}_{seed}'
    targets = sorted(names, key=lambda n: int(n.split('_')[1]))
    how = ('each blocked cube has a 90-degree rotated, graspable blocker beside it; moving the blocker first clears it'
           if family == 'oneway' else
           'each blocked pair is two same-orientation cubes side by side that occupy each other\'s finger slots')
    spec = {'schema': SCHEMA, 'id': ident, 'seed': seed, 'family': ident, 'split': 'test', 'category': 'mixed',
            'object_set': 'number', 'count': 10, 'command': DEFAULT_COMMAND, 'research_targets': targets,
            'layout': layout,
            'note': (f'Held-out instructor test scene with blocking ({family}): {k} displaced cube(s), {n_pairs} blocked pair(s); '
                     f'{how}. Not used for training, debugging or model selection. '
                     'Category is a geometric diagnostic, not a proof that a physical grasp is impossible.')}
    report = validate_layout(spec)
    spec['category'] = report['diagnostic_category']
    return spec, report, pairs, singles


def verify(args):
    V1.setup(args.project)
    V1.below_normal_priority()
    from vlapipe import config as C
    from vlapipe.final_project_v2.common import dump
    from vlapipe.final_project_v2.scenes import freeze
    shard, shards = (int(v) for v in args.shard.split('/'))
    out = args.work / 'verified_v2'
    out.mkdir(parents=True, exist_ok=True)
    strata = [(family, k) for family in FAMILIES for k in KS]
    for number, (family, k) in enumerate(strata):
        if number % shards != shard:
            continue
        for index in range(args.candidates):
            seed = candidate_seed(family, k, index)
            dest = out / f'{seed}.json'
            if dest.exists():
                continue
            began = time.perf_counter()
            row = {'seed': seed, 'family': family, 'k': k, 'index': index, 'accepted': False}
            try:
                spec, report, pairs, singles = make_spec(family, k, index)
                row.update(id=spec['id'], pairs=pairs, singles=singles, diagnostic_category=report['diagnostic_category'])
                if report['errors']:
                    row['reject'] = 'layout_error: ' + '; '.join(report['errors'])
                else:
                    frozen = freeze(spec)
                    settled = frozen['settled_diagnostics']['accessibility']
                    blocked = sorted(n for n in spec['research_targets'] if not settled[n]['accessible'])
                    want = sorted(a for a, _ in pairs) if family == 'oneway' else sorted(n for pair in pairs for n in pair)
                    check = V1.d_rollout(frozen, C.FINAL_PROJECT_MAX_STEPS)
                    row['d_check'] = check
                    row['blocked_after_settling'] = blocked
                    if set(check['initial_wrong']) != set(spec['research_targets']):
                        row['reject'] = 'initial_mismatch'
                    elif blocked != want:
                        row['reject'] = 'blocking_changed_after_settling'
                    elif family == 'oneway' and check['stop_reason'] != 'scene_completed':
                        row['reject'] = 'd_not_completed: ' + check['stop_reason']
                    elif family == 'mutual' and not (
                            check['stop_reason'] == 'plan_exhausted' and sorted(check['order']) == sorted(singles)
                            and abs(check['final_acc'] - (10 - 2 * len(pairs)) / 10) < 1e-9):
                        row['reject'] = 'd_did_not_place_exactly_the_free_cubes: ' + check['stop_reason']
                    else:
                        row['accepted'] = True
                        row['spec'] = frozen
            except Exception as exc:  # 후보 하나의 실패가 나머지를 막지 않게 기록만 한다
                row['reject'] = f'exception: {type(exc).__name__}: {exc}'
            row['seconds'] = round(time.perf_counter() - began, 2)
            dump(dest, row)
            print(seed, family, 'k', k, 'OK' if row['accepted'] else row.get('reject'),
                  (row.get('d_check') or {}).get('steps'), row['seconds'], flush=True)


def schematic(spec, pairs, family, size=256):
    """위에서 본 배치 그림(좌표를 그린 것). 주황 = 막힌 큐브, 파랑 테두리 = 90도 돌아간(집을 수 있는) 막는 큐브."""
    from PIL import Image, ImageDraw
    from vlapipe import config as C
    image = Image.new('RGB', (size, size), (236, 236, 232))
    draw = ImageDraw.Draw(image)
    x0, x1 = C.WORK_X
    y0, y1 = C.WORK_Y
    scale = (size - 16) / (x1 - x0)
    top = (size - (y1 - y0) * scale) / 2

    def pixel(x, y):
        return 8 + (x - x0) * scale, top + (y1 - y) * scale

    colors = {'left_zone': (198, 214, 236), 'center_zone': (224, 224, 220), 'right_zone': (236, 208, 198)}
    for name, zone in C.ZONES.items():
        a, b = pixel(zone['x'][0], zone['y'][1]), pixel(zone['x'][1], zone['y'][0])
        draw.rectangle([a[0], a[1], b[0], b[1]], fill=colors[name], outline=(150, 150, 150))
    blocked = {a for a, _ in pairs} | ({b for _, b in pairs} if family == 'mutual' else set())
    rotated = {b for _, b in pairs} if family == 'oneway' else set()
    targets = set(spec['research_targets'])
    half = C.NUMBER_FULL[0] / 2
    for name, (x, y, _yaw) in spec['layout'].items():
        a, b = pixel(x - half, y + half), pixel(x + half, y - half)
        fill = (255, 190, 110) if name in blocked else (255, 255, 255) if name in targets else (205, 205, 205)
        outline = (30, 70, 200) if name in rotated else (200, 40, 40) if name in targets else (120, 120, 120)
        draw.rectangle([a[0], a[1], b[0], b[1]], fill=fill, outline=outline, width=2 if name in targets else 1)
        number = name.split('_')[1]
        draw.text(((a[0] + b[0]) / 2 - 3 * len(number), (a[1] + b[1]) / 2 - 5), number, fill=(20, 20, 20))
    return image


def bundle(args):
    V1.setup(args.project)
    from PIL import Image
    from vlapipe.final_project_v2.common import digest, dump
    from vlapipe.final_project_v2.scenes import SCHEMA, validate_layout
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((args.work / 'verified_v2').glob('*.json'))]
    existing = json.loads(args.existing.read_text(encoding='utf-8'))
    owners = {key: {r[key] for r in existing if r.get(key) is not None} for key in ('id', 'seed', 'family', 'state_sha256')}
    known = [r for r in existing if 'layout' in r]

    def nearest(spec):
        """기존 train/validation/development 씬 중 가장 비슷한 것. 차이는 가장 많이 다른 큐브의 위치 차이로 잰다."""
        best = None
        for row in known:
            if set(row['layout']) != set(spec['layout']):
                continue
            gap = max(math.hypot(a[0] - b[0], a[1] - b[1])
                      for a, b in ((spec['layout'][n], row['layout'][n]) for n in spec['layout']))
            if best is None or gap < best[0]:
                best = (gap, row['id'], row['split'])
        return best

    chosen, rejected, unused = [], [], []
    for family in FAMILIES:
        for k in KS:
            group = sorted((r for r in rows if r['family'] == family and r['k'] == k), key=lambda r: r['index'])
            picked = 0
            for r in group:
                if picked == PER_STRATUM:
                    unused.append(r['seed'])
                elif not r['accepted']:
                    rejected.append(r)
                else:
                    r['nearest'] = nearest(r['spec'])
                    if r['nearest'] is not None and r['nearest'][0] < NOVELTY_MIN_DISTANCE:
                        rejected.append({**r, 'reject': f'too_close_to_seen_scene: {r["nearest"][0]:.4f} m from {r["nearest"][1]}'})
                    else:
                        chosen.append(r)
                        picked += 1
            if picked < PER_STRATUM:
                raise SystemExit(f'{family} k={k}: 통과 {picked}개뿐이다. --candidates 를 늘려 verify 를 더 실행해야 한다.')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'preview').mkdir()
    manifest, registry, thumbs = [], [], []
    for number, row in enumerate(chosen):
        spec = row['spec']
        if digest(spec['initial_state']) != spec['state_sha256'] or spec['split'] != 'test':
            raise SystemExit('고정 상태 해시 또는 분할이 맞지 않는다: ' + spec['id'])
        if validate_layout(spec)['errors']:
            raise SystemExit('배치 오류: ' + spec['id'])
        for key in owners:
            if spec[key] in owners[key]:
                raise SystemExit(f'기존 씬과 {key} 가 겹친다: {spec["id"]}')
            owners[key].add(spec[key])
        path = f'test/scene_{number:04d}.json'
        dump(args.out / path, spec)
        preview = f'preview/scene_{number:04d}.png'
        image = schematic(spec, row['pairs'], row['family'])
        image.save(args.out / preview)
        thumbs.append(image)
        close, check = row['nearest'], row['d_check']
        registry.append({'path': path, 'family': spec['family'], 'split': 'test',
                         'state_sha256': spec['state_sha256'], 'preview': preview})
        manifest.append({'path': path, 'id': spec['id'], 'seed': spec['seed'], 'blocking': row['family'], 'k': row['k'],
                         'blocked_pairs': [{'blocked': a, 'blocker': b} for a, b in row['pairs']],
                         'blocked_cubes': row['blocked_after_settling'], 'free_cubes': row['singles'],
                         'targets': spec['research_targets'], 'category': spec['category'],
                         'state_sha256': spec['state_sha256'], 'scene_sha256': digest(spec),
                         'd_check_stop': check['stop_reason'], 'd_check_steps': check['steps'],
                         'd_check_final_acc': check['final_acc'], 'd_check_order': check['order'],
                         'nearest_seen_scene': None if close is None else
                         {'max_cube_distance_m': round(close[0], 4), 'id': close[1], 'split': close[2]}})
    note = ('Held-out instructor test scenes with blocked cubes. Selected only by layout rule and a D-state check; '
            'no learned model was run on them. Run with --split test --final-test after models and strategies are fixed.')
    dump(args.out / 'suite.json', {'schema': SCHEMA, 'scenes': registry, 'source': 'instructor_generated', 'note': note})
    subsets = {}
    for family in FAMILIES:
        members = [i for i, m in enumerate(manifest) if m['blocking'] == family]
        subsets[family] = members
        subsets[family + '_core10'] = [i for k in KS for i in [j for j in members if manifest[j]['k'] == k][:2]]
    for name, picks in subsets.items():
        dump(args.out / name / 'suite.json',
             {'schema': SCHEMA, 'scenes': [{**registry[i], 'path': '../' + registry[i]['path'],
                                            'preview': '../' + registry[i]['preview']} for i in picks],
              'source': 'instructor_generated', 'note': note + f' Subset {name}.'})
        if name in FAMILIES:
            sheet = Image.new('RGB', (PER_STRATUM * 256, len(KS) * 256), (255, 255, 255))
            for slot, i in enumerate(picks):
                sheet.paste(thumbs[i], ((slot % PER_STRATUM) * 256, (slot // PER_STRATUM) * 256))
            sheet.save(args.out / f'overview_{name}.png')
    dump(args.out / 'manifest.json', {
        'version': VERSION, 'scenes': manifest, 'subsets': subsets,
        'generator_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                             for name in ('make_test_scenes_v2.py', 'make_test_scenes.py')},
        'design': {'blocking': list(FAMILIES), 'k': list(KS), 'per_stratum': PER_STRATUM, 'seed_base': SEED_BASE,
                   'max_pairs': MAX_PAIRS, 'area_x_m': AREA_X, 'area_y_m': AREA_Y, 'min_distance_m': MIN_DISTANCE,
                   'pair_distance_m': PAIR_DISTANCE, 'pair_offset_y_m': PAIR_OFFSET_Y,
                   'novelty_min_distance_m': NOVELTY_MIN_DISTANCE},
        'd_check': {'method': 'D-state classical controller, accelerated physics, no rendering; mirrors run_trial D path',
                    'oneway_rule': 'accepted only if D completes the scene',
                    'mutual_rule': 'accepted only if D places exactly the unpaired cubes and then stops (plan_exhausted)',
                    'candidates_checked': len(chosen) + len(rejected),
                    'rejected_before_selection': [{'seed': r['seed'], 'blocking': r['family'], 'k': r['k'],
                                                   'reason': r.get('reject')} for r in rejected],
                    'verified_but_unused_seeds': unused},
        'existing_scenes_compared': {'total': len(existing), 'with_layout': len(known)}})
    print('씬', len(chosen), '| 선택 전 탈락', len(rejected), '| 쓰지 않은 후보', len(unused))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('verify')
    p.add_argument('--shard', default='0/1')
    p.add_argument('--candidates', type=int, default=CANDIDATES_PER_STRATUM)
    q = sub.add_parser('bundle')
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--existing', type=Path, required=True)
    for item in (p, q):
        item.add_argument('--project', type=Path, required=True)
        item.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    args.project = args.project.resolve()
    {'verify': verify, 'bundle': bundle}[args.command](args)


if __name__ == '__main__':
    main()
