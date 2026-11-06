"""처음 보는 테스트 씬 묶음(instructor_test_v1)을 만든다. GPU·렌더링을 쓰지 않는다.

단계
  verify  후보 배치를 만들고(순수 계산), 물리로 고정(freeze)한 뒤, D 제어기가 끝까지 정리하는지 확인한다.
  bundle  확인을 통과한 후보를 층마다 앞에서부터 골라 suite 폴더를 만든다.

씬 설계 (층 20개 = 어질러진 큐브 수 k 1~10 x 배치 방식 2가지, 층마다 5개 -> 100개)
  grid  VLM 학습용 초기 화면과 같은 규칙(중앙 2x5 칸, 칸마다 +-6 mm 흔들림)에 새 seed
  free  같은 범위 안의 연속 무작위 위치(중심 간 거리 7 cm 이상). 칸에 맞지 않고 간격이 더 좁다.
        집을 수 있는 큐브부터 치우는 순서가 존재하는 배치만 쓴다(서로 막는 쌍은 D도 풀지 못한다)
  공통: 고른 k개만 중앙 구역에 두고 나머지는 정리된 자리, 회전 0, split=test, seed 700100~701099.
        기존 train/validation/development 씬과 3 cm 안으로 겹치는 배치는 쓰지 않는다

D 확인은 runner.run_trial 의 D 경로(classical_next + ScriptedSortingPolicy, 운반당 최대 240스텝,
물체당 3회, 6000스텝)를 렌더링 없이 그대로 따라 한 가속 물리 실행이다. 공식 D trial 기록이 아니다.

사용법 (vla_pipeline 밖에서 실행해도 된다)
  python -B -I -X utf8 make_test_scenes.py verify --project <vla_pipeline> --work <작업 폴더> [--shard i/n]
  python -B -I -X utf8 make_test_scenes.py bundle --project <vla_pipeline> --work <작업 폴더> --out <새 폴더> --existing <inventory json>
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

VERSION = 'instructor_test_v1'
SEED_BASE = 700000
STYLES = ('grid', 'free')
KS = tuple(range(1, 11))
PER_STRATUM = 5
CANDIDATES_PER_STRATUM = 8
GRID_CELLS = [(x, y) for y in (.205, .280, .355, .430, .505) for x in (-.075, .075)]
GRID_JITTER = 0.006
FREE_X = (-0.085, 0.085)
FREE_Y = (0.195, 0.515)
FREE_MIN_DISTANCE = 0.07
FREE_RESTARTS = 5000
NOVELTY_MIN_DISTANCE = 0.03
ACTION_STEPS = 240
MAX_ATTEMPTS = 3
POLICY_SEED = 42


def candidate_seed(k, style, index):
    return SEED_BASE + k * 100 + (0 if style == 'grid' else 50) + index


def setup(project):
    """프로젝트 모듈을 불러온다. 숫자 텍스처는 이미 있으므로 다시 쓰지 않게 막는다(공유 assets 보호)."""
    sys.path.insert(0, str(project))
    from vlapipe import numbered
    textures = project / 'assets/number_textures'
    if not all((textures / f'num_{n:02d}.png').exists() for n in range(1, 11)):
        raise SystemExit('숫자 텍스처가 없다. 이 도구는 텍스처를 만들지 않는다.')
    numbered.write_number_textures = lambda *args, **kwargs: []
    numbered.activate(count=10)


def below_normal_priority():
    try:
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00004000)
    except (AttributeError, OSError):
        pass


def removal_order_exists(layout, names):
    """집을 수 있는 큐브부터 차례로 치웠을 때 전부 치워지는가(기하 판정).

    서로의 손가락 자리를 막는 큐브 쌍이 있으면 D 제어기는 어느 쪽도 집지 못하고 멈춘다.
    첫 후보 80개에서 이 판정과 D 물리 실행 결과가 80개 모두 일치했다.
    """
    from vlapipe import accessibility as access, scene
    tidy = scene.tidy_layout()
    pos = {name: list(pose[:2]) for name, pose in layout.items()}
    yaw = {name: pose[2] for name, pose in layout.items()}
    remaining = set(names)
    while remaining:
        ready = [name for name in remaining if access.is_accessible(name, pos, yaw)[0]]
        if not ready:
            return False
        for name in ready:
            pos[name], yaw[name] = list(tidy[name][:2]), 0.0
            remaining.discard(name)
    return True


def make_spec(k, style, index):
    from vlapipe import config as C, scene
    from vlapipe.final_project_v2.scenes import SCHEMA, DEFAULT_COMMAND, validate_layout
    seed = candidate_seed(k, style, index)
    rng = random.Random(seed)
    names = rng.sample(C.OBJECT_NAMES, k)
    layout = {name: [float(v) for v in pose] for name, pose in scene.tidy_layout().items()}
    if style == 'grid':
        cells = GRID_CELLS.copy()
        rng.shuffle(cells)
        for name, (x, y) in zip(names, cells):
            layout[name] = [round(x + rng.uniform(-GRID_JITTER, GRID_JITTER), 4),
                            round(y + rng.uniform(-GRID_JITTER, GRID_JITTER), 4), 0.0]
    else:
        from vlapipe import accessibility as access
        fixed = {name: pose[:2] for name, pose in layout.items() if name not in names}
        for _ in range(FREE_RESTARTS):
            placed = {}
            for name in names:
                for _try in range(400):
                    x, y = round(rng.uniform(*FREE_X), 4), round(rng.uniform(*FREE_Y), 4)
                    if any(math.hypot(x - px, y - py) < FREE_MIN_DISTANCE for px, py in placed.values()):
                        continue
                    # 같은 크기·같은 방향의 큐브끼리는 막는 관계가 대칭이므로 새 큐브만 검사하면 된다.
                    if access.blockers(name, {**fixed, **placed, name: [x, y]}, {}):
                        continue
                    placed[name] = [x, y]
                    break
                else:
                    break
            if len(placed) == k:
                trial = {**layout, **{name: [x, y, 0.0] for name, (x, y) in placed.items()}}
                if removal_order_exists(trial, names):
                    layout = trial
                    break
        else:
            raise ValueError('free 배치를 만들지 못했다')
    ident = f'{VERSION}_k{k:02d}_{style}_{seed}'
    targets = sorted(names, key=lambda n: int(n.split('_')[1]))
    spec = {'schema': SCHEMA, 'id': ident, 'seed': seed, 'family': ident, 'split': 'test', 'category': 'open',
            'object_set': 'number', 'count': 10, 'command': DEFAULT_COMMAND, 'research_targets': targets,
            'layout': layout,
            'note': (f'Held-out instructor test scene: {k} displaced cube(s), placement style {style}. '
                     'Not used for training, debugging or model selection. '
                     'Category is a geometric diagnostic, not a proof that a physical grasp is impossible.')}
    report = validate_layout(spec)
    spec['category'] = report['diagnostic_category']
    return spec, report


def d_rollout(frozen, max_steps):
    """run_trial 의 D 경로를 렌더링 없이 따라 한다."""
    import numpy as np
    from vlapipe import config as C
    from vlapipe.policy import ScriptedSortingPolicy
    from vlapipe.final_project_v2.common import digest
    from vlapipe.final_project_v2.metrics import observe_outcome
    from vlapipe.final_project_v2.runner import classical_next
    from vlapipe.final_project_v2.scenes import environment, state_of
    random.seed(POLICY_SEED)
    np.random.seed(POLICY_SEED)
    with environment(frozen) as env:
        if digest(state_of(env)) != frozen['state_sha256']:
            raise ValueError('복원한 초기 상태가 고정 상태와 다르다')
        outcome = initial = observe_outcome(env)
        wrong = sorted(n for n, row in initial['objects'].items() if not row['correct'])
        stable = 5 if outcome['scene_success'] else 0
        attempts, order = {}, []
        steps = current_steps = 0
        current = teacher = stop = None
        while steps < max_steps:
            if outcome['scene_success'] and stable >= 5:
                stop = 'scene_completed'
                break
            if current is None:
                chosen = classical_next(env, attempts, MAX_ATTEMPTS)
                if chosen is None:
                    stop = 'plan_exhausted'
                    break
                name = chosen['object']
                if attempts.get(name, 0) >= MAX_ATTEMPTS:
                    stop = 'object_attempt_limit'
                    break
                attempts[name] = attempts.get(name, 0) + 1
                current, current_steps = chosen, 0
                teacher = ScriptedSortingPolicy(env)
                teacher.max_attempts = 1
                teacher.set_subtask(chosen)
                order.append(name)
            action = np.asarray(teacher.select_action(), dtype=np.float32)
            if action.shape != (C.ACTION_DIM,) or not np.isfinite(action).all() or np.max(np.abs(action)) > 1.00001:
                raise ValueError('제어기 행동이 범위를 벗어났다')
            env.step(action)
            steps += 1
            current_steps += 1
            outcome = observe_outcome(env)
            stable = stable + 1 if outcome['scene_success'] else 0
            if current_steps >= ACTION_STEPS or teacher.done:
                current = None
        return {'stop_reason': stop or 'step_limit', 'steps': steps, 'simulation_seconds': steps / C.CONTROL_HZ,
                'order': order, 'attempts': attempts, 'initial_wrong': wrong,
                'initial_acc': initial['acc'], 'final_acc': outcome['acc'], 'final_success': bool(outcome['scene_success'])}


def verify(args):
    setup(args.project)
    below_normal_priority()
    from vlapipe import config as C
    from vlapipe.final_project_v2.common import dump
    from vlapipe.final_project_v2.scenes import freeze
    shard, shards = (int(v) for v in args.shard.split('/'))
    out = args.work / 'verified'
    out.mkdir(parents=True, exist_ok=True)
    strata = [(k, style) for k in KS for style in STYLES]
    for number, (k, style) in enumerate(strata):
        if number % shards != shard:
            continue
        for index in range(args.candidates):
            seed = candidate_seed(k, style, index)
            dest = out / f'{seed}.json'
            if dest.exists():
                continue
            began = time.perf_counter()
            row = {'seed': seed, 'k': k, 'style': style, 'index': index, 'accepted': False}
            try:
                spec, report = make_spec(k, style, index)
                row['id'] = spec['id']
                row['diagnostic_category'] = report['diagnostic_category']
                row['layout_warnings'] = report['warnings']
                if report['errors']:
                    row['reject'] = 'layout_error: ' + '; '.join(report['errors'])
                else:
                    frozen = freeze(spec)
                    check = d_rollout(frozen, C.FINAL_PROJECT_MAX_STEPS)
                    row['d_check'] = check
                    if set(check['initial_wrong']) != set(spec['research_targets']):
                        row['reject'] = 'initial_mismatch'
                    elif check['stop_reason'] != 'scene_completed':
                        row['reject'] = 'd_not_completed: ' + check['stop_reason']
                    else:
                        row['accepted'] = True
                        row['spec'] = frozen
            except Exception as exc:  # 후보 하나의 실패가 나머지를 막지 않게 기록만 한다
                row['reject'] = f'exception: {type(exc).__name__}: {exc}'
            row['seconds'] = round(time.perf_counter() - began, 2)
            dump(dest, row)
            print(seed, 'k', k, style, 'OK' if row['accepted'] else row.get('reject'),
                  (row.get('d_check') or {}).get('steps'), row['seconds'], flush=True)


def schematic(spec, size=256):
    """위에서 본 배치 그림. 카메라 화면이 아니라 좌표를 그린 것이다."""
    from PIL import Image, ImageDraw
    from vlapipe import config as C
    image = Image.new('RGB', (size, size), (236, 236, 232))
    draw = ImageDraw.Draw(image)
    x0, x1 = C.WORK_X
    y0, y1 = C.WORK_Y
    scale = (size - 16) / (x1 - x0)
    top = (size - (y1 - y0) * scale) / 2

    def pixel(x, y):  # 로봇은 그림 아래쪽, 왼쪽 구역은 그림 왼쪽
        return 8 + (x - x0) * scale, top + (y1 - y) * scale

    colors = {'left_zone': (198, 214, 236), 'center_zone': (224, 224, 220), 'right_zone': (236, 208, 198)}
    for name, zone in C.ZONES.items():
        a, b = pixel(zone['x'][0], zone['y'][1]), pixel(zone['x'][1], zone['y'][0])
        draw.rectangle([a[0], a[1], b[0], b[1]], fill=colors[name], outline=(150, 150, 150))
    targets = set(spec['research_targets'])
    half = C.NUMBER_FULL[0] / 2
    for name, (x, y, _yaw) in spec['layout'].items():
        a, b = pixel(x - half, y + half), pixel(x + half, y - half)
        draw.rectangle([a[0], a[1], b[0], b[1]], fill=(255, 255, 255) if name in targets else (205, 205, 205),
                       outline=(200, 40, 40) if name in targets else (120, 120, 120), width=2 if name in targets else 1)
        number = name.split('_')[1]
        draw.text(((a[0] + b[0]) / 2 - 3 * len(number), (a[1] + b[1]) / 2 - 5), number, fill=(20, 20, 20))
    return image


def bundle(args):
    setup(args.project)
    from PIL import Image
    from vlapipe.final_project_v2.common import digest, dump
    from vlapipe.final_project_v2.scenes import SCHEMA, validate_layout
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((args.work / 'verified').glob('*.json'))]
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
    for k in KS:
        for style in STYLES:
            group = sorted((r for r in rows if r['k'] == k and r['style'] == style), key=lambda r: r['index'])
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
                raise SystemExit(f'k={k} {style}: 통과 {picked}개뿐이다. --candidates 를 늘려 verify 를 더 실행해야 한다.')
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
        image = schematic(spec)
        image.save(args.out / preview)
        thumbs.append(image)
        close = row['nearest']
        check = row['d_check']
        registry.append({'path': path, 'family': spec['family'], 'split': 'test',
                         'state_sha256': spec['state_sha256'], 'preview': preview})
        manifest.append({'path': path, 'id': spec['id'], 'seed': spec['seed'], 'k': row['k'], 'style': row['style'],
                         'targets': spec['research_targets'], 'category': spec['category'],
                         'state_sha256': spec['state_sha256'], 'scene_sha256': digest(spec),
                         'd_check_steps': check['steps'], 'd_check_seconds': check['simulation_seconds'],
                         'd_check_transports': len(check['order']),
                         'nearest_seen_scene': None if close is None else
                         {'max_cube_distance_m': round(close[0], 4), 'id': close[1], 'split': close[2]}})
    note = ('Held-out instructor test scenes. Selected only by layout rule and a D-state feasibility check; '
            'no learned model was run on them. Run with --split test --final-test after models and strategies are fixed.')
    dump(args.out / 'suite.json', {'schema': SCHEMA, 'scenes': registry, 'source': 'instructor_generated', 'note': note})
    subsets = {'core10': [next(i for i, m in enumerate(manifest) if m['k'] == k and m['style'] == STYLES[(k + 1) % 2])
                          for k in KS],
               'core20': [next(i for i, m in enumerate(manifest) if m['k'] == k and m['style'] == style)
                          for k in KS for style in STYLES]}
    for name, picks in subsets.items():
        dump(args.out / name / 'suite.json',
             {'schema': SCHEMA, 'scenes': [{**registry[i], 'path': '../' + registry[i]['path'],
                                            'preview': '../' + registry[i]['preview']} for i in picks],
              'source': 'instructor_generated', 'note': note + f' Fixed subset {name} of the 100 scenes.'})
    sheet = Image.new('RGB', (PER_STRATUM * 2 * 256, len(KS) * 256), (255, 255, 255))
    for number, image in enumerate(thumbs):
        sheet.paste(image, ((number % 10) * 256, (number // 10) * 256))
    sheet.save(args.out / 'overview.png')
    generator = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dump(args.out / 'manifest.json', {
        'version': VERSION, 'scenes': manifest, 'subsets': subsets, 'generator_sha256': generator,
        'design': {'k': list(KS), 'styles': list(STYLES), 'per_stratum': PER_STRATUM, 'seed_base': SEED_BASE,
                   'grid_cells': GRID_CELLS, 'grid_jitter_m': GRID_JITTER, 'free_x_m': FREE_X, 'free_y_m': FREE_Y,
                   'free_min_distance_m': FREE_MIN_DISTANCE,
                   'novelty_min_distance_m': NOVELTY_MIN_DISTANCE},
        'feasibility_check': {'method': 'D-state classical controller, accelerated physics, no rendering; mirrors run_trial D path',
                              'action_steps': ACTION_STEPS, 'max_attempts': MAX_ATTEMPTS, 'policy_seed': POLICY_SEED,
                              'candidates_checked': len(chosen) + len(rejected),
                              'rejected_before_selection': [{'seed': r['seed'], 'k': r['k'], 'style': r['style'],
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
