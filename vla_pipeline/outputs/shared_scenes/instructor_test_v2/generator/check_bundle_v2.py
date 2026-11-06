"""instructor_test_v2 묶음 검사. GPU·렌더링 없음.

1. 모든 씬을 scenes.read_scene 으로 읽고 고정 상태 해시, 분할, 중복, 막힌 큐브 기록을 본다.
2. 부분집합 suite 의 상대 경로가 실제 파일을 가리키는지 본다.
3. --replay <부분집합 이름> 을 주면 그 씬들을 새 환경에 복원해 D 확인을 다시 실행하고 기록과 비교한다.
사용법: python -B -I -X utf8 check_bundle_v2.py <묶음 폴더> <vla_pipeline 폴더> [--replay oneway_core10]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('bundle', type=Path)
parser.add_argument('project', type=Path)
parser.add_argument('--replay', action='append', default=[])
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).parent))
import make_test_scenes as V1
V1.setup(args.project.resolve())
from vlapipe import config as C
from vlapipe.final_project_v2.common import digest
from vlapipe.final_project_v2.scenes import read_scene, validate_layout

manifest = json.loads((args.bundle / 'manifest.json').read_text(encoding='utf-8'))
by_path = {m['path']: m for m in manifest['scenes']}
registry = json.loads((args.bundle / 'suite.json').read_text(encoding='utf-8'))
ids, owners = set(), {'family': set(), 'seed': set(), 'state_sha256': set()}
strata, categories, pair_counts = collections.Counter(), collections.Counter(), collections.Counter()
for row in registry['scenes']:
    spec = read_scene(args.bundle / row['path'])
    meta = by_path[row['path']]
    assert spec['id'] not in ids and spec['split'] == row['split'] == 'test'
    assert spec.get('initial_state') and digest(spec['initial_state']) == spec['state_sha256'] == row['state_sha256']
    assert digest(spec) == meta['scene_sha256'] and (args.bundle / row['preview']).exists()
    access = validate_layout(spec)['accessibility']
    blocked = sorted(n for n in spec['research_targets'] if not access[n]['accessible'])
    assert blocked == meta['blocked_cubes'] and blocked, spec['id']
    assert len(spec['research_targets']) == meta['k']
    for pair in meta['blocked_pairs']:
        assert pair['blocker'] in access[pair['blocked']]['blockers']
        assert access[pair['blocker']]['accessible'] == (meta['blocking'] == 'oneway')
    ids.add(spec['id'])
    for key in owners:
        assert spec[key] not in owners[key], key
        owners[key].add(spec[key])
    strata[(meta['blocking'], meta['k'])] += 1
    categories[(meta['blocking'], spec['category'])] += 1
    pair_counts[(meta['blocking'], len(meta['blocked_pairs']))] += 1
print('씬', len(ids), '모두 읽힘, 고정 상태 해시 일치, 중복 없음, 모든 씬에 막힌 큐브가 있음')
print('층별 개수:', sorted(strata.items()))
print('기하 분류:', sorted(categories.items()))
print('막힌 쌍의 수:', sorted(pair_counts.items()))
for name, picks in manifest['subsets'].items():
    sub = json.loads((args.bundle / name / 'suite.json').read_text(encoding='utf-8'))
    for row, i in zip(sub['scenes'], picks):
        assert (args.bundle / name / row['path']).resolve() == (args.bundle / registry['scenes'][i]['path']).resolve()
        assert (args.bundle / name / row['path']).exists()
    print(name, len(sub['scenes']), '개, k =', sorted(collections.Counter(manifest['scenes'][i]['k'] for i in picks).items()))
for family in ('oneway', 'mutual'):
    scenes = [m for m in manifest['scenes'] if m['blocking'] == family]
    print(family, 'D 확인: 종료', dict(collections.Counter(m['d_check_stop'] for m in scenes)),
          '| 스텝', min(m['d_check_steps'] for m in scenes), '~', max(m['d_check_steps'] for m in scenes),
          '| D 최종 정확도', sorted(collections.Counter(m['d_check_final_acc'] for m in scenes).items()))
gaps = [m['nearest_seen_scene']['max_cube_distance_m'] for m in manifest['scenes'] if m['nearest_seen_scene']]
print('기존 train/validation/development 씬 중 가장 비슷한 것과의 차이: 최소', min(gaps), 'm, 중앙값', sorted(gaps)[len(gaps) // 2], 'm')
rejected = manifest['d_check']['rejected_before_selection']
print('선택 전 탈락', len(rejected), dict(collections.Counter((r['blocking'], r['reason'].split(':')[0]) for r in rejected)))
for name in args.replay:
    same = 0
    for i in manifest['subsets'][name]:
        meta = manifest['scenes'][i]
        check = V1.d_rollout(read_scene(args.bundle / meta['path']), C.FINAL_PROJECT_MAX_STEPS)
        ok = (check['stop_reason'], check['steps'], check['final_acc']) == (meta['d_check_stop'], meta['d_check_steps'], meta['d_check_final_acc'])
        same += ok
        if not ok:
            print('  다름:', meta['path'], check['stop_reason'], check['steps'], check['final_acc'])
    print(f'{name}: 재실행 {len(manifest["subsets"][name])}개 중 {same}개가 기록과 같음 ({sys.executable})')
