"""기존 씬 JSON의 id/seed/family/split/state_sha256 목록을 만든다(읽기 전용).

layout은 train/validation/development 분할만 저장한다. test/sft_test 씬의 배치는 읽어서 쓰지 않는다.
사용법: python -B -I -X utf8 inventory_existing.py <vla_pipeline 폴더> <출력 json>
"""
import collections
import json
import sys
from pathlib import Path

PROJECT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
rows, seen = [], set()
skipped = 0
for root in (PROJECT / 'outputs', PROJECT / 'final_project_v2'):
    for path in root.rglob('*.json'):
        try:
            if path.stat().st_size > 400_000:
                continue
            text = path.read_text(encoding='utf-8-sig')
            if 'final_project_scene_v2' not in text:
                continue
            spec = json.loads(text)
        except (OSError, ValueError):
            skipped += 1
            continue
        if not isinstance(spec, dict) or spec.get('schema') != 'final_project_scene_v2' or 'layout' not in spec:
            continue
        key = (spec.get('id'), spec.get('seed'), spec.get('split'), spec.get('state_sha256'))
        if key in seen:
            continue
        seen.add(key)
        row = {'id': spec.get('id'), 'seed': spec.get('seed'), 'family': spec.get('family'),
               'split': spec.get('split'), 'state_sha256': spec.get('state_sha256'),
               'source': path.relative_to(PROJECT).as_posix()}
        if spec.get('split') in ('train', 'validation', 'development'):
            row['layout'] = spec['layout']
        rows.append(row)
OUT.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
by_split = collections.Counter(r['split'] for r in rows)
seeds = sorted({r['seed'] for r in rows if isinstance(r['seed'], int)})
print('씬', len(rows), dict(by_split), '읽기 실패', skipped)
bands = collections.Counter((s // 10000) * 10000 for s in seeds)
print('seed 대역(1만 단위):', sorted(bands.items()))
print('seed 최소/최대:', seeds[0], seeds[-1])
