# 3. Colab에서 A, B, C 실험 돌리기

조교가 고정 배포본과 연결 검사를 확인한 뒤, 네 사람(학생 3명과 조교)이 **각자 자기 Colab에서 동시에** 돌립니다. 한 사람당 2시간 남짓 걸립니다.

**이번 설정은 같은 부분 작업 VLA를 A/B/C 모두에 사용하는 비교입니다.** A는 전체 정리 지시를 그대로 받고 VLM을 호출하지 않습니다. B/C는 VLM이 만든 부분 지시를 받습니다. 전체 과제용으로 따로 학습한 A와의 비교가 아니며, A는 학습 때와 다른 지시 형식 때문에 불리할 수 있습니다. 보고서에는 "동일한 부분 작업 학습 VLA에서 VLM·피드백을 추가한 비교"라고 적으세요. 현재 모델은 예비 실험용이며 전체 정리60%가 검증된 상태가 아닙니다.

> 이 절차는 코드와 노트북을 읽고 쓴 것이고, Colab에서 처음부터 끝까지 실제로 돌려 보지는 못했습니다. 조교가 먼저 한 묶음을 돌려 확인한 뒤 나머지가 시작하는 것을 권합니다. 전략 파일 3개가 실제 실행기에서 오류 없이 도는 것은 가짜 모델로 확인했습니다.

## 전체 그림

| | 내용 |
|---|---|
| **누가** | 4명. 각자 씬 5개씩 맡음 |
| **무엇을** | 맡은 씬 5개 × 조건 5개(A, B, C0, C1, C2) = 25번 실행 |
| **어디서** | 각자의 Colab GPU |
| **얼마나** | 준비 15~20분 + 실행 약 2시간 |
| **끝나면** | 결과 폴더를 조교에게 전달 → 조교가 합쳐서 표로 만듦 |

D는 이미 끝났으므로 Colab에서 다시 돌리지 않습니다.

## 누가 어느 묶음을 맡는가

20개 씬을 5개씩 나눴습니다. 묶음마다 큐브 2, 4, 6, 8, 10개인 씬이 하나씩 들어 있습니다.

| 담당 | `PART` 값 | 계열 | 씬 |
|---|---|---|---|
| 조교 | `part1_oneway_a` | oneway | scene_0000, 0010, 0020, 0030, 0040 |
| 학생 1 | `part2_oneway_b` | oneway | scene_0001, 0011, 0021, 0031, 0041 |
| 학생 2 | `part3_mutual_a` | mutual | scene_0050, 0060, 0070, 0080, 0090 |
| 학생 3 | `part4_mutual_b` | mutual | scene_0051, 0061, 0071, 0081, 0091 |

**한 사람이 자기 씬의 모든 조건을 돌립니다.** "나는 A만, 너는 C만"처럼 조건으로 나누면 안 됩니다. Colab마다 GPU 속도가 달라서, 그 차이가 조건의 차이처럼 보이게 됩니다.

## 시작 전에 조교가 준비할 것

1. **씬과 전략 파일을 저장소에 올리기.** `repo_overlay/` 폴더의 내용을 저장소의 같은 위치에 복사해 `c-loop-validation` 브랜치에 올립니다.
   - `vla_pipeline/outputs/shared_scenes/instructor_test_v2/` (씬 묶음)
   - `vla_pipeline/final_project_v2/student_strategy_c0_v2.py`, `c1_v2.py`, `c2_v2.py` (C 전략)
   - 저장소에 올리지 못했다면 `repo_overlay.zip`을 학생에게 직접 나눠 줍니다. 아래 4단계의 "저장소에 없을 때" 칸을 쓰면 됩니다.
2. **모델 확정.** 이번에는 `models/release_manifest_deadline_20261106.json`의 VLM R5 + motion7k를 고정합니다. 가중치는 GitHub Releases에서 받고, 코드·명세는 `instructor/runtime-conda-fixes` 브랜치에서 받습니다. 실험 중 모델을 바꾸면 다른 실험 묶음으로 나누어야 합니다.
3. **GPU 종류 정하기.** 네 사람이 같은 종류를 쓰도록 미리 정합니다. 지금까지 모델 검증은 RTX PRO 6000에서만 했고 T4에서는 하지 않았습니다.

## 학생이 할 일 (한 사람 기준)

### 1단계: 노트북 열고 GPU 확인

1. `vla_pipeline/notebooks/Final_project_v2_0.ipynb`를 Colab에서 엽니다. (VS Code에서 열고 커널을 Colab으로 골라도 됩니다.)
2. 런타임 유형을 GPU로 바꿉니다.
3. 새 칸을 만들어 `!nvidia-smi`를 실행하고 **GPU 이름을 적어 둡니다.** 조교가 정한 종류와 다르면 런타임을 다시 연결합니다.

### 2단계: 설정 칸 고치기 (첫 번째 코드 칸)

```python
TEAM = 'part2'        # 본인 묶음 번호로: part1, part2, part3, part4
REPO_REF = 'instructor/runtime-conda-fixes'
MODEL_SOURCE = 'local'
```

나머지는 그대로 둡니다. `--shared-vla`가 없는 예전 실행 명령과 섞지 마세요.

### 3단계: 환경 준비와 모델 받기

1. 두 번째 코드 칸("코드·실행 환경·저장 위치")을 실행합니다. 저장소를 받고 환경을 설치합니다. Google Drive 연결을 물으면 허용합니다. 결과가 Drive에 백업됩니다.
2. 기본 "배포 모델 받기" 셀 대신 새 코드 칸에서 아래를 실행합니다.

```python
run.run('scripts/download_models.py',
        '--manifest', 'models/release_manifest_deadline_20261106.json',
        '--roles', 'vlm,vla_hier', '--models-dir', MODELS_DIR, '--upgrade')
```

VLM과 공통 VLA 두 개(압축 파일 합계 약1.9GB)를 받고 해시를 검사합니다. A도 같은 `vla_hier`를 사용하므로 기존 `vla_single`을 따로 받을 필요가 없습니다. 기존 파일은 검증 후 백업하며 같은 버전은 다시 받지 않습니다.

**그 아래 칸들(검증 씬 만들기, A~E 비교, 결과 취합)은 실행하지 않습니다.** 이번에는 아래 4단계의 칸으로 대신합니다.

### 4단계: 테스트 실행 칸 붙여 넣기

노트북 맨 아래에 새 코드 칸을 만들고 아래를 붙여 넣습니다. **첫 줄의 `PART`만 본인 것으로 바꿉니다.**

```python
PART = 'part2_oneway_b'     # ← 본인 묶음 이름

# (저장소에 씬이 없을 때만) repo_overlay.zip 을 Colab 파일 창에 올린 뒤 아래 두 줄의 # 을 지우고 실행
# import zipfile
# zipfile.ZipFile('/content/repo_overlay.zip').extractall(PROJECT.parent)

SUITE = PROJECT/'outputs/shared_scenes/instructor_test_v2/colab_split4'/PART
STRATEGY = lambda name: PROJECT/'final_project_v2'/f'student_strategy_{name}_v2.py'
assert (SUITE/'suite.json').is_file(), '씬 묶음이 없습니다. 조교에게 알려 주세요.'
assert all(STRATEGY(n).is_file() for n in ('c0', 'c1', 'c2')), '전략 파일이 없습니다.'

OUT = run.root/('test_shared_'+PART)
COMMON = ['suite', '--scenes', SUITE, '--split', 'test', '--final-test',
          '--seconds', 300, '--max-steps', 6000, '--action-steps', 240, '--observer-steps', 60,
          '--max-calls', 12, '--repeat-seeds', '42',
          '--model-source', MODEL_SOURCE, '--models-dir', MODELS_DIR, '--shared-vla']
JOBS = {'AB':   ['--conditions', 'A,B'],
        'C_c0': ['--conditions', 'C', '--strategy', STRATEGY('c0')],
        'C_c1': ['--conditions', 'C', '--strategy', STRATEGY('c1')],
        'C_c2': ['--conditions', 'C', '--strategy', STRATEGY('c2')]}
RUN_NOW = ['AB', 'C_c0', 'C_c1', 'C_c2']     # 일부만 다시 돌릴 때는 이 목록을 줄입니다

for label in RUN_NOW:
    print('=====', label, '시작')
    try:
        run.run('scripts/final_project_v2.py', *COMMON, *JOBS[label], '--out', OUT/label)
    except subprocess.CalledProcessError:
        print('!!!!!', label, ': 일부 실행에 오류가 있습니다.', OUT/label/'launch_errors.json', '을 확인하세요.')
print('끝. 결과 폴더:', OUT)
```

실행하면 차례로 네 묶음이 돕니다.

| 이름 | 내용 | 실행 수 | 예상 시간 |
|---|---|---|---|
| `AB` | A와 B | 10 | 약 40분 |
| `C_c0` | C, 전략 C0 | 5 | 최대 약 28분 |
| `C_c1` | C, 전략 C1 | 5 | 최대 약 28분 |
| `C_c2` | C, 전략 C2 | 5 | 최대 약 28분 |

실행 한 번은 모델을 불러오는 30초쯤과 최대 300초로 이루어집니다. 화면에 진행 상황이 계속 찍힙니다.

### 5단계: 끝났는지 확인

마지막에 `끝. 결과 폴더: ...`가 나오면 끝입니다. 새 칸에서 아래를 실행해 25개가 다 있는지 봅니다.

```python
import json
from pathlib import Path
rows = []
for f in sorted(OUT.rglob('report.json')):
    r = json.loads(f.read_text(encoding='utf-8'))
    rows.append((f.parts[-5], f.parts[-4], r['condition'], Path((r.get('strategy') or {}).get('path', '')).stem, r['status'], r['stop_reason'],
                 r['initial']['acc'], r['final']['acc'], round(r['summary']['total_wall_seconds'])))
print(len(rows), '개 (25개여야 합니다)')
for row in rows:
    print(*row)
first = json.loads(next(OUT.rglob('report.json')).read_text(encoding='utf-8'))
print('GPU:', first['hardware'].get('gpu'))
```

- `status`가 모두 `completed`여야 합니다. `error`가 있으면 그 줄을 조교에게 알려 주세요.
- 마지막 줄에 찍힌 GPU 이름을 조교에게 알려 주세요.

### 6단계: 결과 전달

결과는 Google Drive의 `VLA2026/Final_project_v2/<TEAM>/<날짜시간>/test_shared_<PART>/`에 백업됩니다. 이 `test_shared_<PART>` 폴더를 통째로 조교에게 공유합니다. Drive를 연결하지 않았다면 Colab 파일 창에서 폴더를 압축해 내려받습니다.

```python
import shutil
shutil.make_archive('/content/test_'+PART, 'zip', OUT)   # /content/test_partN_....zip 이 생깁니다
```

## 지켜야 할 것

- **실험 도중에 전략 파일, 프롬프트, 설정 값을 바꾸지 않습니다.** 결과를 본 뒤에 고치면 이 씬들은 더는 "처음 보는 씬"이 아닙니다.
- **숫자를 바꾸지 않습니다.** 제한 300초, VLM 호출 12회, 운반 240스텝은 네 사람이 같아야 합니다.
- **한 Colab에서 두 개를 동시에 돌리지 않습니다.** 모델이 생각하는 시간도 300초에 들어가므로, 느려지면 결과가 나빠집니다.
- **브라우저 탭을 닫지 않습니다.** Colab은 오래 방치하면 연결을 끊습니다.

## 문제가 생기면

| 증상 | 할 일 |
|---|---|
| `씬 묶음이 없습니다` | 저장소에 아직 안 올라온 것입니다. 조교에게 `repo_overlay.zip`을 받아 4단계의 "저장소에 없을 때" 두 줄을 씁니다 |
| 모델 받기에서 `Existing model is preserved` | 예전 모델이 남아 있는 것입니다. 같은 칸의 `UPGRADE_MODELS` 줄을 `UPGRADE_MODELS = True`로 바꿔 다시 실행합니다 |
| 도중에 연결이 끊김 | 1~3단계를 다시 실행합니다. 새 실행은 새 결과 폴더를 쓰므로, 4단계에서 `RUN_NOW`를 끝나지 않은 묶음만 남기고 실행합니다. 예: `RUN_NOW = ['C_c1', 'C_c2']`. 먼저 끝난 묶음은 Drive의 이전 폴더에 남아 있습니다. 두 폴더를 모두 조교에게 전달합니다 |
| 같은 세션에서 한 묶음만 다시 돌리고 싶음 | 같은 이름의 폴더가 있으면 오류가 납니다. `JOBS['C_c1_retry'] = JOBS['C_c1']`을 추가하고 `RUN_NOW = ['C_c1_retry']`로 실행합니다. 다시 돌린 이유를 조교에게 알립니다 |
| `Final benchmark requires final_test=True` | 4단계의 칸을 그대로 쓰지 않은 것입니다. `--final-test`가 들어 있는지 확인합니다 |
| GPU 메모리 부족 | 런타임을 다시 시작하고 1단계부터 합니다. 다른 노트북이 같은 GPU를 쓰고 있지 않은지 봅니다 |

## 조교가 결과를 합치는 법

네 사람의 `test_shared_<PART>` 폴더를 한 폴더 아래에 모은 뒤 실행합니다. 별도 A 모델을 썼던 이전 실행과 이번 공통 VLA 실행은 섞지 않습니다.

```powershell
python scripts/final_project_v2.py aggregate --root <모은 폴더>
```

`<모은 폴더>/comparison/`에 `results.csv`(실행별), `summary.json`(평균과 같은 씬에서의 차이), `index.html`이 생깁니다. D 결과(`d_results/`)도 같은 폴더에 넣으면 함께 집계됩니다. 다만 D는 조교 PC에서 돌렸으므로 시간 값은 Colab 결과와 직접 비교하지 않습니다.

## 결과가 나오면 볼 것

- **A와 B의 차이**: 같은 부분 작업 학습 VLA에서 전체 과제 지시와 VLM의 부분 지시를 비교합니다. A를 별도로 전체 과제로 학습한 실험은 아닙니다.
- **B와 C의 차이**: 같은 VLA·VLM과 같은 첫 프롬프트에서 루프 정책을 비교합니다. 적은 씬의 차이를 루프의 일반적 우수성으로 단정하지 말고, 실제 재계획·호출 수·실패 원인을 함께 확인합니다.
- **C0, C1, C2의 차이**: 정확도와 함께 VLM 호출 횟수(`planner_calls`)와 걸린 시간을 같이 봅니다.
- **oneway**: VLM이 낸 계획의 순서가 D의 순서(2번 문서의 표)와 같은지 `plan_001.json`에서 확인합니다.
- **mutual**: 끝 정확도가 D의 값(0.8 또는 0.6)을 넘었는지 봅니다. 넘었다면 영상에서 막힌 쌍이 어떻게 떨어졌는지 확인합니다.
- **씬이 5개씩뿐**이므로 한두 씬의 차이로 결론을 내리지 않습니다. 실패한 실행의 영상을 직접 보고 "왜 실패했는가"를 분류하는 것이 숫자보다 쓸모 있습니다.
