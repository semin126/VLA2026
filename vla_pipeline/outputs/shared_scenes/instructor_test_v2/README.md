# instructor_test_v2: 막힌 큐브가 있는 테스트 씬 100개 (2026-11-06)

대상 독자: 실험을 실행하는 사람(조교, Codex 세션)과 결과를 해석하는 학생.

모든 씬에 지금 바로는 집을 수 없는 큐브가 1개 이상 있다. 학습, 디버깅, 모델 선택에 쓰지 않은 씬이고, 만들 때 학습 모델(VLM, VLA)은 실행하지 않았다. 막힌 큐브가 없는 묶음은 `instructor_test_v1`이다.

## 두 계열

"막혔다"는 것은 큐브 양옆의 손가락이 내려올 자리에 다른 큐브가 있다는 뜻이다(`vlapipe/accessibility.py`의 기하 판정).

| 계열 | 막는 방식 | 풀 수 있는가 |
|---|---|---|
| `oneway` (50개) | 막힌 큐브 옆에 90도 돌아간 큐브가 있다. 돌아간 큐브는 손가락 방향이 달라 집을 수 있다 | 막는 큐브를 먼저 치우면 풀린다. 50개 모두 D가 끝까지 정리했다 |
| `mutual` (50개) | 같은 방향의 두 큐브가 5.6~7.2cm 간격으로 나란히 놓여 서로의 손가락 자리를 막는다. 학생 검증 씬 `valid_blocked`와 같은 방식이다 | D는 이 쌍을 집지 않는다. 쌍이 아닌 큐브만 옮기고 멈춘다. 학습 모델이 밀거나 비집어 풀 수 있는지는 확인하지 않았다 |

`mutual`에서 D가 도달한 정확도는 막힌 쌍이 1개면 0.8, 2개면 0.6, 3개면 0.4다. 운반만으로 얻을 수 있는 값이고, 그 이상은 쌍을 실제로 떼어 내야 나온다.

## 구성

| 경로 | 내용 |
|---|---|
| `suite.json` | 전체 100개 (`split: test`) |
| `oneway/suite.json`, `mutual/suite.json` | 계열별 50개 |
| `oneway_core10/suite.json`, `mutual_core10/suite.json` | 고정 부분집합 10개씩. 큐브 수마다 2개 |
| `test/scene_0000.json` ~ `scene_0049.json` | `oneway`. 10개씩 어질러진 큐브 수 2, 4, 6, 8, 10 순서 |
| `test/scene_0050.json` ~ `scene_0099.json` | `mutual`. 같은 순서 |
| `manifest.json` | 씬별 계열, 큐브 수, 막힌 쌍, D 확인 결과, 탈락한 후보와 이유 |
| `overview_oneway.png`, `overview_mutual.png`, `preview/` | 위에서 본 배치 그림(좌표를 그린 것). 주황은 막힌 큐브, 파란 테두리는 90도 돌아간 큐브 |
| `generator/` | 생성·검사 스크립트 사본 |

부분집합은 결과를 보기 전에 규칙으로 정했다. 각 층에서 고른 10개 중 seed가 작은 2개다.

## 씬 설계

- 어질러진 큐브 수 k는 2, 4, 6, 8, 10이고 계열마다 10개씩이다. 나머지 큐브는 정리된 자리에 있다.
- 막힌 쌍의 수는 1~3개(k/2 이하)에서 무작위다. `oneway`는 1쌍 26개, 2쌍 16개, 3쌍 8개이고 `mutual`은 1쌍 26개, 2쌍 12개, 3쌍 12개다.
- 쌍이 아닌 큐브는 모두 바로 집을 수 있는 위치에 있다. 위치 범위는 v1과 같다(x ±8.5cm, y 19.5~51.5cm).
- seed는 710200~716099 대역이다. 기존 씬과 id, seed, family, 고정 상태가 겹치지 않는다. 기존 train/validation/development 씬 중 가장 비슷한 것과도, 가장 많이 다른 큐브의 위치가 7.3cm 이상 다르다.

## 선정 절차와 수치

1. 층(계열 × 큐브 수)마다 후보 16개, 모두 160개를 만들고 물리로 고정했다.
2. 후보마다 D 제어기를 실행했다. 통과 기준은 계열마다 다르다.
   - `oneway`: D가 씬을 끝까지 정리할 것. 80개 중 75개 통과.
   - `mutual`: D가 쌍이 아닌 큐브만 정확히 다 옮기고 멈출 것. 80개 중 75개 통과.
3. 층마다 seed 순서대로 앞에서부터 10개를 골랐다. 그 과정에서 4개가 탈락했다(`oneway` 3개, `mutual` 1개).
4. 고정 뒤에도 막힌 큐브가 의도한 그대로인지 씬마다 확인했다.

D 확인은 `run_trial`의 D 경로를 렌더링 없이 따라 한 가속 물리 실행이다. 두 `core10`은 conda `vla2026` 환경에서 다시 실행해 20개 모두 기록과 같았다.

## 실행 방법

이 폴더를 `vla_pipeline/outputs/shared_scenes/instructor_test_v2/`에 둔 뒤 `vla_pipeline`에서 실행한다. `--scenes`만 바꾸면 v1과 명령이 같다.

```powershell
$py = Join-Path $env:USERPROFILE 'miniforge3\envs\vla2026\python.exe'
$S = 'outputs/shared_scenes/instructor_test_v2/oneway_core10'   # 또는 mutual_core10, oneway, mutual, 전체

& $py scripts/final_project_v2.py suite --scenes $S --split test --final-test --conditions D --no-video --out outputs/<새 폴더>
```

A, B, C 명령과 소요 시간은 `instructor_test_v1/README.md`와 같다.

## 해석할 때 주의

- **D 결과는 선정 기준이다.** `oneway`에서 D는 구성상 100%이고, `mutual`에서 D는 구성상 씬을 끝내지 못한다. D 수치를 성능으로 보고하지 않는다.
- **`mutual`의 씬 완료율은 낮게 나오는 것이 정상이다.** 조건 간 비교는 정확도와, D 도달값(0.8/0.6/0.4)을 넘었는지로 본다.
- **90도 돌아간 큐브는 VLA 학습 자료에 드물다.** 대부분의 학습 시연은 회전 0인 큐브다. `oneway`의 실패가 막힘 때문인지 회전 때문인지는 이 묶음만으로 가를 수 없다. v1(회전 없음, 막힘 없음)과 함께 보면 일부 구분된다.
- **이 묶음으로 모델, 프롬프트, 전략을 고치지 않는다.** 고쳐야 하면 새 seed 대역으로 다시 만든다.
- **한 방향으로 막힌 배치는 회전이 다른 큐브가 있어야만 생긴다.** 같은 크기·같은 방향의 큐브끼리는 막는 관계가 항상 양쪽이다.

## 다시 만들기

`generator/`의 두 스크립트를 같은 폴더에 두고 실행한다.

```
python -B -I -X utf8 generator/make_test_scenes_v2.py verify --project <vla_pipeline> --work <작업 폴더>
python -B -I -X utf8 generator/make_test_scenes_v2.py bundle --project <vla_pipeline> --work <작업 폴더> --out <새 폴더> --existing <기존 씬 목록 json>
python -B -I -X utf8 generator/check_bundle_v2.py <묶음 폴더> <vla_pipeline> --replay oneway_core10 --replay mutual_core10
```

## Colab 4계정으로 나눠 돌릴 때

`colab_split4/` 아래 네 묶음은 두 `core10`의 20개를 5개씩 나눈 것이다. 묶음마다 큐브 수 2, 4, 6, 8, 10이 1개씩 들어 있다.

| 묶음 | 계열 |
|---|---|
| `colab_split4/part1_oneway_a`, `part2_oneway_b` | `oneway` |
| `colab_split4/part3_mutual_a`, `part4_mutual_b` | `mutual` |

한 계정이 맡은 묶음의 모든 조건(A, B, C)을 돌린다. 조건별로 계정을 나누면 기기 차이가 조건 차이에 섞인다. 네 계정이 같은 GPU 종류를 쓰는지 `report.json`의 `hardware`로 확인한다.
