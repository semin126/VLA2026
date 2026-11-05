# Final project ver2.0

학생 씬의 지속적인 Git 공유 경로와 명령은 [씬 push/pull 안내](SCENE_GIT.md)를 따릅니다.

이 버전의 학생 연구는 **VLM 프롬프트의 실행 전 검사·선택**과 **VLM–VLA 피드백 루프**입니다. 기존 `student_design.py`의 정답 좌표 기반 규칙 과제와 다릅니다. 기존 코드·노트북은 보존합니다.

**학생은 [실행·수정 상세 안내](FINAL_PROJECT_V2_STUDENT_GUIDE.md)부터 읽으세요. A·B·D는 완성 코드이며 C의 빈칸과 독립 실행됩니다.** B와 C의 시작 전략은 `final_project_v2/prompts/starter_strategy.txt`에서 함께 읽고, C의 루프는 `final_project_v2/student_strategy.py`에 작성합니다.

## 시작 파일

| 파일 | 용도 |
|---|---|
| [Final_project_v2_0.ipynb](../notebooks/Final_project_v2_0.ipynb) | A~E 개별 실행, 여러 씬 비교, 결과 취합·영상 |
| [Scene_designer_v2_0.ipynb](../notebooks/Scene_designer_v2_0.ipynb) | 씬 설계·검사·미리보기·초기 상태 고정 |
| [Finetuning_data_v2_0.ipynb](../notebooks/Finetuning_data_v2_0.ipynb) | 추가 데이터 생성·검토·학습 연결 |
| [student_strategy.py](../final_project_v2/student_strategy.py) | 학생이 작성하는 프롬프트와 루프 |

Windows 로컬 GPU에서는 [conda 환경·커널 안내](LOCAL_CONDA.md)에 따라 `VLA2026 (conda, Python 3.12, CUDA)`를 선택합니다. 기존 `.venv`도 명시적으로 선택할 수 있습니다. 학생 원격 환경은 VS Code의 Google Colab·Jupyter 확장 → `Select Kernel → Colab` → GPU 서버입니다. 로컬 `.py` 변경은 Colab에 자동 반영되지 않습니다. 기본 체크아웃은 `c-loop-validation`이며, 수정본을 push하거나 Upload to Colab 후 실제 원격 `SOURCE_ROOT`를 선택하세요. commit/push는 실행기가 하지 않습니다.

처음에는 기본 D만 실행해 설치·씬·기록을 확인합니다. 연구용 기본 예산은 300초·6000스텝이며 C가 최초 전체 계획 후 재계획할 여유를 둡니다. `SMOKE_TEST=True`의 12초는 연결 검사이며 연구 성능 예산이 아닙니다. 예비시험으로 공통 예산을 확정하세요. 모델 로딩 후의 과제 시간과 로딩 시간을 나누어 기록합니다.

## 비교 조건

| 조건 | 판단과 실행 |
|---|---|
| A | 전체 과제 명령＋관측 → VLA. 정답으로 다음 물체를 골라주지 않음 |
| B | 초기 이미지＋과제＋학생 전략 → VLM 1회 → 고정 계획을 VLA로 실행 |
| C | B와 같은 초기 입력 → VLA 실행 → 학생의 재관측·피드백·재계획 루프 |
| D-state | 정답 위치 인식＋결정적 접근성·목적지 점유 규칙＋고전 운반 제어 |
| E | 같은 씬을 Windows 로컬에서 조이스틱으로 조작 |

A~E의 **과제 명령**은 같습니다. 전략 프롬프트는 B/C에 적용하며 최초 프롬프트와 이미지는 같습니다. C의 후속 호출에는 학생이 구성한 실행 이력이 추가됩니다. B의 VLA도 현재 관측은 계속 사용합니다. B에서 금지되는 것은 VLM 재호출입니다.

D-state는 고전 CV가 아닙니다. CV 생략을 명시한 특권 정보 기준선입니다. D-cv는 이 버전에 포함하지 않습니다. D의 동률 처리와 선택 순서는 고정되어 있지만 부동소수점·물리 엔진·GPU 차이까지 비트 단위 결정성을 보장하지 않습니다.

A는 전체 정리 명령을, B/C는 부분 운반 명령을 학습한 모델이 필요합니다. 같은 실제 궤적을 전체/부분 라벨로 만들어 두 모델을 학습할 수 있습니다. B/C는 같은 VLA 체크포인트와 VLM·전략을 사용하세요. 학습 업데이트 수·데이터량·언어 범위의 차이를 보고서에 남기세요. A에 부분 운반 모델만 넣은 결과는 연결 검사로 구분합니다.

A에 전달하는 언어는 전체 명령과 완전한 분류 규칙을 담은 짧은 문장입니다. VLA의 언어 길이 제한 때문에 공개 과제 JSON 전체를 액션 모델에 넣지 않습니다. 기본 문장은 사용 모델 토크나이저에서38토큰이며 양쪽 분류 규칙이 보존되는지 확인했습니다. B/C의 VLM에는 같은 과제와 함께 정적 목표 정의·학생 전략·출력 형식을 제공합니다. 원시 입력 문자열 전체가 모델 종류 사이에서 같다는 뜻은 아닙니다.

## 내부 프롬프트와 학생 루프

`build_prompt(public_task, context)`의 기본 연결은 제공합니다. 내부 전략은 `prompts/starter_strategy.txt`에서 편집합니다. 과제·정적 물체 설명·목표 구역·고정 놓기 자리와 출력 JSON 계약은 기반 코드가 붙입니다. 고정 놓기 좌표는 공개 과제 정의이며 물체의 현재 정답 좌표가 아닙니다. VLM에는 현재 상단 카메라 이미지를 전달합니다. 현재 정답 좌표·접근성 검사 결과·남은 물체 정답 목록은 전달하지 않습니다. `TASK_COMMAND`와 `CUSTOM_SCENE`으로 노트북에서 시작 명령과 씬을 지정합니다.

`loop_decision(context)`는 C에서 다음을 반환합니다.

- `continue`: 현재 동작 또는 남은 계획 계속.
- `replan`: 현재 실행을 끊고 새 이미지로 VLM 호출.
- `stop`: 연구 전략에 따른 중단.
- 또는 `{'decision': 위 문자열, 'feedback': JSON값}`으로 학생의 실패 해석을 다음 프롬프트에 전달.

루프 함수는 `after_action`, `observation_tick`, `plan_exhausted`, `invalid_plan` 사건에서 호출됩니다. 실행 지시·스텝 수·이력·남은 예산·이전 VLM 응답과 현재 두 카메라 RGB 이미지를 받습니다. `OBSERVER_STEPS`로 동작 중 관측 간격을 정합니다. 자동으로 정답 성공/실패 라벨을 받아 판단하는 구조가 아닙니다. 성공 판단을 VLM에 맡길지, 이미지 변화로 판단할지, 고정 주기로 재검사할지는 학생 연구입니다.

VLA에는 이미지 2종·상태16·부분 명령을 전달합니다. B/C는 공통 `ACTION_STEPS`가 지나면 부분 지시를 끝냅니다. 정답 상태로 개별 성공을 알려주거나 이미 정리된 물체를 몰래 건너뛰지 않습니다. 전체 정리 완료 시의 공통 평가 종료만 적용합니다. C는 중간에 재계획할 수 있습니다.

잘못된 JSON·없는 ID·지원하지 않는 동작은 기록하고 실행하지 않습니다. 잘못된 목적지처럼 형식은 유효하지만 전략이 틀린 응답은 자동 교정하지 않습니다. 모델이 올바른 형식을 못 내면 프롬프트·모델·학습을 연구해야 합니다.

## 씬과 분할

기본 `scenes` 명령은 개발 3개(open/blocked/mixed), 학습 혼합 4개, validation 혼합 2개, 별도 SFT 시험 혼합 2개, 최종 비교용 혼합 시험 10개를 만듭니다. `--tests`로 마지막 개수를 늘릴 수 있습니다. 씬은 좌표와 로봇·물체의 초기 동역학 상태를 함께 저장합니다. 실행마다 상태 해시를 대조합니다.

모든 정리 대상이 막힌 경우에도 먼저 이동할 수 있는 방해물이 필요합니다. 기본 blocked는 정리 대상 2개와 이미 목표 구역에 있지만 통로를 막는 물체를 구분합니다. 기본 스킬은 운반입니다. 학습되지 않은 밀기·적층 해체를 요구하지 않습니다.

씬 검사: 물체 ID·수, 유한 좌표, 회전한 사각형의 초기 관통, 작업 범위·도달 반경, 정착 높이, 접근성 진단. **실제 IK·접촉·파지와 전체 해결 가능성은 D-state 또는 사람의 실행으로 따로 확인**하세요. 기하 진단과 실제 성공이 다르면 관찰 결과로 남깁니다.

기본 혼합 생성기는 같은 방해물 쌍 구조를 변형한 좁은 분포입니다. 10개가 있다는 사실만으로 넓은 일반화를 주장하지 마세요. 학생이 배치·대상·방해물 관계를 다양화하고 별도 family로 관리해야 합니다. 같은 원본 배치를 조금 바꾼 파생 씬은 같은 family를 유지하세요.

직접 만든 씬은 `split`·`seed`·`family`·고유 ID를 정한 뒤 `freeze`로 저장하고, `register-scenes --scene 첫파일.json --scene 둘째파일.json --out 새_suite`로 등록합니다. 같은 family/seed/초기 상태가 서로 다른 분할에 들어가면 거절합니다. 노트북 `SCENES_ROOT`에 새 suite 위치를 지정하면 같은 A~E 실행기를 쓸 수 있습니다.

개발 씬은 자유롭게 보고 수정합니다. 프롬프트·모델 선택에는 validation을 사용하고 최종 `test`는 확정 후에 실행합니다. 최종 시험 실행은 `--final-test` 또는 노트북 `FINAL_TEST=True`로 명시합니다. 최종 시험을 학습 데이터로 가져오는 경로는 거절합니다.

## 시각적 확인

기본 경로는 노트북의 JSON 좌표 편집 → 초기 상태 고정 → 4방향 PNG 렌더 → D-state 시험입니다. 추가 UI 없이 Windows와 Colab에서 사용할 수 있습니다. 선택 UI는 Gradio 표 편집기이며 별도 `.venv-ui`를 권장합니다. 모델 실행 환경을 바꾸지 않고 UI 의존성을 설치할 수 있습니다.

렌더된 실제 상단·정면·측면·손목 이미지를 함께 확인합니다. 정면 영상은 좌우가 상단 영상과 반대로 보이므로 판단용 주 카메라와 혼동하지 마세요. 모델의 학습 관측 스키마는 바꾸지 않습니다.

E는 기존 pygame 입력 경로를 사용합니다. Enter/녹화 버튼으로 시간을 시작하고 Esc로 종료합니다. Space는 그리퍼입니다. 평가 중 리셋·폐기는 적용하지 않습니다. 조이스틱 버튼은 기존 `gamepad_map.json`을 사용하며 실제 장치의 조작성은 사람의 확인이 필요합니다. 키보드 대체는 `--input-kind keyboard`이며 조이스틱 시연으로 표기하지 않습니다.

## 지표와 시간

- **ACC**: 전체 물체 중 올바른 목표 구역에 안정적으로 놓인 비율.
- **macro-F1**: 실제 목표 구역별 F1의 평균. 미완료·구역 밖·불안정한 물체는 오답.
- **씬 성공률**: 모든 물체가 올바르게 정리된 비율.
- **시간·비용**: 실제 경과 시간, 시뮬레이션 시간, 모델 로딩 시간, 추론 시간, 호출 수, 제어 스텝 수.

정리 상태는 중심점만이 아니라 전체 물체 바닥 면적이 구역 안에 있고 테이블 높이에서 안정·저속이며 손에 들리지 않았는지 확인합니다. 숫자 큐브는 어느 면으로 놓여도 됩니다. 처음부터 올바른 물체 수와 새로 올바르게 정리한 물체 수를 함께 저장합니다. micro-F1은 이 단일 분류 설정에서 ACC와 같으므로 별도 성능처럼 사용하지 않습니다.

제어는 모든 조건에서 실제 시간 기준 최대 20Hz입니다. 추론이 느리면 물리 스텝도 늦어지며 그 비용을 숨기지 않습니다. 동기 GPU 추론·학생 함수는 기한 직후 반환할 수 있습니다. 초과 시간을 별도 기록하고 늦게 나온 계획·액션·적분 스텝은 기한 내 성과에 반영하지 않습니다. GPU 호출을 강제 종료하는 하드 실시간 시스템은 아닙니다.

MP4는 녹화 상태를 오프라인으로 렌더한 **시뮬레이션 시간 영상**입니다. 실제 경과 시간·계획 시점은 영상의 wall 표시와 `events.jsonl`, `trace.npz`를 확인하세요. 영상 렌더링은 물리적 액션 재실행 검증이 아닙니다. 영상 생성·백업은 과제 시간에서 제외합니다.

## 결과 취합

각 조건을 따로 실행하거나 suite로 실행할 수 있습니다. 같은 씬·정책 seed끼리 비교하며 장비·공통 예산·코드가 다른 실행은 다른 cohort로 나눕니다. 결과는 `results.csv`, `summary.json`, `index.html`입니다. 모델 오류도 물리 최종 점수와 오류 개수를 함께 남깁니다. 실행 자체가 시작되지 못한 항목은 suite의 coverage·launch_errors에 남습니다.

장비가 다른 E는 A~D와 실행 시간을 직접 동일 조건의 속도 비교로 해석하지 않습니다. 조작자의 숙련도·연습 횟수·초기 미리보기 시간·시험 순서도 기록하세요. 기본 E는 타이머 시작 전에 씬을 볼 수 있으므로 사람의 준비·관찰 시간이 자동화 모델의 첫 관측과 다르다는 점을 보고합니다.

## 추가 데이터와 학습

VLM 응답은 후보입니다. 이미지·계획을 확인하고 `review.json`에 검토자·수정 응답·accepted/rejected를 기록한 뒤 내보냅니다. 후보 해시와 검토 대상이 일치해야 합니다. family·seed·상태가 분할을 넘나들면 거절합니다. 초기 관측 후보와 실제 C 실행의 피드백 후보를 모두 만들 수 있습니다. VLM 응답만으로 접근성의 물리적 정답이 입증되지는 않습니다.

VLA 데이터는 D-state의 실제 제어 궤적을 수집합니다. 전체 성공 씬은 A용, 실제 들어서 놓은 성공 구간은 B/C용으로 저장합니다. 이미지 2종·상태16·action7·T+1 관측/T 액션을 유지합니다. 실패도 보관합니다. `prepare-actions`는 family·seed·초기 상태의 분할을 확인한 뒤 기존 LeRobot 변환기를 호출합니다. validation/sft_test는 학습에서 제외하며 원래 분할 라벨은 `v2_lineage.json`에 남습니다.

VLM 학습은 기존 `finetune_planner.py`의 답변 토큰 손실을 사용합니다. ver2.0 계획 응답 평가는 `evaluate-vlm`을 사용하세요. v1 계획 파서를 쓰는 `evaluate_planner.py`와 응답 형식이 다릅니다. 계획 라벨 F1을 실제 로봇 정리 F1로 부르지 않습니다.

## 모듈 실행 예시

프로젝트 루트에서 PowerShell로 실행합니다. 출력 폴더는 새 이름을 사용하세요.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py scripts/final_project_v2.py scenes --out outputs/my_scenes --tests 10
& $py scripts/final_project_v2.py trial --condition D --scene outputs/my_scenes/development/development_open_100.json --out outputs/my_D --seconds 120
& $py scripts/final_project_v2.py suite --scenes outputs/my_scenes --split development --conditions D --out outputs/my_suite --seconds 120
& $py scripts/final_project_v2.py aggregate --root outputs/my_suite
& $py scripts/check_final_project_v2.py --out outputs/my_contract_checks
```

공식 UI 참고: [Colab VS Code 사용자 가이드](https://github.com/googlecolab/colab-vscode/wiki/User-Guide), [MuJoCo Python 뷰어](https://mujoco.readthedocs.io/en/stable/python.html), [Gradio 로컬·Colab 공유 방식](https://gradio.app/guides/sharing-your-app).
