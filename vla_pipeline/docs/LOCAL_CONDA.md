# Windows 로컬 GPU와 노트북 커널

학생 브랜치 `c-loop-validation`의 로컬 실행·학습용 환경입니다. 기존 `.venv`와 수집 데이터는 유지합니다. Colab은 `requirements-colab-lock.txt`를 사용하는 별도 환경이며 Windows 잠금 파일을 설치하지 않습니다.

## 설치 및 커널 선택

Miniforge를 설치한 뒤 `vla_pipeline` 폴더에서 PowerShell로 실행합니다.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_conda.ps1
```

자동 발견이 안 되면 `-CondaExe '실제경로\Scripts\conda.exe'`를 추가합니다. 전용 conda 환경 `vla2026`과 `VLA2026 (conda, Python 3.12, CUDA)` Jupyter 커널을 등록합니다. 시스템 Python이나 기존 `.venv`를 덮어쓰지 않습니다. 환경 정의는 `environment-windows.yml`, ML 패키지 잠금 파일은 `requirements-windows-ml-lock.txt`입니다. Git for Windows도 필요합니다.

VS Code에서 v2 노트북을 열고 오른쪽 위 **Select Kernel → Jupyter Kernel/Python Environments → VLA2026 (conda, Python 3.12, CUDA)**를 선택합니다. 목록에 즉시 보이지 않으면 창을 다시 로드합니다. 다른 위치에 conda를 설치했다면 기본 Python 인터프리터도 직접 선택합니다.

```python
import sys, torch
print(sys.executable)
print(torch.__version__, torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

bootstrap이 출력하는 `Kernel Python`과 `Execution Python`이 같은 conda Python인지 확인합니다. `FPV2_PYTHON`을 직접 지정하면 그것이 우선하므로, 이전 `.venv` 경로를 넣었다면 지우고 커널을 재시작합니다. conda 셸 활성화 없이도 등록된 Jupyter 커널로 실행할 수 있습니다.

## 수정된 비교 노트북

- 기본 체크아웃은 `c-loop-validation`; `FPV2_REPO_REF`로 확정 commit을 지정할 수 있습니다. **수정본이 원격에 push되어야 새 Colab clone에 포함됩니다.** 로컬 수정본은 업로드 후 SOURCE_ROOT로 지정할 수도 있습니다. 기존 checkout을 자동 pull/reset하지 않습니다.
- bootstrap은 실제 소스 commit/branch와 학생 전략 hash를 출력·저장합니다. 오래된 Python 소스를 발견하면 수정 브랜치를 받도록 오류를 표시합니다.
- 기본 비교 씬은 학생의 3개 validation 레이아웃입니다. `SCENES_ROOT`를 직접 지정하면 그 suite를 재사용합니다. 씬 생성·편집은 매번 새 폴더에 보존하고 이전 파일을 삭제하지 않습니다.
- validation 전용 suite도 자동으로 첫 씬을 선택합니다. 개발/검증 씬이 전혀 없으면 CUSTOM_SCENE를 명시해야 합니다. 최종 시험에는 기존 FINAL_TEST 절차가 적용됩니다.
- C는 학생이 구현한 **계획 전체 소진 후 재계획** 정책을 유지합니다. 최초 10개 작업을 실행한 뒤에도 기회가 남도록 연구용 기본 예산을 **300초·6000스텝**으로 변경했습니다. VLM 호출은 최대 8회입니다. 실제 추론 속도와 필요한 운반 시간에 맞춰 예비시험 후 모든 조건에 같은 예산을 고정하세요.
- **SMOKE_TEST=True**는 12초·2400스텝의 짧은 연결 검사입니다. C 연구 조건에서 첫 재계획에 도달할 수 없는 예산은 실행 전에 오류를 표시합니다. 이 검사 모드의 수치를 연구 성능으로 사용하지 않습니다.
- 여러 씬을 한꺼번에 돌리는 `RUN_SUITE`는 기본 False입니다. True로 바꾸거나 `FPV2_RUN_SUITE=1`을 지정합니다. 기본 SPLIT은 validation이며, 재실행은 새 suite 출력 폴더를 만듭니다.

## VLA 파인튜닝

`Finetuning_data_v2_0.ipynb`에서 정상 작업 흐름은 다음과 같습니다.

1. 독립된 train/validation/sft_test 씬을 준비합니다. 최종 test를 학습에 넣지 않습니다.
2. `COLLECT_ACTIONS=True`로 실제 성공 궤적을 수집합니다.
3. `PREPARE_VLA=True`로 train-only LeRobot 데이터와 분할 명세를 만듭니다.
4. `TRAIN_VLA=True`로 학습합니다. 기본 2,000스텝도 성능을 보장하는 값은 아니며 validation으로 선택합니다.
5. 뒤의 재로딩 셀이 저장된 전체 추론 폴더를 다시 읽고 실제 시뮬레이터 한 스텝을 확인합니다. 정리 성공률은 별도의 충분한 실행 예산으로 평가합니다.

이미 변환된 데이터로 학습할 때는 학습 셀에서 다음처럼 지정할 수 있습니다.

```python
# 아래는 조교의 기존 train-only 부분 작업 데이터로 연결만 확인하는 예시입니다.
os.environ['FPV2_DATASET_HIER'] = str(PROJECT/'outputs/final_v2_prepare_hier_004/lerobot')
os.environ['FPV2_TRAIN_SYSTEMS'] = 'hier'
os.environ['FPV2_TRAIN_STEPS'] = '2'
os.environ['FPV2_TRAIN_VLA'] = '1'
```

이 경로는 조교 로컬 데이터이며 Git clone만으로 생기지 않습니다. 학생은 실제 전달받은 train-only 데이터 폴더로 바꿔야 합니다. 위 설정 후 `train_models` 셀부터 실행합니다. 또는 셀의 TRAIN_VLA/VLA_SYSTEMS/VLA_STEPS를 직접 편집할 수 있습니다. 새로운 학습마다 별도 출력 폴더를 만들며, 존재하는 모델을 덮어쓰거나 자동 재개하지 않습니다.

학습 후 출력하는 `TRAINED_VLA['single']`은 A용, `TRAINED_VLA['hier']`는 B/C용 경로입니다. 비교 노트북의 VLA_SCENE/VLA_SUBTASK에 전달합니다. 단순히 student_strategy.py를 실행하거나 VLM의 JSON 계획을 생성하는 것은 VLA 학습이 아닙니다.

1~2스텝은 설치·입출력·저장·재로딩을 확인하는 검사입니다. 유용하게 학습한 모델, 60% 성공, 전체 정리 완료를 입증하지 않습니다. Windows에서는 영상 코덱에 의존하지 않는 이미지 기반 LeRobot 데이터셋을 사용합니다.
