# Final_project 시작 안내

실행 파일은 [notebooks/Final_project.ipynb](../notebooks/Final_project.ipynb)입니다. 연구에서 스스로 결정할 부분은 [여섯 가지 판단 지점](RESEARCH_DECISIONS.md)에 정리했습니다. 설치와 파일 형식 문제를 연구의 핵심 과제로 삼을 필요는 없습니다.

## 어디에서 무엇이 실행될까?

| 위치 | 역할 | 대표 경로 |
|---|---|---|
| 여러분의 Windows·VS Code | 노트북 편집, Git commit/push/pull, 로컬 사람 시연 입력 | `VLA2026/vla_pipeline` |
| Colab의 Linux GPU 서버 | 시뮬레이션·자동 수집·VLM/VLA 학습·평가 | `/content/...` |
| Google Drive | 연결이 종료되어도 남겨 둘 결과 | `MyDrive/VLA2026/Final_project/TEAM/RUN_ID` |

로컬 파일을 고쳤다고 원격 코드가 자동 갱신되지는 않습니다. push한 브랜치를 노트북의 `REPO_REF`로 선택하거나, VS Code Colab 확장의 **Upload to Colab** 후 원격 경로를 지정하세요. 이 노트북은 기존 작업 폴더를 강제로 reset/pull하지 않습니다.

## 처음 실행할 순서

1. VS Code의 Google Colab·Jupyter 확장을 설치하고 노트북을 엽니다.
2. `Select Kernel → Colab`에서 사용 가능한 GPU 서버를 선택하고 로그인합니다. Auto Connect만으로 GPU가 보장되지는 않습니다.
3. `TEAM`과 코드 브랜치를 확인합니다. 처음에는 `PROFILE="connection"`으로 둡니다.
4. Drive 마운트를 허용하고 출력된 보관 경로를 확인합니다. 권한 오류가 나면 저장을 확인하기 전 긴 학습을 시작하지 않습니다.
5. 셀을 위에서부터 실행합니다. 프로젝트용 Python 3.12 환경과 잠금 파일을 원격에 설치하므로 Windows에 CUDA를 설치할 필요가 없습니다.
6. 기본 영상·로그, 수집 시연, 학습 결과, 저장 모델 평가, 연구 기본 A의 결과를 확인합니다.
7. 마지막 `ARTIFACTS.md`와 `stage_paths.json`에서 파일 위치를 찾고 Drive에서도 실제로 열리는지 확인합니다.

`connection`의 1~2스텝 학습은 연결 검사입니다. 충분히 학습한 분류 로봇이 아닙니다. `study`도 시작 설정일 뿐이며 학습량과 데이터는 validation 결과로 정해야 합니다. 특히 다른 큰 VLM으로 바꾸고 동일한 메모리에서 전량 학습할 수 있다고 가정하지 마세요.

## 모듈을 어떻게 읽을까?

| 기능 | 코드 | 알아야 할 핵심 |
|---|---|---|
| 장면·로봇 실행 | `vlapipe/scene.py`, `env.py`, `controller.py` | 관측 두 카메라·상태16, 액션7의 의미 |
| 고전 제어 | `vlapipe/policy.py`, `scripts/collect_scripted.py` | 알고 있는 위치로 실제 접근·파지·운반·놓기. 녹화 중 순간이동 아님 |
| VLM 계획 | `vlapipe/planner.py` | 이미지·명령에서 작업 제안. 형식 오류·물체·목적지 오류를 구분 |
| 시연 저장·검사 | `episode.py`, `demo_validation.py` | T액션과 T+1관측, 성공 후보와 사람 검토 구분 |
| VLM 데이터·학습·평가 | `collect_planner_dataset.py`, `finetune_planner.py`, `evaluate_planner.py` | 원본 seed 분할, 답변 토큰 loss, 저장 모델 재평가 |
| VLA 데이터·학습 | `prepare_vla_training.py`, `train_vla.py` | train-only 변환, 두 카메라·상태와 실제 액션 학습 |
| 공정한 연구 실행 | `research_lab.py`, `run_research_lab.py` | 동일 정보·초기 배치·행동 예산, 실제 피드백과 중단 사유 |
| 학생 설계 | `vlapipe/student_design.py` | 접근성 기준, 다음 물체 선택, 재계획 조건을 직접 구현 |

설명과 명령은 노트북에 함께 있습니다. 전체 모듈을 노트북 안에 복제하지 않고 같은 저장소의 코드를 사용하므로, 실제 실행 코드와 수정 파일을 연결해서 볼 수 있습니다.

## 산출물 찾기

출력 폴더명은 매번 달라집니다. 고정 파일 이름을 추측하지 말고 RUN_ID와 색인을 보세요.

- `configuration.json`, `environment.json`, `renderer_setup.json`: 설정과 실제 GPU·OpenGL 실행 환경.
- `source_manifest.json`, `source_snapshot.tar.gz`, `student_design_used.py`: 실제 실행 코드.
- `commands.json`, `logs/`: 실행 명령, 종료 코드, 오류 원문.
- collection 단계의 `accepted/`, `rejected/`, `collection_report.json`: 원본 시연과 실패 분모.
- `demo_preview.../agentview.gif`, `wrist.gif`: 두 카메라의 전체 시연 미리보기.
- planner_data 단계의 JSONL·이미지·manifest: VLM 학습/검증/시험 분할.
- vlm_train 단계의 `pretrained_model/`, `loss.jsonl`: 저장 VLM과 학습 곡선.
- vla_train 단계의 `checkpoints/숫자/pretrained_model/`: 저장 VLA·정규화·전후처리. `last` 링크 대신 숫자 폴더를 전달.
- 평가·연결 단계의 `summary.json`, `report.json`, 원문 계획, 영상: 모델 성능과 실제 작업을 구분할 근거.
- `results.csv`, `ARTIFACTS.md`, `artifact_manifest.json`: 연구 결과 취합·파일 색인·해시.

연결이 갑자기 종료되면 마지막 단계의 미복사 파일은 없어질 수 있습니다. 기본 백업은 단계가 끝날 때 수행되며 학습 자동 재개를 보장하지 않습니다. 기존 RUN_ID를 덮어쓰지 말고, 재사용할 데이터와 모델을 명시적으로 선택해 새 실행을 시작하세요.

## 저장한 모델을 다시 사용하기

새 GPU 세션에서는 먼저 Drive와 실행 환경을 준비합니다. 학습을 다시 할 필요가 없다면 수집·학습 셀을 건너뛰고 아래처럼 과거 실행 폴더에서 모델 위치를 지정한 뒤, 노트북의 저장 모델 실행 셀을 사용합니다. `저장된_RUN_ID`는 본인 Drive에 실제로 있는 실행 폴더 이름으로 바꿉니다.

```python
past = Path('/content/drive/MyDrive/VLA2026/Final_project/team01/저장된_RUN_ID')
paths = json.loads((past / 'stage_paths.json').read_text(encoding='utf-8'))
SAVED_VLM = past / paths['vlm_train'] / 'pretrained_model'
SAVED_VLA = sorted(
    (past / paths['vla_train'] / 'checkpoints').glob('[0-9]*/pretrained_model')
)[-1]
```

이 방법은 **추론에 모델을 재사용**하는 절차입니다. optimizer 상태까지 복구해 학습을 이어가는 자동 재개 기능과 구분합니다. 다른 사람의 MyDrive 폴더는 본인 계정에서 같은 경로로 보이지 않을 수 있으므로, 공유·복사 후 실제 마운트 경로를 확인하세요.

## 사람의 키보드·조이스틱 입력

장치는 Windows PC에 있고 시뮬레이터 커널은 Colab에 있으므로, 원격 pygame 창이 로컬 조이스틱을 직접 읽지 못합니다. 자동 수집은 노트북에서, 사람 시연은 [로컬 수집 안내](../TELEOP_STUDENT.md)의 CPU 실행기로 수행합니다. 검토한 HDF5 파일을 Colab에 올린 뒤 노트북의 `EXTRA_ACCEPTED_DIR`에 원격 폴더를 지정하세요.

## 연구 실험으로 넘어가기

설치·기본 제어·저장 모델 재로딩이 정상인지 먼저 확인합니다. 이후 학생 설계 세 함수를 구현하고 A/B/C/D를 비교합니다. 완성 전의 NotImplemented 오류는 의도된 연구 빈칸입니다. 검사·재계획 조건에만 더 많은 정보나 시도 횟수를 주지 말고, 실패와 비용도 함께 분석하세요.

평면 큐브와 고정 목표 자리를 썼다면 결론도 그 범위로 제한합니다. 실제 적층·지지 관계·임시 공간·동적 유입은 추가 구현과 검증이 필요한 주제입니다.

## 실제 확인 범위 — 2026-11-05

동일한 노트북 파일을 실제 Colab Tesla T4에서 코드 셀16개 끝까지 실행했습니다. 새 Python 3.12 설치, 고전 수집, VLM 1스텝·VLA 2스텝 학습과 저장 모델 실행, 연구 A, 산출물 생성을 확인했습니다. 이후 Drive 저장을 켠 기본 설정으로도 전체 실행을 완료했습니다. 두 모델 폴더17파일·약2.94GB의 원본/Drive SHA256 일치, Drive 모델로 VLM 재평가와 VLA6제어 스텝 실행까지 확인했습니다. Windows VS Code 확장 화면에서의 조작은 별도 확인이 필요합니다.

고전 제어 기준선은5개 물체를 정리했습니다. 짧게 학습한 VLM은 validation에서 형식을 지키지 못했고, 저장 VLA의 짧은 실행은 새 운반 성공0이었습니다. 따라서 프로그램 연결 검사 통과를 충분히 학습된 모델의 작업 성능으로 해석하지 않습니다. 조교의 자세한 실행 기록을 확인하세요.

공식 UI 안내: [Google Colab VS Code 사용자 가이드](https://github.com/googlecolab/colab-vscode/wiki/User-Guide).
