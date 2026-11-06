# 배포 모델과 로컬/Hub 선택

모델 가중치는 GitHub Releases에서 받고, 코드·다운로드 명세는 Git으로 받습니다. 추론 폴더 전체가 필요합니다. 새 모델을 내려받을 때 기존 모델은 자동으로 덮어쓰지 않습니다.

## 학생이 처음 실행할 때

현재 B/C의 부분 운반 모델은 [motor preview r2](https://github.com/semin126/VLA2026/releases/tag/models-motor-preview-20261106-r2)입니다.
동일 고정 씬10개에서 로컬은 최종 안정 운반4/10, 실제 Colab RTX PRO6000은5/10이었습니다.
두 결과는 같은 모델·씬의 반복 검증이며 독립20개 표본이나 학습 개선으로 합치지 않습니다.
60% 목표에는 아직 못 미칩니다. A는 이전r1 디버그 모델0/10이므로,
현재 A/B 조합으로 계층화의 연구 효과를 주장하면 안 됩니다.
**기본 VLM은 R5**입니다. 공개 파일을 실제 Colab RTX PRO6000에서 추론하여,
같은 계획 입력132개에서 목표 쌍 F1은 R2 .6053 / R3 .8099 / R5 .9259였습니다.
R5의 응답 형식 준수는131/132이고, 별도 실제 B/C 프롬프트4개는 목표쌍과 done이 모두 맞았습니다.
이것은 계획 점수이며 로봇 운반 성공률이나 독립 씬132개의 결과가 아닙니다.
[모델·검증 입력·원본 응답](https://github.com/semin126/VLA2026/releases/tag/models-planner-20261106-r5)을 공개했습니다.
기본 R5와 비교용 R2/R3 manifest 모두 같은 B/C 운반 모델을 가리킵니다.

이미 받은 VLM을 R5로 바꾸려면 다음을 실행하세요. 기존 VLM은 백업됩니다.

```powershell
python scripts/download_models.py --roles vlm --upgrade
```

이미 모델을 받았다면 다음으로 B/C만 교체할 수 있습니다. 이전 모델은 백업됩니다.

```powershell
python scripts/download_models.py --roles vla_hier --upgrade
```

새 모델의 내부 XYZ 형식은 absolute_position이며 제공 실행기가 기존action7로 변환합니다.
직접 저수준 루프를 작성한다면 매 행동 전에
`policy.set_controller_target(env.ctrl.target_pos)`를 호출하세요.
Colab 검증은 추론·평가만 수행했으며 모든 학습과 학습용 시연 수집은 로컬 GPU에서 했습니다.
씬별 판정과 실행 코드는 Release의 `motor_preview_r2_colab_summary_evidence.zip`에 있습니다.
전체 스텝 행동 궤적은 이 증거 ZIP에 포함하지 못했습니다.

1. 수정한 코드를 본인 브랜치에 보관한 뒤 `c-loop-validation`의 최신 코드를 받습니다.
2. 로컬은 `VLA2026 (conda, Python 3.12, CUDA)`, Colab은 GPU 커널을 선택하고 bootstrap까지 실행합니다.
3. 다운로드 셀에서 `DOWNLOAD_MODELS=True`로 한 번 실행하거나 다음을 실행합니다.

```python
run.run('scripts/download_models.py', '--models-dir', MODELS_DIR)
```

명령줄에서는 `python scripts/download_models.py`를 사용합니다. B/C만 필요하면 `--roles vlm,vla_hier`로 다운로드 용량을 줄일 수 있습니다. archive SHA256과 폴더 안 파일 해시를 확인한 후 설치합니다.

이미 받은 모델을 새 버전으로 교체할 때는 노트북에서 `DOWNLOAD_MODELS=True`,
`UPGRADE_MODELS=True`를 함께 설정하세요. 명령줄에서는
`python scripts/download_models.py --upgrade`를 사용합니다. 새 파일을 모두 검증한 뒤
기존 모델을 `models/_previous/`에 백업하고 교체합니다. 새 파일 검증이 실패하면 기존 모델을 그대로 둡니다.

폴더 구조:

```text
models/
  vlm/pretrained_model/        # 이미지 + 과제/전략/실행 문맥 → 접근성·계획 JSON
  vla_single/pretrained_model/ # A: 전체 과제 지시 → action7
  vla_hier/pretrained_model/   # B/C: 물체별 운반 지시 → action7
  release_manifest.json       # 다운로드 주소, 해시, 실제 검증 범위
```

## 기본값: models 우선

```python
MODEL_SOURCE = 'auto'
VLM_SOURCE = 'auto'
VLA_SOURCE = 'auto'
MODELS_DIR = 'models'
VLM_MODEL = ''
VLA_SCENE = ''
VLA_SUBTASK = ''
RUN_CONDITIONS = ['A', 'B', 'C', 'D']
```

- `auto`: models의 해당 모델이 있으면 사용하고, 없을 때만 Hub ID를 사용합니다.
- `local`: models 또는 명시한 로컬 폴더만 사용합니다. 없으면 다운로드 안내와 함께 종료합니다.
- `hub`: Hub에서 불러오도록 명시합니다. 역할별 기본 Hub ID는 models/model_sources.json에서 설정합니다.

VLM_SOURCE와 VLA_SOURCE를 따로 정할 수 있습니다. 예를 들어 VLM만 원래 Hub 기반 모델과 비교하려면 다음처럼 지정합니다.

```python
VLM_SOURCE = 'hub'
VLM_MODEL = 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
VLA_SOURCE = 'auto'
```

VLA_SCENE/VLA_SUBTASK에 로컬 경로나 호환 Hub ID를 직접 지정하는 기존 방식도 지원합니다. 명시적 경로/ID가 기본 폴더보다 우선합니다. 로컬 모델이 일부만 있거나 손상된 경우에는 Hub로 조용히 바꾸지 않고 오류를 표시합니다. 실제 선택한 위치와 source는 report.json의 models에 기록합니다.

**일반 `lerobot/smolvla_base`는 학습 초기화용입니다.** 이 시뮬레이터용 상태16·행동7·카메라2종·정규화 통계가 맞아야 하므로, 일반 base를 완성된 정리 정책처럼 실행하지 않습니다. 기본 Hub fallback이 이 규격과 다르면 설명 있는 오류로 종료합니다. Hub에서 VLA를 바로 실행하려면 이 환경에 맞춘 학습 체크포인트 ID를 설정하세요. 이번 배포는 Releases 모델을 models에 설치하는 경로가 기본입니다.

CLI에서도 `--model-source auto/local/hub`, `--vlm-source`, `--vla-source`, `--models-dir`, `--vlm-model`, `--checkpoint`, `--vla-hub-id`를 지원합니다.

## 학습·검증의 의미

VLM은 학생의 초기/재계획 프롬프트 형식으로 학습합니다. 교사 정답은 시뮬레이터 기하 규칙으로 만들었으며 사람 검토 정답으로 표시하지 않습니다. VLM의 계획 F1은 실제 운반 성공률이 아닙니다.

VLA는 실제 성공한 숫자 큐브 운반 궤적을 학습하며 원본 씬 단위로 학습·검증을 분리합니다. r1에서는 A가 전체 성공 궤적만, B/C가 전체 실패 궤적 중 성공한 부분 운반도 포함했으므로 동일 학습 조건의 통제 실험이 아니었습니다. motor r2는 전체와 모든 부분 작업이 성공한 공통 원본에서 B/C를 학습했지만, 이에 대응하는 새 A는 아직 배포하지 않았습니다. 정식 A/B 비교는 같은 원본·행동·영상과 학습 조건을 맞춘 A가 필요합니다. 숫자를 볼 수 있도록 고해상도 카메라를 사용하며 실행 때도 체크포인트의 학습 해상도로 관측합니다. B와 C는 같은 부분 작업 모델을 사용합니다.

단일 운반 평가는 처음부터 정리되어 있던 다른 9개 물체를 성공으로 세지 않습니다. 지시한 대상의 들어 올리기·운반·놓기를 구분하고, 전체 씬 ACC는 별도로 기록합니다. 학습 loss와 목표 성공률은 서로 다른 지표입니다.

preview 배포는 빠른 실험과 실패 분석용입니다. release_manifest와 Release 설명에 있는 표본 수·실측 성공률을 확인하세요. 짧은 튜닝이나 연결 검사를 60% 성공 또는 전체 씬 정리 성능으로 해석하지 않습니다. 기본 기반 모델, 다른 모델, 다른 프롬프트를 비교할 때는 선택한 모델과 코드 버전을 함께 기록합니다.

수집 시 `collect-actions --accelerated`는 물리 궤적을 빠르게 만드는 용도이며 A~E 시간 비교에 사용하지 않습니다. 정식 비교는 기존 20Hz 상한·동일한 wall-time 예산을 유지합니다.

## 2026-11-06 planner r2: VLM만 추가 학습

[planner r2 공개 파일](https://github.com/semin126/VLA2026/releases/tag/models-planner-20261106-r2)은
로컬 RTX3090Ti에서 초기·재계획·완료 문맥으로 300업데이트 추가 학습한 VLM입니다.
층화 validation 12개에서 JSON 형식 11/12, 계획 pair F1 0.7885,
접근성 라벨 정확도 0.9000, done 정확도 11/12였습니다.
한 단일 운반 씬에서는 반복 출력 뒤 JSON이 잘렸습니다.
씬 안의 여러 행이 포함되고 자동 생성 라벨이므로 독립적인 12개 과제 성공률로 해석하지 않습니다.

이 VLM-only Release 당시에는 A/B/C용 VLA가 r1 파일을 참조했습니다.
현재 B/C 모델은 위의 motor r2 안내를 확인하세요. A는 여전히 r1입니다.
`DOWNLOAD_MODELS=True`, `UPGRADE_MODELS=True`로 새 VLM을 받고 이전 버전을 보존할 수 있습니다.

**공개 배포본 Colab 실측:** ZIP과 내부 파일 해시를 확인한 뒤 RTX PRO6000 Blackwell 약96GB에서
52개 응답 생성을 완료했습니다. JSON 형식43/52(82.69%), 계획 pair F1 0.7481,
done 정확도43/52였습니다. 형식 실패9개와 계획 누락이 남아 있습니다.
학생의 실제 `valid_open` 시작 프롬프트는10개 대상 중6개를 올바르게 제안하고4개를 빠뜨렸습니다
(그 한 씬 pair F1=0.75, 불필요한 추가 물체0).
[Colab raw 응답과 측정 요약](https://github.com/semin126/VLA2026/releases/download/models-planner-20261106-r2/planner_r2_colab_evidence.zip)을 확인할 수 있습니다.
Colab은 추론만 실행했으며, 학습·데이터 생성은 모두 로컬에서 진행했습니다.

## 2026-11-06 preview r1의 실제 결과

**이 VLA는 아직 운반이 가능한 기준 모델이 아닙니다.** 체크포인트 로드, 학생 전략과의 연결, 실패 분석을 위한 미리보기입니다. 이 가중치로 얻은 A/B/C 차이를 계층 전략의 효과라고 결론 내리지 마세요.

| 모델 | 학습 | 별도 validation 결과 |
|---|---|---|
| VLM | 70개 이미지/프롬프트, 100 업데이트 | 16건: JSON 형식 100%, 계획 pair F1 0.8644, 순서까지 일치 75% |
| A용 VLA | 25개 성공 궤적, 400+600 업데이트 | 대상 운반 0/10, 모두 파지 단계 실패 |
| B/C용 VLA | 79개 성공 부분 운반, 400+600 업데이트 | 대상 운반 0/10, 모두 파지 단계 실패 |

VLM 정답은 기하 기반 교사의 자동 라벨이며, 사람 검토·숫자 인식·실제 운반 성공의 점수가 아닙니다. 두 번째 VLA 라운드는 첫 라운드의 가중치에서 시작하되 optimizer는 새로 초기화했습니다. 검증 씬은 각 번호 한 번씩을 포함하며, 다른 아홉 개가 처음부터 정리되어 있어도 성공으로 가산하지 않았습니다. 원래 학습한 단일 씬에 대한 추가 진단도 별도 기록합니다. 전체 씬·가림 씬에 대한 성공률은 아직 확보하지 못했습니다.

Releases의 `evaluation_and_recipe.zip`에는 검증 요약·전후 이미지·씬 정의·학습 명령과 재현 보조 코드를 제공합니다. 원본 HDF5와 optimizer 체크포인트는 로컬 `outputs/model_release_20261106_round1/`에 보존합니다.

**후속 실제 Colab 확인:** 공개 r1 ZIP을 다운로드해 SHA256을 확인하고,
RTX PRO6000 Blackwell 약96GB에서 A/B용 VLA의 CUDA 추론과 유한한 action7 출력을 확인했습니다.
최신 정밀도 보존 로더(`5b1e518`)에서도 통과했습니다. 이 확인은 T4 검증이나 물리 운반 성공을 뜻하지 않습니다.
VLM은 학생의 `valid_open` 초기 문맥에서 반복 응답 후 JSON이 잘린 사례가 있어,
위의 16행 라벨 평가만으로 실제 전략 문맥의 응답 계약까지 통과했다고 판단하지 않습니다.
새 모델의 측정 범위는 `models/release_manifest.json`과 해당 Release 설명에서 확인하세요.

학습 가중치의 FP32 보장, 텐서별 추론 정밀도 복원, 추가 행동 타깃과 엄격한 운반 평가는
[운반 모델 검증 안내](MOTOR_MODEL_VALIDATION.md)에 설명되어 있습니다.

다음 학습의 통과 기준은 (1) 한 시연 씬의 실제 파지·운반·놓기를 재현하고, (2) 학습하지 않은 단일 운반에서 모든 번호·양쪽 목적지를 검증한 뒤, (3) B/C 공통 모델을 전체 씬으로 확장하는 것입니다. 학습 loss만 감소하면 통과한 것으로 보지 않습니다.
