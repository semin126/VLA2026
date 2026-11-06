# 학생용 실험 자료 묶음 (2026-11-06)

보고서를 쓰고 남은 실험을 돌리는 데 필요한 자료입니다. 순서대로 읽으면 됩니다.

| 문서 | 내용 | 언제 읽나 |
|---|---|---|
| [1_experiment_setup.md](1_experiment_setup.md) | 실험 환경, 네 조건, 테스트 씬, 모델 학습 과정과 겪은 문제 | 보고서의 "실험 방법"을 쓸 때 |
| [2_D_results.md](2_D_results.md) | 먼저 끝난 D 실험의 결과와 해석, 영상과 자료 위치 | 보고서의 "결과"를 쓸 때, A·B·C 결과와 비교할 때 |
| [3_colab_guide.md](3_colab_guide.md) | Colab에서 A, B, C를 돌리는 순서 (네 사람이 동시에) | 조교가 시작을 알린 뒤 |
| [STUDENT_COMMENTS.md](STUDENT_COMMENTS.md) | C 전략 파일에서 조교가 손본 부분과 스스로 생각해 볼 질문 | C 전략을 설명하거나 발표를 준비할 때 |

## 폴더

| 폴더 | 내용 |
|---|---|
| `images/` | 문서에 쓰인 그림과 움짤. 보고서에 가져다 써도 됩니다 |
| `videos/` | D 실행 영상 26개 (MP4, 실제 속도) |
| `d_results/` | D 실행의 원본 기록 30개 (점수, 사건, 시작·끝 화면, 상태 기록) |

## 현재 본실험 안내

**oneway와 mutual 모두**, **A·B·C1**을 실행합니다. C0·C2는 이번 본실험에서 제외합니다.
코드는 `instructor/runtime-conda-fixes`, 모델은 고정 배포 VLM R5 + motion7k입니다.
전용 노트북 `vla_pipeline/notebooks/Mutual_oneway_C1_20261106.ipynb`와
[Colab 안내](3_colab_guide.md)를 사용하세요. 네 묶음을 합쳐 20씬 × 3조건 = 60회입니다.

C1은 실패 대응과 호출 비용을 고려한 사전 선택이며 최고 성능이 입증된 것은 아닙니다.
D는 기존 참고 자료를 보관하고, C 전략 세 종류의 설명은 배경 학습용으로 남깁니다.
실제 Colab 검사 범위와 한계는 `vla_pipeline/docs/C1_COLAB_VALIDATION_20261106.md`에 기록합니다.
