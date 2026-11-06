# oneway・mutual / A・B・C1 전용 노트북 검증

2026-11-06, `instructor/runtime-conda-fixes` 배포용.
실행 진입점은 [전용 노트북](../notebooks/Mutual_oneway_C1_20261106.ipynb), 학생 순서는 [Colab 안내](../../research/3_colab_guide.md)입니다.

## 확정한 실험

- oneway 10씬 + mutual 10씬, 씬마다 A·B·C1: 총 60회.
- 네 담당자는 자기 묶음 5씬 × 3조건 = 15회씩 실행합니다.
- 고정 VLM R5 + motion7k. A/B/C1 모두 `--shared-vla`, 모델 선택은 `local`.
- A에는 전체 과제 지시, B/C1에는 VLM의 부분 지시. A의 VLM 호출은 0회입니다.
- C1 v2는 사전 선택입니다. C0/C2보다 실제 정리 성능이 높다고 입증한 결과는 아닙니다.

## 통과한 검사

| 검사 | 확인한 범위 |
|---|---|
| 로컬 노트북 검사 | nbformat, 모든 코드 셀 Python 문법, 네 묶음 60건의 CLI 인자, 실제 씬 파일과 frozen state 해시, 실행 잠금 |
| 회귀 확인 | 로컬 smoke 5/5, 기존 messy/seed3 실행 4/7·종료 코드0 |
| 실제 Colab 준비 | 지정 강사 브랜치 새 clone, Panda 에셋 다운로드·패치, Python 3.12, EGL 렌더링, CUDA·OpenCV 확인 |
| 실제 모델 검사 | 공식 downloader로 기존 공개 R5·motion7k의 모든 파일 해시·역할·버전 검사. 두 역할만 사용 |
| 최종 씬 실행 목록 | 네 실제 suite에서 oneway 30건 + mutual 30건 생성. 이 단계에서 최종 씬 모델 추론은 하지 않음 |
| 실제 GPU 실행 | 기존 검증 씬 1개에 A·B·C1을 각각 실행, 3회 모두 코드 오류 없이 종료 |
| 비교 조건 확인 | 같은 VLA 가중치 해시, 같은 frozen state, B/C1 같은 초기 프롬프트, A VLM 호출0·B1·C1의 호출1 |
| 결과 저장 | 세 trial의 MP4, report, 집계 CSV·JSON·HTML, 전달용 ZIP 생성 확인 |

Colab 장비는 **NVIDIA RTX PRO 6000 Blackwell Server Edition**입니다.
실행한 전용 노트북 SHA256:
`2a28d0ac324067cd5725e0dab491d23073ae4f134dbd245bddd1b73849ef2beb`.
검사한 실행 기반 커밋은 `e20789dc28cc7194cc2134286d6f0f6a1e4ed7cd`이며,
전용 노트북 셀은 위 해시의 파일을 별도로 전달해 그대로 실행했습니다.

[원본 GPU 검사 결과](evidence/C1_COLAB_20261106.json)의 SHA256은
`16498b47825b520cc17c2d69a78c540041df50dabcf9eed35e6d168714355e7b`입니다.

## 실제 실행으로 발견해서 고친 오류

첫 검사에서는 A·B·C1 추론과 영상 생성이 끝난 뒤 집계가 `KeyError: output`으로 멈췄습니다.
노트북의 `schedule.json`에 trial별 출력 상대 경로가 빠져 있었기 때문입니다.
각 항목에 `output`을 기록하고 실제 실행 폴더도 같은 값을 사용하도록 고쳤습니다.
수정본은 새 작업 폴더에서 처음부터 다시 실행해 집계와 ZIP 생성까지 통과했습니다.
원본 실패 로그는 조교의 `evidence/motor_r2_colab/student_c1_progress3.json`에 보관합니다.

## 검증의 한계

이번 실제 GPU 연결 검사는 기존 검증 씬을 사용했고 예산은 30초/300스텝/최대2회 호출로 줄였습니다.
본실험 설정 300초/6000스텝/최대12회 호출의 인자는 별도로 검사했지만,
**최종 oneway·mutual 20씬의 60회를 미리 실행한 것은 아닙니다.**
검증 씬의 C1은 한 번의 계획으로 끝났으므로, 이 검사만으로 실패 후 재계획의 효과를 입증하지 않습니다.

기존 잠금 파일로 준비한 Python 3.12 GPU 환경을 재사용했습니다.
이번 검사에서 패키지를 전부 재설치하거나 Google Drive를 다시 인증하지는 않았습니다.
모델 ZIP의 최초 공개 다운로드는 앞선 검증에서 했고, 이번에는 같은 공개 파일의 설치본을 공식 downloader로 재검사했습니다.
따라서 T4, 완전히 새 런타임의 설치, 이번 노트북의 Drive 인증·백업을 새로 실증했다고 쓰지 않습니다.

검사 중에는 **학습하지 않았습니다.** 전체 정리 성공률이나 C1의 전략 우위를 주장하는 성능 평가가 아닙니다.
