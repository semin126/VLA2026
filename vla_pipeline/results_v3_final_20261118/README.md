# 최종 발표용 평가 자료 — h2 / a2 / R8 / c0

최종 선정 기록에 따라 h2 자료를 준비했습니다. 공개 여부는 실제 GitHub 게시 영수증으로 확인하세요.

모델과 규칙을 고정한 뒤 만든 흩어진 새 장면24개에서 목표120개를 비교했습니다. 실제 MuJoCo 시뮬레이션 실행 기록입니다. 실물 로봇 실험으로 표기하지 마세요. 300초 실제 시간/최대6000스텝/운반240스텝, 정책seed42를 사용했습니다. 이 장면으로 모델을 선택하지 않았습니다.

| 조건 | 방법 | 남은 목표 ↓ | 정리한 목표 | 정돈 큐브 교란 |
|---|---|---:|---:|---:|
| A | 전체 명령 한 문장 + VLA a2 | 90 | 30 | 0 |
| B | R8이 처음 한 번 계획 + VLA h2 | 85 | 35 | 1 |
| C | R8 + VLA h2 + 새 사진으로 재계획하는 c0 | 67 | 53 | 2 |
| D | 정답 좌표를 쓰는 스크립트 교사 | 0 | 120 | 0 |

D는 신경망 조건보다 많은 정답 정보를 쓰는 상한 참고값입니다. C의 최종 목표 정리 수 증가는 운반 한 번의 성공률 증가와 구분해야 합니다.

| 비교 | 장면당 잔여 차이 | 짝 bootstrap 95% 구간 |
|---|---:|---|
| A-B | +0.21 | [-0.38, 0.88] |
| A-C | +0.96 | [0.38, 1.54] |
| B-C | +0.75 | [0.12, 1.33] |

양수는 뒤 조건이 더 적게 남겼다는 뜻입니다. 24장면을 단위로 재표집한 구간이며 여러 정책·학습 시드나 장비 변동까지 포함하지 않습니다. B 단독이 A보다 낫다는 근거는 이 측정에서 뚜렷하지 않습니다.

## PPT용 그림

- [실험 세팅과 두 카메라](fig1_setup.png)
- [A·B·C·D 흐름](fig2_conditions.png)
- [조건별 남은 목표](fig3_remaining.png)
- [장면별 짝 비교](fig4_paired.png)
- [시간에 따른 정리 수](fig5_time_curve.png): 위치·높이 기반 보조 곡선으로 공식 엄격 점수와 구분합니다.
- [운반 결과 유형](fig6_failures.png)

[집계 JSON](results_summary.json)과 [장면별 CSV](results_table.csv)를 함께 제공합니다. 공식 점수는 큐브 바닥 네 꼭짓점·바로 선 자세·정지·놓음을 확인합니다.

## 실제 평가 영상과 정리 전후 사진

결과를 보고 예시를 고르지 않고 장면 목록의 처음4개를 사용했습니다. A·B·C·D 영상16개와 전후 이미지4개가 있습니다.

| 장면 | 전후 사진 | A | B | C | D |
|---|---|---|---|---|---|
| scene_0000 | [사진](examples/scene_0000_before_after.png) | [A](examples/scene_0000_A.mp4) | [B](examples/scene_0000_B.mp4) | [C](examples/scene_0000_C.mp4) | [D](examples/scene_0000_D.mp4) |
| scene_0001 | [사진](examples/scene_0001_before_after.png) | [A](examples/scene_0001_A.mp4) | [B](examples/scene_0001_B.mp4) | [C](examples/scene_0001_C.mp4) | [D](examples/scene_0001_D.mp4) |
| scene_0002 | [사진](examples/scene_0002_before_after.png) | [A](examples/scene_0002_A.mp4) | [B](examples/scene_0002_B.mp4) | [C](examples/scene_0002_C.mp4) | [D](examples/scene_0002_D.mp4) |
| scene_0003 | [사진](examples/scene_0003_before_after.png) | [A](examples/scene_0003_A.mp4) | [B](examples/scene_0003_B.mp4) | [C](examples/scene_0003_C.mp4) | [D](examples/scene_0003_D.mp4) |

영상은 기록된 시뮬레이터 상태를 재생하며 계획 모델의 실제 대기 시간을 생략합니다. 영상 길이로300초 성능을 비교하지 마세요. 각 종료 사진은 해당 조건이 멈춘 시점입니다. 전체96영상의 별도 릴리스 묶음은44,440프레임을 디코딩해 검사했습니다. 이 폴더의16영상은 검증된 원본과 바이트가 같습니다.

## 실행과 출처

- 측정 코드: [1963e47ca195a0710406d2a1453eb6186e4dcb8d](https://github.com/semin126/VLA2026/tree/1963e47ca195a0710406d2a1453eb6186e4dcb8d/vla_pipeline).
- [v3 노트북](https://github.com/semin126/VLA2026/blob/1963e47ca195a0710406d2a1453eb6186e4dcb8d/vla_pipeline/notebooks/Final_project_v3_0.ipynb)과 [모델 다운로드 매니페스트](https://github.com/semin126/VLA2026/blob/1963e47ca195a0710406d2a1453eb6186e4dcb8d/vla_pipeline/models/release_manifest_v3_scattered_20261113.json).
- A는 a2, B·C는 h2, 계획은 R8을 사용했습니다. C는 각 운반 직후 무조건 재계획하는 C2와 구분합니다.
- 로컬 RTX3090Ti 환경에서 측정했습니다. 다른 장비에서는 같은 실제 시간에 실행하는 스텝 수가 달라질 수 있습니다.
- 새 Colab 체크아웃 전에 `os.environ["FPV3_REPO_REF"]`를 위 커밋으로 지정하고 SOURCE VERSION을 확인하세요. 기존 PROJECT 폴더는 환경변수만으로 바뀌지 않습니다. 새 Colab 전체 실행을 재검증한 것은 아닙니다.
- 이 폴더에는 실행 코드·노트북·모델 파일을 복사하지 않았습니다. 원래 v3 실행 경로와 모델 다운로드 주소를 사용합니다.
- 미디어·결과 파일의 SHA와 원본 평가 출처는 [PROVENANCE.json](PROVENANCE.json)을 따릅니다.

그림·영상 출처: “조교팀 MuJoCo 시뮬레이션 측정, v3/h2·R8·a2, seed42”.
