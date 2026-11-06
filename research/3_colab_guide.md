# Colab 본실험: oneway・mutual에서 A・B・C1 비교

이번 본실험은 **oneway와 mutual 모두**, **A·B·C1만** 실행합니다. C0·C2는 이번 본실험에서 제외합니다.
실험 도중 전략을 바꾸지 않습니다. C1은 실패 대응과 VLM 호출 비용을 고려해 사전에 선택했습니다.
실제 성능에서 C0·C2보다 우수하다고 확인된 것은 아닙니다.

## 사용할 노트북과 코드

- 코드 브랜치: **`instructor/runtime-conda-fixes`**
- 전용 노트북: **[`Mutual_oneway_C1_20261106.ipynb`](../vla_pipeline/notebooks/Mutual_oneway_C1_20261106.ipynb)**
- [Colab에서 열기](https://colab.research.google.com/github/semin126/VLA2026/blob/instructor/runtime-conda-fixes/vla_pipeline/notebooks/Mutual_oneway_C1_20261106.ipynb)
- C 전략: `vla_pipeline/final_project_v2/student_strategy_c1_v2.py`
- 모델 명세: `models/release_manifest_deadline_20261106.json` — **VLM R5 + VLA motion7k**

기존 `Final_project_v2_0.ipynb`는 범용 개발용입니다. 이번 실험은 위 전용 노트북을 사용하며,
이전 안내의 C0·C1·C2 실행 셀이나 기본 모델 다운로드 셀을 섞어 쓰지 않습니다.
Colab에서는 지정 브랜치를 새 폴더에 받습니다. GitHub에서 노트북을 열었다고 로컬 수정 파일까지 자동 전송되지는 않습니다.

## 비교하는 조건

| 조건 | 실행 |
|---|---|
| A | 전체 정리 지시를 VLA에 직접 전달. VLM 호출 없음 |
| B | VLM이 처음 한 번 계획하고 VLA가 부분 지시를 실행 |
| C1 | B와 같은 초기 프롬프트로 시작. 실패가 의심되면 학생 C1 전략에 따라 재계획 |

**`--shared-vla`로 세 조건 모두 같은 motion7k를 사용합니다.** A의 지시문은 전체 과제로 유지합니다.
부분 작업 위주로 학습한 모델이므로 A의 전체 지시는 학습 분포와 다릅니다.
따라서 별도로 전체 과제를 학습한 A와 비교했다고 쓰면 안 됩니다.
현재 배포 모델은 예비 실험용이며 **전체 정리 성공률 60%가 검증된 모델이 아닙니다.**

## 담당 묶음

| 담당 | 설정 셀의 `PART` | 계열 | 씬 수 | 실행 수 |
|---|---|---|---|---|
| 조교 | `part1_oneway_a` | oneway | 5 | 15 |
| 학생 1 | `part2_oneway_b` | oneway | 5 | 15 |
| 학생 2 | `part3_mutual_a` | mutual | 5 | 15 |
| 학생 3 | `part4_mutual_b` | mutual | 5 | 15 |

전체를 합쳐 **oneway 10씬 + mutual 10씬**, 씬당 A·B·C1, 총 **60회**입니다.
각 묶음은 어질러진 큐브가 2·4·6·8·10개인 씬을 하나씩 포함합니다.
**한 사람이 자기 씬의 모든 조건을 실행합니다.** 조건별로 사람을 나누지 않습니다.
개인당 한 계열을 맡아도 전체 실험에서는 두 계열을 모두 평가합니다.
D는 기존 결과를 보관하며 Colab에서 새로 실행하지 않습니다.

## 실행 순서

1. 위 전용 노트북을 열고 GPU 런타임에 연결합니다. VS Code에서는 Colab 커널을 선택합니다.
2. 첫 설정 셀의 **`PART`만 본인 묶음으로 변경**합니다. `TEAM = PART`는 그대로 둡니다.
3. 코드·환경 준비 셀을 실행합니다. Google Drive 연결을 허용하면 매 trial 명령 종료 후 백업됩니다.
   출력의 Execution Python과 GPU 이름을 확인하세요. CUDA가 없으면 본실험 전에 중단합니다.
4. 모델 다운로드 셀을 실행합니다. R5와 motion7k 두 개만 받으며 SHA256을 검사합니다.
   같은 버전은 유지하고 이전 모델은 새 파일 검증 후 백업합니다. `vla_single`은 받지 않습니다.
5. 실행 전 검사 셀에서 **본인 계열, 씬 5개, 실행 15개, C1**인지 확인합니다.
   이 검사는 파일과 버전을 확인하며 최종 씬에서 모델을 실행하지 않습니다.
6. 설정 셀에서 **`RUN_EXPERIMENT=True`로 변경**하고 설정 셀을 실행한 다음,
   **본실험 실행 셀**을 실행합니다. `PART`는 바꾸지 마세요. 환경 준비 셀을 다시 실행할 필요는 없습니다.
7. 결과 확인 셀을 실행합니다. 보고서 15개와 오류 유무를 확인하고 결과 폴더·압축 파일을 조교에게 전달합니다.

실험은 순차 실행됩니다. **15회 × 최대 300초 = 75분**에 모델 로딩·녹화·백업 시간이 더해집니다.
GPU 종류에 따라 조기 종료와 준비 시간이 달라지므로 실제 완료 시각을 보장하는 값은 아닙니다.
한 Colab에서 동시에 다른 GPU 작업을 돌리지 않습니다.

## 고정 설정과 결과

- 제한 시간 300초, 최대 6000스텝, 운반 240스텝, 관측 간격 60스텝, VLM 호출 최대 12회, 정책 seed 42.
- 모델 선택은 `local`입니다. 모델이 없으면 Hub의 다른 모델로 자동 대체하지 않고 중단합니다.
- 씬마다 조건 순서를 고정 seed로 섞고, 매번 같은 frozen state에서 시작합니다.
- B의 제공 전략과 C1은 같은 초기 VLM 프롬프트를 사용합니다.
- 결과: `test_shared_c1_<PART>/<scene>/42/<A 또는 B 또는 C>/report.json`.
  폴더의 C는 **C1**을 뜻하며 보고서에는 실제 전략 파일과 해시가 기록됩니다.
- `completed`는 코드가 정상 종료했다는 뜻입니다. **씬 정리 성공은 `scene_success`**, 물체 정확도는 `acc`, 구역 F1은 `macro_f1`을 봅니다.
- `process.mp4`, `plan_*.json`, `events.jsonl`로 동작과 재계획을 확인할 수 있습니다.
- Drive: `VLA2026/Final_project_v2/<PART>/<실행ID>/test_shared_c1_<PART>/`.
  실행 중 연결이 끊긴 trial은 아직 백업되지 않았을 수 있습니다.

## 오류와 재실행

낮은 점수나 실패도 정상 결과이면 그대로 보관합니다. 성공할 때까지 반복해 가장 좋은 결과를 고르지 않습니다.
기술적 오류가 있으면 **묶음·씬·조건·오류 로그**를 함께 보내 주세요.
실행 셀을 같은 폴더에서 다시 돌리면 덮어쓰지 않고 중단합니다.
조교가 중단된 trial만 확인해 새 위치에서 재실행하며, 원본과 재실행 이유를 모두 남깁니다.
집계에는 같은 씬·조건·seed의 공식 결과 하나만 포함하고, 기술적 재시도 원본은 별도로 보관합니다.

## 조교 집계와 해석

전용 노트북은 개인 묶음의 `comparison/results.csv`, `summary.json`, `index.html`을 만듭니다.
전체 결과를 받을 때는 **oneway용 폴더와 mutual용 폴더를 나눠**, 각 계열에 두 담당자의 결과를 넣고 각각 실행합니다.

```powershell
python scripts/final_project_v2.py aggregate --root <oneway 결과를 모은 폴더>
python scripts/final_project_v2.py aggregate --root <mutual 결과를 모은 폴더>
```

각 계열은 10씬 × 3조건 = 30개 보고서입니다. 다른 모델 버전이나 과거 C0·C2 결과는 섞지 않습니다.
D는 정답 좌표를 사용하는 참고 기준이며 씬 선정에도 사용됐습니다. PC와 Colab의 시간은 직접 비교하지 않습니다.
oneway는 순서와 90도 회전 큐브 처리, mutual은 막힌 쌍과 정확도 변화를 영상으로 함께 확인하세요.
시작부터 정리된 큐브가 있으므로 **최종 ACC와 시작 ACC의 차이**도 함께 봅니다.
C1이 B보다 낮더라도 실패 의심 판정, VLM 호출 비용, 하위 VLA의 실패를 살펴보고 해석합니다.

## 검증 범위

실제 Colab GPU에서 공개 R5 + motion7k로 동일 검증 씬의 A/B/C0/C1/C2 실행이 오류 없이 완료됐습니다.
A는 VLM 호출 0회, 다섯 조건의 VLA 가중치 해시가 같고 B/C 초기 프롬프트와 frozen state가 같음을 확인했습니다.
이는 **연결 검사**이며 최종 oneway·mutual 20씬의 성능이나 C1의 우수성을 검증한 결과는 아닙니다.
이번 전용 노트북의 추가 검증 기록은 `vla_pipeline/docs/C1_COLAB_VALIDATION_20261106.md`에서 확인하세요.
