# 2026-11-06 13:30 학생 전달용 모델

이 버전은 모델 로딩·운반·전략 연결을 실행하는 **고정된 예비 실험용**입니다.
전체 정리 성공률60%나 공정한 A/B 연구 기준이 확보된 모델은 아닙니다.

| 역할 | 제공 모델 | 현재 실측 |
|---|---|---|
| B/C 공통 VLM | R5 | 실제 Colab 계획 F1 .9259, 응답 형식131/132 |
| B/C 부분 작업 VLA | motion7k | 로컬 단일 운반14/20, 여러 큐브 중 지정 번호 선택3/20 |
| A 공통 VLA 비교 | `--shared-vla`로 같은 motion7k | 전체 정리 지시를 주고 VLM은 호출하지 않음. 별도 A 성능 측정 필요 |

계획 F1은 실제 운반 성공률이 아닙니다. 단일 운반70%를 여러 큐브 정리70%로
해석하지 마세요. 이번 비교는 부분 작업을 학습한 동일 VLA에 VLM·루프를 추가하는 실험입니다.
전체 과제를 별도 학습한 VLA의 일반적 성능까지 결론 내리지 않습니다.
새로운 전체 데이터 학습은 로컬에서 계속 진행하며, 이 고정 파일을 중간에 바꾸지 않습니다.

## 모델 받기

학생 수정본을 보관한 뒤 `c-loop-validation`의 최신 코드를 받으세요.
터미널의 작업 폴더가 `vla_pipeline`일 때:

```powershell
python scripts/download_models.py --manifest models/release_manifest_deadline_20261106.json --roles vlm,vla_hier --upgrade
```

이미 받은 같은 버전은 검증 후 유지하고, 교체하는 역할의 기존 모델은 백업합니다.
A/B/C 모두 이 공통 VLA를 사용하려면 실행 명령에 `--shared-vla`를 지정합니다.
명세의 기존 `vla_single` 항목은 이 방식에서 사용하지 않습니다.

실행 노트북에서 bootstrap과 설정을 실행한 뒤에는 다음 셀로 받을 수 있습니다.

```python
run.run('scripts/download_models.py',
        '--manifest', 'models/release_manifest_deadline_20261106.json',
        '--roles', 'vlm,vla_hier', '--models-dir', MODELS_DIR, '--upgrade')
```

이후에는 저장소 루트의 [공통 VLA Colab 안내](../../research/3_colab_guide.md)를 따르세요.
A/B/C는 동일한 `vla_hier`를 사용하며, A의 지시문은 전체 과제로 유지합니다.
모든 비교 조건에서 씬·시작 명령·GPU·시간 예산을 맞추고, 보고서의 실제 모델 버전을 보관하세요.
실험 도중 모델을 갱신하면 별도 실험 묶음으로 나누어야 합니다.

[VLM R5와 Colab 검증](https://github.com/semin126/VLA2026/releases/tag/models-planner-20261106-r5)
및 [motion7k 배포](https://github.com/semin126/VLA2026/releases/tag/models-motor-deadline-20261106-motion7k)에서 파일과 측정 범위를 확인할 수 있습니다.
