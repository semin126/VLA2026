# VLM·VLA 전체 연구 자료

2026-11-04에 정리한 전체 자료를 2026-11-05에 업로드한 묶음입니다.
JSON·JSONL·이미지·VLA 시연 HDF5·학습된 모델·평가 결과·코드·실행 환경·기존 배포본을 11개 주제로 분류했습니다.
원본 ZIP은 50,515,727,574 bytes이며, GitHub 첨부 한도에 맞춰 26개 바이너리 조각으로 나눴습니다. 원래 ZIP의 내용은 바꾸지 않았습니다.

## 다운로드와 압축 해제

1. 첨부된 `download_archive.py`를 내려받습니다.
2. 저장한 폴더에서 `python download_archive.py`를 실행합니다. 전체 조각을 다운로드하고 SHA-256을 검사한 뒤 원래 ZIP으로 합칩니다.
3. 생성된 `VLM_VLA_ALL_TOPICS_20261104.zip`을 압축 해제합니다. ZIP 안의 `00_START_HERE_KO.md`와 `00_PACKAGING_UPDATES_KO.md`를 참고하세요.

이미 모든 `.001`~`.026` 조각과 `archive_manifest.json`을 내려받았다면 `python download_archive.py --join-only`로 합칠 수 있습니다. 개별 조각은 단독으로 압축 해제할 수 없습니다.
다운로드한 조각과 복원 ZIP을 함께 저장하는 동안 약 101GB의 여유 공간이 필요하며, ZIP 압축을 풀 공간은 별도로 필요합니다.

## 주요 데이터

| 주제 | 위치와 수량 |
|---|---|
| Qwen 최종 VLM 데이터 | `01_VLM_Qwen_datasets/qwen_final_data`: 학습 39 / 검증 11 / 시험 33 |
| Qwen 확장 원본 | `qwen_scale_data`: 425 / 70 / 330 |
| SmolVLM2 확장 데이터 | `02_VLM_SmolVLM2_datasets_and_models/session10_improvement_20261104/augmented`: 120 / 30 / 60 |
| VLA 성공 시연 | `03_VLA_demonstration_dataset/vla10_pilot_20261104`: 학습 20 에피소드 / 검증 5 에피소드 |
| 학습된 모델 | `04_Trained_models` 및 `05_VLA_training_and_comparison` |

각 데이터셋의 상대 이미지 경로를 유지했습니다. 정제본·부분집합·기존 ZIP 사이에 중복 자료가 있으므로 수량을 더해 고유 표본 수로 해석하면 안 됩니다.
기존 보고서 중 일부는 과거 실험 상태입니다. VLA 2,000스텝 학습 기록은 05 폴더에 있습니다.
별도로 진행 중이던 VLA 비교에서 압축 중 추가된 자료는 ZIP의 업데이트 안내에 기록했습니다. 이후 생성된 결과까지 자동으로 포함하는 묶음은 아닙니다.

원본 ZIP SHA-256: `b405feff9bcf7a56a66e8c7bac3164599b6b0811ad40bd9e3b6ace4c018f6fe2`
