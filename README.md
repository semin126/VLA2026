# VLA2026

MuJoCo 기반 VLM·VLA 연구 코드와 데이터 자료입니다.

- [기존 파이프라인 코드·실행 안내](vla_pipeline/README.md)
- [VLM·VLA 전체 자료 다운로드](https://github.com/semin126/VLA2026/releases/tag/dataset-full-20261104)
- [데이터 구성·다운로드 방법](datasets/README.md)

## 전체 자료 받기

전체 ZIP은 약 50.5GB이며, Releases에 26개 분할 파일로 제공됩니다. JSON·JSONL과 연결된 이미지, VLA 행동 시연 HDF5, 학습된 모델, 평가 결과, 코드와 실행 환경을 포함합니다.

Python 3.11 이상에서 다음 명령을 실행하면 모든 조각을 내려받고 SHA-256을 검사한 뒤 원래 ZIP으로 합칩니다.

```sh
python datasets/download_archive.py --directory downloaded_data
```

조각 파일과 복원 ZIP을 함께 저장할 약 101GB, 압축을 풀 공간은 별도로 필요합니다.

## 주요 데이터

| 자료 | 학습 / 검증 / 시험 |
|---|---|
| Qwen 최종 데이터 `qwen_final_data` | 39 / 11 / 33 |
| Qwen 확장 원본 `qwen_scale_data` | 425 / 70 / 330 |
| SmolVLM2 확장 데이터 `augmented` | 120 / 30 / 60 |
| VLA 성공 시연 | 학습 20 / 검증 5 에피소드 |

자료는 2026-11-04 수집본이며 2026-11-05에 게시했습니다. 정제본·부분집합·기존 배포본 사이에 중복이 있으므로 수량을 더해 고유 표본 수로 해석하지 않습니다. 상세 분류와 압축 중 추가된 실험 결과 안내는 ZIP 내부 안내문에 있습니다.
