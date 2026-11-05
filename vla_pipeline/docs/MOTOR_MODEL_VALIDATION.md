# 운반 모델의 학습과 배포 검증

데이터 생성과 VLM/VLA 파인튜닝은 조교의 **로컬 GPU**에서 진행합니다.
Colab에서는 공개 배포 파일 다운로드, 무결성 확인, GPU 추론만 검사합니다.

## 무엇을 성공으로 세는가

`scripts/evaluate_motor_skill.py`는 지정한 숫자 큐브를 실제로 집어 올리고,
공중에서 운반한 뒤 올바른 구역에 안정적으로 놓았는지 검사합니다.
지시하지 않은 물체를 집거나 다른 정답 물체를 훼손하면 엄격 성공에서 제외합니다.
처음부터 정돈돼 있던 다른 아홉 물체는 운반 성공으로 세지 않습니다.

- `single`: 방해물 없는 단일 운반입니다. 숫자로 여러 후보 중 하나를 고르는 능력을 입증하지 않습니다.
- `selection`: 여러 미정리 큐브 중 지시한 하나만 운반합니다. 단일 운반과 따로 보고합니다.
- `--full-horizon`: 중간에 한 번 성공해도 실행을 계속하여 이후 재파지나 실패 역전을 확인합니다.

모델과 캐시 길이 선택에는 `validation`만 사용합니다. 모델/설정을 고정한 뒤
별도 `sft_test`를 한 번 평가합니다. 이것은 A~E의 최종 mixed benchmark와 별도입니다.
이 진단은 가속 시뮬레이션이므로 A~E 제한 시간 실험의 속도로 해석하지 않습니다.
씬 단위 ACC/F1과 함께 실제 운반 성공 수, 시행 수, 신뢰구간을 확인하세요.

## 학습 가중치의 정밀도

Transformers가 모델의 일부를 BF16으로 구성할 수 있어, 설정 하나만으로는
FP32 학습 가중치가 보장되지 않았습니다. 현재 `train_vla.py`는
`vlapipe/model_loading.py`의 프로세스 내부 factory를 사용해 가중치를 읽기 전에
학습 파라미터를 FP32로 만듭니다. `--mixed-precision bf16`은 연산 정밀도에 적용됩니다.
추론은 저장된 각 텐서의 정밀도를 복원합니다. backbone 전체를 임의의 한 정밀도로
바꾸지 않습니다. 설치된 LeRobot/Transformers 소스는 이 기능을 위해 수정하지 않습니다.

VLM/VLA 학습기는 같은 checkout의 학습 잠금을 공유합니다. 다른 학습이 진행 중이면
기본적으로 중단하고, `--wait-for-gpu`를 주면 기다립니다. 추론에는 이 잠금이 없습니다.

## 행동 타깃 실험

기본 타깃은 기존 `delta`입니다. 학습 데이터 준비 시 선택할 수 있는
`--action-representation absolute_position`은 신경망의 XYZ 출력만 다음 **로봇 제어 목표 위치**로
학습합니다. 실행 직전에 현재 로봇 제어 목표와의 차이를 기존 delta7 명령으로 바꿉니다.
물체 정답 위치를 추론 입력에 넣거나 규칙 제어기로 행동을 대체하지 않습니다.
RGB 두 카메라, state16, 외부 action7 및 기존 제어기 수치는 유지합니다.

이 모델은 `action_representation.json`이 필요합니다. 제공 실행기는 이를 자동 적용합니다.
직접 실행 루프를 작성하면 매 행동 전에 다음 연결을 넣으세요.

```python
policy.set_controller_target(env.ctrl.target_pos)
action = policy.select_action({**env.get_obs(), 'task': instruction})
env.step(action)
```

이는 로봇 actuator의 명령 상태이며 물체 인식 결과가 아닙니다. 원시 시연 파일은 바꾸지 않습니다.
서로 다른 행동 타깃으로 학습한 A와 B/C는 공정한 비교군으로 바로 묶지 않습니다.
A는 같은 원본 시연에 전체 과제 라벨을 붙여 별도로 학습하고, B/C는 부분 지시문 모델을 공유합니다.

## 배포본을 사용할 때

`models/release_manifest.json`의 측정 범위와 `validated_motor_policy`를 먼저 확인하세요.
`models-preview-20261106-r1`은 운반 성공률 0/10이었던 연결 점검용 프리뷰입니다.
코드의 연결 검사 통과나 학습 loss 감소만으로 정상 운반 모델이라고 부르지 않습니다.
새 버전은 다른 immutable Release에 배포하며, 기존 모델은 자동으로 덮어쓰지 않습니다.
새 폴더에 설치하려면 `download_models.py --models-dir <새 경로>`를 사용하고
노트북의 모델 폴더도 같은 경로로 지정하세요.
기본 `models/` 경로를 유지하며 새 버전을 받으려면 `download_models.py --upgrade`를 사용하세요.
새 번들 검증이 끝난 뒤 기존 모델을 `models/_previous/`에 백업하고 교체합니다.
