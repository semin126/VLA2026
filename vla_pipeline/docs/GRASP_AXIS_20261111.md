# 집는 축 계약 — 2026-11-11

접근성은 두 단계로 생각합니다. 먼저 큐브를 집을 수 있는 축이 있는지 판단하고,
두 축 모두 어렵다면 이웃을 먼저 옮기거나 밀어 공간을 만드는 방법을 검토합니다.
한 축의 집기 실패만으로 물리적으로 집을 수 없다고 결론 내리지 마세요.

## 공유 지시문과 teacher

`vlapipe.planner.axis_instruction(name, zone, axis=None)`은 영어 문자열 하나를 반환합니다.
Claude 수집기 tools/axis_teacher.py의 기존 문구와 같습니다.

| grasp_axis | 의미 | object_6, left_zone 지시문 |
|---|---|---|
| left-right | 손가락이 월드 x, agentview 이미지 좌우로 벌어짐 | `Pick up the cube number 6 with the fingers left-right and place it in the left storage zone.` |
| front-back | 손가락이 월드 y, 로봇 앞뒤로 벌어짐 | `Pick up the cube number 6 with the fingers front-back and place it in the left storage zone.` |
| 생략 | 기존 grasp_pose와 지시문 | `Pick up the cube number 6 and place it in the left storage zone.` |

이는 팔의 이동 방향이나 큐브의 긴 변을 말하는 것이 아니라 손가락 개폐축입니다.
임의로 world yaw=0 또는 pi/2로 고정하지 않고, 현재 큐브의 면 축 네 개 중 지정한
월드 축에 가장 가까운 것을 씁니다. `grasp_axis_yaw(cube_yaw, axis)`는 Claude의
phys_access2.face_yaw와 동률 선택 순서까지 같습니다.

```python
from vlapipe.planner import axis_instruction

label = axis_instruction('object_6', 'left_zone', 'front-back')
teacher.set_subtask({'object': 'object_6', 'action': 'transport',
                     'target_zone': 'left_zone', 'grasp_axis': 'front-back'})
```

teacher는 set_subtask 및 하강 직전 _refresh_pick에서 축을 반영합니다.
앞으로 공유 subtask 키를 쓰면 env.grasp_pose를 프로세스에서 감쌀 필요가 없습니다.
수집기는 위 label을 학습 지시문으로 사용하고, teacher의 subtask와 같은 축을 기록해야 합니다.
축을 생략하면 기존 env.grasp_pose와 운반 액션을 유지합니다.
직사각형 등 기존 색 물체에는 새 숫자 큐브 지시문을 생성하지 않습니다.

## VLM 응답과 runner

```json
{"assessments": [
  {"object": "object_6", "accessible": true, "blockers": [],
   "free_axes": ["front-back"], "reason": "visual evidence"}
], "plan": [
  {"object": "object_6", "action": "transport", "target_zone": "left_zone",
   "grasp_axis": "front-back", "reason": "strategy"}
], "done": false}
```

이는 형식 예시입니다. 물체와 축은 이미지에서 판단하세요.
transport의 grasp_axis와 assessment의 free_axes는 생략할 수 있습니다.
free_axes는 두 축의 부분집합이며 빈 리스트도 허용합니다. 중복·알 수 없는 축·문자열 이외 값은 거부합니다.
grasp_axis를 지정하면 두 문자열 중 하나여야 합니다. null과 오타는 거부합니다.
push에 grasp_axis를 붙이는 것도 거부합니다. 자동으로 축을 보충하거나 교정하지 않습니다.
accessible/free_axes의 시각 판단을 정답 라벨로 바꾸지 않습니다.

B/C의 VLA에는 공유 함수의 영어 지시문을, D teacher에는 같은 grasp_axis를 전달합니다.
after_action.action 안의 계획 항목과 after_action.grasp_axis로 축을 추적할 수 있습니다.
축을 바꿔도 운반 시도는 같은 물체의 summary.attempts에 더해집니다.
축마다 별도의 운반 시도 횟수를 두지 않습니다. 밀기는 기존 별도 한도 2회를 유지합니다.
A의 전체 지시·VLM 미사용은 유지합니다. 기본 D의 축 선택 규칙은 변경하지 않습니다.

## 학생이 검토할 질문

- 이미지에서 한 축만 막힌 것과 두 축 모두 막힌 것을 무엇으로 구분하나요?
- 큐브가 기울어져 있으면 left-right 지시는 실제로 어느 면 축을 선택하나요?
- 방향만 바꾸는 것과 이웃을 먼저 옮기는 것의 시간·성공률은 어떻게 비교하나요?
- 두 축이 모두 막혔다고 판단했을 때 어느 물체를 어느 쪽으로 밀어야 하나요?

밀기 전략 연결 예제 student_strategy_push_example.py에도 축 선택 안내를 추가했습니다.
기존 C0/C1/C2와 starter_strategy는 유지합니다. 비교할 때 B와 C가 같은 초기 프롬프트를
쓰는지 확인하고, 축 계약 추가 전후 실험을 같은 모델·프롬프트 실험으로 합치지 마세요.
모델 배포와 GPU 성능 검증은 별도입니다. 새 문구가 기존 모델에 학습됐다고 가정하지 않습니다.

## CPU 검증 범위

check_grasp_axis_contract.py: 문구, 축 변환, teacher 재선택, schema 검사.
check_push_contract.py: 실제 CPU runner에 가짜 모델·빈 이미지를 써서 축 지시·피드백·D 전달 확인.
두 검사 모두 실제 영상·학습 모델·GPU를 쓰지 않으며 집기 성능을 측정하지 않습니다.
