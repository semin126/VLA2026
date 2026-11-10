# 밀기 행동 계약 — 2026-11-10

밀기는 새 실행 인터페이스입니다. 기존 배포 모델 R5/motion7k가 이 지시를
학습했거나 수행한다는 뜻이 아닙니다. Claude가 수집·학습·평가하며,
검증한 후보를 사용자가 승인한 뒤 Codex가 models에 반영합니다.
이번 코드 변경에는 모델 교체나 새 성능 측정이 없습니다.

## 수집기와 실행기가 공유하는 함수

`vlapipe.planner.push_instruction(obj_name, direction)`은 영어 문자열 하나를 반환합니다.
`push_destination(current_xy, direction)`은 현재 월드 XY에서 7cm 이동한 목표 XY를 반환합니다.
고정 거리는 `config.PUSH_DISTANCE`, 물체별 밀기 한도는 `config.MAX_PUSH_ATTEMPTS`입니다.
좌표는 자동 클리핑하지 않습니다. 공간이 비었는지와 경계 안인지는 별도로 검증해야 합니다.

| direction | 좌표 변화 | object_3 지시문 |
|---|---|---|
| toward_robot | −y | `Push object 3 toward the robot.` |
| away | +y | `Push object 3 away from the robot.` |
| left | −x, agentview 이미지 왼쪽 | `Push object 3 to the left.` |
| right | +x, agentview 이미지 오른쪽 | `Push object 3 to the right.` |

존의 x 범위와 `scene.py`의 agentview 카메라 축을 기준으로 정했습니다.
마주보는 frontview의 좌우와 혼동하지 마세요. 학습과 추론에서 문구를 똑같이 씁니다.

```python
from vlapipe.planner import push_instruction, push_destination

label = push_instruction('object_3', 'toward_robot')
# 고전 제어기/수집기만 현재 정답 좌표를 읽습니다.
teacher_task = {'object': 'object_3', 'action': 'push',
                'push_to': push_destination(env.object_pos('object_3')[:2], 'toward_robot')}
teacher.set_subtask(teacher_task)
```

## VLM 출력과 실행

```json
{"assessments": [], "plan": [
  {"object": "object_3", "action": "push", "direction": "toward_robot", "reason": "visual evidence"},
  {"object": "object_3", "action": "transport", "target_zone": "right_zone", "reason": "strategy"}
], "done": false}
```

이는 형식 예시이며 object_3을 이 방향으로 밀어야 한다는 정답이 아닙니다.
운반은 `target_zone`, 밀기는 네 방향 중 `direction`이 필요합니다. 밀기에는
`target_zone`을 요구하지 않으며, `push_to` 좌표를 VLM이 출력하면 거부합니다.
방향 오타·누락을 고치거나 정답 좌표로 계획을 자동 수정하지 않습니다.
기존처럼 action을 생략한 plan 항목은 운반입니다.

B/C의 VLA에는 방향별 영어 지시문만 전달합니다. 현재 물체 정답 좌표는 전달하지 않습니다.
D-state teacher에 밀기 항목이 선택되면 그때 현재 XY에서 `push_to`를 계산합니다.
**기본 D 계획기와 기존 RulePlanner의 선택 규칙은 이번에 바꾸지 않았습니다.**
기본 D는 기존 운반 전용 기준선이며, 새로운 밀기 기준선은 별도 선택 규칙과 검증이 필요합니다.
A는 기존 전체 명령 한 줄과 정리 규칙만 받으며 VLM을 호출하지 않습니다.

## 시도와 피드백

- 운반: 물체마다 `TrialConfig.max_attempts`(기본 3)회, `summary.attempts`에 기록합니다.
- 밀기: 물체마다 2회, `summary.push_attempts`에 따로 기록합니다.
- 시도는 시작할 때 셉니다. 도중 재계획으로 중단되어도 사용한 시도입니다.
- 초과한 다음 항목을 시작하려 하면 운반은 `object_attempt_limit`, 밀기는
  `push_attempt_limit`로 trial을 종료합니다. 계획을 자동으로 건너뛰지 않습니다.
- 문맥의 `execution_attempts.transport/push`에는 전체 실행의 누적 횟수가 있습니다.
  최근 20개 history만 세면 오래된 시도나 중단된 시도를 놓칠 수 있습니다.
- `after_action.action`은 기존처럼 계획 항목 dict입니다. 그 안의 action/direction과
  새 `after_action.action_type`, `after_action.direction`을 볼 수 있습니다.
- 밀기 후에도 자동 운반을 하지 않습니다. 다음 계획에 운반이 있어야 합니다.
  고정 시간의 종료는 성공 판정이 아닙니다. 점수와 씬 완료 기준은 그대로입니다.

## 학생의 선택과 비교

`final_project_v2/student_strategy_push_example.py`는 명시적으로 선택하는 연결 예제입니다.
기존 C0/C1/C2와 공통 `starter_strategy.txt`는 운반 전용 실험용으로 유지합니다.
기존 C1의 완료 이미지·시도 계산을 그대로 밀기에 적용하면, 밀어서 중앙에 남긴 정상 동작을
실패로 의심하거나 밀기를 운반 횟수에 더할 수 있습니다. 새 전략에서 행동별로 판단하세요.
새 예제는 매 행동 후 재계획하는 단순 예시이며 우수성이 검증된 전략이 아닙니다.
예제의 운반 한도 3은 실행기의 설정을 바꿀 때 함께 맞춰야 합니다.

먼저 예상과 판단 근거를 적고 검증 씬에서 비교해 보세요.

- 언제 바로 옮기고, 언제 방해물을 먼저 옮기고, 언제 밀어야 할까요?
- 어느 쪽이 비었는지 이미지의 무엇으로 판단하나요? 그리퍼가 접근할 자리도 비었나요?
- 밀기로 공간을 만든 뒤 어떤 관찰이 있어야 운반을 시작할 수 있나요?
- 밀기는 성공했지만 정리 점수는 그대로일 수 있습니다. 무엇을 기록해야 할까요?
- 밀기 호출과 재계획에 드는 시간보다 운반 성공의 이득이 큰가요?

기존 transport 응답 파싱·액션 생성·시도 한도는 유지하지만, 공통 task/출력 계약과
문맥에는 새 필드가 추가되어 **VLM 프롬프트가 바뀝니다**. 과거 결과를 정확히 재현하려면
과거 코드 커밋·전략·모델·씬·예산을 함께 고정해야 합니다. 이전 결과 파일은 수정하지 않습니다.
기존 운반 전용 분석 도구로 밀기 횟수를 운반 성공률에 합산하지 마세요.
봉인 시험 씬으로 수집·전략 선택·모델 선택을 하지 않습니다.

## 검증 범위

`python scripts/check_push_contract.py`: CPU MuJoCo와 가짜 모델·빈 이미지로
파싱, 지시문, 시도 분리, 피드백, A 유지, D teacher 좌표 연결을 확인합니다.
GPU·체크포인트·실제 렌더링을 사용하지 않습니다. 물리 밀기 성공률은 검증하지 않습니다.
R1 수정 전후 기존 messy/seed3의 첫 운반 151스텝은 매 액션과 단계가 비트 단위로 같았습니다.
