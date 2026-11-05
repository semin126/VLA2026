# 직접 정리 시연을 모으기 전에

조교가 제공한 저장·학습 코드를 사용하되, 이번에 모을 장면과 실패 조건의 이유는 여러분이 정합니다. HDF5를 많이 만드는 것보다 task와 실제 동작이 맞는 데이터를 모으는 것이 먼저입니다.

1. 조이스틱이 없으면 `--input keyboard-gamepad`로 공통 패드 입력 경로를 사용합니다. W/S앞뒤, A/D좌우, Q/E높이, Z/C회전, SPACE그리퍼, ENTER녹화, SHIFT정밀입니다. 카메라 뷰 창에 포커스를 두세요.
2. 실물 조이스틱은 `--probe-gamepad --seconds 10`으로 장치·축·버튼을 확인하고, 필요하면 조교와 `--calibrate`를 실행합니다. 가상 입력 시험이 실제 USB 매핑의 증거는 아닙니다.
3. 기본 학습 데이터는 `--direction tidy`로 직접 정리하는 시연입니다. 역순 영상은 실제 실행 가능한 정리 시연을 보장하지 않습니다.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py scripts/teleop_record.py --input keyboard-gamepad --direction tidy --objects number --count 10 --task-level subtask --object object_10 --target-zone left_zone --layout messy --seed 3 --episodes 1 --operator student --out outputs/my_tidy
```

실물 기기로는 `--input gamepad --gamepad-map my_gamepad_map.json`으로 바꿉니다. 녹화 시작 후 물체를 실제로 잡아 들어 옮기고, 목표 구역에 놓고, 그리퍼를 열고, 물체가 멈춘 뒤 저장합니다. reset이나 장면 편집으로 물체를 옮긴 결과는 시연이 아닙니다.

각 실행은 새 폴더를 만듭니다. candidates는 기계 검사 통과 후보이고 rejected는 실패 기록입니다. 실패 기록도 지우지 마세요. 후보를 조교와 검토한 뒤 `review_teleop.py`로 새 검토본을 만듭니다. 검토 전 후보는 학습 준비 단계에서 거부됩니다.

저장 영상을 먼저 확인하세요.

```powershell
& $py scripts/preview_episode.py 'outputs/실제_수집폴더/candidates/ep_실제번호.hdf5' --camera agentview --stride 2 --out outputs/my_review/demo_top.gif
```

`--camera robot0_eye_in_hand`로 손목 영상도 볼 수 있습니다. 기본은 전체 시연이며, `.preview.json`의 `contains_final_observation=true`와 `truncated=false`를 확인해 마지막 놓기까지 검토하세요. 저장 영상을 보는 것과 액션을 물리적으로 재실행하는 것은 다른 검사입니다.

각 파일에 대해 다음을 기록하세요: 원본 장면 seed, 수집자, task, 실제 움직인 물체, 성공/실패와 이유, 프레임 수, 다른 물체에 준 영향. 같은 seed의 재시도는 서로 독립된 장면이 아닙니다.

학습 데이터가 준비되면 조교 안내의 `prepare_vla_training.py`→`train_vla.py`를 사용합니다. 여러분이 결정할 일은 어떤 실패를 줄이려는지, 어떤 장면을 학습에서 빼고 시험할지, 데이터가 늘면 어떤 지표가 달라질 것인지입니다.
