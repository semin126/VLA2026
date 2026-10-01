# AGENTS.md — `vla_pipeline` 에서 일하는 AI 에이전트를 위한 안내

> 이 폴더는 **단독 배포본**입니다. 상위 저장소 없이 이것만 있어도 돌아갑니다.
> 사용자는 대전과학고 2026 심화자율연구에 참여하는 **고등학생**입니다.
> 마지막 검증: 2026-09-03, 윈도우 11 / Python 3.14.3 / RTX 3090 Ti

---

## 1. 무엇을 하는 코드인가

계층적 VLA 파이프라인의 실행 코드입니다.

```
이미지 + 자연어 명령 ("테이블을 정리하시오")
  │
  ├─ [1단계 VLM]  SmolVLM2  : 명령을 순차 지시문으로 분해
  │                           "빨간 블록을 집어 오른쪽 보관 구역에 놓으시오"
  └─ [2단계 VLA]  SmolVLA   : 지시문 + 이미지 → 7차원 관절 제어
```

물리 엔진 MuJoCo, 로봇 Franka Emika Panda.
관측·액션 규격은 **LIBERO(robosuite OSC_POSE) 호환**입니다.

연구 가설: 단일 end-to-end VLA 는 가림·밀집·접근성 제약에 취약하다.
"무엇을 할 것인가"와 "어떻게 할 것인가"를 분리하면 복잡한 씬에서 더 안정적이다.

---

## 2. 먼저 이것부터

**어떤 작업이든 시작 전에 환경이 살아 있는지 확인하세요.**

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py     # 설치 상태
.\.venv\Scripts\python.exe scripts\smoke_test.py    # 실제 동작 (1~2분)
```

`smoke_test.py` 는 씬·IK·녹화·역재생·데이터규격을 사람 조작 없이 끝까지 확인합니다.
**코드를 고친 뒤에도 이걸로 회귀 확인하세요.** 5/5(또는 `--vlm` 시 6/6)가 나와야 합니다.

`.venv` 가 없으면 `SETUP_STUDENT.md` 3절대로 만들면 됩니다.

---

## 3. 절대 바꾸면 안 되는 것 — LIBERO 호환 규격

```
action = [dx, dy, dz, drx, dry, drz, gripper]      # 7차원
  [0:3] 위치 델타,   [-1,1] → 최대 ±0.05 m
  [3:6] 회전 델타(축각), [-1,1] → 최대 ±0.5 rad
  [6]   그리퍼, -1 = 열기 / +1 = 닫기
  제어 20 Hz (시뮬 timestep 0.002 × 25 스텝)

관측: agentview_image, robot0_eye_in_hand_image,
     robot0_joint_pos(7), robot0_gripper_qpos(2),
     robot0_eef_pos(3), robot0_eef_quat(4)
state 벡터 = 뒤 4개를 이어붙인 16차원
```

나중에 LIBERO 사전학습 가중치를 붙일 수 있도록 맞춰 둔 것입니다.
**여기를 바꾸면 그 길이 막힙니다.** 학생이 "차원을 늘리자"고 하면 이 이유를 설명하세요.

수치는 전부 `vlapipe/config.py` 에 있습니다. 다른 파일에 상수를 새로 박지 마세요.

### 녹화되는 카메라는 2대뿐입니다 (자주 혼동됨)

| | |
|---|---|
| **저장됨** | `agentview`(top-down) · `robot0_eye_in_hand`(손목) — `C.CAMERA_NAMES` |
| 쓸 수 있는 카메라 전체 | `agentview` · `frontview` · `sideview` · `robot0_eye_in_hand` — `C.ALL_CAMERAS` |
| 텔레옵 뷰 창(`camview.py`) | **보기 전용.** 무엇을 띄우든 저장물은 안 바뀜 |

에피소드 HDF5 에는 이미지 2종 + state 16차원 + `actions` + **`qpos`(전체 상태)** +
`target_pos`/`target_quat` 가 들어갑니다. `qpos` 덕분에 **나중에 `frontview`·`sideview` 를
원하는 해상도로 다시 렌더할 수 있습니다** — 발표 영상 때문에 다시 녹화하지 마세요.

사용자가 "카메라를 늘리자"고 하면, **보기용인지 학습용인지 먼저 물어보세요.**
보기용이면 `--panel-cameras`, 학습용이면 LIBERO 스키마가 깨진다는 점을 알려야 합니다.

---

## 4. 설계 결정 — 이유를 모르면 고치다 망가뜨립니다

1. **LIBERO 를 robosuite 째 설치하지 않았다.** robosuite 는 윈도우 비공식 지원이라
   학생 노트북에서 깨집니다. 대신 액션/관측/데이터셋 **스키마만** 동일하게 맞췄습니다.
   → "LIBERO 를 설치하자"는 제안은 거절하세요.

2. **주 카메라 `agentview` 는 top-down 이다.** 로봇을 마주보는 카메라는 영상 좌우가
   월드 좌우와 **뒤집힙니다.** 그대로 학습하면 VLM 이 말하는 "왼쪽"과 로봇의 "왼쪽"이
   반대가 됩니다. 마주보는 시점은 `frontview` 로 남겨 뒀습니다(발표 영상용).

3. **중력 보상은 필수다.** 없으면 위치 제어 정상상태 오차가 6~16 mm 생겨 파지가
   어긋납니다. `env._sim_step()` 에서 `qfrc_applied[팔+손가락] = qfrc_bias`.

4. **단계별 속도 제한이 있다.** 액션을 최대치로 포화시키면 이송 중 관성으로 물체를
   놓칩니다. `policy.PHASE_SPEED` 의 carry 구간 0.30 배가 그 때문입니다.

5. **역재생 수학.** 위치·회전 델타는 부호 반전이면 됩니다: `a'[k] = -a[T-1-k]`.
   그리퍼는 **절대 명령이라 부호 반전이 안 됩니다.** "역재생 기준 다음 프레임에서
   손가락이 닫혀 있는가"로 다시 만듭니다. 그래야 원본의 *놓기* 가 역재생의 *잡기* 가 됩니다.
   낙하·투척 구간은 `reverse.check_reversibility()` 가 걸러냅니다.

6. **윈도우 한글 경로 우회.** MuJoCo 의 C 로더는 UTF-8 경로를 CP949 로 잘못 읽어
   한글 경로에서 실패합니다. `config.mujoco_path()` 가 ASCII 정션을 만들어 우회합니다.
   → **MuJoCo 에 경로를 넘기는 코드를 새로 쓸 때는 반드시 `C.mujoco_path()` 를 거치세요.**

7. **텔레옵 키 입력을 MuJoCo 뷰어에서 받지 않는다.** 뷰어(simulate)는 알파벳 대부분에
   렌더링 토글 단축키를 갖고 있고, 우리 조작키 11개가 **전부** 충돌합니다
   (`W`=Wireframe, `D`=Static Body, `S`=Shadow, `Q`=Camera, `E`=Equality, `Z`=Light,
   `C`=Contact Point, `R`=Reflection, `X`=Texture, `P`=Contact Split, `A`=Auto Connect).
   뷰어에서 WASD 를 누르면 로봇이 안 움직이고 화면이 깨집니다. 키 처리는 C++ 쪽이라
   파이썬에서 막을 수 없습니다.
   → **키는 `camview.CameraPanel`(pygame 창)에서 받습니다. `launch_passive` 에
   `key_callback` 을 넘기지 마세요.** 되돌리면 그 버그가 그대로 돌아옵니다.
   목록 확인: `mujoco.mjVISSTRING` / `mujoco.mjRNDSTRING` 의 3번째 필드.
   뷰어에서 실수로 켠 mjVIS 플래그는 매 프레임 복원합니다(`viewer.opt.flags`).
   mjRND(Wireframe/Shadow/Reflection)는 `scn` 에 접근할 수 없어 복원 불가입니다.

8. **텔레옵 z 축은 기본 배율이 낮다 (`teleop.AXIS_GAIN['gain_z'] = 0.35`).**
   한 제어 스텝의 최대 이동이 `MAX_DPOS = 0.05 m` 라, speed 0.55 면 스텝당 27.5 mm 로
   **타일 높이(30 mm)와 맞먹어 물체를 지나칩니다.** z 를 해트에 매핑한 기기는 값이
   `-1/0/+1` 뿐이라 더 심합니다. 실측: 1.0 → 23.2 mm, 0.35 → 8.1 mm, 정밀 모드 → 2.0 mm.
   `--speed-z` 또는 매핑 파일의 `gain_z` 로 조절. 정밀 모드(`SLOW_SCALE = 0.25`)는
   키보드 SHIFT / 패드 `btn_slow` 를 **누르고 있는 동안**만 적용됩니다.
   → 사용자가 "잡기 어렵다"고 하면 여기부터 보세요. IK 나 제어기 문제가 아닙니다.

9. **Colab 노트북은 공유 드라이브에서 바로 돌지 않는다.** `session12` 의 `0-2` 셀이
   `MyDrive/VLA_Team_2026/vla_pipeline` 을 `/content/vla_pipeline` 으로 복사한 뒤
   거기서 작업합니다. 공유 폴더에서 바로 돌리면 (a) `SortingEnv` 가 매번 새로 쓰는
   `assets/.../vla_scene.xml` 을 여러 명이 동시에 덮어써 `ParseXML` 오류가 나고,
   (b) 학습 체크포인트가 GB 단위로 공유 드라이브에 쌓이며, (c) Drive FUSE 가 느립니다.
   → **`os.chdir` 를 공유 경로로 되돌리지 마세요.** 배포는 `make_student_zip.py` 결과물을
   `VLA_Team_2026/` 아래에 푸는 방식입니다(폴더째 업로드 금지 — `.venv` 가 딸려 갑니다).

10. **게임패드는 SDL 배경 이벤트가 필요하다.** pygame 창이 열려 있으면 SDL 은 포커스가
   없을 때 조이스틱 이벤트를 주지 않습니다. `teleop.py`·`camview.py` 가 임포트 시점에
   `SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS=1` 을 설정합니다. **pygame.init() 보다 먼저여야
   하므로 함수 안으로 옮기지 마세요.**

---

## 5. 파일 지도

```
vla_pipeline/
├── SETUP_STUDENT.md      학생용 설치 안내 (설치 질문은 여기로)
├── AGENTS.md             이 파일
├── README.md             상세 문서 — 규격·실측값·문제해결
├── setup_assets.py       Panda 에셋 다운로드 + 손목카메라/TCP 패치 (보통 불필요)
├── vlapipe/
│   ├── config.py         ★ 모든 수치 (씬 규격·액션 규격·경로·한글경로 우회)
│   ├── scene.py          MJCF 생성, tidy/messy/random/cluttered 배치
│   ├── accessibility.py  가림·도달 판정 — 과제명의 '물체 접근성' 구현부
│   ├── metrics.py        정량 지표 5군 + Wilson CI + 짝지은 부트스트랩
│   ├── controller.py     감쇠최소자승 IK + OSC 유사 제어기
│   ├── env.py            SortingEnv — reset/step/render/성공판정
│   ├── policy.py         ScriptedSortingPolicy(teacher) · SmolVLAPolicy
│   ├── planner.py        RulePlanner(정답) · VLMPlanner(SmolVLM2) · 채점
│   ├── teleop.py         게임패드/키보드 입력, 캘리브레이션, 배치 저장·불러오기
│   ├── camview.py        텔레옵용 다중 카메라 뷰 창 (보기 전용, 녹화와 무관)
│   ├── episode.py        HDF5 녹화 포맷
│   ├── reverse.py        역재생 변환 + 역재생 불가 구간 검출
│   └── to_lerobot.py     LeRobot 데이터셋 변환·검증
├── scripts/
│   ├── check_env.py      설치 점검
│   ├── smoke_test.py     ★ 전 구간 자동 검증 (에이전트가 쓰기 좋음)
│   ├── run_pipeline.py   엔드투엔드 실행
│   ├── teleop_record.py  텔레옵 녹화 (GUI — 사람이 필요)
│   ├── build_dataset.py  역재생 + 검증 + LeRobot 변환
│   ├── make_student_zip.py  학생 배포용 zip 만들기 (조교용)
│   ├── run_isolated.sh   리눅스 전용 — 윈도우에서는 무시
│   └── check_server.sh   리눅스 전용 — 윈도우에서는 무시
├── notebooks/            Colab 학습 노트북
├── assets/               Panda 에셋 (패치본 panda_vla.xml 포함)
├── data/                 녹화 데이터 (없을 수 있음)
└── outputs/              실행 결과물
```

---

## 6. 실측값 — 추측한 값과 섞지 마세요

**전부 이 코드로 실제로 잰 값입니다.** 사용자가 수치를 물으면 여기 있는 것만 쓰고,
없으면 "재 보지 않았다"고 하거나 직접 돌려서 재세요.

### 윈도우 PC (2026-09-03, Python 3.14.3 / RTX 3090 Ti)
| 항목 | 값 |
|---|---|
| `run_pipeline --layout messy --seed 3` (규칙 플래너) | 서브태스크 5/6, 정리완료 True, 12.4초 |
| `run_pipeline --layout cluttered --seed 3` (가림 있음) | 서브태스크 8/9, 정리완료 True, 13.1초 |
| `run_pipeline --planner vlm` (messy seed 3) | 서브태스크 9/9, 정리완료 **False**, 54.4초, 계획 정답 1/5 |
| SmolVLM2-500M 로드 / 추론 | 9.8초 / 2.0초 (CUDA) |
| 〃 CPU 폴백 (`CUDA_VISIBLE_DEVICES=-1`) | 9.1초 / **10.6초** (float32). 죽지 않고 돌아감 |
| 노트북 채점 루프 (messy, seed 3·7·11·19·23) | recall **0.19** / precision **0.16**, 5씬 17.3초 |
| 역재생 액션 적분 오차 | **0.000 mm** |
| `smoke_test.py` | 5/5 (`--vlm` 시 6/6) |
| 한글 경로에서 재실행 | 5/5 통과 (정션 우회 동작 확인) |

### 리눅스 데스크톱 (앞선 측정, RTX 4080)
| 항목 | 값 |
|---|---|
| IK 수렴 | 목표점 60개 중 57개. 도달 하한 접근높이 0.30 m |
| 제어 — 가림 없음(`messy`) | 운반 17/20 (85%) · 완전정리 4/4 |
| 제어 — 가림 있음(`cluttered`) | 운반 20/34 (59%) · 완전정리 1/5 · 밀기 11/12 |
| VLM 계획 | recall 0.23 (500M) / 0.31 (2.2B, 55초) |

> ⚠️ **위 제어 수치는 씬 5개 기준이라 통계적으로 의미가 없습니다.**
> `4/5` 와 `1/5` 의 95% 신뢰구간이 겹칩니다. 본 실험은 조건당 30씬 + 동일 시드입니다.
> 학생이 이 숫자로 결론을 내려 하면 이 점을 반드시 짚어 주세요.

### 아직 아무도 끝까지 못 돌린 것
- **SmolVLA 파인튜닝** — 데이터가 없어 미실행
- **LeRobot 데이터셋 실제 변환** — 프레임 구조까지만 검증. Colab 확인 필요
- **뷰어 GUI 실조작** — 창은 뜨지만 사람이 실제로 몰아 봐야 확인됩니다(자동 검증 불가).
  키 입력은 뷰어 창 포커스가 있어야 하고, 패드 입력은 포커스와 무관합니다
  (pygame 이 SDL 조이스틱을 직접 읽기 때문 — 첫 수업에서 확인할 것)

### 게임패드 매핑 — 확정됨 (2026-09-03 캘리브레이션 완료)

수업용 **Thrustmaster T.16000M**(비행 스틱, 축 4·버튼 16·해트 1)의 매핑은
`gamepad_map.json` 에 있고, `--gamepad-map` 을 생략해도 자동으로 적용됩니다.

```
x = 축 0        y = 축 1(반전)      yaw = 축 2(반전, 트위스트)
z = hat0_y      (엄지 십자키)
그리퍼 = 버튼 0(트리거)   녹화 = 15   리셋 = 9   폐기 = 4
```

**z 를 축 3(스로틀 레버)으로 되돌리지 마세요.** 스프링이 없어 손을 떼도 제자리에
머무는데 축은 전부 속도 명령이라, 로봇이 멈추지 않고 계속 오르내립니다.
`--calibrate` 가 이런 축을 감지하면 경고하고 해트로 바꿀지 물어봅니다.

---

## 7. VLM 정확도가 낮은 것은 버그가 아닙니다

SmolVLM2 는 이 과제를 제대로 못 합니다. **이건 고칠 버그가 아니라 연구 문제입니다.**

관찰된 것:
1. **2.2B 는 값어치를 못 한다** — recall 0.23 → 0.31 로 조금 오르는 대신 추론이 40배 느려짐
2. **2.2B 는 이미지를 거의 안 본다** — 씬을 바꿔도 출력이 거의 같음. 프롬프트 규칙을 베낌
3. **`lines` 프롬프트는 실패했다** — 한 줄만 쓰고 멈춤. JSON 보다 나빴음
   (추측이 실측으로 뒤집힌 사례. 기본값이 `json` 인 이유)
4. **틀리는 방식이 한쪽으로 쏠린다** (500M, messy 5씬, 2026-09-03 측정).
   5씬 합계로 **없는 물체 13개**(이미 제자리에 있는 것까지 옮기라고 함),
   **구역 틀림 6개**, **빠뜨림 11개**. seed 23 은 아예 한 줄만 내고 멈췄습니다.
   구역 배정도 "긴 블록=오른쪽 / 정육면체·타일=왼쪽" 규칙이 아니라 목록 앞뒤로
   갈라 놓는 식이었습니다 → **이미지도 규칙도 제대로 안 보고 있다**는 근거.
   `plan_accuracy` 가 돌려주는 `missed`/`hallucinated`/`wrong_zone` 로 확인할 수 있습니다.

학생이 "VLM 이 틀리는데 고쳐 달라"고 하면, **고쳐 주지 말고** 위 사실을 알려 주고
어느 길로 갈지 같이 정하세요:
1. 프롬프트 개선 (`vlapipe/planner.py` 의 `_PROMPT`)
2. 문제 쪼개기 — "가운데에 무엇이 있나?"만 묻고 목적지는 규칙으로
3. **이 과제만 VLM 파인튜닝** — 위 관찰(특히 2번)이 "왜 파인튜닝이 필요한가"의 근거

`--planner rule` 로 언제든 정답 플래너를 쓸 수 있으므로, VLM 정확도와 무관하게
2단계(VLA) 연구는 계속 진행할 수 있습니다.

---

## 8. 작업 규약

1. **실측값만 보고하세요.** "될 것 같다"가 아니라 "돌려 봤더니 이렇다".
   검증 못 한 것은 검증 못 했다고 명시하세요. 이 연구의 기본 원칙입니다.
2. **코드를 고쳤으면 `smoke_test.py` + `run_pipeline.py` 로 회귀 확인**하세요.
3. **3절 규격과 4절 설계 결정은 사용자 승인 없이 바꾸지 마세요.**
4. `scripts/*.sh` 두 개는 리눅스 전용입니다. 윈도우에서는 필요 없습니다.
5. `data/` 는 손으로 모은 연구 데이터입니다. **덮어쓰거나 지우지 마세요.**
   테스트 결과물은 `outputs/` 에 쓰세요.
6. 사용자는 고등학생입니다. 막히면 대신 해 주기보다 **어디를 보면 되는지** 알려 주세요.
   다만 환경 설치 문제는 그냥 해결해 주는 편이 낫습니다(연구의 본질이 아니므로).

---

## 9. 자주 쓰는 명령

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py                              # 설치 점검
.\.venv\Scripts\python.exe scripts\smoke_test.py                             # 전 구간 검증
.\.venv\Scripts\python.exe scripts\smoke_test.py --vlm                       # VLM 포함
.\.venv\Scripts\python.exe scripts\run_pipeline.py --layout messy --seed 3   # 엔드투엔드
.\.venv\Scripts\python.exe scripts\run_pipeline.py --planner vlm             # VLM 플래너로
.\.venv\Scripts\python.exe scripts\teleop_record.py --input keyboard         # 텔레옵 (GUI)
.\.venv\Scripts\python.exe scripts\teleop_record.py --input keyboard --panel-cameras agentview,frontview,sideview,robot0_eye_in_hand --panel-cols 2
.\.venv\Scripts\python.exe scripts\teleop_record.py --calibrate              # 패드 축 확인
.\.venv\Scripts\python.exe scripts\build_dataset.py                          # 역재생 + 검증
```

`--layout` 은 `tidy` / `messy` / `random` / `cluttered`(가림 있음) 중에서 고릅니다.
`--seed` 를 바꾸면 배치가 달라집니다. 비교 실험은 **반드시 같은 시드**로 하세요.
