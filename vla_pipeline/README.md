# VLA 계층적 파이프라인 — 실행 코드

2026 대전과학고 심화자율연구 / NMAIL

```
이미지 + 자연어 명령
   └─[1단계 VLM]  SmolVLM2 가 명령을 순차 지시문으로 분해
        └─[2단계 VLA]  SmolVLA 가 지시문+이미지를 받아 관절 제어
```

물리 환경은 MuJoCo, 로봇은 Franka Emika Panda(MuJoCo Menagerie),
관측·액션 규격은 **LIBERO(robosuite OSC_POSE) 호환**입니다.

---

> **처음 설치하는 학생은 [SETUP_STUDENT.md](SETUP_STUDENT.md) 를 보세요.**
> 이 문서는 설치가 끝난 사람을 위한 것입니다.
> AI 에이전트로 작업한다면 [AGENTS.md](AGENTS.md) 입니다.

## 0. 명령을 어떻게 읽나 (윈도우 기준)

이 문서의 명령은 **PowerShell** 에서, **`vla_pipeline` 폴더 안에서** 실행합니다.
VSCode 로 이 폴더를 열고 `` Ctrl + ` `` 로 터미널을 여는 것이 가장 편합니다.

지금 어디에 있는지 확인:

```powershell
Get-Location            # ...\vla_pipeline 로 끝나야 합니다
Test-Path .\.venv\Scripts\python.exe    # True 여야 합니다
```

`False` 가 나오면 폴더를 잘못 잡았거나 가상환경을 아직 안 만든 것입니다
([SETUP_STUDENT.md](SETUP_STUDENT.md) 3절).

### 왜 `.\.venv\Scripts\python.exe` 를 매번 쓰나

윈도우에서 그냥 `python` 을 치면 **다른 파이썬이 잡히거나 Microsoft Store 안내창이 뜹니다.**
가상환경 안의 파이썬을 확실히 쓰려고 전체 경로로 적습니다. 그대로 복사해 붙여 넣으면 됩니다.

매번 치기 번거로우면 변수에 담아 두세요. **터미널을 새로 열 때마다 다시 해야 합니다.**

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py scripts\check_env.py
```

이 문서에서는 오해가 없도록 항상 전체 경로로 적습니다.

---

## 1. 5분 안에 돌려보기

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
.\.venv\Scripts\python.exe scripts\check_env.py
.\.venv\Scripts\python.exe scripts\smoke_test.py
.\.venv\Scripts\python.exe scripts\run_pipeline.py --layout messy --seed 3
```

`py -3.14` 가 없다고 나오면 자기 버전으로 바꾸세요 (`py -3.12 -m venv .venv`).
Python 3.12 이상이면 됩니다.

| 명령 | 무엇을 보나 | 걸리는 시간 |
|---|---|---|
| `check_env.py` | 패키지·에셋이 **설치돼 있나** | 5초 |
| `smoke_test.py` | 파이프라인이 **실제로 도나** (5/5 나와야 정상) | 1~2분 |
| `run_pipeline.py` | 눈으로 확인 — `rollout.gif` 생성 | 12초 |

`setup_assets.py` 는 **보통 필요 없습니다.** 패치본 에셋(`panda_vla.xml`)이 함께 들어
있습니다. 에셋을 지웠을 때만 git 을 깔고 돌리세요.

결과는 `outputs\run_messy3_rule_scripted\` 에 `before.png`, `after.png`,
`rollout.gif`, `report.json` 으로 남습니다.

```powershell
Invoke-Item outputs\run_messy3_rule_scripted\rollout.gif    # 기본 앱으로 열기
```

### 자주 쓰는 실행 옵션

```powershell
# 배치 바꾸기 — tidy / messy / random / cluttered(가림 있음)
.\.venv\Scripts\python.exe scripts\run_pipeline.py --layout cluttered --seed 3

# 시드 바꾸기 (같은 배치를 다시 만들고 싶으면 같은 시드를 쓰세요)
.\.venv\Scripts\python.exe scripts\run_pipeline.py --layout messy --seed 7

# VLM 플래너로 (torch 필요 — 6절)
.\.venv\Scripts\python.exe scripts\run_pipeline.py --planner vlm

# 영상 없이 빠르게
.\.venv\Scripts\python.exe scripts\run_pipeline.py --no-video
```

> **비교 실험은 반드시 같은 시드로.** 시드가 다르면 배치가 달라져서
> 무엇 때문에 결과가 바뀌었는지 알 수 없게 됩니다.

---

## 2. 씬 규격 (260827 문서 기준)

| 항목 | 값 |
|---|---|
| 작업 영역 | 0.8 m (가로) × 0.5 m (세로), 3 × 1 분할 |
| 구역 | `left_zone`(초록) · `center_zone`(올리브) · `right_zone`(파랑) |
| 로봇 | Franka Panda, 한 변의 중심에서 살짝 뒤 (`(0, -0.12, 0.80)`) |
| 물체 | 10개 — 0.06×0.04×0.04 ×5, 0.05³ ×3, 0.04×0.04×0.03 ×2 |
| 카메라 | `agentview`(상단) · `frontview` · `sideview` · `robot0_eye_in_hand`(손목) |
| 조명 | 주광(로봇 뒤 상단 45°, 그림자 O) + 보조광(반대편 30%, 그림자 X) |

**정돈 상태 정의**: 긴 블록 5개 → 오른쪽 보관 구역, 정육면체·타일 5개 → 왼쪽 보관 구역,
가운데 작업 구역은 비어 있음.

### 카메라 배치에서 바꾼 것 하나

문서의 "로봇을 마주보는 면 상단" 카메라(`frontview`)는 **영상의 좌우가 월드 좌우와 뒤집힙니다.**
그 시점으로 학습하면 VLM 이 말하는 "왼쪽"과 로봇 기준 "왼쪽"이 반대가 됩니다.
그래서 학습·추론용 주 카메라 `agentview` 는 **테이블 바로 위(top-down)** 에 두었습니다
(영상 오른쪽 = 월드 +x, 영상 위 = 월드 +y). `frontview` 는 발표용 데모 영상으로 남겨 두었습니다.

---

## 3. 관측 / 액션 규격 (LIBERO 호환 — 바꾸지 마세요)

**액션 (7차원)** — `[dx, dy, dz, drx, dry, drz, gripper]`
- 앞 3개: 월드 좌표 위치 델타, `[-1,1]` → 최대 ±0.05 m
- 중간 3개: 월드 좌표 회전 델타(축각), `[-1,1]` → 최대 ±0.5 rad
- 마지막: 그리퍼, **−1 = 열기 / +1 = 닫기**
- 제어 주기 20 Hz (시뮬 timestep 0.002 s × 25)

**관측**
```
agentview_image            (H, W, 3) uint8
robot0_eye_in_hand_image   (H, W, 3) uint8
robot0_joint_pos           (7,)      robot0_gripper_qpos (2,)
robot0_eef_pos             (3,)      robot0_eef_quat     (4,)
```
`state` 벡터는 위 4개를 이어붙인 16차원입니다.

내부적으로는 감쇠 최소자승 IK 로 델타 자세를 관절 목표로 바꾸고,
실제 Franka 제어기처럼 **중력 보상**을 겁니다(안 하면 정상상태 오차가 6~16 mm 생겨
파지가 어긋납니다).

수치는 전부 `vlapipe/config.py` 에 있습니다. 직접 확인해 보려면:

```powershell
.\.venv\Scripts\python.exe -c "from vlapipe import config as C; print(C.ACTION_DIM, C.STATE_DIM, C.CONTROL_HZ, C.MAX_DPOS, C.MAX_DROT)"
```

---

## 4. 데이터 수집 — 어지르고 나서 뒤집기

정확한 위치에 **내려놓기**는 손으로 하기 어렵지만, 구역에서 **집어서 아무 데나 놓기**는 쉽습니다.
그래서 정돈된 씬을 어지르는 쪽을 녹화하고, 그걸 시간 역순으로 뒤집어 정돈 데모로 씁니다.

```powershell
.\.venv\Scripts\python.exe scripts\teleop_record.py --input keyboard             # 패드 없을 때
.\.venv\Scripts\python.exe scripts\teleop_record.py --input gamepad --episodes 5 # 패드
```

창이 **두 개** 뜹니다. 조작하는 **MuJoCo 뷰어**와, 고정 카메라를 모아 보는 **카메라 뷰**입니다.
**로봇은 저절로 움직이지 않습니다** — 넣는 만큼만 20 Hz 로 움직입니다.

### 카메라 뷰 창 — 팔에 가려 안 보일 때

뷰어는 시점이 하나뿐이라 **팔이 물체를 가리면 무엇을 집는지 보이지 않습니다.**
시점을 돌리면 이번엔 조작 방향 감각이 틀어집니다. 그래서 조작은 뷰어에서 하고,
확인은 고정 카메라 여러 대를 동시에 보는 편이 낫습니다.

기본으로 `agentview`(위) · `frontview`(앞) · `wrist`(손목) 세 개가 뜹니다.

```powershell
# 옆에서 보는 시점 추가 — 높이(집는 깊이)를 판단하기에 sideview 가 가장 좋습니다
.\.venv\Scripts\python.exe scripts\teleop_record.py --input keyboard `
    --panel-cameras agentview,frontview,sideview,robot0_eye_in_hand --panel-cols 2

.\.venv\Scripts\python.exe scripts\teleop_record.py --input keyboard --panel-size 320   # 크게
.\.venv\Scripts\python.exe scripts\teleop_record.py --input keyboard --panel-every 2    # 느릴 때
.\.venv\Scripts\python.exe scripts\teleop_record.py --input keyboard --no-panel         # 끄기
```

쓸 수 있는 카메라: `agentview` · `frontview` · `sideview` · `robot0_eye_in_hand`.

> 여기에 무엇을 띄우든 **녹화되는 데이터는 달라지지 않습니다**
> (녹화 카메라는 `--cameras` 가 정합니다 — 아래 참고).

### ★ 키보드 입력은 이 창에서만 받습니다

**MuJoCo 뷰어(simulate)는 알파벳 대부분에 자기 렌더링 토글 단축키를 갖고 있고,
우리 조작키 11개가 전부 충돌합니다.**

```
W=Wireframe   S=Shadow        A=Auto Connect   D=Static Body   Q=Camera
E=Equality    Z=Light         C=Contact Point  R=Reflection
X=Texture     P=Contact Split
```

뷰어 창에 포커스를 두고 `W` 를 누르면 로봇이 움직이는 대신 화면이 와이어프레임이 되고,
`D` 를 누르면 테이블(static body)이 사라집니다. 뷰어의 키 처리는 C++ 쪽이라
파이썬에서 막을 수 없습니다. 목록은 직접 확인할 수 있습니다:

```powershell
.\.venv\Scripts\python.exe -c "import mujoco; print([(s[2],s[0]) for s in mujoco.mjVISSTRING if s[2]])"
.\.venv\Scripts\python.exe -c "import mujoco; print([(s[2],s[0]) for s in mujoco.mjRNDSTRING if s[2]])"
```

그래서 **조작 키는 카메라 뷰 창에서만 받습니다.** 뷰어에는 `key_callback` 을 넘기지 않습니다.

- 마우스(시점 회전, `Ctrl+드래그` 물체 이동)는 뷰어 창에서 그대로 씁니다. 문제 없습니다.
- 실수로 뷰어에서 키를 눌러 표시가 바뀌어도, **시각화 플래그(mjVIS)는 매 프레임 원상복구**합니다
  (테이블이 사라지는 `D` 등). 다만 `W`·`S`·`R` 은 scene 쪽 플래그(mjRND)라
  파이썬에서 되돌릴 수 없습니다 — **같은 키를 한 번 더 누르면** 돌아옵니다.
- **표시만 바뀌는 것이라 녹화 데이터에는 영향이 없습니다.**
- 키보드 모드에서 `--no-panel` 을 주면 조작 수단이 사라지므로 **실행을 거부**합니다.

**게임패드는 포커스와 무관하게 동작합니다** — pygame 창이 열려 있으면 SDL 이 기본적으로
비포커스 상태의 조이스틱 이벤트를 막는데, `SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS=1` 로 풀어 두었습니다.
다만 `R`·`X`·`P`·`ESC` 키는 카메라 뷰 창에서 눌러야 합니다.

측정: 카메라 3대 240px 렌더가 프레임당 2.9 ms, 뷰어까지 합친 루프가 10.7 ms
(20 Hz 예산 50 ms). 여유가 있어 기본으로 켜 둡니다.

| 입력 | 조작 |
|---|---|
| 키보드 | `W/S` 앞뒤 · `A/D` 좌우 · `Q/E` 위아래 · `Z/C` 손목 · `SPACE` 그리퍼 · `ENTER` 녹화 · `R` 리셋 · `X` 폐기 · `P` 배치 저장 · `ESC` 종료 |
| 게임패드 | 기기마다 다릅니다. 아래 캘리브레이션 참고 |

**수업용 Thrustmaster T.16000M** (캘리브레이션 완료, `gamepad_map.json`):

| 조작 | T.16000M | 값 |
|---|---|---|
| 좌우 (x) / 앞뒤 (y) | 스틱 기울이기 | 축 0 / 축 1(반전) |
| 손목 회전 (yaw) | 손잡이 비틀기 | 축 2(반전) |
| **높이 (z)** | **엄지 십자키 위/아래** | `hat0_y` |
| 그리퍼 | 트리거 | 버튼 0 |
| 녹화 / 리셋 / 폐기 | 받침대 버튼 | 15 / 9 / 4 |

> **높이가 스로틀 레버가 아니라 십자키입니다.** 스로틀은 스프링이 없어 손을 떼도
> 그 자리에 머물고, 축은 전부 속도 명령이라 로봇이 멈추지 않고 계속 오르내립니다.
> **스로틀 레버는 쓰지 않습니다.**

패드 모드에서도 뷰어 키 `P`(배치 저장) · `R`(리셋) · `X`(폐기) · `ESC`(종료)는 함께 씁니다.
패드에는 배치 저장에 해당하는 버튼이 없기 때문입니다.

### 게임패드 캘리브레이션 (기기마다 한 번)

**축·버튼 번호는 기기마다 다릅니다.** 시키는 대로 움직이면 번호를 찾아
`gamepad_map.json` 으로 저장해 줍니다.

```powershell
.\.venv\Scripts\python.exe scripts\teleop_record.py --calibrate
.\.venv\Scripts\python.exe scripts\teleop_record.py --input gamepad
```

만들어진 `gamepad_map.json` 이 폴더에 있으면 **`--gamepad-map` 을 생략해도 자동으로 씁니다**
(`[매핑] gamepad_map.json 를 자동으로 씁니다` 가 찍힙니다).
다른 파일을 쓰려면 `--gamepad-map 경로` 로 지정하세요.

값을 눈으로 직접 보고 싶으면 진단용으로:

```powershell
.\.venv\Scripts\python.exe scripts\teleop_record.py --calibrate-raw
```

매핑 파일은 **적은 항목만 덮어씁니다.**

```json
{ "z": "hat0_y", "invert_z": false, "btn_record": 5 }
```

축 항목에는 축 번호(정수) 대신 **해트(십자키)** 를 넣을 수 있습니다: `"hat0_y"`.
비행 스틱의 스로틀처럼 **중립 복귀가 없는 축을 z 에 쓰면 로봇이 멈추지 않으므로**,
그런 기기에서는 높이를 해트로 옮기세요. `--calibrate` 가 그런 축을 감지하면 경고하고
바꿔 줄지 물어봅니다.

### 높이(z)가 너무 거칠 때 — 속도 조절

한 제어 스텝의 최대 이동은 `MAX_DPOS = 0.05 m` 입니다. `--speed 0.55` 면 **스텝당 27.5 mm** 라,
**타일 높이(30 mm)와 맞먹어서 한 스텝에 물체를 지나칩니다.** 게다가 z 를 해트(십자키)에
매핑한 기기에서는 값이 `-1/0/+1` 뿐이라 살살 누를 수도 없습니다.

그래서 **z 는 기본 배율 0.35** 로 낮춰 두었습니다 (`teleop.AXIS_GAIN`).

| 설정 | 스텝당 z 이동 (실측) |
|---|---|
| 옛 동작 (`gain_z=1.0`) | 23.2 mm |
| **기본 (`gain_z=0.35`)** | **8.1 mm** |
| 기본 + 정밀 모드 | 2.0 mm |
| `--speed-z 0.2` | 4.6 mm |
| `--speed-z 0.2` + 정밀 모드 | 1.1 mm |

x·y 배율은 그대로입니다(23.2 mm). 더 세밀하게 하려면:

```powershell
.\.venv\Scripts\python.exe scripts\teleop_record.py --input keyboard --speed-z 0.2
.\.venv\Scripts\python.exe scripts\teleop_record.py --input keyboard --speed 0.35   # 전 축을 함께
```

**정밀 모드**(전 축 ×0.25, `teleop.SLOW_SCALE`)는 **누르고 있는 동안만** 적용됩니다.

- 키보드: `SHIFT` 를 누른 채로 조작
- 게임패드: `btn_slow` 로 지정한 버튼. `--calibrate` 의 마지막(선택) 단계에서 정하거나,
  `gamepad_map.json` 에 `"btn_slow": 3` 처럼 직접 적습니다.

배율은 매핑 파일에도 넣을 수 있습니다 (`gain_x` `gain_y` `gain_z` `gain_yaw`).
`--speed-z` 는 파일 값보다 우선합니다.

```json
{ "z": "hat0_y", "invert_z": false, "gain_z": 0.25, "btn_slow": 3 }
```

실행하면 지금 값이 표시됩니다 — `높이 배율 0.35 -> 한 스텝 약 9.6 mm`.

### 씬 직접 편집

뷰어에서 `Ctrl + 왼쪽드래그`로 물체를 끌어 배치를 바꿀 수 있습니다.
바꾼 뒤 `P` 를 누르면 `data\layouts\*.json` 으로 저장되고, 다시 부를 때는:

```powershell
.\.venv\Scripts\python.exe scripts\teleop_record.py --layout data\layouts\layout_0903_1530.json
```

**실험 조건을 손으로 만들 때 쓰는 기능입니다.**

> **반드시 지킬 것**: 물체를 던지거나 떨어뜨리지 마세요.
> 낙하 구간은 뒤집으면 물체가 저절로 날아오르는 물리적으로 불가능한 장면이 됩니다.
> 항상 **집어서 → 옮겨서 → 내려놓기**로 천천히.
> `build_dataset.py` 가 이런 구간을 자동으로 찾아내 **해당 에피소드를 통째로 제외**합니다.

### 뒤집기의 수학

`target[t+1] = target[t] + a[t]` 이므로 위치·회전 델타는 **부호만 뒤집으면** 됩니다.

```
a'[k][:6] = -a[T-1-k][:6]
```

그리퍼는 델타가 아니라 절대 명령이라 부호 반전이 안 됩니다. 대신
"역재생 기준 다음 프레임에서 손가락이 닫혀 있어야 하는가"로 다시 만듭니다.
그러면 원본의 *놓기* 순간이 역재생의 *잡기* 순간으로 정확히 대응됩니다.

```powershell
.\.venv\Scripts\python.exe scripts\build_dataset.py          # 역재생 + 검증
```

`data\episodes\` 의 원본을 뒤집어 `data\episodes_reversed\` 에 넣고, 규격까지 검사해
`build_report.json` 을 남깁니다.

### 무엇이 녹화되는가 (보는 것과 다릅니다)

에피소드 HDF5 에 **실제로 저장되는 것**은 아래가 전부입니다.

| 저장되는 것 | 내용 |
|---|---|
| `obs/agentview_image` | **위(top-down) 카메라**, 256×256 uint8 |
| `obs/robot0_eye_in_hand_image` | **손목 카메라**, 256×256 uint8 |
| `obs/robot0_joint_pos` (7) · `robot0_gripper_qpos` (2) · `robot0_eef_pos` (3) · `robot0_eef_quat` (4) | state 16차원 |
| `actions` | 7차원 액션 (T개) |
| `qpos` | **MuJoCo 전체 상태** (T+1개) — 나중에 다른 카메라로 재렌더할 때 씀 |
| `target_pos` · `target_quat` | 제어기 목표 자세 (역재생 검증에 씀) |

- 이미지 카메라는 **2대뿐**입니다 (`agentview`, `robot0_eye_in_hand`).
  LIBERO 와 같은 구성이라 이렇게 맞춰 두었습니다. `--cameras` 로 바꿀 수 있지만,
  **학습 스키마가 달라지므로 바꾸지 마세요.**
- 해상도는 `--image-size` (기본 256).
- **뷰어를 어느 각도로 돌려 놓든, 카메라 뷰 창에 무엇을 띄우든 저장되는 것은 위와 같습니다.**
- `qpos` 를 함께 저장하므로, 나중에 **`frontview`·`sideview` 를 원하는 해상도로
  다시 렌더할 수 있습니다.** 발표 영상은 이렇게 만들면 됩니다 — 다시 녹화할 필요가 없습니다.

```powershell
# 저장된 에피소드에 무엇이 들어 있는지 직접 확인
.\.venv\Scripts\python.exe -c "import h5py,sys; f=h5py.File(sys.argv[1]); f.visit(print); print(dict(f.attrs))" data\episodes\ep_0903_153012.hdf5
```

---

## 5. 학습 (Colab)

`notebooks/session12_vla_pipeline.ipynb` 를 Colab 에서 엽니다.

### 공유 드라이브에 올리는 방법

노트북의 `0-2` 셀이 아래 경로를 찾습니다.

```
MyDrive/VLA_Team_2026/
├── vla_pipeline/          ← 여기. make_student_zip.py 로 만든 zip 을 푼 것
│   ├── vlapipe/  scripts/  assets/  notebooks/
│   └── README.md ...
└── results/               ← 노트북 마지막 셀이 결과를 여기에 저장
```

```powershell
.\.venv\Scripts\python.exe scripts\make_student_zip.py
```
이 zip 을 `VLA_Team_2026/` 아래에 풀면 `vla_pipeline/` 폴더 하나가 생깁니다. 그대로 두면 됩니다.
`.venv`·`outputs`·`data` 는 zip 에서 빠지고 **`assets` 는 들어갑니다**(없으면 씬을 못 만듭니다).

> **폴더째 업로드하지 마세요.** `.venv` 가 함께 올라가는데, 그 안에는 만든 PC 의
> 절대경로가 박혀 있어 다른 곳에서 동작하지 않습니다. 용량만 수 GB 늘어납니다.

### 공유 폴더에서 바로 돌리지 않습니다

`0-2` 셀은 공유 폴더의 코드를 **런타임 로컬(`/content/vla_pipeline`)로 복사**한 뒤
거기서 작업합니다. 원본은 읽기만 합니다. 이유가 셋 있습니다.

1. `SortingEnv` 는 만들 때마다 `assets/.../vla_scene.xml` 을 **새로 씁니다.**
   여러 명이 동시에 돌리면 서로의 씬 파일을 덮어써 `ParseXML` 오류가 납니다.
2. 학습 체크포인트(`outputs/smolvla_sorting`)가 **GB 단위**로 쌓여 공유 드라이브를 채웁니다.
3. Drive(FUSE)는 느려서 메시 파일 로딩이 오래 걸립니다.

녹화한 에피소드는 `VLA_Team_2026/vla_pipeline/data/episodes/` 에 올리면,
`0-2` 가 실행될 때마다 `data/` 를 지우고 다시 복사하므로 **새로 올린 것이 반영**됩니다.
셀이 몇 개를 찾았는지 찍어 주니 확인하고 넘어가세요.

> **LeRobot 변환과 SmolVLA 학습은 윈도우 로컬에서 못 합니다.**
> `lerobot` 은 Python 3.14 윈도우 휠이 없습니다. 아래 명령은 **Colab(리눅스)** 기준입니다.

```bash
pip install -r requirements-colab.txt
pip install "lerobot[smolvla] @ git+https://github.com/huggingface/lerobot.git"
python scripts/build_dataset.py --lerobot --repo-id dshs2026/vla_sorting
```

`observation.images.agentview`, `observation.images.wrist`, `observation.state`,
`action`, `task` 를 가진 LeRobot 데이터셋이 생깁니다. 그대로 SmolVLA 파인튜닝에 넣습니다.

학습이 끝나면 체크포인트를 받아 와 로컬에서도 돌려 볼 수 있습니다:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --policy smolvla --checkpoint outputs\ckpt\last
```

---

## 6. VLM 을 로컬에서 돌리려면 — torch 설치 (선택)

기본 설치(`requirements-windows.txt`)에는 torch 가 없습니다.
`--planner vlm` 이나 `smoke_test.py --vlm` 을 쓰려면 추가로 설치하세요.

**NVIDIA GPU 가 있는 경우** — 반드시 `--index-url` 을 붙여야 CUDA 판이 옵니다.

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install transformers accelerate num2words
```

**GPU 가 없는 경우** (CPU 판, 느립니다)

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision
.\.venv\Scripts\python.exe -m pip install transformers accelerate num2words
```

확인:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\.venv\Scripts\python.exe scripts\smoke_test.py --vlm
```

`2.11.0+cu128 True` 처럼 나오면 GPU 판입니다. `+cpu` 이거나 `False` 면 CPU 판이니,
GPU 를 쓰려면 위 `--index-url` 명령으로 다시 설치하세요.

---

## 7. 파일 구조

```
vla_pipeline/
├── SETUP_STUDENT.md         학생용 설치 안내
├── AGENTS.md / CLAUDE.md    AI 에이전트용 안내
├── setup_assets.py          Panda 에셋 다운로드 + 손목카메라/TCP 패치
├── vlapipe/
│   ├── config.py            씬·액션·관측 규격 (수치는 전부 여기)
│   ├── scene.py             MJCF 생성, tidy/messy/random/cluttered 배치
│   ├── accessibility.py     가림·도달 판정 ('물체 접근성' 구현부)
│   ├── metrics.py           정량 지표 + Wilson CI + 짝지은 부트스트랩
│   ├── controller.py        DLS IK + OSC 유사 제어기
│   ├── env.py               SortingEnv (reset/step/render/성공판정)
│   ├── policy.py            ScriptedSortingPolicy(teacher) · SmolVLAPolicy
│   ├── planner.py           RulePlanner(정답) · VLMPlanner(SmolVLM2) · 채점
│   ├── teleop.py            게임패드/키보드 입력, 캘리브레이션, 배치 저장·불러오기
│   ├── episode.py           HDF5 녹화 포맷
│   ├── reverse.py           역재생 변환 + 역재생 가능성 검사
│   └── to_lerobot.py        LeRobot 데이터셋 변환·검증
├── scripts/                 check_env · smoke_test · run_pipeline · teleop_record ·
│                            build_dataset · make_student_zip
└── notebooks/               Colab 학습 노트북
```

---

## 8. 지금 어디까지 됐나 (전부 이 코드로 실측)

### 윈도우 PC (2026-09-03, Python 3.14.3 / RTX 3090 Ti)

| 항목 | 값 |
|---|---|
| `run_pipeline --layout messy --seed 3` | 서브태스크 5/6, 정리완료 True, 11.6~12.4초 |
| `run_pipeline --layout cluttered --seed 3` | 서브태스크 8/9, 정리완료 True, 13.1초 |
| `run_pipeline --planner vlm` (messy seed 3) | 서브태스크 9/9, 정리완료 **False**, 54.4초, 계획 정답 1/5 |
| SmolVLM2-500M 로드 / 추론 | 9.8초 / 2.0초 (CUDA) |
| 역재생 액션 적분 오차 | **0.000 mm** |
| `smoke_test.py` | 5/5 (`--vlm` 시 6/6) |
| 한글 경로에서 재실행 | 5/5 통과 (경로 우회 동작 확인) |

### 리눅스 데스크톱 (앞선 측정, RTX 4080)

| 항목 | 상태 |
|---|---|
| 씬 · IK · 렌더 | IK 목표점 60개 중 57개 수렴 (실패 3개는 로봇 베이스 바로 앞) |
| Scripted teacher 파지 | 서브태스크 **26/30 (87%)** |
| 씬 전체 정리 (2패스) | **6개 씬 중 5개 완전 정리** (seed 42 에서 `green_tile` 1개 남음) |
| 제어 — 가림 있음(`cluttered`) | 운반 20/34 (59%) · 완전정리 1/5 · 밀기 11/12 |

> ⚠️ **위 제어 수치는 씬 5~6개 기준이라 통계적으로 의미가 없습니다.**
> `4/5` 와 `1/5` 의 95% 신뢰구간이 겹칩니다. 본 실험은 조건당 30씬 + 동일 시드입니다.

### 아직 아무도 끝까지 못 돌린 것

- **SmolVLA 파인튜닝** — 데이터가 없어 미실행. `data/` 는 아직 비어 있습니다
- **LeRobot 데이터셋 실제 변환** — 프레임 구조까지만 검증. Colab 확인 필요
- **게임패드 버튼 번호** — 기기마다 다름. `--calibrate` 로 각자 확인
- **뷰어 GUI 실조작** — 사람이 직접 몰아 봐야 확인됨

남은 파지 실패는 대부분 납작한 타일(높이 0.03 m)이고, 앞선 작업이 물체를 건드린 뒤에 생깁니다.
`--passes 2` 의 재계획으로 대부분 회수됩니다.

---

## 9. VLM 플래너 정확도 — 실측

`messy` 씬 3개(seed 3·7·11), 정답은 `RulePlanner` 기준, RTX 4080.

| 모델 | 프롬프트 | recall | precision | 추론 |
|---|---|---|---|---|
| SmolVLM2-500M | `json` | **0.23** (3/13) | 0.16 (3/19) | 1.5 s |
| SmolVLM2-500M | `lines` | 0.00 (0/13) | 0.00 (0/3) | 0.5 s |
| SmolVLM2-2.2B | `json` | ~0.31 (4/13) | 0.21 (4/19) | **55 s** |

### 여기서 읽어야 할 것

**둘 다 이 과제를 제대로 못 합니다.** 버그가 아니라 모델의 한계이고,
**학생들이 풀어야 할 진짜 문제**입니다.

1. **2.2B 는 값어치를 못 합니다** — recall 이 0.23 → 0.31 로 조금 오르는 대신
   추론이 **40배** 느려집니다(1.5 s → 55 s). 씬 하나 계획하는 데 1분은 수업에서 못 씁니다.
2. **2.2B 는 이미지를 거의 안 봅니다** — 씬을 바꿔도 출력 앞부분이
   `brown → left, green → left, blue → ...` 로 거의 똑같습니다.
   프롬프트의 규칙을 그대로 베껴 쓰는 것에 가깝습니다.
3. **`lines` 프롬프트는 실패했습니다** — "한 줄에 하나씩"이 작은 모델에 쉬울 것 같지만
   실제로는 **한 줄만 쓰고 멈춰서** JSON 보다 나빴습니다. 그래서 기본값은 `json` 입니다.
   추측이 실측으로 뒤집힌 사례라 프롬프트 배틀 도입부로 쓰기 좋습니다.
4. **JSON 이 잘리는 문제** — 토큰 한계로 배열 중간에서 끊기면 파싱이 통째로 실패합니다.
   `_repair_json_array()` 가 마지막 온전한 원소까지 살려 냅니다(토큰 예산도 400 으로 상향).

갈 수 있는 길:
1. 프롬프트 개선 (`vlapipe/planner.py` 의 `_PROMPT`) — 노트북 SECTION 3 에 채점 루프
2. 문제 쪼개기 — "가운데에 무엇이 있나?" 만 묻고 목적지는 규칙으로 정하기
3. **이 과제만 VLM 파인튜닝** — 10회차에서 만든 SFT 데이터셋이 정확히 여기에 쓰입니다.
   위 3가지 관찰(특히 2번)이 "왜 파인튜닝이 필요한가"의 근거가 됩니다.

파이프라인은 `--planner rule` 로 언제든 정답 플래너를 쓸 수 있으므로,
VLM 정확도와 무관하게 2단계(VLA) 연구는 계속 진행할 수 있습니다.

---

## 10. 수업 진행 순서 (제안)

| 시간 | 내용 |
|---|---|
| 10분 | `check_env.py` → `smoke_test.py` — 3명 PC 모두 5/5 확인 |
| 15분 | `run_pipeline.py --layout messy --seed 3` 돌려 보고 `rollout.gif` 확인 |
| 20분 | 노트북 SECTION 1~2 — 씬 규격, 7차원 액션을 직접 넣어 보기 |
| 25분 | SECTION 3 — VLM 플래너 채점. **프롬프트 배틀 시작** |
| 30분 | `teleop_record.py --calibrate` → 어지르기 녹화 연습 |
| 20분 | `build_dataset.py` 로 역재생 확인. 던지면 왜 안 되는지 직접 실험 |

숙제: 1인당 텔레옵 에피소드 5개 + 프롬프트 배틀 비교표.

---

## 11. 문제 해결 — 윈도우

### `python` 을 쳤더니 Microsoft Store 가 열림 / 다른 파이썬이 잡힘

윈도우가 넣어 둔 가짜 `python.exe` 입니다. 이 문서대로
`.\.venv\Scripts\python.exe` 전체 경로를 쓰세요. 지금 무엇이 잡히는지 확인:

```powershell
Get-Command python | Select-Object Source
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
```

### `.\.venv\Scripts\python.exe : 용어가 인식되지 않습니다`

`vla_pipeline` 폴더 바깥에 있거나 가상환경이 없는 것입니다.

```powershell
Get-Location                            # ...\vla_pipeline 이어야 함
Test-Path .\.venv\Scripts\python.exe    # True 여야 함
```

### `Activate.ps1 ... 스크립트를 실행할 수 없습니다` (실행 정책)

`activate` 를 쓰려다 나오는 오류입니다. **이 문서 방식(전체 경로)에서는 안 씁니다.**
굳이 쓰겠다면 현재 창에서만 허용하세요:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### `ParseXML: Error opening file` (파일은 분명히 있는데)

**경로에 한글이 있으면** MuJoCo 가 파일을 열지 못합니다.
윈도우 사용자 이름이 한글이면(`C:\Users\권민규\...`) 바로 걸립니다.

원인은 인코딩입니다. 파이썬 바인딩은 경로를 UTF-8 바이트로 넘기는데,
윈도우의 `fopen` 은 그 바이트를 ANSI 코드페이지(한국어 윈도우는 CP949)로 해석합니다.
8.3 짧은 경로(`ACTIVI~1`)로 바꿔도 한글이 남아 있어서 소용없습니다.

**이미 코드에서 처리해 두었습니다.** `config.mujoco_path()` 가 ASCII 이름의
디렉터리 정션(`%LOCALAPPDATA%\vlapipe_links\<해시>`)을 만들어 그 경로로 넘깁니다.
정션은 관리자 권한이 필요 없고, 복사가 아니라 링크라서 `scene.py` 가 새로 쓴
XML 도 즉시 반영됩니다. MuJoCo 에 경로를 넘기는 코드를 새로 쓸 때는
반드시 `C.mujoco_path(...)` 를 거치세요.

그래도 실패하면 폴더째 `C:\vla\vla_pipeline` 처럼 한글 없는 곳으로 옮기세요.

### `pygame` 설치가 소스 빌드로 넘어가면서 실패

Python 3.13 이상에는 pygame 휠이 없습니다(`distutils` 제거 때문).
`requirements-windows.txt` 가 환경 마커로 `pygame-ce` 를 대신 설치하도록 돼 있으니
그냥 다시 설치하면 됩니다. 수동으로는 `pip install pygame-ce`
(import 이름이 그대로 `pygame` 이라 코드는 손댈 것이 없습니다).

### 폴더를 옮기거나 이름을 바꿨더니 이상해짐

**가상환경 안에는 만들 때의 절대경로가 박혀 있습니다.** 지우고 다시 만드세요.

```powershell
Remove-Item .venv -Recurse -Force
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
```

### VSCode 가 빨간 줄로 "패키지 없음" 이라는데 실행은 잘 됨

인터프리터가 `.venv` 로 안 잡힌 것입니다.
`Ctrl + Shift + P` → `Python: Select Interpreter` → `.\.venv\Scripts\python.exe`.
(`.vscode/settings.json` 을 넣어 두어 대개는 자동으로 잡힙니다.)

### 키를 눌러도 로봇이 안 움직이고, 화면만 이상해짐

**MuJoCo 뷰어 창에 포커스가 있는 것입니다.** 조작 키는 **카메라 뷰 창**에서만 받습니다.
카메라 뷰 창을 한 번 클릭한 뒤 조작하세요. (위 4절 "키보드 입력은 이 창에서만 받습니다")

로봇이 점박이로 보이거나 테이블이 사라졌다면 뷰어의 표시 옵션이 토글된 것입니다.
`W`(와이어프레임) · `S`(그림자) · `R`(반사)은 **뷰어 창에서 같은 키를 한 번 더** 누르면
돌아옵니다. 나머지는 매 프레임 자동 복구됩니다. **데이터에는 영향이 없습니다.**

### 게임패드를 움직여도 반응이 없음

캘리브레이션부터 확인하세요 — `--calibrate-raw` 로 축 값이 실제로 변하는지 봅니다.
값이 변하는데 로봇이 안 움직이면 매핑 문제이니 `--calibrate` 를 다시 하세요.

(포커스 때문일 가능성은 없습니다. `SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS=1` 로
어느 창을 보고 있든 패드가 동작하도록 해 두었습니다.)

### 뷰어 창이 안 뜨거나 바로 꺼짐

먼저 `smoke_test.py` 를 돌려 보세요. 창 없이 도는 검사라, 통과한다면
시뮬레이션 자체는 정상이고 창(그래픽 드라이버)만 문제입니다.

### `torch.cuda.is_available()` 이 False

CPU 판이 설치된 것입니다. 6절의 `--index-url` 을 붙여 다시 설치하세요.

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Colab 에서 `AssertionError: Torch not compiled with CUDA enabled`

**런타임에 GPU 가 붙어 있지 않은 것입니다.** Colab 은 기본이 CPU 런타임이라
`VLMPlanner` / `SmolVLAPolicy` 를 만들 때 여기서 걸립니다.

**런타임 > 런타임 유형 변경 > T4 GPU** 로 바꾸고 노트북을 처음부터 다시 실행하세요.
(런타임을 바꾸면 세션이 새로 뜨므로 드라이브 마운트 셀부터 다시 돌려야 합니다.)

GPU 를 못 쓰는 상황이면 그냥 두어도 **CPU 로 자동 전환되어 돌아갑니다**(안내 문구가 출력됨).
다만 SmolVLM2-500M 추론이 실측 2.0초 → 10.6초로 느려집니다(윈도우 PC 기준).
`device='cpu'` 를 직접 넘겨도 같습니다.

```python
vlm = VLMPlanner()                # 기본값 'auto' — GPU 있으면 GPU, 없으면 CPU
vlm = VLMPlanner(device='cpu')    # 강제로 CPU
```

### `pip install lerobot` 이 실패

윈도우 + Python 3.14 조합에는 휠이 없습니다. **정상입니다.**
LeRobot 변환·SmolVLA 학습은 Colab 에서 하세요 (5절).

### `symlinks ... your machine does not support them` 경고

Hugging Face 캐시가 심볼릭 링크 대신 복사를 쓴다는 안내일 뿐입니다. 무시하세요.

---

## 12. 문제 해결 — 작업 중 VSCode 가 통째로 꺼질 때 (리눅스 전용)

> **윈도우에서는 이 절이 해당되지 않습니다.** `run_isolated.sh` · `check_server.sh` 도
> 리눅스 전용이라 실행할 필요가 없습니다.

**증상**: 무거운 작업(대형 VLM 로드, 파인튜닝) 중에 작업이 중단되고 VSCode 창까지 함께 닫힘.

**원인**: `systemd-oomd`. 확인 방법:

```bash
journalctl -u systemd-oomd --since "1 hour ago" --no-pager | tail
# Killed .../app-code-XXXX.scope due to memory pressure for
# /user.slice/user-1000.slice/user@1000.service being 90.28% > 50.00% for > 20s
```

VSCode 통합 터미널에서 띄운 프로세스는 **VSCode 와 같은 cgroup**(`app-code-*.scope`)에서
돕니다. 사용자 슬라이스의 메모리 압박(PSI)이 50%를 20초 넘게 넘으면 oomd 가 그 scope 를
통째로 죽이는데, 거기엔 VSCode 본체가 들어 있습니다. 그래서 편집기까지 같이 꺼집니다.
절대 사용량이 아니라 **압박(PSI)** 기준이라, "메모리 여유 15GB" 여도 대용량 파일을 읽으며
할당하면 발동합니다.

### 대처 (위에서부터 권장)

**1. 무거운 작업은 격리 scope 에서 실행** — sudo 불필요, 즉시 적용
```bash
./scripts/run_isolated.sh python scripts/run_pipeline.py --planner vlm
MEM=16G ./scripts/run_isolated.sh python train.py
```

**2. 브라우저 정리**
```bash
ps -eo rss,comm | awk '/chrome/{s+=$1} END{printf "Chrome %.1fGB\n", s/1048576}'
```

**3. oomd 임계 완화** (sudo 필요, 영구)
```bash
sudo tee /etc/systemd/oomd.conf >/dev/null <<'EOF'
[OOM]
DefaultMemoryPressureLimit=80%
DefaultMemoryPressureDurationSec=60s
EOF
sudo systemctl restart systemd-oomd
```

**4. 사용자 슬라이스에서만 oomd 해제** (3번으로 부족할 때)
```bash
sudo systemctl edit user@1000.service
#   [Service]
#   ManagedOOMMemoryPressure=auto
sudo systemctl daemon-reload
```

### 코드 쪽에서 이미 한 것

`VLMPlanner` 는 `device_map` + `low_cpu_mem_usage` 로 가중치를 GPU 에 바로 흘려보냅니다.
기본 방식(CPU 에 전부 올린 뒤 `.to('cuda')`)은 호스트 RAM 피크가 모델 크기의 약 2배입니다.

| | 변경 전 | 변경 후 |
|---|---|---|
| SmolVLM2-500M 로드 | 78 s | **3 s** (리눅스) / 9.8 s (윈도우, 첫 로드) |
| 호스트 RAM 피크 | — | 3.25 GB |

> **SmolVLM2-2.2B 를 리눅스 데스크톱에서 부르지 마세요.** 로드 180초 동안 지속적인
> 메모리 압박을 만들어 위 크래시를 일으킵니다. 2.2B 실험은 Colab 에서 하거나
> `run_isolated.sh` 로 감싸세요.
