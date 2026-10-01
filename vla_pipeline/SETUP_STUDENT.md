# 학생용 설치 안내 — vla_pipeline

2026 대전과학고 심화자율연구 / NMAIL

`vla_pipeline.zip` 을 받은 상태에서 **처음부터 끝까지 따라 하면 되는 문서**입니다.
윈도우 노트북 + VSCode 기준이고, 30분이면 끝납니다.

막히면 **8절 문제 해결**을 먼저 보세요. 거기 없는 문제면 조교에게 알려 주세요.

---

## 0. 무엇을 설치하는 것인가

로봇 팔(Franka Panda)이 테이블 위 블록을 정리하는 **시뮬레이터**입니다.
물리 엔진은 MuJoCo 이고, 우리가 연구할 VLM/VLA 모델이 이 위에서 돕니다.

이 문서를 마치면 아래가 됩니다.

| 되는 것 | 필요한 것 |
|---|---|
| 씬 만들기 · 렌더링 · 로봇 제어 | 1~4절만 (설치 5분, 용량 약 400 MB) |
| 게임패드/키보드 텔레옵 녹화 | 1~4절 |
| 역재생 → 학습 데이터 만들기 | 1~4절 |
| **VLM(SmolVLM2) 로 계획 세우기** | 5절 추가 (torch, 약 3 GB) |
| **SmolVLA 파인튜닝** | Colab 에서 (7절) |

**1~4절이 기본입니다.** 5절은 GPU 가 있거나 VLM 실습을 할 때만 하세요.

---

## 1. 압축 풀 위치 정하기 — 여기서 제일 많이 틀립니다

압축을 풀 위치는 **경로에 한글도 공백도 없는 곳**을 권합니다.

```
좋음   C:\vla\vla_pipeline
좋음   C:\Users\minkyu\Documents\vla_pipeline
피하기 C:\Users\권민규\바탕 화면\vla_pipeline
```

한글 경로에서도 **돌아가긴 합니다.** MuJoCo 가 한글 경로를 못 여는 문제가 있는데
(`ParseXML: Error opening file` — 파일은 멀쩡히 있는데도 실패), 코드가
`vlapipe/config.py` 의 `mujoco_path()` 로 자동 우회하도록 만들어져 있습니다.
실제로 한글 경로에서 전 항목 통과를 확인했습니다. 그래도 다른 도구에서 또 걸릴 수
있으니 **ASCII 경로를 쓰는 편이 안전합니다.**

> 압축을 풀면 `vla_pipeline` 폴더 하나가 나옵니다.
> 그 안에 `README.md`, `vlapipe/`, `scripts/`, `assets/` 가 보이면 제대로 푼 것입니다.
> `vla_pipeline/vla_pipeline/...` 처럼 두 겹이 됐다면 한 겹을 올려 주세요.

---

## 2. 파이썬 설치

**Python 3.12 이상**이면 됩니다. (조교 PC 는 3.14.3 으로 확인했습니다.)

이미 있는지 먼저 확인하세요. PowerShell 을 열고:

```powershell
py -0p
```

목록에 3.12 이상이 있으면 넘어가도 됩니다. 없으면 <https://www.python.org/downloads/>
에서 설치하고, 설치 화면에서 **"Add python.exe to PATH"** 를 반드시 체크하세요.

> Microsoft Store 판 파이썬은 권한 문제로 종종 말썽입니다. python.org 판을 쓰세요.

---

## 3. 가상환경 만들고 패키지 설치

VSCode 에서 **`vla_pipeline` 폴더를 여세요** (File → Open Folder).
`Ctrl + ~` 로 터미널을 열고 (PowerShell 인지 확인) 순서대로:

```powershell
py -3.14 -m venv .venv
```
3.14 가 없으면 자기 버전으로 바꾸세요 (`py -3.12 -m venv .venv`).

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
```

2~3분 걸립니다. `Successfully installed ...` 가 나오면 성공입니다.

### 명령 앞에 왜 계속 `.\.venv\Scripts\python.exe` 를 붙이나요

가상환경 안의 파이썬을 **확실히** 쓰기 위해서입니다.
`activate` 를 해도 되지만, 실수로 다른 파이썬이 잡히는 사고가 자주 나서
이 문서에서는 전부 전체 경로로 씁니다. 그대로 복사해 쓰세요.

### VSCode 인터프리터 잡기

`Ctrl + Shift + P` → `Python: Select Interpreter` → `.\.venv\Scripts\python.exe` 선택.

이걸 안 하면 코드가 멀쩡히 돌아가는데도 편집기가 빨간 줄로
"패키지가 없다"고 표시합니다. **폴더에 `.vscode/settings.json` 을 넣어 두었으니
대개는 자동으로 잡힙니다.** 그래도 빨간 줄이 남으면 위 절차로 직접 고르세요.

---

## 4. 제대로 깔렸는지 확인 — 두 단계

### 4-1. 설치 점검

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py
```

```
[필수 패키지]
  OK  mujoco / numpy / h5py / imageio / PIL
[에셋]
  OK  Panda 원본 ... / 패치본 ...
[렌더링 테스트]
  OK  agentview (128, 128, 3), 평균밝기 135.1
  OK  물체 10개, 잘못 놓인 것 4개
모두 정상입니다.
```

`[선택 패키지]` 의 `torch` · `transformers` · `lerobot` 이 **"없음" 이어도 정상입니다.**
1~4절만 했다면 원래 없는 게 맞습니다.

### 4-2. 실제로 도는지 확인 (스모크 테스트)

`check_env.py` 는 "설치되어 있는가"를 보고, 이건 **"실제로 돌아가는가"** 를 봅니다.

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

1~2분 안에 5개 항목이 전부 `OK` 로 나와야 합니다.

```
  [ OK ] 1. 씬 생성 · 렌더링
  [ OK ] 2. IK · 제어 (스크립트 정책)
  [ OK ] 3. 녹화 (HDF5 저장 → 재읽기)
  [ OK ] 4. 역재생 (부호 반전 + 그리퍼 재생성)
         적분 오차 0.000mm (허용 0.5mm)
  [ OK ] 5. 학습 데이터 규격 검증
5/5 통과
```

**여기까지 통과하면 설치는 끝입니다.**

### 4-3. 눈으로 확인

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --layout messy --seed 3
```

`outputs\run_messy3_rule_scripted\` 에 `before.png`, `after.png`, `rollout.gif` 가 생깁니다.
`rollout.gif` 를 열어 로봇이 블록을 옮기는지 보세요. 조교 PC 기준 약 12초, 서브태스크 5/6 성공입니다.
(seed 를 바꾸면 결과도 달라집니다. 몇 개 돌려 보세요.)

---

## 5. (선택) VLM 을 로컬에서 돌리고 싶다면 — torch 설치

VLM 계획 실습을 **자기 노트북에서** 하려면 torch 가 필요합니다.
용량이 크니(2~3 GB) 먼저 4절까지 끝낸 뒤에 하세요.

### GPU(NVIDIA)가 있는 경우

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install transformers accelerate num2words
```

다운로드 2.8 GB 입니다. 학교 와이파이보다 집에서 받으세요.

### GPU 가 없는 경우 (내장 그래픽 노트북)

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision
.\.venv\Scripts\python.exe -m pip install transformers accelerate num2words
```

CPU 로도 **돌긴 합니다.** 다만 한 장 추론에 수십 초 걸리므로,
VLM 실습은 Colab 에서 하는 편이 낫습니다.

### 확인

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py --vlm
```

처음 실행할 때 SmolVLM2-500M 모델을 약 1 GB 받습니다(한 번만).
조교 PC(RTX 3090 Ti) 기준 로드 9.8초, 추론 2.0초였습니다.

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --planner vlm --layout messy --seed 3
```

> **정확도가 낮게 나오는 것이 정상입니다.** 조교 PC 에서 정답 1/5 였습니다.
> 이건 버그가 아니라 작은 VLM 의 한계이고, **여러분이 풀어야 할 연구 문제**입니다.
> 자세한 내용은 `README.md` 7절을 보세요.

---

## 6. 텔레옵 녹화 — 연구 데이터를 만드는 부분

정돈된 씬을 **어지르는** 걸 녹화한 뒤, 시간을 거꾸로 뒤집어 **정돈하는** 데모로 씁니다.
구역 안에 정확히 내려놓기는 손으로 어렵지만, 집어서 아무 데나 놓기는 쉽기 때문입니다.

### 창이 두 개 뜹니다

| 창 | 무엇 |
|---|---|
| **카메라 뷰** | **키보드 조작은 여기서 합니다.** 고정 카메라 여러 대를 모아 보여 줍니다 |
| **MuJoCo 뷰어** | 마우스로 시점을 돌리고, `Ctrl+드래그`로 물체를 옮깁니다 |

### ⚠ 키는 반드시 "카메라 뷰" 창에서 누르세요

**MuJoCo 뷰어 창에서 `W`/`A`/`S`/`D` 를 누르면 로봇이 움직이지 않고 화면이 깨집니다.**

버그가 아닙니다. MuJoCo 뷰어는 알파벳 대부분에 **자기 표시 옵션 단축키**를 갖고 있는데,
하필 우리 조작키와 전부 겹칩니다.

| 누른 키 | 뷰어에서 실제로 일어나는 일 |
|---|---|
| `W` | 와이어프레임 켜짐 → 로봇이 점박이처럼 보임 |
| `D` | Static Body 꺼짐 → **테이블이 사라짐** |
| `S` | 그림자 꺼짐 |
| `Q` `E` `Z` `C` `R` `X` `P` `A` | 각각 다른 표시 옵션이 토글됨 |

그래서 **키보드 조작은 카메라 뷰 창에서만 받도록** 만들었습니다.
카메라 뷰 창을 한 번 클릭해 포커스를 준 뒤 조작하세요.

이미 화면이 깨졌다면 **같은 키를 한 번 더 누르면 되돌아옵니다**(뷰어 창에서).
잘 모르겠으면 그냥 껐다 다시 켜세요. 표시만 바뀐 것이라 **데이터에는 아무 영향이 없습니다.**

> **게임패드는 포커스와 상관없이 동작합니다.** 어느 창을 보고 있든 됩니다.
> 다만 `R`(리셋) · `X`(폐기) · `P`(배치 저장) · `ESC` 키는 카메라 뷰 창에서 눌러야 합니다.

**로봇이 저절로 움직이지는 않습니다** — 여러분이 스틱이나 키를 넣는 만큼만
20 Hz 로 움직입니다. 가만히 두면 가만히 있습니다.

### 카메라 뷰 창 — 팔에 가려서 안 보일 때

뷰어는 시점이 하나뿐이라, **로봇 팔이 물체를 가리면 무엇을 집는지 안 보입니다.**
시점을 돌리면 이번엔 "앞으로"가 화면 어느 쪽인지 헷갈립니다.
그래서 **조작은 뷰어에서 하고, 확인은 카메라 뷰 창에서** 하세요.

기본으로 세 개가 뜹니다: `agentview`(위에서) · `frontview`(앞에서) · `wrist`(손목).

**집는 높이가 헷갈리면 옆에서 보는 시점을 추가하세요.** 깊이 판단에 제일 좋습니다.

```powershell
.\.venv\Scripts\python.exe scripts\teleop_record.py --input keyboard --panel-cameras agentview,frontview,sideview,robot0_eye_in_hand --panel-cols 2
```

| 옵션 | 뜻 |
|---|---|
| `--panel-cameras` | 띄울 카메라. `agentview` `frontview` `sideview` `robot0_eye_in_hand` |
| `--panel-cols` | 열 수 (4대면 `2` 가 보기 좋습니다) |
| `--panel-size` | 칸 크기(px). 기본 240, 화면이 크면 `320` |
| `--panel-every` | 버벅이면 `2` (2프레임에 한 번만 그림) |
| `--no-panel` | 창을 아예 안 띄움 |

> **이 창에 무엇을 띄우든 녹화되는 데이터는 달라지지 않습니다.**
> 무엇이 저장되는지는 아래 "무엇이 녹화되나" 를 보세요.

> `--no-panel` 로 이 창을 끄면 **키보드로는 조작할 수 없습니다**(위 경고 참고).
> 그래서 키보드 모드에서 `--no-panel` 을 주면 실행되지 않고 안내가 나옵니다.

**마우스로 보는 각도 바꾸기** — 여기는 **MuJoCo 뷰어 창**에서 합니다 (로봇은 안 움직입니다)

| 마우스 | 동작 |
|---|---|
| 왼쪽 드래그 | 시점 회전 |
| 오른쪽 드래그 | 시점 이동(팬) |
| 휠 | 확대/축소 |
| **`Ctrl` + 왼쪽 드래그** | **물체를 직접 끌어 옮기기** (씬 편집) |

마우스는 뷰어 창에서 써도 안전합니다. **문제가 되는 것은 키보드뿐입니다.**

`Ctrl+드래그` 로 배치를 바꾼 뒤 (**카메라 뷰 창에서**) `P` 를 누르면
`data/layouts/*.json` 으로 저장됩니다.
나중에 `--layout data/layouts/xxx.json` 으로 그 배치를 다시 부를 수 있습니다.
**실험 조건을 만들 때 쓰는 기능입니다.**

**처음 5분 — 이 순서로 익히세요**

1. **카메라 뷰 창**을 한 번 클릭해 포커스를 줍니다. (키가 안 먹으면 십중팔구 이것 때문입니다)
2. 로봇을 좌우·앞뒤로만 움직여 봅니다. 방향이 화면과 맞는지 확인하세요.
3. 위아래로 움직여 봅니다. **손을 뗐을 때 멈추는지** 꼭 확인하세요.
   안 멈추면 아래 "중립으로 안 돌아오는 축" 항목을 보세요.
4. 그리퍼를 열고 닫아 봅니다. 손가락이 실제로 벌어지고 오므라드는지 봅니다.
5. 블록 하나를 **집어서 → 들어올려 → 옮겨서 → 천천히 내려놓기.** 녹화는 아직 하지 마세요.
6. 이게 편해지면 그때 `B`(또는 `ENTER`)로 녹화를 시작합니다.

**녹화 한 판의 흐름**

```
정돈된 상태에서 시작
  → 녹화 시작 (B / ENTER)
  → 물체를 하나씩 집어서 가운데로 옮겨 어지른다  ← 여기가 데이터
  → 녹화 정지 (B / ENTER)  = data/episodes/ 에 저장
  → 리셋 (X / R) 하고 다음 판
```

망친 판은 저장하지 말고 **폐기(Y / X)** 하세요. 나쁜 데이터는 안 쓰는 편이 낫습니다.

### 키보드 (패드 없이)

```powershell
.\.venv\Scripts\python.exe scripts\teleop_record.py --input keyboard
```

MuJoCo 뷰어 창이 뜹니다. **창을 클릭해 포커스를 준 뒤** 조작하세요.

| 키 | 동작 |
|---|---|
| `W`/`S` | 앞/뒤 | 
| `A`/`D` | 좌/우 |
| `Q`/`E` | 위/아래 |
| `Z`/`C` | 손목 회전 |
| `SPACE` | 그리퍼 열기/닫기 |
| `ENTER` | 녹화 시작 / 정지(저장) |
| `R` | 정돈 상태로 리셋 |
| `X` | 지금 녹화 버리기 |
| `P` | 현재 배치를 JSON 으로 저장 |
| `ESC` | 종료 |

뷰어에서 `Ctrl + 왼쪽드래그` 로 물체를 직접 끌어 배치를 바꿀 수 있습니다.

### 게임패드 — 수업용 조종간은 이미 설정돼 있습니다

수업에서 쓰는 **Thrustmaster T.16000M** 은 조교가 캘리브레이션을 끝내
`gamepad_map.json` 으로 넣어 두었습니다. 그냥 이렇게 쓰면 됩니다.

```powershell
.\.venv\Scripts\python.exe scripts\teleop_record.py --input gamepad
```

`gamepad_map.json` 이 폴더에 있으면 **자동으로 적용**됩니다(실행할 때 `[매핑] ...` 이라고 알려 줍니다).

| 조작 | T.16000M |
|---|---|
| 좌우 (x) | 스틱을 좌우로 기울임 |
| 앞뒤 (y) | 스틱을 앞뒤로 기울임 |
| 손목 회전 (yaw) | 손잡이를 비틂(트위스트) |
| **높이 (z)** | **엄지 십자키(해트) 위/아래** |
| 그리퍼 열고닫기 | 트리거 (버튼 0) |
| 녹화 시작/정지 | 버튼 15 |
| 씬 리셋 | 버튼 9 |
| 녹화 폐기 | 버튼 4 |

> 높이가 받침대의 **스로틀 레버가 아니라 십자키**인 것에 주의하세요.
> 스로틀은 손을 떼도 그 자리에 머물러서, 로봇이 멈추지 않고 계속 오르내립니다.
> 십자키는 손을 떼면 즉시 멈춥니다. **스로틀 레버는 쓰지 않습니다.**

### 다른 패드를 쓴다면 — 캘리브레이션

**기기마다 축·버튼 번호가 다릅니다.** 다른 패드를 가져왔다면 이걸 먼저 하세요.

```powershell
.\.venv\Scripts\python.exe scripts\teleop_record.py --calibrate
```

시키는 대로 스틱을 밀고 버튼을 누르면, 번호를 알아서 찾아
`gamepad_map.json` 으로 저장해 줍니다. 2분이면 끝납니다.

```
[x] 로봇을 **오른쪽**(월드 +x)으로 보낼 조작을 끝까지 하세요
      준비되면 ENTER 를 누르고 3초 안에 조작하세요...
      -> 축 0, 부호 반전 아니오
[btn_grip] 그리퍼를 여닫을 버튼 을 누르세요. 준비되면 ENTER...
      -> 버튼 0
```

다음부터는 이렇게 씁니다.

```powershell
.\.venv\Scripts\python.exe scripts\teleop_record.py --input gamepad --gamepad-map gamepad_map.json
```

> 번호가 화면에 어떻게 찍히는지 직접 보고 싶으면 `--calibrate-raw` 를 쓰세요(진단용).
> `gamepad_map.json` 은 사람이 읽을 수 있는 JSON 이라 나중에 손으로 고쳐도 됩니다.

### 조작

| 조작 | 기본 |
|---|---|
| 평면 이동 (앞뒤·좌우) | 왼쪽 스틱 |
| 높이 | 오른쪽 스틱 상하 |
| 손목 회전 | 오른쪽 스틱 좌우 |
| 그리퍼 열고닫기 | A |
| 녹화 시작 / 정지(저장) | B |
| 씬 리셋 | X |
| 녹화 버리기 | Y |

패드로 몰더라도 **뷰어 창의 키 몇 개는 함께 씁니다.**

| 키 | 동작 |
|---|---|
| `P` | 현재 배치를 JSON 으로 저장 (패드에는 이 버튼이 없습니다) |
| `R` | 리셋 · `X` 녹화 폐기 |
| `ESC` | 종료 |

키를 쓰려면 **뷰어 창을 한 번 클릭**해 포커스를 주세요. 스틱은 포커스와 무관합니다.

### ⚠ 중립으로 안 돌아오는 축에 주의

모든 축은 **속도 명령**입니다. 스틱을 기울인 만큼 그 방향으로 계속 움직이고,
손을 떼면 스틱이 가운데로 돌아오면서 멈춥니다.

그런데 **비행 스틱의 스로틀 레버**처럼 손을 떼도 그 자리에 머무는 축이 있습니다.
그런 축을 높이(z)에 배정하면 **로봇이 멈추지 않고 계속 오르내려** 테이블에 부딪힙니다.

`--calibrate` 가 이런 축을 감지하면 경고해 줍니다. 그럴 때는 높이를 십자키(해트)로
옮기세요. 십자키는 손을 떼면 반드시 0 으로 돌아옵니다.

`gamepad_map.json` 에서:
```json
{ "z": "hat0_y", "invert_z": false }
```

> 조교 PC 에 연결된 **Thrustmaster T.16000M** 이 정확히 이 경우입니다.
> 축 4개 중 3번이 스로틀 레버라, 높이는 십자키로 옮겨 쓰는 것을 권합니다.

### 높이가 확확 움직여서 잡기 어렵다면

**정상입니다. 원래 그랬고, 기본값을 낮춰 두었습니다.**

로봇은 한 번에 최대 5 cm 씩 움직일 수 있는데, 타일 두께는 3 cm 밖에 안 됩니다.
그대로 두면 **한 번 내리는 순간 물체를 지나쳐 버립니다.**
그래서 높이만 배율을 낮춰 한 스텝에 약 8 mm 씩 움직이게 해 두었습니다.

그래도 거칠면 **정밀 모드**를 쓰세요. **누르고 있는 동안만** 천천히 움직입니다.

| | 정밀 모드 |
|---|---|
| 키보드 | `SHIFT` 를 누른 채로 `Q`/`E` |
| 게임패드 | `--calibrate` 에서 정한 정밀 버튼 (선택 항목) |

잡을 때 이렇게 하면 편합니다.

1. 물체 **위쪽**까지는 평소 속도로 이동
2. 내려갈 때만 `SHIFT` 를 누른 채로 — 한 스텝에 2 mm 씩 내려갑니다
3. 손끝(손목 카메라)이 물체를 감쌌을 때 `SPACE`

더 느리게 하고 싶으면:

```powershell
.\.venv\Scripts\python.exe scripts\teleop_record.py --input keyboard --speed-z 0.2
```

실행할 때 지금 값이 표시됩니다 — `높이 배율 0.35 -> 한 스텝 약 9.6 mm`.

### ★ 녹화할 때 반드시 지킬 것

**물체를 던지거나 떨어뜨리지 마세요.** 항상 **집어서 → 옮겨서 → 천천히 내려놓기**.

떨어뜨린 구간을 시간 역순으로 뒤집으면 **물체가 저절로 바닥에서 날아오르는**
물리적으로 불가능한 장면이 됩니다. 그런 데이터로 배운 모델은 망가집니다.
`build_dataset.py` 가 그런 구간을 자동으로 찾아 **에피소드째 제외**하므로,
난폭하게 녹화하면 애써 만든 데이터가 통째로 버려집니다.

### 뒤집어서 학습 데이터로

```powershell
.\.venv\Scripts\python.exe scripts\build_dataset.py
```

`data\episodes\` 의 원본을 뒤집어 `data\episodes_reversed\` 에 넣고 검증까지 합니다.

### 무엇이 녹화되나 (보는 것과 다릅니다)

카메라 뷰 창에 네 개를 띄워도, **파일에 저장되는 이미지는 두 개뿐**입니다.

| 저장되는 것 | 내용 |
|---|---|
| `agentview_image` | **위에서 내려다본** 화면 (256×256) |
| `robot0_eye_in_hand_image` | **손목** 카메라 (256×256) |
| state 16차원 | 관절 7 + 그리퍼 2 + 손끝 위치 3 + 손끝 회전 4 |
| `actions` | 7차원 액션 |
| `qpos` | MuJoCo 전체 상태 |

이미지가 2대인 것은 **LIBERO 와 같은 구성**에 맞추려고 그렇게 정한 것입니다.
바꾸지 마세요.

- **뷰어를 어느 각도로 돌려 놓든 저장되는 것은 같습니다.** 마음 편히 돌려 보세요.
- `qpos` 를 함께 저장하므로 **나중에 `frontview`·`sideview` 로 다시 렌더할 수 있습니다.**
  발표 영상은 이렇게 만들면 되고, 다시 녹화할 필요가 없습니다.

직접 확인해 보려면 (파일 이름은 자기 것으로 바꾸세요):

```powershell
.\.venv\Scripts\python.exe -c "import h5py,sys; f=h5py.File(sys.argv[1]); f.visit(print)" data\episodes\ep_0903_153012.hdf5
```

---

## 7. Colab (SmolVLA 파인튜닝)

노트북 학습은 Colab 에서 합니다. `notebooks\session12_vla_pipeline.ipynb` 를 여세요.
LeRobot 데이터셋 변환도 Colab 에서 합니다.

**공유 드라이브에 이렇게 들어 있어야 노트북이 찾습니다** (조교가 올려 둡니다).

```
MyDrive/VLA_Team_2026/
├── vla_pipeline/     ← 여러분이 받은 것과 같은 폴더
└── results/          ← 노트북이 결과를 여기에 저장
```

`0-2` 셀은 공유 폴더의 코드를 **Colab 런타임으로 복사한 뒤** 거기서 돌립니다.
공유 폴더 원본은 건드리지 않으니, 여러 명이 동시에 돌려도 서로 방해하지 않습니다.

여러분이 녹화한 에피소드를 학습에 쓰려면
`VLA_Team_2026/vla_pipeline/data/episodes/` 에 올리세요. `0-2` 를 다시 실행하면 반영되고,
**몇 개를 찾았는지 화면에 찍힙니다.** 0개로 나오면 올린 위치가 틀린 것입니다.

> **여기부터는 자기 노트북이 아니라 Colab(리눅스) 에서 치는 명령입니다.**
> `.\.venv\Scripts\python.exe` 가 아니라 그냥 `python` 입니다.
> 윈도우에서 `pip install lerobot` 은 실패합니다 — Python 3.14 윈도우 휠이 없어서이고,
> **정상입니다.**

```bash
# Colab 셀에서
!pip install -r requirements-colab.txt
!pip install "lerobot[smolvla] @ git+https://github.com/huggingface/lerobot.git"
!python scripts/build_dataset.py --lerobot --repo-id dshs2026/vla_sorting
```

> 이 부분(LeRobot 변환·SmolVLA 학습)은 **아직 아무도 끝까지 돌려 보지 않았습니다.**
> 프레임 구조까지만 검증돼 있습니다. 처음 돌리는 사람은 막히는 지점을 기록해 주세요.

완성한 데이터·노트북은 공유 드라이브 `VLA_Team_2026` 에 올립니다.

---

## 8. 문제 해결

### `ParseXML: Error opening file` — 파일은 분명히 있는데
경로에 한글이 있어서입니다. 코드가 자동 우회하지만 실패했다면,
프로젝트를 `C:\vla\vla_pipeline` 처럼 한글 없는 곳으로 옮기세요. (1절)

### `pygame` 설치가 소스 빌드로 넘어가며 실패
Python 3.13+ 에는 pygame 휠이 없습니다. `requirements-windows.txt` 가
자동으로 `pygame-ce` 를 대신 설치하도록 돼 있으니, 그냥
`pip install -r requirements-windows.txt` 를 다시 하세요.
수동으로 할 때는 `pip install pygame-ce` 입니다 (import 이름은 그대로 `pygame`).

### VSCode 가 빨간 줄로 "패키지 없음" 이라고 하는데 실행은 됨
인터프리터가 `.venv` 로 안 잡힌 것입니다. 3절 마지막 항목을 보세요.

### `python` 을 쳤더니 Microsoft Store 가 열림
윈도우의 가짜 `python.exe` 입니다. 이 문서대로 `.\.venv\Scripts\python.exe`
전체 경로를 쓰거나, `py` 명령을 쓰세요.

### 뷰어 창은 떴는데 키가 안 먹음
**창을 한 번 클릭해 포커스를 주세요.** 터미널에 포커스가 있으면 키가 안 갑니다.

### 뷰어 창이 아예 안 뜨거나 바로 꺼짐
그래픽 드라이버 문제일 수 있습니다. 먼저 `scripts\smoke_test.py` 를 돌려 보세요.
그건 창 없이 돌기 때문에, 통과한다면 시뮬레이션 자체는 정상이고 창만 문제입니다.
노트북 그래픽 드라이버를 업데이트해 보세요.

### 모델 다운로드가 너무 느림 / 중간에 끊김
다시 실행하면 받다 만 곳부터 이어집니다. 캐시는
`C:\Users\<이름>\.cache\huggingface\` 에 쌓입니다.

### `symlinks ... your machine does not support them` 경고
경고일 뿐 정상 동작합니다. 무시하세요.

### 그 밖에
`README.md` 9절에 더 있습니다. 그래도 안 되면 **에러 메시지 전문을 그대로 복사해서**
조교에게 보내 주세요. 스크린샷보다 텍스트가 낫습니다.

---

## 9. Claude Code 를 쓸 때

이 폴더에는 AI 에이전트용 안내 `AGENTS.md` 가 함께 들어 있습니다.
Claude Code 로 이 폴더를 열면 자동으로 읽습니다.

에이전트에게 시킬 때 도움이 되는 말:
- "설치가 잘 됐는지 확인해 줘" → `scripts/smoke_test.py` 를 돌려 줍니다.
- "액션 규격이 뭐야" → `vlapipe/config.py` 를 봐야 한다는 걸 압니다.
- 수치를 물어보면 **직접 돌려서 확인하라고 요구하세요.** 추측한 값과
  실제로 잰 값은 다릅니다. 우리 연구는 잰 값만 씁니다.

---

## 요약 — 명령어만

```powershell
# 설치
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt

# 확인
.\.venv\Scripts\python.exe scripts\check_env.py      # 설치됐나
.\.venv\Scripts\python.exe scripts\smoke_test.py     # 실제로 도나

# 써 보기
.\.venv\Scripts\python.exe scripts\run_pipeline.py --layout messy --seed 3
.\.venv\Scripts\python.exe scripts\teleop_record.py --input keyboard
.\.venv\Scripts\python.exe scripts\build_dataset.py

# VLM 까지 (선택)
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install transformers accelerate num2words
.\.venv\Scripts\python.exe scripts\smoke_test.py --vlm
```
