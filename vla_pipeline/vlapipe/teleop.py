"""텔레오퍼레이션 — 게임패드/키보드로 로봇을 직접 몰고, 그 결과를 녹화한다.

수업에서의 사용 흐름:
  1) 정돈된 씬에서 시작한다.
  2) 사람이 물체를 집어 아무 데나 옮기며 **어지른다** (녹화).
  3) build_dataset.py 가 이 녹화를 뒤집어 **정돈하는** 데모로 만든다.

왜 어지르는 쪽을 녹화하나?
  구역 안에 정확히 내려놓기는 손으로 하기 어렵지만, 구역에서 집어 아무 데나 놓기는 쉽다.
  뒤집으면 '정밀 배치' 데모가 공짜로 생긴다.

주의: 물체를 던지거나 떨어뜨리면 역재생이 물리적으로 불가능한 장면이 된다.
      항상 '집어서 - 옮겨서 - 내려놓기'로 천천히 움직일 것.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from . import config as C

# pygame 창이 하나라도 열려 있으면 SDL 은 그 창에 포커스가 없을 때 조이스틱 이벤트를
# 주지 않는다. 텔레옵은 카메라 뷰 창과 MuJoCo 뷰어 창을 오가며 쓰므로 반드시 켜야 한다.
# pygame.init() 보다 먼저여야 해서 모듈 임포트 시점에 둔다. (camview.py 에도 같은 설정)
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

DEADZONE = 0.15

# 정밀 모드 배율. 키보드는 Shift 를 누르고 있는 동안, 패드는 btn_slow 를 누르고 있는 동안.
SLOW_SCALE = 0.25

# 축별 기본 배율 (--speed 에 곱해진다).
#
# z 를 낮춰 둔 이유: 한 제어 스텝의 최대 이동이 MAX_DPOS = 0.05 m 인데, speed 0.55 면
# 스텝당 27.5 mm 다. 타일 높이가 30 mm 라서 **한 스텝만 내려도 물체를 지나친다.**
# 게다가 z 를 해트(십자키)에 매핑한 기기에서는 값이 -1/0/+1 뿐이라 살살 누를 수가 없다.
# 0.35 면 스텝당 9.6 mm 로, 타일 높이의 1/3 씩 내려간다.
AXIS_GAIN = {"gain_x": 1.0, "gain_y": 1.0, "gain_z": 0.35, "gain_yaw": 1.0}


def _dz(v: float, dz: float = DEADZONE) -> float:
    return 0.0 if abs(v) < dz else float(np.sign(v) * (abs(v) - dz) / (1 - dz))


class GamepadInput:
    """pygame joystick. 축·버튼 번호는 기기마다 다르므로 --calibrate 로 확인해 조정한다.

    축 항목("x"/"y"/"z"/"yaw")에는 두 가지를 넣을 수 있다.
      정수      : 아날로그 축 번호. 예) 0
      "hatN_x"  : 해트(십자키/POV)의 가로. "hatN_y" 는 세로.
    해트는 -1/0/+1 만 나오지만 **손을 떼면 반드시 0 으로 돌아온다.**
    비행 스틱의 스로틀 레버처럼 중립 복귀가 없는 축을 z 에 쓰면 로봇이 계속
    올라가거나 내려가므로, 그런 기기에서는 z 를 해트로 옮기는 편이 안전하다.
    """

    DEFAULT_MAP = {
        "x": 0,          # 왼쪽 스틱 좌우  -> 월드 +x
        "y": 1,          # 왼쪽 스틱 상하  -> 월드 +y (위로 밀면 앞으로)
        "z": 3,          # 오른쪽 스틱 상하 -> 월드 +z
        "yaw": 2,        # 오른쪽 스틱 좌우 -> 손목 회전
        "invert_x": False,
        "invert_y": True,
        "invert_z": True,
        "invert_yaw": False,
        "btn_grip": 0,       # A  : 그리퍼 토글
        "btn_record": 1,     # B  : 녹화 시작/정지
        "btn_reset": 2,      # X  : 정돈 상태로 리셋
        "btn_discard": 3,    # Y  : 현재 녹화 폐기
        # 누르고 있는 동안 모든 축이 느려진다(정밀 모드). 없으면 None.
        "btn_slow": None,
        **AXIS_GAIN,
    }

    AXIS_ROLES = ("x", "y", "z", "yaw")
    BUTTON_ROLES = ("btn_grip", "btn_record", "btn_reset", "btn_discard")

    def __init__(self, index: int = 0, mapping: dict | None = None, speed: float = 0.6):
        import pygame

        self.pygame = pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("게임패드를 찾지 못했습니다. USB 연결을 확인하거나 --input keyboard 를 쓰세요.")
        self.js = pygame.joystick.Joystick(index)
        self.js.init()
        self.map = {**self.DEFAULT_MAP, **(mapping or {})}
        self.speed = speed
        self.gripper = C.GRIPPER_OPEN_CMD
        self._prev_btn: dict[int, bool] = {}
        self.events: list[str] = []
        self.quit = False
        print(f"게임패드: {self.js.get_name()}  축 {self.js.get_numaxes()}개, "
              f"버튼 {self.js.get_numbuttons()}개, 해트 {self.js.get_numhats()}개")

    def _hat(self, spec: str) -> float:
        """'hat0_y' 같은 표기를 읽는다. 해트는 -1/0/+1 이고 손을 떼면 0 으로 돌아온다."""
        try:
            n, comp = spec[3:].split("_")
            idx, n = int(n), None
        except (ValueError, IndexError):
            raise ValueError(f"해트 표기가 잘못됐습니다: {spec!r} (예: 'hat0_y')")
        if idx >= self.js.get_numhats():
            return 0.0
        hx, hy = self.js.get_hat(idx)
        return float(hx if comp == "x" else hy)

    def _axis(self, key: str) -> float:
        i = self.map[key]
        if i is None:
            return 0.0
        if isinstance(i, str):                     # 해트 (중립 복귀 있음)
            v = self._hat(i)
        elif i >= self.js.get_numaxes():
            return 0.0
        else:
            v = _dz(self.js.get_axis(i))
        return -v if self.map.get(f"invert_{key}") else v

    def _pressed(self, key: str) -> bool:
        i = self.map[key]
        if i is None or not isinstance(i, int) or i >= self.js.get_numbuttons():
            return False
        cur = bool(self.js.get_button(i))
        edge = cur and not self._prev_btn.get(i, False)
        self._prev_btn[i] = cur
        return edge

    def _held(self, key: str) -> bool:
        """눌린 순간(edge)이 아니라 **누르고 있는 동안** 참. 정밀 모드용."""
        i = self.map.get(key)
        if i is None or not isinstance(i, int) or i >= self.js.get_numbuttons():
            return False
        return bool(self.js.get_button(i))

    def poll(self) -> np.ndarray:
        self.pygame.event.pump()
        # events 를 여기서 비우지 않는다. 뷰어 스레드의 key_callback 이 프레임 사이에
        # 넣은 이벤트가 지워져 버리기 때문. 비우는 것은 메인 루프가 처리한 뒤에 한다
        # (KeyboardInput 과 같은 방식).
        if self._pressed("btn_grip"):
            self.gripper = (C.GRIPPER_CLOSE_CMD if self.gripper == C.GRIPPER_OPEN_CMD
                            else C.GRIPPER_OPEN_CMD)
        for name, key in (("record", "btn_record"), ("reset", "btn_reset"), ("discard", "btn_discard")):
            if self._pressed(key):
                self.events.append(name)
        # 정밀 모드: btn_slow 를 누르고 있는 동안 전 축이 느려진다.
        self.slow = self._held("btn_slow")
        s = self.speed * (SLOW_SCALE if self.slow else 1.0)
        a = np.zeros(C.ACTION_DIM, dtype=np.float32)
        a[0] = self._axis("x") * s * self.map["gain_x"]
        a[1] = self._axis("y") * s * self.map["gain_y"]
        a[2] = self._axis("z") * s * self.map["gain_z"]
        a[5] = self._axis("yaw") * s * self.map["gain_yaw"]
        a[6] = self.gripper
        return a

    # 패드로 몰더라도 뷰어의 키 몇 개는 받는다. 특히 P(배치 저장)는 패드에 버튼이 없어서
    # 이게 없으면 게임패드 모드에서 씬 편집 결과를 저장할 방법이 아예 사라진다.
    # 이동은 패드가 맡으므로 여기서는 '사건' 키만 처리한다.
    KEY_HELP = ("[보조 키] R: 리셋   X: 녹화 폐기   P: 현재 배치를 JSON 으로 저장   ESC: 종료\n"
                "          (이동·그리퍼·녹화는 패드로. 키는 '카메라 뷰' 창에서 받습니다 —\n"
                "           MuJoCo 뷰어 창에서 누르면 화면 표시 옵션이 바뀝니다)")

    def key_callback(self, keycode: int) -> None:
        if keycode == 256:                    # Esc
            self.quit = True
            return
        try:
            ch = chr(keycode).upper()
        except ValueError:
            return
        if ch == "R":
            self.events.append("reset")
        elif ch == "X":
            self.events.append("discard")
        elif ch == "P":
            self.events.append("dump_layout")

    def describe(self) -> str:
        """지금 적용된 매핑을 사람이 읽을 수 있게. 기본값인지 캘리브레이션 결과인지 헷갈리지 않도록."""
        def ax(k):
            v = self.map[k]
            name = f"해트 {v}" if isinstance(v, str) else f"축 {v}"
            return name + (" (반전)" if self.map.get(f"invert_{k}") else "")
        mm = self.speed * self.map["gain_z"] * C.MAX_DPOS * 1000
        slow = (f"  정밀=버튼 {self.map['btn_slow']}" if self.map.get("btn_slow") is not None
                else "  정밀버튼 없음(--calibrate 에서 btn_slow 지정 가능)")
        return (f"[패드] 좌우={ax('x')}  앞뒤={ax('y')}  높이={ax('z')}  손목={ax('yaw')}\n"
                f"       그리퍼=버튼 {self.map['btn_grip']}  녹화=버튼 {self.map['btn_record']}  "
                f"리셋=버튼 {self.map['btn_reset']}  폐기=버튼 {self.map['btn_discard']}{slow}\n"
                f"       높이 배율 {self.map['gain_z']} -> 한 스텝 약 {mm:.1f} mm "
                f"(--speed-z 로 조절)")

    def record_hint(self) -> str:
        return f"버튼 {self.map['btn_record']}"

    def _info(self) -> str:
        return (f"{self.js.get_name()} — 축 {self.js.get_numaxes()}개, "
                f"버튼 {self.js.get_numbuttons()}개, 해트 {self.js.get_numhats()}개")

    def calibrate_raw(self) -> None:
        """살아 있는 축·버튼 값을 그대로 보여 준다 (진단용)."""
        print(self._info())
        print(f"현재 매핑: {json.dumps(self.map, ensure_ascii=False)}")
        print("축을 움직이고 버튼을 눌러 어떤 번호가 반응하는지 보세요. Ctrl+C 로 종료.")
        try:
            while True:
                self.pygame.event.pump()
                ax = [f"{i}:{self.js.get_axis(i):+.2f}" for i in range(self.js.get_numaxes())]
                bt = [i for i in range(self.js.get_numbuttons()) if self.js.get_button(i)]
                ht = [self.js.get_hat(i) for i in range(self.js.get_numhats())]
                print(f"\r축 {' '.join(ax)} | 해트 {ht} | 눌린버튼 {bt}    ", end="", flush=True)
                time.sleep(0.05)
        except KeyboardInterrupt:
            print()

    # ---- 캘리브레이션 마법사 --------------------------------------------
    # 학생이 축 번호를 눈으로 읽어 적는 대신, 시키는 대로 움직이면 매핑을 만들어 준다.

    AXIS_THRESH = 0.45          # 이만큼은 움직여야 '의도한 조작'으로 본다

    def _pump(self, seconds: float) -> None:
        """SDL 이 장치 상태를 채우도록 일정 시간 이벤트를 돌린다.

        pygame 은 첫 event.pump() 직후에는 축 값이 아직 0.0 으로 비어 있을 수 있다.
        그 상태를 기준값으로 잡으면, 실제로는 한쪽에 가 있는 스로틀 축이
        '방금 크게 움직인 축'으로 잘못 잡힌다. 그래서 항상 예열한 뒤 읽는다.
        """
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            self.pygame.event.pump()
            time.sleep(0.02)

    def _snapshot(self) -> tuple[list[float], list[tuple[int, int]]]:
        self.pygame.event.pump()
        return ([self.js.get_axis(i) for i in range(self.js.get_numaxes())],
                [self.js.get_hat(i) for i in range(self.js.get_numhats())])

    def _countdown(self, n: int = 3) -> None:
        for k in range(n, 0, -1):
            print(f"\r      {k} ...   ", end="", flush=True)
            self._pump(1.0)
        print("\r      >>> 지금 조작하세요 (3초)      ", flush=True)

    def _detect_axis(self, duration: float = 3.0):
        """직전 상태를 기준으로, duration 동안 **가장 크게 변한** 축/해트를 고른다.

        (스펙, 부호, 변화량, 기준값) 또는 None.
        '먼저 문턱을 넘은 축'이 아니라 '가장 크게 변한 축'을 고르는 것이 핵심이다.
        전자는 이미 한쪽으로 밀려 있는 스로틀 축이 매번 이겨 버린다.
        """
        base_ax, base_ht = self._snapshot()     # 이 단계 직전 상태가 기준
        best = (0.0, None, 0)
        hat_hit = None
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            ax, ht = self._snapshot()
            for i, (h, hb) in enumerate(zip(ht, base_ht)):
                for comp, cur, b in (("x", h[0], hb[0]), ("y", h[1], hb[1])):
                    if cur != b and hat_hit is None:
                        hat_hit = (f"hat{i}_{comp}", 1 if cur - b > 0 else -1)
            for i, (a, b) in enumerate(zip(ax, base_ax)):
                d = a - b
                if abs(d) > best[0]:
                    best = (abs(d), i, 1 if d > 0 else -1)
            time.sleep(0.02)

        if hat_hit:            # 해트는 디지털이라 눌렀다는 것 자체가 분명한 의도다
            return hat_hit[0], hat_hit[1], 1.0, 0.0
        mag, i, sign = best
        if i is None or mag < self.AXIS_THRESH:
            return None
        return i, sign, mag, base_ax[i]

    def _returns_to_neutral(self, spec, base_val: float, wait: float = 1.2) -> bool:
        """손을 뗀 뒤 원래 위치로 돌아오는 축인지 본다 (스프링이 있는가)."""
        if isinstance(spec, str):
            return True                      # 해트는 항상 돌아온다
        self._pump(wait)
        ax, _ = self._snapshot()
        return abs(ax[spec] - base_val) < 0.25

    def _detect_button(self, duration: float = 10.0):
        """**새로** 눌린 버튼 하나를 찾는다. 이미 눌려 있던 것은 무시한다."""
        self._pump(0.3)
        n = self.js.get_numbuttons()
        held = {i for i in range(n) if self.js.get_button(i)}
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            self.pygame.event.pump()
            for i in range(n):
                if not self.js.get_button(i):
                    held.discard(i)
                elif i not in held:
                    while self.js.get_button(i):      # 뗄 때까지 기다린다
                        self.pygame.event.pump()
                        time.sleep(0.02)
                    return i
            time.sleep(0.02)
        return None

    @staticmethod
    def _retry(msg: str) -> bool:
        return input(f"      {msg}\n      다시 하려면 ENTER, 건너뛰려면 s + ENTER: "
                     ).strip().lower() != "s"

    def calibrate(self, out_path: Path | str | None = None) -> dict:
        """시키는 대로 움직이면 매핑 JSON 을 만들어 준다."""
        AXIS_STEPS = [
            ("x",   "로봇을 **오른쪽**(월드 +x)으로 보낼 조작을 끝까지 하세요"),
            ("y",   "로봇을 **앞쪽**(월드 +y, 로봇에서 멀어지는 쪽)으로 보낼 조작을 하세요"),
            ("z",   "로봇을 **위로** 올릴 조작을 하세요"),
            ("yaw", "손목을 **반시계 방향**으로 돌릴 조작을 하세요"),
        ]
        BUTTON_STEPS = [
            ("btn_grip",    "그리퍼를 여닫을 버튼"),
            ("btn_record",  "녹화를 시작/정지할 버튼"),
            ("btn_reset",   "씬을 리셋할 버튼"),
            ("btn_discard", "녹화를 버릴 버튼"),
            # 선택 항목. 잡을 때 천천히 내려가려면 있는 편이 훨씬 편하다.
            ("btn_slow",    "[선택] 누르고 있는 동안 **천천히** 움직일 버튼"
                            " (물체를 잡을 때 씀. 없으면 s 로 건너뛰기)"),
        ]

        print(f"\n캘리브레이션 — {self._info()}")
        print("높이(z)는 스로틀 레버보다 **십자키(해트)** 를 권합니다. 손을 떼면 멈추기 때문입니다.\n")
        print("각 단계에서 ENTER 를 누르면 3초 카운트다운 뒤 '지금 조작하세요' 가 뜹니다.")
        print("그때 해당 조작을 **끝까지, 3초 동안 유지**하세요. 한 번에 하나만 움직입니다.")
        print("Ctrl+C 로 중단합니다.\n")

        print("장치 상태를 읽는 중...")
        self._pump(0.8)                       # SDL 이 실제 축 값을 채울 때까지 기다린다
        ax0, ht0 = self._snapshot()
        print(f"현재 축 값: {[round(v, 2) for v in ax0]}   해트: {ht0}")
        if any(abs(v) > 0.5 for v in ax0):
            hot = [i for i, v in enumerate(ax0) if abs(v) > 0.5]
            print(f"(축 {hot} 은 지금 중립이 아닙니다 — 스로틀/슬라이더일 수 있습니다)")
        print()

        mapping: dict = {}
        used_axis: dict = {}          # 축·해트 전용
        used_btn: dict = {}           # 버튼 전용 (섞으면 축 0 과 버튼 0 이 충돌한다)
        no_spring: list = []
        try:
            for role, prompt in AXIS_STEPS:
                while True:
                    input(f"[{role}] {prompt}\n      준비되면 ENTER... ")
                    self._countdown()
                    got = self._detect_axis()
                    if got is None:
                        if self._retry("움직임을 감지하지 못했습니다."):
                            continue
                        print(f"      -> 건너뜀. 기본값({self.map[role]}) 유지\n")
                        break
                    spec, sign, mag, base_val = got
                    if spec in used_axis:
                        if self._retry(f"이미 '{used_axis[spec]}' 에 쓴 조작입니다."):
                            continue
                    kind = "해트" if isinstance(spec, str) else "축"
                    print(f"      -> {kind} {spec}, 부호 반전 {'예' if sign < 0 else '아니오'} "
                          f"(변화량 {mag:.2f})")
                    if not self._returns_to_neutral(spec, base_val):
                        no_spring.append((role, spec))
                        print(f"      ! 이 축은 손을 뗀 뒤에도 제자리로 돌아오지 않습니다.")
                    mapping[role] = spec
                    mapping[f"invert_{role}"] = (sign < 0)
                    used_axis[spec] = role
                    print()
                    break

            for role, prompt in BUTTON_STEPS:
                while True:
                    # 선택 항목은 처음부터 건너뛸 수 있어야 한다. 10초를 기다리게 하면 안 된다.
                    ans = input(f"[{role}] {prompt}\n"
                                f"      누를 준비가 되면 ENTER (건너뛰려면 s + ENTER): ")
                    if ans.strip().lower() == "s":
                        print(f"      -> 건너뜀. 기본값({self.map[role]}) 유지\n")
                        break
                    b = self._detect_button()
                    if b is None:
                        if self._retry("버튼 입력을 감지하지 못했습니다."):
                            continue
                        print(f"      -> 건너뜀. 기본값({self.map[role]}) 유지\n")
                        break
                    if b in used_btn:
                        if self._retry(f"버튼 {b} 는 이미 '{used_btn[b]}' 에 배정됐습니다."):
                            continue
                    mapping[role] = b
                    used_btn[b] = role
                    print(f"      -> 버튼 {b}\n")
                    break
        except KeyboardInterrupt:
            print("\n중단했습니다. 여기까지 찾은 것만 저장합니다.")

        if not mapping:
            print("\n아무것도 감지하지 못했습니다. `--calibrate-raw` 로 직접 확인해 보세요.")
            return {}

        # 중립 복귀가 없는 축을 높이(z)에 쓰면 로봇이 멈추지 않는다.
        for role, spec in no_spring:
            if role == "z":
                print(f"\n⚠ 높이(z)에 배정한 축 {spec} 는 스프링이 없습니다.")
                print("  손을 떼도 로봇이 계속 오르내려 테이블에 부딪힙니다.")
                if self.js.get_numhats() > 0:
                    print('  십자키(해트)로 바꾸는 것을 권합니다.  "z": "hat0_y"')
                    if input("      지금 바꿀까요? [Y/n]: ").strip().lower() not in ("n", "no"):
                        mapping["z"] = "hat0_y"
                        mapping["invert_z"] = False
                        print('      -> z 를 "hat0_y" 로 바꿨습니다.')

        print(f"\n찾은 매핑:\n{json.dumps(mapping, ensure_ascii=False, indent=2)}")
        if out_path:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n저장했습니다: {p}")
            print(f"다음부터는:  --input gamepad --gamepad-map {p}")
        return mapping


def load_gamepad_map(path: Path | str | None) -> dict | None:
    """--gamepad-map 으로 넘긴 JSON 을 읽는다. 소스를 고치지 않고 축/버튼 번호를 바꾸기 위한 것.

    DEFAULT_MAP 과 같은 키만 쓰며, 적은 키만 덮어쓴다. 예:
        {"z": 2, "yaw": 3, "invert_z": false, "btn_record": 5}
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"매핑 파일이 없습니다: {p}\n"
            f"먼저 `python scripts/teleop_record.py --calibrate` 로 만드세요.")
    # utf-8-sig: 윈도우 메모장이나 PowerShell 의 Out-File 은 BOM 을 붙이는데,
    # 그냥 utf-8 로 읽으면 JSONDecodeError 가 난다. BOM 이 없어도 그대로 읽힌다.
    m = json.loads(p.read_text(encoding="utf-8-sig"))
    unknown = set(m) - set(GamepadInput.DEFAULT_MAP)
    if unknown:
        raise ValueError(f"매핑 파일에 모르는 키가 있습니다: {sorted(unknown)}\n"
                         f"쓸 수 있는 키: {sorted(GamepadInput.DEFAULT_MAP)}")
    return m


class KeyboardInput:
    """MuJoCo viewer 의 key_callback 으로 받는 키보드 입력 (패드가 없을 때)."""

    HELP = ("[조작] W/S: 앞뒤(+y/-y)  A/D: 좌우(-x/+x)  Q/E: 위/아래(z)  Z/C: 손목회전\n"
            "       SHIFT 를 누른 채로 = 정밀 모드 (천천히)\n"
            "       SPACE: 그리퍼 열고닫기   ENTER: 녹화 시작/정지   R: 정돈 상태로 리셋\n"
            "       X: 녹화 폐기   P: 현재 배치를 JSON 으로 저장   ESC: 종료\n"
            "       ※ 키는 '카메라 뷰' 창에서만 먹습니다. MuJoCo 뷰어 창에서 누르면\n"
            "          로봇이 아니라 화면 표시 옵션(Wireframe 등)이 바뀝니다.")

    def __init__(self, speed: float = 0.55, gain: dict | None = None):
        self.speed = speed
        # 축별 배율. z 를 낮춰 두는 이유는 AXIS_GAIN 주석 참고.
        self.gain = {**AXIS_GAIN, **(gain or {})}
        self.gripper = C.GRIPPER_OPEN_CMD
        self.held: set[str] = set()
        self.events: list[str] = []
        self.quit = False
        self.slow = False        # SHIFT 를 누르고 있는 동안 참 (카메라 뷰 창이 설정)

    def key_callback(self, keycode: int) -> None:
        try:
            ch = chr(keycode).upper()
        except ValueError:
            return
        if ch == " ":
            self.gripper = (C.GRIPPER_CLOSE_CMD if self.gripper == C.GRIPPER_OPEN_CMD
                            else C.GRIPPER_OPEN_CMD)
        elif keycode in (257, 335):          # Enter
            self.events.append("record")
        elif ch == "R":
            self.events.append("reset")
        elif ch == "X":
            self.events.append("discard")
        elif ch == "P":
            self.events.append("dump_layout")
        elif keycode == 256:                 # Esc
            self.quit = True
        elif ch in "WSADQEZC":
            self.held.add(ch)

    # 키 -> (액션 인덱스, 부호, 어느 축의 배율을 쓰는가)
    _KEY_AXIS = {
        "W": (1, +1, "gain_y"), "S": (1, -1, "gain_y"),
        "D": (0, +1, "gain_x"), "A": (0, -1, "gain_x"),
        "Q": (2, +1, "gain_z"), "E": (2, -1, "gain_z"),
        "Z": (5, +1, "gain_yaw"), "C": (5, -1, "gain_yaw"),
    }

    def poll(self) -> np.ndarray:
        a = np.zeros(C.ACTION_DIM, dtype=np.float32)
        s = self.speed * (SLOW_SCALE if self.slow else 1.0)
        for ch in self.held:
            idx, sign, gain_key = self._KEY_AXIS.get(ch, (None, 0, None))
            if idx is not None:
                a[idx] += sign * s * self.gain[gain_key]
        a[6] = self.gripper
        self.held.clear()      # 키를 누르고 있는 동안 반복 입력이 들어온다
        return a


def dump_layout(env, path: Path) -> Path:
    """현재 물체 배치를 JSON 으로 저장 (씬 편집 결과 보존)."""
    lay = {}
    for n in C.OBJECT_NAMES:
        p = env.object_pos(n)
        lay[n] = [round(float(p[0]), 4), round(float(p[1]), 4), round(float(env.object_yaw(n)), 4)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lay, indent=2), encoding="utf-8")
    return path


def load_layout(env, path: Path) -> None:
    lay = json.loads(Path(path).read_text(encoding="utf-8-sig"))   # BOM 허용
    for n, (x, y, yaw) in lay.items():
        if n in C.OBJECT_NAMES:
            env.set_object_pose(n, x, y, yaw, forward=False)
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    env._sim_step(80)
    env.ctrl.sync_target_to_current()
