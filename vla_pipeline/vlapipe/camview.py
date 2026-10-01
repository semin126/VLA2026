"""텔레옵 중에 여러 카메라 시점을 한 창에 띄운다.

왜 필요한가?
  MuJoCo 뷰어는 시점이 하나뿐이라, 팔이 물체를 가리면 무엇을 집는지 보이지 않는다.
  시점을 돌리면 이번엔 조작 방향 감각이 틀어진다. 그래서 조작은 뷰어에서 하고,
  확인은 **고정 카메라 여러 대를 동시에** 보는 편이 낫다.

보기 전용이 아니라 **키보드 조작도 이 창에서 받는다.** 이유가 있다:

  MuJoCo 뷰어(simulate)는 알파벳 거의 전부에 렌더링 토글 단축키를 갖고 있다.
  우리 조작키 11개가 **전부** 충돌한다:
    W=Wireframe  S=Shadow  A=Auto Connect  D=Static Body  Q=Camera
    E=Equality   Z=Light   C=Contact Point R=Reflection    X=Texture  P=Contact Split
  그래서 뷰어 창에 포커스를 두고 WASD 를 누르면 로봇이 움직이는 대신
  화면이 와이어프레임이 되고 테이블(static body)이 사라진다.
  키 목록은 `mujoco.mjVISSTRING` / `mujoco.mjRNDSTRING` 에서 확인할 수 있다.

  피할 방법이 없다(뷰어의 키 처리는 C++ 쪽이라 파이썬에서 막지 못한다).
  그래서 **조작 키는 이 pygame 창에서 받는다.** 여기에는 충돌하는 단축키가 없다.

녹화되는 관측과는 무관하다(그건 env.cameras 가 정한다). 여기에 무엇을 띄우든
학습 데이터는 달라지지 않는다.
"""

from __future__ import annotations

import math
import os

import numpy as np

from . import config as C
from .env import SortingEnv

# 게임패드가 '창이 포커스를 잃으면' 죽는 문제를 막는다.
# pygame 창이 하나라도 있으면 SDL 은 기본적으로 포커스가 없을 때 조이스틱 이벤트를
# 주지 않는다. 조작은 이 창에서, 시선은 뷰어 창에 두는 일이 잦으므로 반드시 켜야 한다.
# pygame.init() 보다 먼저 설정돼야 하므로 모듈 임포트 시점에 둔다.
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

# 화면에 붙일 이름표. ASCII 로만 쓴다 — PIL 기본 폰트에 한글 글리프가 없어서
# 한글을 넣으면 네모(두부)로 나온다.
LABELS = {
    "agentview": "agentview (top, RECORDED)",
    "frontview": "frontview (front)",
    "sideview": "sideview (side)",
    "robot0_eye_in_hand": "wrist (RECORDED)",
}

DEFAULT_CAMERAS = ("agentview", "frontview", "robot0_eye_in_hand")
HEADER_H = 16
PAD = 4
BG = (24, 24, 28)
FG = (235, 235, 235)


def _label_strip(text: str, width: int) -> np.ndarray:
    """이름표 띠 하나를 그린다."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (width, HEADER_H), BG)
    d = ImageDraw.Draw(img)
    d.text((3, 2), text[: max(1, width // 6)], fill=FG)
    return np.asarray(img)


def tile(env: SortingEnv, cameras=DEFAULT_CAMERAS, size: int = 240,
         cols: int | None = None, labels: bool = True) -> np.ndarray:
    """카메라 여러 대를 렌더해서 격자 한 장으로 붙인다 (창 없이도 되는 순수 함수)."""
    cams = [c for c in cameras if c]
    if not cams:
        raise ValueError("카메라를 하나 이상 지정하세요.")
    unknown = [c for c in cams if c not in C.ALL_CAMERAS]
    if unknown:
        raise ValueError(f"모르는 카메라: {unknown}\n쓸 수 있는 것: {list(C.ALL_CAMERAS)}")

    cols = cols or min(len(cams), 3)
    rows = math.ceil(len(cams) / cols)
    cell_h = size + (HEADER_H if labels else 0)
    out = np.full((rows * cell_h + (rows + 1) * PAD,
                   cols * size + (cols + 1) * PAD, 3), BG, dtype=np.uint8)

    for i, cam in enumerate(cams):
        r, c = divmod(i, cols)
        y = PAD + r * (cell_h + PAD)
        x = PAD + c * (size + PAD)
        if labels:
            out[y:y + HEADER_H, x:x + size] = _label_strip(LABELS.get(cam, cam), size)
            y += HEADER_H
        out[y:y + size, x:x + size] = env.render(cam, size, size)
    return out


# 계속 누르고 있는 동안 움직이는 키 (이동·회전). 매 프레임 눌림 상태를 읽는다.
CONTINUOUS = "WSADQEZC"
# 한 번 누를 때 한 번만 반응해야 하는 키 (토글·사건).
# pygame 키코드 -> KeyboardInput.key_callback 이 기대하는 GLFW 키코드로 맞춘다.
ONESHOT = {
    32: 32,        # SPACE : 그리퍼
    13: 257,       # Enter : 녹화 시작/정지
    271: 335,      # 키패드 Enter
    27: 256,       # Esc   : 종료
    ord("r"): ord("r"),
    ord("x"): ord("x"),
    ord("p"): ord("p"),
}


class CameraPanel:
    """고정 카메라 여러 대를 pygame 창 하나에 실시간으로 띄운다.

    pygame 은 이미 게임패드 때문에 깔려 있으므로 새로 설치할 것이 없다.
    """

    def __init__(self, env: SortingEnv, cameras=DEFAULT_CAMERAS, size: int = 240,
                 cols: int | None = None, every: int = 1,
                 title: str = "카메라 뷰 — 조작하려면 이 창을 클릭하세요"):
        import pygame

        self.pygame = pygame
        self.env = env
        self.cameras = tuple(cameras)
        self.size = size
        self.cols = cols
        self.every = max(1, int(every))
        self._n = 0
        self.alive = True
        self.quit = False

        pygame.init()
        pygame.display.init()
        frame = tile(env, self.cameras, size, cols)          # 창 크기를 정하려고 한 장 그려 본다
        self.screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
        pygame.display.set_caption(title)
        self._blit(frame)

    def _blit(self, frame: np.ndarray) -> None:
        surf = self.pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        self.screen.blit(surf, (0, 0))
        self.pygame.display.flip()

    def update(self) -> None:
        """한 프레임 갱신. --panel-every 로 몇 프레임에 한 번만 그릴 수도 있다."""
        if not self.alive:
            return
        self._n += 1
        if self._n % self.every:
            return
        try:
            self._blit(tile(self.env, self.cameras, self.size, self.cols))
        except Exception:
            # 보기용 창 때문에 녹화가 끊기면 안 된다. 창만 포기한다.
            self.alive = False

    def pump_input(self, src) -> None:
        """이 창의 키 입력을 입력 객체(src)로 흘려보낸다.

        이동 키는 **누르고 있는 동안** 계속 들어가야 하므로 `key.get_pressed()` 로 읽고,
        토글·사건 키는 한 번만 반응해야 하므로 KEYDOWN 이벤트로 읽는다.
        (키 반복에 기대면 OS 의 '반복 시작 지연' 때문에 처음 0.5초가 끊긴다.)

        창에 포커스가 없으면 get_pressed() 가 전부 False 라 저절로 멈춘다.
        게임패드는 SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS 덕분에 포커스와 무관하게 동작한다.
        """
        if not self.alive:
            return
        cb = getattr(src, "key_callback", None)

        for ev in self.pygame.event.get():
            if ev.type == self.pygame.QUIT:
                self.close()
                return
            if ev.type == self.pygame.KEYDOWN and cb is not None:
                mapped = ONESHOT.get(ev.key)
                if mapped is not None:
                    cb(mapped)

        if cb is None:
            return
        pressed = self.pygame.key.get_pressed()
        for ch in CONTINUOUS:
            if pressed[getattr(self.pygame, f"K_{ch.lower()}")]:
                cb(ord(ch.lower()))

        # SHIFT 를 누르고 있는 동안 정밀 모드. 누름/뗌이 아니라 '상태'라서 매 프레임 갱신한다.
        # 패드는 자기 btn_slow 를 스스로 보므로 건드리지 않는다.
        if hasattr(src, "slow") and not hasattr(src, "js"):
            src.slow = bool(self.pygame.key.get_mods() & self.pygame.KMOD_SHIFT)

    def focus_hint(self, input_kind: str = "keyboard") -> str:
        if input_kind == "gamepad":
            # 패드는 SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS 덕분에 포커스와 무관하다.
            return ("패드는 어느 창을 보고 있든 동작합니다. "
                    "다만 R/X/P/ESC 키는 이 '카메라 뷰' 창에서만 먹습니다.")
        return ("★ 조작하려면 이 '카메라 뷰' 창을 클릭해 포커스를 주세요. "
                "MuJoCo 뷰어 창에서 키를 누르면 로봇이 아니라 화면 표시가 바뀝니다.")

    def close(self) -> None:
        if self.alive:
            self.alive = False
            try:
                self.pygame.display.quit()
            except Exception:
                pass
