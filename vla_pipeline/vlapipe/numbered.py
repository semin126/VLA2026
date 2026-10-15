"""숫자로 물체를 구분하는 실험 조건 — 색·모양 단서를 없앤 조건.

무엇이 달라지는가
-----------------
물체 10개가 **전부 똑같은 흰 정육면체**(0.05 m)이고, 여섯 면에 1~10 숫자만 찍혀 있다.
색도 크기도 모양도 같으므로 숫자 말고는 구별할 방법이 없다.

    1단계 VLM : 숫자를 '읽어야' 계획을 세울 수 있다
                ("object 7 을 왼쪽 보관 구역으로")
    2단계 VLA : 지시문에 적힌 숫자로 대상을 찾아야 한다

바뀌지 않는 것 (비교가 성립하려면 반드시 같아야 하는 것)
    - 액션/관측 규격 (7차원 델타 액션, 16차원 state — LIBERO 호환)
    - 테이블·구역·조명·카메라·로봇
    - 배치 생성(tidy/messy/random/cluttered)과 시드
    - 지시문 틀 ("Pick up the X and place it in the Y.")
      -> X 자리에 들어가는 말만 "red block" -> "object 7" 로 바뀐다

정돈 규칙
---------
색 세트의 '모양 규칙'(긴 블록=오른쪽, 정육면체·타일=왼쪽)과 같은 자리를 숫자 규칙이 차지한다.
    1~5 번  -> 오른쪽 보관 구역
    6~10 번 -> 왼쪽 보관 구역
숫자와 구역 개수는 `config.NUMBER_SPLIT` / `NUMBER_COUNT` 에 있다.

쓰는 법
-------
    from vlapipe import numbered
    from vlapipe.env import SortingEnv

    numbered.activate()          # 물체 세트 교체 + 숫자 텍스처 생성  <- env 보다 먼저!
    env = SortingEnv(layout="messy", seed=3)

    # 규칙 플래너(정답)는 그대로 쓰면 된다
    from vlapipe.planner import RulePlanner, VLMPlanner
    plan = RulePlanner(env).plan()
    # VLM 플래너는 숫자용 프롬프트로
    plan = VLMPlanner(prompt_style="number").plan(env.render("agentview", 512, 512))

명령줄에서는 `scripts/run_pipeline.py --objects number`,
숫자가 제대로 찍혔는지 눈으로 확인하려면 `scripts/check_numbers.py`.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from . import config as C
from . import planner as planner_mod
from . import scene as scene_mod

# 숫자 텍스처가 저장되는 곳. 그림 파일이라 git 에 넣지 않고, 없으면 그때 만든다.
TEXTURE_DIR = C.ASSET_DIR / "number_textures"

# 숫자를 그릴 글꼴 후보. 윈도우 -> Arial Bold, 리눅스/Colab -> DejaVu.
# 하나도 없으면 Pillow 내장 글꼴로 떨어진다(Pillow 10.1+ 는 크기 지정이 된다).
_FONT_CANDIDATES = ("arialbd.ttf", "ariblk.ttf", "segoeuib.ttf",
                    "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial Bold.ttf")


# --------------------------------------------------------------------------
# 숫자 텍스처 만들기
# --------------------------------------------------------------------------
def _font(px: int):
    from PIL import ImageFont

    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=px)     # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()           # 아주 작게 나온다 (최후의 수단)


def _fit_font(draw, text: str, max_w: float, max_h: float, start_px: int):
    """글자가 면 안에 들어오는 가장 큰 글꼴 크기를 찾는다 ('10' 은 두 자라서 작아진다)."""
    px = start_px
    while px > 8:
        f = _font(px)
        l, t, r, b = draw.textbbox((0, 0), text, font=f)
        if (r - l) <= max_w and (b - t) <= max_h:
            return f
        px = int(px * 0.92)
    return _font(8)


def _draw_number(n: int, size: int):
    """숫자 한 개가 찍힌 정사각형 텍스처 이미지.

    테두리 : 밝은 테이블(0.82) 위에서 물체 윤곽이 보이도록
    밑줄   : 6 과 9 처럼 뒤집히면 헷갈리는 숫자의 '아래쪽'을 알려 준다
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), C.NUMBER_FACE_RGB)
    d = ImageDraw.Draw(img)
    edge = max(2, size // 32)
    d.rectangle([0, 0, size - 1, size - 1], outline=C.NUMBER_EDGE_RGB, width=edge)

    s = str(n)
    inner = size - 2 * edge - size * 0.06
    f = _fit_font(d, s, inner, inner * 0.72, int(size * 0.80))
    l, t, r, b = d.textbbox((0, 0), s, font=f)
    w, h = r - l, b - t
    cx, cy = size / 2.0, size * 0.46
    d.text((cx - w / 2.0 - l, cy - h / 2.0 - t), s, font=f, fill=C.NUMBER_DIGIT_RGB)

    uy = cy + h / 2.0 + size * 0.07
    d.rectangle([cx - w / 2.0, uy, cx + w / 2.0, uy + max(2, size // 26)],
                fill=C.NUMBER_DIGIT_RGB)
    return img


def write_number_textures(size: int | None = None, force: bool = False) -> list[Path]:
    """1~10 숫자 텍스처를 assets/number_textures/ 에 쓴다. 이미 있으면 건너뛴다."""
    size = size or C.NUMBER_TEX_PX
    TEXTURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for o in C.NUMBER_OBJECTS:
        p = TEXTURE_DIR / f"num_{o['number']:02d}.png"
        if force or not p.exists():
            _draw_number(o["number"], size).save(p)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------
# 씬 에셋 (<texture>/<material>)
# --------------------------------------------------------------------------
def _xml_attr(value: str) -> str:
    return (str(value).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def asset_xml() -> str:
    """숫자 텍스처 10개를 씬의 <asset> 에 넣을 XML 조각.

    type="cube" + gridsize 기본값(1 1) 이면 한 장의 그림이 정육면체 여섯 면에 모두
    찍힌다. 그래서 top-down(agentview) 에서도 손목 카메라에서도 숫자가 보인다.
    경로는 반드시 C.mujoco_path() 를 거친다 (한글 경로에서 MuJoCo 가 파일을 못 연다).
    """
    rows = []
    for o in C.NUMBER_OBJECTS:
        tag = f"num_{o['number']:02d}"
        f = _xml_attr(C.mujoco_path(TEXTURE_DIR / f"{tag}.png"))
        rows.append(
            f'    <texture type="cube" name="{tag}_tex" file="{f}"/>\n'
            f'    <material name="{o["material"]}" texture="{tag}_tex" '
            f'specular="0.10" shininess="0.08" reflectance="0.0"/>'
        )
    return "\n".join(rows)


# --------------------------------------------------------------------------
# 켜기 / 끄기
# --------------------------------------------------------------------------
def activate(regenerate: bool = False) -> list[dict]:
    """숫자 조건으로 바꾼다. **SortingEnv 를 만들기 전에** 불러야 한다.

    env 는 생성 시점에 물체 이름으로 MuJoCo 조인트 주소를 잡아 두기 때문에,
    env 를 만든 뒤에 세트를 바꾸면 이름이 맞지 않아 KeyError 가 난다.
    """
    write_number_textures(force=regenerate)
    scene_mod.EXTRA_ASSETS = asset_xml()
    return C.use_object_set("number")


def deactivate() -> list[dict]:
    """색 조건으로 되돌린다 (한 프로세스에서 두 조건을 번갈아 돌릴 때)."""
    scene_mod.EXTRA_ASSETS = ""
    return C.use_object_set("color")


# --------------------------------------------------------------------------
# 숫자가 몇 픽셀로 찍히는가 — 계산값(실측 아님)
# --------------------------------------------------------------------------
def face_pixels(image_size: int = C.IMAGE_SIZE) -> float:
    """agentview(top-down) 이미지에서 물체 윗면 한 변이 차지하는 픽셀 수.

    **카메라 기하로 계산한 값이고 렌더 실측이 아니다.** 숫자를 읽히게 하려면
    VLM 에 넣는 이미지를 충분히 크게 렌더해야 하는데, 그 기준을 잡기 위한 것이다.
    숫자 자체의 높이는 윗면의 약 0.7 배이므로 이 값의 70% 쯤으로 보면 된다.
    """
    m = re.search(r'name="agentview"\s+pos="\S+\s+\S+\s+([\d.]+)"[^/]*?fovy="([\d.]+)"',
                  scene_mod._CAMERAS)
    if not m:
        return 0.0
    cam_z, fovy = float(m.group(1)), float(m.group(2))
    dist = cam_z - (C.TABLE_TOP_Z + C.NUMBER_FULL[2])      # 카메라에서 물체 윗면까지
    visible = 2.0 * dist * math.tan(math.radians(fovy) / 2.0)
    return image_size * C.NUMBER_FULL[0] / visible


# --------------------------------------------------------------------------
# 숫자용 VLM 프롬프트
#
# 색 세트의 프롬프트(planner._PROMPT)와 틀을 똑같이 맞추고, 구분 단서만 숫자로 바꿨다.
#   "number"      : 숫자를 읽고 + 규칙까지 적용해서 목적지를 말하게 한다
#   "number_read" : 가운데에 있는 숫자만 읽게 하고 목적지는 코드 규칙에 맡긴다
#                   (parse_instruction 은 구역 언급이 없으면 home_zone 으로 폴백한다)
# 두 프롬프트의 점수 차이가 "숫자를 못 읽는 것"과 "규칙을 못 쓰는 것"을 갈라 준다.
# 아직 돌려 보지 않았으므로 recall 값은 비어 있다 — 실제로 재서 채워 넣을 것.
# --------------------------------------------------------------------------
_PROMPT_NUMBER = """You are the high-level planner of a robot that tidies a table.

The image is a top-down view of a table with three zones:
- LEFT part of the image  = left storage zone (green)
- MIDDLE of the image     = center workspace (olive) -- this must end up EMPTY
- RIGHT part of the image = right storage zone (blue)

Every object is the same white cube. The cubes differ ONLY by a black NUMBER
printed on them, from 1 to <COUNT>. Read the printed number to tell the cubes
apart -- their colour, size and shape carry no information.

Where each cube belongs:
- cubes numbered 1 to <SPLIT>          -> right storage zone
- cubes numbered <SPLIT1> to <COUNT>   -> left storage zone
- a cube already inside its own zone must NOT be moved.

User command: "{command}"

List ONLY the cubes that are in the wrong place, in the order the robot should
move them. Answer with a JSON array and nothing else. Each element must be:
{{"object": "cube number N", "target_zone": "left storage zone|right storage zone"}}
where N is the number printed on that cube.

JSON:"""


_PROMPT_NUMBER_READ = """You are the vision module of a robot that tidies a table.

The image is a top-down view of a table with three zones:
- LEFT part of the image  = left storage zone (green)
- MIDDLE of the image     = center workspace (olive)
- RIGHT part of the image = right storage zone (blue)

Every object is the same white cube with a black NUMBER printed on it (1 to <COUNT>).

Look ONLY at the olive middle area. Ignore cubes that are inside the green or
blue areas. Read the number printed on every cube lying in the middle area and
answer with a JSON array of those numbers, like [3, 7]. If the middle area is
empty, answer []. Do not say anything else -- no zones, no explanations.

User command: "{command}"

JSON:"""


def _fill(template: str) -> str:
    # <SPLIT1> 을 <SPLIT> 보다 먼저 치환해야 한다 ("<SPLIT1>" 안에 "<SPLIT" 가 들어 있다).
    return (template.replace("<SPLIT1>", str(C.NUMBER_SPLIT + 1))
            .replace("<SPLIT>", str(C.NUMBER_SPLIT))
            .replace("<COUNT>", str(C.NUMBER_COUNT)))


PROMPT_NUMBER = _fill(_PROMPT_NUMBER)
PROMPT_NUMBER_READ = _fill(_PROMPT_NUMBER_READ)

# VLMPlanner(prompt_style=...) 에서 바로 고를 수 있게 등록한다.
planner_mod.PROMPTS["number"] = PROMPT_NUMBER
planner_mod.PROMPTS["number_read"] = PROMPT_NUMBER_READ
