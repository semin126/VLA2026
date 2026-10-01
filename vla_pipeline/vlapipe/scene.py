"""LIBERO 호환 MuJoCo 씬 생성.

- Franka Panda(Menagerie) + 테이블 + 3x1 구역 + 물체 10개 + 카메라 3대 + 2점 조명
- 씬 XML 은 반드시 panda_vla.xml 과 같은 폴더에 써야 meshdir="assets" 가 해석된다.
"""

from __future__ import annotations

import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from . import config as C

# --------------------------------------------------------------------------
# 조명
#   기준: 260827 문서 (주광 = 로봇 뒤쪽 상단에서 작업구역을 향해 45도,
#                      보조광 = 반대편에서 30% 세기, 그림자 소프트닝)
#   260910 변경: 조명 위치에 따라 물체 색이 급변하는 문제를 줄이려고 문서 기준에서 조정
#     - 보조광 세기 30% → 약 60% (diffuse 0.36 / 0.60): 암부를 더 밝게 채움
#     - exponent 10→3, 6→2 / cutoff 60→70, 75→80: 스팟 가장자리 감쇠를 완만하게
#     - specular 0.22→0.08, 0.04→0.02: 하이라이트가 물체 고유색을 덮지 않도록
#     - ambient 추가 (0.10 / 0.08): 직접광을 받지 않는 면의 색 유지
#   그림자: 주광만 투사(castshadow). 보조광과 ambient가 암부를 채워 부드럽게 보이게 함
# --------------------------------------------------------------------------
_LIGHTS = """
    <light name="key_light" pos="0 -0.90 2.00" dir="0 0.7071 -0.7071" directional="false"
           castshadow="true" ambient="0.10 0.10 0.10" diffuse="0.60 0.60 0.58"
           specular="0.08 0.08 0.08" cutoff="70" exponent="3"/>
    <light name="fill_light" pos="0 1.55 1.90" dir="0 -0.737 -0.676" directional="false"
           castshadow="false" ambient="0.08 0.08 0.08" diffuse="0.36 0.36 0.37"
           specular="0.02 0.02 0.02" cutoff="80" exponent="2"/>
"""

# 카메라.
#  agentview  : 테이블 바로 위에서 내려다본다(top-down). 영상의 오른쪽=월드 +x,
#               위쪽=월드 +y 라서 VLM 이 말하는 좌/우가 로봇 기준 좌/우와 일치한다.
#               로봇 뒤/앞 시점은 팔이 화면을 가리거나(뒤) 좌우가 뒤집혀(앞) 보이므로
#               학습·추론용 주 카메라는 이 top-down 을 쓴다. (LIBERO agentview 역할)
#  frontview  : 260827 문서의 '로봇을 마주보는 면 상단'. 데모 영상용.
#               주의: 이 시점은 좌우가 뒤집혀 보인다.
#  sideview   : 260827 문서의 '로봇이 없는 옆면 상단'.
#  robot0_eye_in_hand : 손목 (panda_vla.xml 안에 정의)
_CAMERAS = """
    <camera name="agentview" pos="0 0.37 1.70" xyaxes="1 0 0  0 1 0" fovy="58"/>
    <camera name="frontview" pos="0 0.98 1.34" xyaxes="-1 0 0  0 -0.625 0.781" fovy="46"/>
    <camera name="sideview"  pos="0.95 0.33 1.28" xyaxes="0 1 0  -0.436 0 0.900" fovy="46"/>
"""


def _zone_geoms() -> str:
    out = []
    for name, z in C.ZONES.items():
        cx = 0.5 * (z["x"][0] + z["x"][1])
        cy = 0.5 * (z["y"][0] + z["y"][1])
        hx = 0.5 * (z["x"][1] - z["x"][0])
        hy = 0.5 * (z["y"][1] - z["y"][0])
        out.append(
            f'<geom name="{name}" type="box" pos="{cx:.4f} {cy:.4f} {C.TABLE_TOP_Z + 0.0012:.4f}" '
            f'size="{hx:.4f} {hy:.4f} 0.0010" rgba="{z["rgba"]}" '
            f'contype="0" conaffinity="0" group="1"/>'
        )
    return "\n    ".join(out)


def _object_body(obj: dict, xy: tuple[float, float], yaw: float = 0.0) -> str:
    hx, hy, hz = C.half_size(obj)
    z = C.TABLE_TOP_Z + hz + 0.0015
    qw, qz = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return f"""
    <body name="{obj['name']}" pos="{xy[0]:.4f} {xy[1]:.4f} {z:.4f}" quat="{qw:.5f} 0 0 {qz:.5f}">
      <freejoint name="{obj['name']}_freejoint"/>
      <geom name="{obj['name']}_geom" type="box" size="{hx:.4f} {hy:.4f} {hz:.4f}"
            rgba="{obj['rgba']}" mass="0.060" condim="4"
            friction="1.6 0.10 0.003" solimp="0.95 0.99 0.001" solref="0.004 1"/>
    </body>"""


def build_scene_xml(layout: dict[str, tuple[float, float, float]]) -> str:
    """layout: {물체이름: (x, y, yaw)} -> 씬 XML 문자열."""
    bodies = "\n".join(
        _object_body(C.OBJECT_BY_NAME[n], (layout[n][0], layout[n][1]), layout[n][2])
        for n in C.OBJECT_NAMES if n in layout
    )
    bx, by = C.ROBOT_BASE_XY
    thx, thy = C.TABLE_HALF
    return f"""<mujoco model="vla_sorting_scene">
  <include file="panda_vla.xml"/>

  <option timestep="{C.SIM_TIMESTEP}" integrator="implicitfast" cone="elliptic" impratio="10"/>

  <visual>
    <global offwidth="1280" offheight="960" azimuth="90" elevation="-25"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient="0.34 0.34 0.35" diffuse="0.14 0.14 0.14" specular="0 0 0"/>
    <map znear="0.02" zfar="12"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.60 0.66" rgb2="0.20 0.24 0.30"
             width="512" height="1024"/>
    <texture type="2d" name="floor_tex" builtin="checker" mark="edge"
             rgb1="0.34 0.36 0.39" rgb2="0.29 0.31 0.34" markrgb="0.45 0.47 0.50"
             width="512" height="512"/>
    <material name="floor_mat" texture="floor_tex" texuniform="true" texrepeat="6 6" reflectance="0.03"/>
    <material name="table_mat" rgba="0.82 0.80 0.76 1" specular="0.12" shininess="0.10" reflectance="0.0"/>
    <material name="pedestal_mat" rgba="0.24 0.25 0.28 1" specular="0.2" shininess="0.2"/>
  </asset>

  <worldbody>
{_LIGHTS}
    <geom name="floor" type="plane" pos="0 0 0" size="3 3 0.05" material="floor_mat"
          friction="1 0.005 0.0001"/>

    <!-- 로봇 받침대 -->
    <geom name="pedestal" type="box" material="pedestal_mat"
          pos="{bx:.4f} {by:.4f} {C.TABLE_TOP_Z / 2:.4f}" size="0.13 0.13 {C.TABLE_TOP_Z / 2:.4f}"/>

    <!-- 작업 테이블 -->
    <geom name="table" type="box" material="table_mat" friction="0.9 0.02 0.001"
          pos="0 {C.TABLE_CENTER_Y:.4f} {C.TABLE_TOP_Z - C.TABLE_THICK / 2:.4f}"
          size="{thx:.4f} {thy:.4f} {C.TABLE_THICK / 2:.4f}"/>
    <geom name="table_leg_1" type="box" material="pedestal_mat" pos="{-thx + 0.06:.4f} {C.TABLE_CENTER_Y - thy + 0.06:.4f} {(C.TABLE_TOP_Z - C.TABLE_THICK) / 2:.4f}" size="0.025 0.025 {(C.TABLE_TOP_Z - C.TABLE_THICK) / 2:.4f}"/>
    <geom name="table_leg_2" type="box" material="pedestal_mat" pos="{thx - 0.06:.4f} {C.TABLE_CENTER_Y - thy + 0.06:.4f} {(C.TABLE_TOP_Z - C.TABLE_THICK) / 2:.4f}" size="0.025 0.025 {(C.TABLE_TOP_Z - C.TABLE_THICK) / 2:.4f}"/>
    <geom name="table_leg_3" type="box" material="pedestal_mat" pos="{-thx + 0.06:.4f} {C.TABLE_CENTER_Y + thy - 0.06:.4f} {(C.TABLE_TOP_Z - C.TABLE_THICK) / 2:.4f}" size="0.025 0.025 {(C.TABLE_TOP_Z - C.TABLE_THICK) / 2:.4f}"/>
    <geom name="table_leg_4" type="box" material="pedestal_mat" pos="{thx - 0.06:.4f} {C.TABLE_CENTER_Y + thy - 0.06:.4f} {(C.TABLE_TOP_Z - C.TABLE_THICK) / 2:.4f}" size="0.025 0.025 {(C.TABLE_TOP_Z - C.TABLE_THICK) / 2:.4f}"/>

    <!-- 3 x 1 구역 표시 -->
    {_zone_geoms()}

{_CAMERAS}
{bodies}
  </worldbody>
</mujoco>
"""


# --------------------------------------------------------------------------
# 배치(layout) 생성
# --------------------------------------------------------------------------
def _reachable(x: float, y: float) -> bool:
    bx, by = C.ROBOT_BASE_XY
    d = math.hypot(x - bx, y - by)
    return C.REACH_MIN <= d <= C.REACH_MAX


def tidy_layout() -> dict[str, tuple[float, float, float]]:
    """정돈된 상태 — 모든 물체가 home_zone 안에 격자로 놓이고 가운데는 비어 있다."""
    layout: dict[str, tuple[float, float, float]] = {}
    grids = {
        "left_zone": [(-0.33, 0.14), (-0.20, 0.14), (-0.33, 0.26), (-0.20, 0.26), (-0.33, 0.38)],
        "right_zone": [(0.20, 0.14), (0.33, 0.14), (0.20, 0.26), (0.33, 0.26), (0.20, 0.38)],
    }
    used = {k: 0 for k in grids}
    for obj in C.OBJECTS:
        z = obj["home_zone"]
        x, y = grids[z][used[z]]
        used[z] += 1
        layout[obj["name"]] = (x, y, 0.0)
    return layout


def messy_layout(seed: int = 0, n_moved: int | None = None) -> dict[str, tuple[float, float, float]]:
    """어질러진 상태 — 물체 일부가 가운데 작업 구역으로 흩어져 있다."""
    rng = random.Random(seed)
    layout = tidy_layout()
    names = list(C.OBJECT_NAMES)
    rng.shuffle(names)
    k = n_moved if n_moved is not None else rng.randint(3, 6)
    placed: list[tuple[float, float]] = []

    cz = C.ZONES["center_zone"]
    for name in names[:k]:
        for _ in range(200):
            x = rng.uniform(cz["x"][0] + 0.05, cz["x"][1] - 0.05)
            y = rng.uniform(max(cz["y"][0] + 0.05, 0.12), cz["y"][1] - 0.05)
            if not _reachable(x, y):
                continue
            if all(math.hypot(x - px, y - py) > 0.11 for px, py in placed):
                placed.append((x, y))
                layout[name] = (x, y, rng.uniform(-math.pi / 2, math.pi / 2))
                break
    return layout


def domain_randomized_layout(seed: int = 0) -> dict[str, tuple[float, float, float]]:
    """도메인 랜덤화 — 물체 10개를 작업 영역 전체에 무작위로 배치."""
    rng = random.Random(seed)
    layout: dict[str, tuple[float, float, float]] = {}
    placed: list[tuple[float, float]] = []
    for obj in C.OBJECTS:
        for _ in range(400):
            x = rng.uniform(C.WORK_X[0] + 0.05, C.WORK_X[1] - 0.05)
            y = rng.uniform(C.WORK_Y[0] + 0.06, C.WORK_Y[1] - 0.05)
            if not _reachable(x, y):
                continue
            if all(math.hypot(x - px, y - py) > 0.10 for px, py in placed):
                placed.append((x, y))
                layout[obj["name"]] = (x, y, rng.uniform(-math.pi, math.pi))
                break
        else:
            layout[obj["name"]] = tidy_layout()[obj["name"]]
    return layout


def cluttered_layout(seed: int = 0, n_blocked: int = 2) -> dict[str, tuple[float, float, float]]:
    """접근성 제약이 있는 배치 — 이 연구의 제목이 말하는 조건을 실제로 만든다.

    messy/random 배치는 물체 간 최소 간격을 강제하므로 **가림이 절대 생기지 않는다**.
    그리퍼는 물체 중심에서 ±0.04 m 지점으로 손가락이 내려오므로, 그보다 가까이
    다른 물체가 있어야 '가려짐'이 된다. 여기서는 목표 물체 옆에 장애물을 일부러 붙인다.

    n_blocked 개의 물체가 다른 물체에 막혀 바로 집을 수 없는 상태가 된다.
    """
    rng = random.Random(seed + 9973)
    layout = messy_layout(seed)
    # 가운데로 나온 물체 중에서 목표를 고른다
    cz = C.ZONES["center_zone"]
    in_center = [n for n, (x, y, _) in layout.items()
                 if cz["x"][0] <= x <= cz["x"][1] and cz["y"][0] <= y <= cz["y"][1]]
    rng.shuffle(in_center)
    # 장애물로 쓸 물체 = 아직 보관 구역에 남아 있는 것들
    spare = [n for n in C.OBJECT_NAMES if n not in in_center]
    rng.shuffle(spare)

    made = 0
    for target in in_center:
        if made >= n_blocked or not spare:
            break
        tx, ty, tyaw = layout[target]
        obj = C.OBJECT_BY_NAME[target]
        hx, hy, _ = C.half_size(obj)
        yaw = tyaw + (math.pi / 2 if hx > hy else 0.0)   # 개폐축
        ux, uy = math.cos(yaw), math.sin(yaw)
        side = rng.choice((1.0, -1.0))
        # 손가락이 내려올 자리(±0.04)에 장애물을 겹치게 놓는다
        gap = rng.uniform(0.040, 0.056)
        bx, by = tx + side * gap * ux, ty + side * gap * uy
        if not _reachable(bx, by):
            continue
        blocker = spare.pop()
        layout[blocker] = (bx, by, rng.uniform(-math.pi / 2, math.pi / 2))
        made += 1
    return layout


LAYOUTS = {"tidy": tidy_layout, "messy": messy_layout,
           "random": domain_randomized_layout, "cluttered": cluttered_layout}


def write_scene(layout_name: str = "tidy", seed: int = 0, path: Path | None = None) -> Path:
    fn = LAYOUTS[layout_name]
    layout = fn() if layout_name == "tidy" else fn(seed)
    xml = build_scene_xml(layout)
    ET.fromstring(xml)  # 문법 검증
    out = path or C.SCENE_XML
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml, encoding="utf-8")
    return out
