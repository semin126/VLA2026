"""프로젝트 전역 설정 — 씬 규격, 액션/관측 규격, 경로.

씬 수치는 '260827 심자연.docx'(VLA 씬 구성하기)의 확정 스펙을 따른다.
액션/관측 규격은 LIBERO(robosuite OSC_POSE) 관례를 따라, 나중에
LIBERO 사전학습 체크포인트나 데이터셋으로 갈아끼울 수 있게 맞춰 두었다.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path

# --------------------------------------------------------------------------
# 경로
# --------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PKG_DIR.parent
ASSET_DIR = PROJECT_DIR / "assets"
PANDA_DIR = ASSET_DIR / "franka_emika_panda"
PANDA_PATCHED_XML = PANDA_DIR / "panda_vla.xml"   # setup_assets.py 가 생성
SCENE_XML = PANDA_DIR / "vla_scene.xml"           # scene.py 가 생성 (에셋과 같은 폴더여야 meshdir 해석됨)

DATA_DIR = PROJECT_DIR / "data"
EPISODE_DIR = DATA_DIR / "episodes"          # 텔레옵 원본 (정돈 -> 어지르기)
REVERSED_DIR = DATA_DIR / "episodes_reversed"  # 역재생 (어지럽힘 -> 정돈) = 학습용
LEROBOT_DIR = DATA_DIR / "lerobot"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"


# --------------------------------------------------------------------------
# 윈도우 + 한글 경로 대응
#
# MuJoCo 의 XML/메시 로더는 C 라이브러리의 fopen 을 쓴다. 파이썬 바인딩은 경로를
# UTF-8 바이트로 넘기는데, 윈도우의 fopen 은 그 바이트를 ANSI 코드페이지(한국어면
# CP949)로 해석한다. 그래서 경로에 한글이 하나라도 있으면
#   ValueError: ParseXML: Error opening file '...'
# 로 실패한다. 파일은 분명히 존재하는데도 그렇다. (리눅스/macOS 는 UTF-8 이라 무관)
#
# 저장소 경로가 "2026 대전과학고 심화자율연구/교육용 자료/..." 라서 항상 걸린다.
# 그래서 ASCII 이름의 디렉터리 정션(junction)을 만들어 두고, MuJoCo 에 넘길 때만
# 그 경로로 바꿔 준다. 정션은 관리자 권한 없이 만들 수 있고, 파일 복사가 아니라
# 같은 폴더를 가리키는 링크라 scene.py 가 새로 쓴 XML 도 즉시 보인다.
# --------------------------------------------------------------------------
_LINK_CACHE: dict[str, str] = {}


def _ascii_link_root() -> Path:
    """ASCII 이름만 들어 있는, 쓰기 가능한 디렉터리를 찾는다."""
    candidates = [
        os.environ.get("LOCALAPPDATA"),
        tempfile.gettempdir(),
        r"C:\Users\Public",
    ]
    for base in candidates:
        if not base or not str(base).isascii():
            continue
        root = Path(base) / "vlapipe_links"
        try:
            root.mkdir(parents=True, exist_ok=True)
            return root
        except OSError:
            continue
    raise RuntimeError(
        "ASCII 경로의 임시 폴더를 찾지 못했습니다. "
        "프로젝트를 한글이 없는 경로(예: C:\\vla_pipeline)로 옮겨 주세요."
    )


def mujoco_path(path: Path | str) -> str:
    """MuJoCo 에 넘길 수 있는 경로 문자열로 바꾼다.

    ASCII 경로거나 윈도우가 아니면 그대로 돌려준다.
    한글 등이 섞여 있으면 부모 폴더를 가리키는 ASCII 정션을 만들어 그 경로를 준다.
    """
    p = Path(path).resolve()
    if os.name != "nt" or str(p).isascii():
        return str(p)

    parent = str(p.parent)
    link = _LINK_CACHE.get(parent)
    if link is None:
        tag = hashlib.md5(parent.encode("utf-8")).hexdigest()[:12]
        target = _ascii_link_root() / tag
        # 이미 있는데 다른 곳을 가리키면(폴더를 옮긴 경우) 다시 만든다.
        if target.exists() and Path(os.path.realpath(target)) != Path(parent):
            target.unlink(missing_ok=True)
        if not target.exists():
            # mklink /J = 디렉터리 정션. 심볼릭 링크와 달리 관리자 권한이 필요 없다.
            # text=True 로 받으면 cmd 출력(OEM 코드페이지)에 한글 경로가 섞여
            # UnicodeDecodeError 가 난다. 바이트로 받아서 직접 푼다.
            r = subprocess.run(["cmd", "/c", "mklink", "/J", str(target), parent],
                               capture_output=True)
            if not target.exists():
                msg = (r.stderr or b"").decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"한글 경로 우회용 정션을 만들지 못했습니다: {msg}\n"
                    "프로젝트를 한글이 없는 경로로 옮기면 해결됩니다."
                )
        link = str(target)
        _LINK_CACHE[parent] = link
    return str(Path(link) / p.name)

# --------------------------------------------------------------------------
# 연산 장치 (torch)
#
# 기본값을 'cuda' 로 박아 두면 GPU 가 없는 환경에서 모델 로드가
#   AssertionError: Torch not compiled with CUDA enabled
# 로 죽는다. Colab 은 런타임 유형을 바꾸지 않으면 CPU 이므로 학생들이 자주 만난다.
# 그래서 기본은 'auto' 로 두고 실제로 있는 장치를 고른다.
# --------------------------------------------------------------------------
DEFAULT_DEVICE = "auto"


def pick_device(device: str | None = None) -> str:
    """'auto'/None 이면 실제 장치를 고르고, 없는 GPU 를 요청하면 CPU 로 되돌린다."""
    import torch

    has_cuda = torch.cuda.is_available()
    if device is None or device == "auto":
        return "cuda" if has_cuda else "cpu"
    if str(device).startswith("cuda") and not has_cuda:
        print("[vlapipe] CUDA 를 쓸 수 없어 CPU 로 실행합니다 (느립니다).\n"
              "          Colab 이면 런타임 > 런타임 유형 변경 > GPU(T4) 로 바꾸고 "
              "다시 실행하세요.")
        return "cpu"
    return str(device)


def pick_dtype(device: str, dtype: str | None = None) -> str:
    """CPU 에서는 bfloat16 이 느리고 일부 연산이 없어 float32 로 돌린다."""
    if dtype is not None:
        return dtype
    return "bfloat16" if str(device).startswith("cuda") else "float32"


# --------------------------------------------------------------------------
# 씬 규격 (260827 문서)
# --------------------------------------------------------------------------
TABLE_TOP_Z = 0.80            # 테이블 상판 높이 [m]
TABLE_HALF = (0.45, 0.34)     # 상판 half-extent (x, y) [m]
TABLE_CENTER_Y = 0.37         # 상판 중심 y [m]  -> y in [0.03, 0.71]
TABLE_THICK = 0.02

# 작업 영역: 0.8m(가로) x 0.5m(세로), 로봇 기준 y = 0.15 ~ 0.65
WORK_X = (-0.40, 0.40)
WORK_Y = (0.05, 0.55)   # 로봇 베이스(y=-0.12) 기준 0.17 ~ 0.67 m

# 로봇: 한 변의 중심에서 살짝 뒤
ROBOT_BASE_XY = (0.0, -0.12)
ROBOT_BASE_Z = TABLE_TOP_Z

# 로봇 안정 가동 반경 (문서: 0.15 ~ 0.65 m). 작업영역 먼 모서리는 이 밖으로
# 벗어나므로 물체 배치는 reachability 마스크로 걸러 쓴다.
# 실측(IK 스윕, 작업영역 13x11 격자):
#   파지 높이(z=table+0.02) : 0.183 m 부터 가능
#   접근 높이(z=table+0.16) : 0.297 m 부터 가능  <- 이게 구속 조건
#   이송 높이(z=table+0.22) : 0.257 m 부터 가능
# 접근 높이가 가장 빡빡하므로 그 값으로 하한을 잡는다.
# (8/27 학생 설계 문서의 "0.15~0.65 m"는 팔 길이만 고려한 추정치였고,
#  실제로는 베이스 근처에서 팔이 접히지 않아 안쪽 한계가 더 크다.)
REACH_MIN = 0.30
REACH_MAX = 0.65   # 실측상 0.78 까지 가능하나 자세 여유를 두어 보수적으로

# 3 x 1 분할 — 좌/중/우
ZONE_NAMES = ("left_zone", "center_zone", "right_zone")
_zw = (WORK_X[1] - WORK_X[0]) / 3.0
ZONES = {
    "left_zone": {
        "x": (WORK_X[0], WORK_X[0] + _zw),
        "y": WORK_Y,
        "rgba": "0.20 0.75 0.35 0.30",
        "label_en": "left storage zone",
        "label_ko": "왼쪽 보관 구역",
    },
    "center_zone": {
        "x": (WORK_X[0] + _zw, WORK_X[0] + 2 * _zw),
        "y": WORK_Y,
        "rgba": "0.85 0.85 0.30 0.22",
        "label_en": "center workspace",
        "label_ko": "가운데 작업 구역",
    },
    "right_zone": {
        "x": (WORK_X[0] + 2 * _zw, WORK_X[1]),
        "y": WORK_Y,
        "rgba": "0.30 0.55 0.95 0.30",
        "label_en": "right storage zone",
        "label_ko": "오른쪽 보관 구역",
    },
}


def zone_center(name: str) -> tuple[float, float]:
    z = ZONES[name]
    return (0.5 * (z["x"][0] + z["x"][1]), 0.5 * (z["y"][0] + z["y"][1]))


# --------------------------------------------------------------------------
# 물체 세트 1 — 색 + 모양으로 구분 (color). 기준 조건.
#
# 물체 10개 (260827 문서의 3가지 크기 = 전체 치수 [m]; MuJoCo size 는 half-extent)
#   0.06 x 0.04 x 0.04  x5
#   0.05 x 0.05 x 0.05  x3
#   0.04 x 0.04 x 0.03  x2
# home_zone = '정돈된 상태'에서 이 물체가 있어야 할 구역.
# grasp_axis = 그리퍼가 물체를 물 때 벌어지는 방향 (물체 로컬 축). Panda 최대 개폐 0.08 m.
# --------------------------------------------------------------------------
COLOR_OBJECTS: list[dict] = [
    # 긴 블록 5개 -> 오른쪽 보관 구역
    {"name": "red_block",    "full": (0.06, 0.04, 0.04), "rgba": "0.90 0.14 0.14 1", "home_zone": "right_zone", "grasp_axis": "y", "color_en": "red",    "color_ko": "빨간"},
    {"name": "orange_block", "full": (0.06, 0.04, 0.04), "rgba": "0.95 0.52 0.10 1", "home_zone": "right_zone", "grasp_axis": "y", "color_en": "orange", "color_ko": "주황"},
    {"name": "yellow_block", "full": (0.06, 0.04, 0.04), "rgba": "0.95 0.82 0.15 1", "home_zone": "right_zone", "grasp_axis": "y", "color_en": "yellow", "color_ko": "노란"},
    {"name": "lime_block",   "full": (0.06, 0.04, 0.04), "rgba": "0.55 0.85 0.25 1", "home_zone": "right_zone", "grasp_axis": "y", "color_en": "lime",   "color_ko": "연두"},
    {"name": "cyan_block",   "full": (0.06, 0.04, 0.04), "rgba": "0.20 0.80 0.85 1", "home_zone": "right_zone", "grasp_axis": "y", "color_en": "cyan",   "color_ko": "하늘"},
    # 정육면체 3개 -> 왼쪽 보관 구역
    {"name": "blue_cube",    "full": (0.05, 0.05, 0.05), "rgba": "0.15 0.35 0.92 1", "home_zone": "left_zone",  "grasp_axis": "y", "color_en": "blue",   "color_ko": "파란"},
    {"name": "purple_cube",  "full": (0.05, 0.05, 0.05), "rgba": "0.55 0.25 0.85 1", "home_zone": "left_zone",  "grasp_axis": "y", "color_en": "purple", "color_ko": "보라"},
    {"name": "pink_cube",    "full": (0.05, 0.05, 0.05), "rgba": "0.95 0.45 0.70 1", "home_zone": "left_zone",  "grasp_axis": "y", "color_en": "pink",   "color_ko": "분홍"},
    # 납작한 작은 블록 2개 -> 왼쪽 보관 구역
    {"name": "green_tile",   "full": (0.04, 0.04, 0.03), "rgba": "0.10 0.55 0.30 1", "home_zone": "left_zone",  "grasp_axis": "y", "color_en": "green",  "color_ko": "초록"},
    {"name": "brown_tile",   "full": (0.04, 0.04, 0.03), "rgba": "0.45 0.30 0.18 1", "home_zone": "left_zone",  "grasp_axis": "y", "color_en": "brown",  "color_ko": "갈색"},
]

# --------------------------------------------------------------------------
# 지시 대상 표현(referring expression) — "물체를 무엇으로 가리키는가"
#
# 이 실험에서 바꾸는 것은 이 단서뿐이고, 씬·로봇·액션 규격은 전부 같게 둔다.
# 그래야 "색/모양으로 가리킬 때"와 "숫자로 가리킬 때"를 같은 조건에서 비교할 수 있다.
#   ref_en / ref_ko  : 지시문에 들어가는 이름 ("red block" / "빨간 물체")
#   ref_brief_en     : 장애물 치우기 문장용 짧은 이름 ("red object")
#   patterns         : VLM 출력에서 이 물체를 찾아내는 정규식 — 강한 단서
#   weak_patterns    : 강한 단서가 하나도 안 맞을 때만 쓰는 폴백
# 색 세트의 문장은 기존 실측값과 비교 가능해야 하므로 글자까지 그대로 유지한다.
# --------------------------------------------------------------------------
def _shape_en(full) -> str:
    return "block" if full[0] != full[1] else ("cube" if full[2] == full[0] else "tile")


for _o in COLOR_OBJECTS:
    _o["ref_en"] = f"{_o['color_en']} {_shape_en(_o['full'])}"
    _o["ref_brief_en"] = f"{_o['color_en']} object"
    _o["ref_ko"] = f"{_o['color_ko']} 물체"
    _o["patterns"] = [rf"\b{_o['color_en']}\b", re.escape(_o["color_ko"]), re.escape(_o["name"])]
    _o["weak_patterns"] = []


# --------------------------------------------------------------------------
# 물체 세트 2 — 숫자로 구분 (number)
#
# 색도 모양도 크기도 전부 똑같은 정육면체 10개를 놓고, 면에 1~10 숫자만 찍는다.
# 숫자 말고는 구별할 방법이 없으므로
#   1단계 VLM : 숫자를 '읽어야' 계획을 세울 수 있다
#   2단계 VLA : 지시문에 적힌 숫자로 대상을 찾아야 한다
# 숫자 텍스처 생성과 씬 등록은 vlapipe/numbered.py 가 한다.
# 이름(name)으로 구분하는 세 번째 조건도 여기에 세트를 하나 더 두면 된다.
# --------------------------------------------------------------------------
NUMBER_COUNT = 10
NUMBER_FULL = (0.05, 0.05, 0.05)   # 전부 같은 치수 = 크기/모양이 단서가 되지 않게
NUMBER_SPLIT = 5                   # 1~5 -> 오른쪽, 6~10 -> 왼쪽 (좌우 5개씩, tidy 격자와 맞음)
NUMBER_TEX_PX = 256                # 숫자 텍스처 한 변 [px]
NUMBER_FACE_RGB = (242, 242, 240)  # 면 바탕색 (테이블 0.82 보다 밝게)
NUMBER_DIGIT_RGB = (18, 18, 18)    # 숫자 색
NUMBER_EDGE_RGB = (70, 70, 72)     # 테두리 — 밝은 테이블 위에서 물체 윤곽이 보이도록


def number_home_zone(n: int) -> str:
    """숫자 규칙 — 색 세트의 '모양 규칙'(긴 블록=오른쪽)과 같은 자리를 차지한다."""
    return "right_zone" if n <= NUMBER_SPLIT else "left_zone"


def _number_patterns(n: int) -> list[str]:
    """'object 7' · 'cube 7' · 'no.7' · '#7' · 'object_7' · '7번' 을 잡는다.
    (?<!\\d)7(?!\\d) 로 둘러싸서 '7' 이 '17'·'70' 에 걸리지 않게 한다."""
    d = str(n)
    return [
        rf"(?:object|cube|block|box|number|no\.?|#)[\s_]*(?<!\d){d}(?!\d)",
        rf"(?<!\d){d}(?!\d)\s*번",
    ]


NUMBER_OBJECTS: list[dict] = [
    {
        "name": f"object_{n}", "number": n, "full": NUMBER_FULL,
        "rgba": "0.95 0.95 0.94 1",         # 텍스처를 못 만들었을 때의 폴백 색
        "material": f"num_{n:02d}_mat",     # scene.py 가 이 재질로 그린다
        "home_zone": number_home_zone(n), "grasp_axis": "y",
        # "Pick up the cube number 7 and place it ..." / "7번 물체를 집어 ..."
        # 'cube' 는 10개가 전부 정육면체라 구분 단서가 되지 않는다 — 숫자만 단서다.
        "ref_en": f"cube number {n}", "ref_brief_en": f"cube number {n}",
        "ref_ko": f"{n}번 물체",
        "patterns": _number_patterns(n),
        # VLM 이 설명 없이 숫자만 ["7", "3"] 처럼 내놓는 경우의 폴백
        "weak_patterns": [rf"(?<!\d){n}(?!\d)"],
    }
    for n in range(1, NUMBER_COUNT + 1)
]


# --------------------------------------------------------------------------
# 활성 물체 세트. 기본은 색 세트(기존 동작 그대로).
# --------------------------------------------------------------------------
OBJECT_SETS = {"color": COLOR_OBJECTS, "number": NUMBER_OBJECTS}
ACTIVE_OBJECT_SET = "color"

OBJECTS: list[dict] = COLOR_OBJECTS
OBJECT_NAMES = [o["name"] for o in OBJECTS]
OBJECT_BY_NAME = {o["name"]: o for o in OBJECTS}


def use_object_set(name: str) -> list[dict]:
    """활성 물체 세트를 바꾼다.

    **SortingEnv 를 만들기 전에** 불러야 한다. env 가 생성 시점에 물체 이름으로
    MuJoCo 조인트 주소를 잡아 두기 때문에, 나중에 바꾸면 이름이 맞지 않는다.
    숫자 세트는 vlapipe/numbered.py 의 activate() 를 쓰는 편이 낫다(텍스처까지 챙긴다).
    """
    global OBJECTS, OBJECT_NAMES, OBJECT_BY_NAME, ACTIVE_OBJECT_SET
    if name not in OBJECT_SETS:
        raise ValueError(f"모르는 물체 세트: {name!r} (가능: {sorted(OBJECT_SETS)})")
    OBJECTS = OBJECT_SETS[name]
    OBJECT_NAMES = [o["name"] for o in OBJECTS]
    OBJECT_BY_NAME = {o["name"]: o for o in OBJECTS}
    ACTIVE_OBJECT_SET = name
    return OBJECTS


def half_size(obj: dict) -> tuple[float, float, float]:
    return tuple(v / 2.0 for v in obj["full"])  # type: ignore[return-value]


# --------------------------------------------------------------------------
# 카메라 (260827 문서: 로봇을 마주보는 면 상단 / 로봇 없는 옆면 상단 / 손목)
# LIBERO 관례에 맞춘 이름을 쓴다.
# --------------------------------------------------------------------------
ALL_CAMERAS = ("agentview", "frontview", "sideview", "robot0_eye_in_hand")
# 학습/추론에 쓰는 카메라 (LIBERO 와 동일하게 2개)
CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")
IMAGE_SIZE = 256           # LIBERO 기본 해상도
RENDER_SIZE = (640, 480)   # 사람이 보는 미리보기용

# --------------------------------------------------------------------------
# 제어 / 액션 규격 (LIBERO OSC_POSE 관례)
#   action = [dx, dy, dz, drx, dry, drz, gripper]  (앞 6개는 [-1,1] 정규화 델타)
#   gripper: -1 = 열기, +1 = 닫기   <- LIBERO/robosuite 관례
# --------------------------------------------------------------------------
ACTION_DIM = 7
CONTROL_HZ = 20
SIM_TIMESTEP = 0.002
SIM_STEPS_PER_ACTION = int(round((1.0 / CONTROL_HZ) / SIM_TIMESTEP))  # 25

MAX_DPOS = 0.05   # 한 제어 스텝당 최대 위치 변화 [m]
MAX_DROT = 0.50   # 한 제어 스텝당 최대 회전 변화 [rad]

GRIPPER_OPEN_CMD = -1.0
GRIPPER_CLOSE_CMD = +1.0
GRIPPER_CTRL_OPEN = 255.0    # panda.xml actuator8 ctrlrange (0=닫힘, 255=열림)
GRIPPER_CTRL_CLOSE = 0.0
GRIPPER_QPOS_OPEN = 0.04     # finger joint 최대 (한쪽)

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ARM_ACTUATORS = [f"actuator{i}" for i in range(1, 8)]
GRIPPER_ACTUATOR = "actuator8"
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]
EEF_SITE = "tcp"          # setup_assets.py 가 hand 바디에 추가
EEF_BODY = "hand"

# 홈 자세 (panda.xml keyframe 'home')
HOME_QPOS = [0.0, -0.30, 0.0, -2.20, 0.0, 1.95, 0.785]

# 관측 키 (LIBERO 호환)
OBS_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_joint_pos",
    "robot0_gripper_qpos",
    "robot0_eef_pos",
    "robot0_eef_quat",
)
STATE_KEYS = ("robot0_joint_pos", "robot0_gripper_qpos", "robot0_eef_pos", "robot0_eef_quat")
STATE_DIM = 7 + 2 + 3 + 4  # 16
