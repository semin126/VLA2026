"""Panda 에셋 준비 — MuJoCo Menagerie 에서 franka_emika_panda 를 받아
손목 카메라(robot0_eye_in_hand)와 TCP 사이트(tcp)를 추가한 panda_vla.xml 을 만든다.

윈도우/Colab 모두에서 한 번만 실행하면 된다.

    python setup_assets.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlapipe import config as C  # noqa: E402

# 손목 카메라: hand 프레임 기준으로 TCP(0,0,0.1034)를 바라보도록 계산해 둔 값.
# MuJoCo 카메라는 자기 프레임의 -z 방향을 본다. xyaxes = (x_cam, y_cam).
WRIST_CAM = {
    "name": "robot0_eye_in_hand",
    "pos": "0.055 0 0.015",
    "xyaxes": "0 1 0  0.8486 0 0.5292",   # 영상 오른쪽=월드+x, 위=작업영역 바깥쪽
    "fovy": "58",
}
TCP_SITE = {
    "name": "tcp",
    "pos": "0 0 0.1034",           # Franka 플랜지->TCP 표준 오프셋
    "size": "0.005",
    "rgba": "1 0 0 0.0",           # 렌더에 안 보이게
    "group": "4",
}


def fetch_menagerie(dest: Path) -> Path:
    """Menagerie 의 franka_emika_panda 폴더만 sparse checkout 으로 받는다."""
    if (dest / "panda.xml").exists():
        print(f"✓ 이미 있음: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "menagerie"
        print("Menagerie 다운로드 중 (약 40MB)...")
        subprocess.check_call(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             C.MENAGERIE_URL, str(repo)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.check_call(["git", "-C", str(repo), "sparse-checkout", "set",
                               "franka_emika_panda"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        src = repo / "franka_emika_panda"
        if not (src / "panda.xml").exists():
            raise RuntimeError("Menagerie 에서 panda.xml 을 찾지 못했습니다.")
        shutil.copytree(src, dest, dirs_exist_ok=True)
    print(f"✓ 다운로드 완료: {dest}")
    return dest


def patch_panda(panda_dir: Path) -> Path:
    """panda.xml 의 hand 바디에 손목 카메라와 TCP 사이트를 추가한 사본을 만든다."""
    src = panda_dir / "panda.xml"
    tree = ET.parse(src)
    root = tree.getroot()

    hand = None
    for body in root.iter("body"):
        if body.get("name") == "hand":
            hand = body
            break
    if hand is None:
        raise RuntimeError("panda.xml 에서 <body name='hand'> 를 찾지 못했습니다.")

    existing = {c.get("name") for c in hand}
    if WRIST_CAM["name"] not in existing:
        cam = ET.SubElement(hand, "camera")
        for k, v in WRIST_CAM.items():
            cam.set(k, v)
    if TCP_SITE["name"] not in existing:
        site = ET.SubElement(hand, "site")
        for k, v in TCP_SITE.items():
            site.set(k, v)

    # 로봇 베이스를 받침대 위(테이블 옆)로 올리고, 작업 영역(+y)을 향하도록 90도 회전.
    link0 = None
    for body in root.iter("body"):
        if body.get("name") == "link0":
            link0 = body
            break
    if link0 is None:
        raise RuntimeError("panda.xml 에서 <body name='link0'> 를 찾지 못했습니다.")
    bx, by = C.ROBOT_BASE_XY
    link0.set("pos", f"{bx:.4f} {by:.4f} {C.ROBOT_BASE_Z:.4f}")
    link0.set("quat", "0.70711 0 0 0.70711")   # +x 를 +y 로

    root.set("model", "panda_vla")
    out = panda_dir / "panda_vla.xml"
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=False)
    print(f"✓ 패치 완료: {out}  (손목 카메라 + TCP 사이트 추가)")
    return out


def main() -> int:
    panda_dir = C.PANDA_DIR
    fetch_menagerie(panda_dir)
    patch_panda(panda_dir)

    try:
        import mujoco
    except ImportError:
        print("⚠️ mujoco 가 설치되어 있지 않아 검증을 건너뜁니다. "
              "pip install -r requirements-windows.txt 후 다시 실행하세요.")
        return 0

    model = mujoco.MjModel.from_xml_path(C.mujoco_path(C.PANDA_PATCHED_XML))
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "robot0_eye_in_hand")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    assert cam_id >= 0 and site_id >= 0, "패치된 카메라/사이트를 찾지 못했습니다."
    print(f"✓ 검증 통과 — mujoco {mujoco.__version__}, nq={model.nq}, nu={model.nu}, "
          f"카메라 {model.ncam}개, 사이트 {model.nsite}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
