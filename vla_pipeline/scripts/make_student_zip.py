"""학생 배포용 vla_pipeline.zip 을 만든다 (조교용).

학생은 이 zip 하나만 받아서 SETUP_STUDENT.md 대로 하면 환경이 구성된다.

  python scripts/make_student_zip.py
  python scripts/make_student_zip.py --out C:\\temp --name vla_pipeline_0910.zip

포함:   소스 전부 + assets/ (git 없이도 돌게 하려면 반드시 있어야 함)
        + SETUP_STUDENT.md / AGENTS.md / CLAUDE.md / README.md + .vscode/
제외:   .venv (OS·경로에 묶여 있어 남의 PC 에서 못 씀)
        outputs/ (실행하면 다시 생김)
        data/ (조교의 연구 데이터 — 학생에게 보낼 것이 아님)
        __pycache__, *.pyc, .git
"""
import _bootstrap  # noqa: F401
import argparse
import zipfile
from pathlib import Path

from vlapipe import config as C

ROOT = Path(C.PROJECT_DIR)

# 통째로 뺄 디렉터리 이름 (경로 어디에 있든)
EXCLUDE_DIRS = {".venv", "outputs", "data", "__pycache__", ".git", ".pytest_cache",
                ".ipynb_checkpoints", ".mypy_cache"}
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".hdf5", ".zip"}

# 이것들이 없으면 학생 PC 에서 반드시 문제가 생긴다 -> 빠졌으면 만들지 않는다.
REQUIRED = [
    "README.md",
    "SETUP_STUDENT.md",
    "AGENTS.md",
    "requirements-windows.txt",
    "vlapipe/config.py",
    "scripts/check_env.py",
    "scripts/smoke_test.py",
    "assets/franka_emika_panda/panda_vla.xml",   # 패치본. 없으면 setup_assets.py 가 필요해짐
    "assets/franka_emika_panda/panda.xml",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent), help="zip 을 놓을 폴더")
    ap.add_argument("--name", default="vla_pipeline.zip")
    ap.add_argument("--force", action="store_true", help="필수 파일이 없어도 만든다")
    return ap.parse_args()


def collect() -> list[Path]:
    files = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if EXCLUDE_DIRS & set(rel.parts):
            continue
        if p.suffix.lower() in EXCLUDE_SUFFIX:
            continue
        files.append(p)
    return sorted(files)


def main() -> int:
    args = parse_args()

    missing = [r for r in REQUIRED if not (ROOT / r).exists()]
    if missing:
        print("필수 파일이 없습니다:")
        for m in missing:
            print(f"  - {m}")
        if "assets/franka_emika_panda/panda_vla.xml" in missing:
            print("\n에셋이 없으면 학생이 setup_assets.py 를 돌려야 하고, 그러려면 git 이 필요합니다.")
            print("먼저 `python setup_assets.py` 를 실행해 에셋을 만든 뒤 다시 시도하세요.")
        if not args.force:
            return 1
        print("\n--force 가 붙어 그대로 진행합니다.\n")

    files = collect()
    if not files:
        print("포함할 파일이 없습니다.")
        return 1

    dst = Path(args.out) / args.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    # zip 안에서 한 겹 폴더(vla_pipeline/)로 감싼다. 압축을 풀면 폴더가 하나 나온다.
    top = ROOT.name

    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, arcname=str(Path(top) / p.relative_to(ROOT)))

    total = sum(p.stat().st_size for p in files)
    print(f"파일 {len(files)}개, 원본 {total / 1e6:.1f} MB")
    print(f"만들었습니다: {dst}  ({dst.stat().st_size / 1e6:.1f} MB)")
    print(f"\n압축을 풀면 '{top}' 폴더 하나가 나옵니다.")
    print("학생에게는 SETUP_STUDENT.md 부터 읽으라고 알려 주세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
