import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def py_cmd() -> str:
    """지금 이 스크립트를 돌리고 있는 파이썬을, 사용자가 그대로 복사해 쓸 수 있는 형태로.

    안내 문구에 `python` 이라고 적어 두면 윈도우에서 문제가 된다. 학생 PC 에서
    `python` 은 venv 밖의 다른 파이썬이거나 Microsoft Store 안내창이기 십상이다.
    그래서 실제로 돌고 있는 인터프리터 경로를 알려 준다.
    프로젝트 폴더 안에서 실행했다면 `.\\.venv\\Scripts\\python.exe` 처럼 짧게 나온다.
    """
    exe = Path(sys.executable)
    try:
        rel = exe.relative_to(Path.cwd())
    except ValueError:
        return str(exe)
    return f".\\{rel}" if os.name == "nt" else f"./{rel}"


def cmd(script: str, *args: str) -> str:
    """안내 문구에 넣을 실행 명령 한 줄. 경로 구분자도 OS 에 맞춘다.

        cmd("scripts/run_pipeline.py", "--layout", "messy")
        -> '.\\.venv\\Scripts\\python.exe scripts\\run_pipeline.py --layout messy'
    """
    return " ".join([py_cmd(), str(Path(script)), *args])
