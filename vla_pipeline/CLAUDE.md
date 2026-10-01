# CLAUDE.md

이 폴더의 에이전트 안내는 [AGENTS.md](AGENTS.md) 에 있습니다.
**작업을 시작하기 전에 AGENTS.md 를 먼저 읽으세요.**

요약:
- 계층적 VLA 파이프라인(MuJoCo + Franka Panda) 실행 코드. 사용자는 고등학생입니다.
- 작업 전후로 `.\.venv\Scripts\python.exe scripts\smoke_test.py` 를 돌려 확인하세요.
- 액션/관측 규격(7차원 델타 액션, 16차원 state)은 **LIBERO 호환**입니다. 바꾸지 마세요.
- 수치는 `vlapipe/config.py` 에만 둡니다. 추측한 값을 실측값처럼 보고하지 마세요.
- 설치 문제는 [SETUP_STUDENT.md](SETUP_STUDENT.md) 를 보세요.
