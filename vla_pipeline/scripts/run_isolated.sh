#!/usr/bin/env bash
# 무거운 작업(대형 VLM 로드, 파인튜닝)을 VSCode 와 분리된 cgroup 에서 실행한다.
#
# 왜 필요한가:
#   VSCode 통합 터미널에서 실행하면 그 프로세스는 app-code-*.scope 안에서 돈다.
#   메모리 압박이 커지면 systemd-oomd 가 그 scope 를 통째로 죽이는데,
#   거기엔 VSCode 본체도 들어 있어서 편집기까지 같이 꺼진다.
#   별도 scope 로 띄우면 최악의 경우에도 이 작업만 죽는다.
#
# 사용:
#   ./scripts/run_isolated.sh python scripts/run_pipeline.py --planner vlm
#   MEM=12G ./scripts/run_isolated.sh python train.py
set -euo pipefail
MEM="${MEM:-12G}"
if ! command -v systemd-run >/dev/null 2>&1; then
    echo "systemd-run 이 없습니다. 그냥 실행합니다." >&2
    exec "$@"
fi
echo "격리 실행 (MemoryMax=$MEM): $*" >&2
exec systemd-run --user --scope \
    -p MemoryMax="$MEM" \
    -p ManagedOOMMemoryPressure=auto \
    --quiet -- "$@"
