#!/usr/bin/env bash
# 조교 PC를 GPU 서버로 열기 전 준비 상태 점검.
# 아무것도 바꾸지 않고 확인만 합니다.  사용: ./scripts/check_server.sh
set -uo pipefail
ok(){ printf '  \033[32mOK  \033[0m %s\n' "$1"; }
no(){ printf '  \033[31m없음\033[0m %s\n' "$1"; }
wr(){ printf '  \033[33m주의\033[0m %s\n' "$1"; }

echo "=== 1. SSH 서버 ==="
if systemctl is-active --quiet ssh 2>/dev/null || systemctl is-active --quiet sshd 2>/dev/null; then
  ok "sshd 실행 중 ($(ss -tlnH 'sport = :22' 2>/dev/null | wc -l) 개 리스너)"
else
  no "sshd 미실행 -> sudo apt install -y openssh-server && sudo systemctl enable --now ssh"
fi

echo; echo "=== 2. 네트워크 ==="
ip -br addr show 2>/dev/null | grep -v DOWN | grep -v '^lo' | while read -r n s a rest; do
  case "$a" in
    10.*|192.168.*|172.1[6-9].*|172.2[0-9].*|172.3[01].*) echo "  $n  $a  (사설)";;
    100.*) echo "  $n  $a  (Tailscale)";;
    *) echo "  $n  $a  (공인 가능성)";;
  esac
done
DEF=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'src \K[\d.]+')
[ -n "${DEF:-}" ] && echo "  기본 경로 소스 IP: $DEF"

echo; echo "=== 3. 터널링 도구 ==="
for c in tailscale cloudflared; do
  command -v "$c" >/dev/null && ok "$c 설치됨" || no "$c 미설치"
done

echo; echo "=== 4. 계정 ==="
n=$(awk -F: '$3>=1000 && $3<65534' /etc/passwd | wc -l)
awk -F: '$3>=1000 && $3<65534 {print "  " $1 " (uid " $3 ")"}' /etc/passwd
[ "$n" -le 1 ] && wr "학생 계정이 없습니다. 조교 계정을 공유하지 마세요 (docs/REMOTE_SERVER.md §4-A)"

echo; echo "=== 5. 자원 보호 (★ 중요) ==="
found=0
for d in /etc/systemd/system/user-*.slice.d; do
  [ -d "$d" ] && { ok "$(basename "$d") 에 한도 설정 있음"; found=1; }
done
[ "$found" -eq 0 ] && wr "사용자 슬라이스 메모리 한도가 없습니다. 학생 작업이 조교 VSCode를 죽일 수 있습니다 (§5)"
systemctl is-active --quiet systemd-oomd && wr "systemd-oomd 활성 — 한도 설정을 반드시 하세요" || ok "systemd-oomd 비활성"

echo; echo "=== 6. GPU ==="
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | sed 's/^/  /'
  c=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
  [ "${c:-1}" -le 1 ] && wr "GPU 1장 — 동시에 한 명만 학습 가능. gpu-lock 설치 권장 (§6)"
else
  no "nvidia-smi 없음"
fi
command -v gpu-lock >/dev/null && ok "gpu-lock 설치됨" || no "gpu-lock 미설치 (§6)"

echo; echo "=== 7. 디스크 ==="
df -h / 2>/dev/null | tail -1 | awk '{print "  /  여유 " $4 " (" $5 " 사용)"}'

echo
echo "다음 단계: docs/REMOTE_SERVER.md"
echo "외부 접속 확인은 학교 WiFi가 아닌 회선에서:"
echo "  ssh -o ConnectTimeout=5 <계정>@${DEF:-<공인IP>}"
