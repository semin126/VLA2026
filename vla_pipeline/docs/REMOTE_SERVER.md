# 조교 PC를 학생용 GPU 서버로 열기

RTX 4080(16GB) 데스크톱을 학생 3명이 원격으로 쓰게 만드는 방법.
**sudo가 필요한 단계는 조교가 직접 실행하세요.** 아래는 검토용 절차서입니다.

---

## 0. 먼저 — 정말 필요한가?

Colab 예산은 부족하지 않습니다(아래 표). 서버는 **대체재가 아니라 보완재**로 보세요.

| | Colab Pro+ (3명, 9/3~11/7) | 이 서버 |
|---|---|---|
| 가용 학습 | L4 기준 약 110회 | 시간 제한 없음 |
| 필요 추정 | 13회 | — |
| VRAM | L4 24GB / A100 40GB | **16GB (더 작음)** |
| 동시 사용 | 3명 각자 | **1명씩 (GPU 1장)** |
| 세션 한도 | 24시간 | 없음 |

**서버가 이기는 지점**: ① CU 걱정 없이 마음껏 실패해 볼 수 있음 ② 24시간 넘는 긴 실험
③ 데이터가 한곳에 모임(윈도우 → Drive → Colab 왕복이 사라짐) ④ 조교가 직접 들여다볼 수 있음

**서버가 지는 지점**: VRAM이 16GB뿐이라 배치 크기가 제한되고, GPU가 1장이라 3명이 동시에 학습할 수 없습니다.

> 권장: **데이터 보관 + 반복 학습 + 조교 디버깅**은 서버, **학생 각자의 병렬 실험**은 Colab.

---

## 1. 이 머신의 현재 상태

```
enp3s0   143.248.144.87/24   ← KAIST 공인 IP, 기본 경로 (NAT 아님)
wlx…     192.168.1.84/24     ← WiFi, 보조
openssh-server               미설치
사용자 계정                   nmail (uid 1000) 하나뿐
GPU                          RTX 4080 16GB
디스크 여유                   / 218GB, /media/nmail/DATA 157GB
```

공인 IP가 인터페이스에 직접 붙어 있으므로 **캠퍼스 방화벽만 허용하면 외부에서 바로 접속됩니다.**
다만 KAIST는 외부 inbound를 기본 차단하는 경우가 많아 **반드시 확인이 필요합니다.**

---

## 2. 경로 선택

```
      SSH 서버 설치
            │
   외부에서 22번 포트가 열리는가?
       ├── 예  →  [경로 A] 직접 SSH        (가장 단순)
       └── 아니오 → [경로 B] Tailscale     (방화벽 무관, 권장 대안)
```

### 확인 방법
학교 WiFi가 **아닌** 회선(휴대폰 LTE 테더링 등)에서:
```bash
ssh -o ConnectTimeout=5 nmail@143.248.144.87
```
접속되면 경로 A, `Connection timed out`이면 경로 B입니다.
`scripts/check_server.sh` 가 이 확인을 도와줍니다.

---

## 3. 공통 — SSH 서버 설치

```bash
sudo apt update && sudo apt install -y openssh-server
sudo systemctl enable --now ssh
systemctl status ssh --no-pager
```

---

## 4-A. 경로 A — 직접 SSH (공인 IP가 열려 있을 때)

인터넷에 SSH를 여는 것이므로 **보안 설정을 건너뛰지 마세요.**

```bash
# 1) 학생 공개키를 미리 받아 둔다 (학생이 자기 윈도우에서 실행)
#    ssh-keygen -t ed25519 -C "student1"
#    type $env:USERPROFILE\.ssh\id_ed25519.pub     ← 이 내용을 조교에게 전달

# 2) 학생 계정 생성 (조교)
for u in vla1 vla2 vla3; do
  sudo adduser --disabled-password --gecos "" $u
  sudo mkdir -p /home/$u/.ssh && sudo chmod 700 /home/$u/.ssh
  # 전달받은 공개키를 /home/$u/.ssh/authorized_keys 에 붙여넣기
  sudo chown -R $u:$u /home/$u/.ssh
  sudo chmod 600 /home/$u/.ssh/authorized_keys
done

# 3) 비밀번호 로그인 차단 (키만 허용)
sudo tee /etc/ssh/sshd_config.d/99-vla.conf >/dev/null <<'EOF'
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
AllowUsers nmail vla1 vla2 vla3
EOF
sudo systemctl restart ssh

# 4) 무차별 대입 차단
sudo apt install -y fail2ban && sudo systemctl enable --now fail2ban
```

> 포트를 22가 아닌 다른 번호로 바꾸면 로그 소음이 크게 줄어듭니다.
> `Port 22022` 를 위 설정에 추가하고 학생에게 `ssh -p 22022` 로 안내하세요.

---

## 4-B. 경로 B — Tailscale (방화벽에 막혔을 때)

포트 개방도, 공인 IP도 필요 없습니다. WireGuard 기반 사설 메시 네트워크입니다.

```bash
# 조교 PC
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
sudo tailscale set --ssh          # Tailscale SSH 사용 시 (선택)
tailscale ip -4                   # 100.x.y.z 주소 확인
tailscale status
```

학생 윈도우 노트북: <https://tailscale.com/download/windows> 설치 후 같은 계정으로 로그인.
그다음 학생은 `ssh vla1@<머신이름>` 또는 `ssh vla1@100.x.y.z` 로 접속합니다.

> **계정 수 주의**: Tailscale 무료(Personal) 플랜은 사용자 3명 / 기기 100대입니다.
> 조교 + 학생 3명 = 4명이면 초과합니다. 해결책은 둘 중 하나:
> ① 학생 노트북을 **조교 계정의 기기로** 등록(간단, 다만 계정 공유)
> ② 조교가 이 머신 하나만 학생 각자의 tailnet에 **공유(node sharing)** — 공유받은 노드는
>    별도 사용자로 계산되지 않습니다.

경로 B에서도 **§4-A의 학생 계정 생성과 키 등록은 그대로 해야 합니다.**
Tailscale은 "어떻게 도달하는가"만 해결하고 "누가 무엇을 할 수 있는가"는 해결하지 않습니다.

---

## 5. 필수 — 자원 한도 (이걸 빼먹으면 조교 PC가 죽습니다)

이 머신은 `systemd-oomd` 가 메모리 압박 시 **VSCode를 통째로 종료**시킵니다(README 9절).
학생이 원격에서 무거운 작업을 돌리면 조교 세션이 같이 죽을 수 있습니다.

```bash
# 학생 슬라이스별 메모리 상한 (영구)
for u in vla1 vla2 vla3; do
  uid=$(id -u $u)
  sudo mkdir -p /etc/systemd/system/user-$uid.slice.d
  sudo tee /etc/systemd/system/user-$uid.slice.d/limits.conf >/dev/null <<EOF
[Slice]
MemoryMax=10G
MemoryHigh=8G
CPUQuota=600%
EOF
done
sudo systemctl daemon-reload
```

`MemoryHigh` 는 그 지점부터 회수 압력을 걸어 **oomd가 발동하기 전에** 브레이크를 겁니다.

---

## 6. GPU 한 장을 셋이 나누기

SmolVLA 학습은 배치 8에서 10~16GB를 씁니다. **4080 16GB로는 한 번에 한 명뿐입니다.**

가장 단순한 방법 — 잠금 파일:

```bash
# /usr/local/bin/gpu-lock  (조교가 설치)
sudo tee /usr/local/bin/gpu-lock >/dev/null <<'EOF'
#!/usr/bin/env bash
# 사용: gpu-lock python train.py ...
LOCK=/tmp/vla-gpu.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "GPU 사용 중: $(cat $LOCK.who 2>/dev/null)" >&2
  echo "기다리려면 gpu-lock -w ... 를 쓰세요." >&2
  exit 1
fi
echo "$USER $(date '+%m-%d %H:%M')" > "$LOCK.who"
trap 'rm -f "$LOCK.who"' EXIT
exec "$@"
EOF
sudo chmod 755 /usr/local/bin/gpu-lock
```

학생 사용법: `gpu-lock python -m lerobot.scripts.train ...`
누가 쓰고 있는지: `cat /tmp/vla-gpu.lock.who`

> 더 간단한 대안: **학습은 조교만 실행**하고, 학생은 데이터 준비와 설정 파일까지만 맡습니다.
> 남은 회차가 적은 상황(§일정)에서는 이쪽이 오히려 빠를 수 있습니다.

---

## 7. 학생 쪽 사용법 — VSCode Remote-SSH

학생들이 이미 VSCode를 쓰므로 이게 가장 자연스럽습니다.

1. VSCode 확장 **Remote - SSH** 설치
2. `F1` → `Remote-SSH: Add New SSH Host` → `ssh vla1@143.248.144.87` (또는 Tailscale 이름)
3. 접속하면 편집·터미널·주피터가 전부 서버에서 돌아갑니다. GPU도 그대로 잡힙니다.
4. 노트북(`.ipynb`)도 VSCode에서 바로 열려 Colab과 거의 같은 경험이 됩니다.

**참고 — Colab UI를 유지하고 싶다면**: Colab은 로컬 런타임 연결을 지원합니다.
서버에서 `pip install jupyter_http_over_ws` 후 주피터를 띄우고, SSH 포트 포워딩
(`ssh -L 8888:localhost:8888 vla1@…`)으로 학생 PC에 끌어온 뒤 Colab의
"로컬 런타임에 연결"에 붙입니다. 다만 VSCode Remote-SSH 쪽이 훨씬 단순합니다.

---

## 8. 점검 순서 요약

```bash
# 조교
sudo apt install -y openssh-server && sudo systemctl enable --now ssh
./scripts/check_server.sh                 # 준비 상태 점검
# → 외부 접속 확인 후 경로 A 또는 B 선택
# → 학생 계정 생성 + 키 등록 (§4-A)
# → 자원 한도 설정 (§5)  ★ 빼먹지 말 것
# → gpu-lock 설치 (§6)

# 학생
# Windows에서 ssh-keygen → 공개키 전달 → VSCode Remote-SSH 접속
```
