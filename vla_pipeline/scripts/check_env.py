r"""환경 점검 — 수업 시작 전에 제일 먼저 실행하세요.

    .\.venv\Scripts\python.exe scripts\check_env.py           환경만 점검
    .\.venv\Scripts\python.exe scripts\check_env.py --bench   20Hz 텔레옵을 감당하는지 측정
"""
import _bootstrap  # noqa: F401
import argparse
import importlib.util as iu
import os
import platform
import sys
import time
from pathlib import Path

from vlapipe import config as C

REQUIRED = ["mujoco", "numpy", "h5py", "imageio", "PIL"]
OPTIONAL = {"pygame": "게임패드 텔레옵", "torch": "VLM/VLA 추론·학습",
            "transformers": "SmolVLM2 플래너", "lerobot": "SmolVLA 학습(Colab)",
            "matplotlib": "그래프"}


def benchmark() -> bool:
    """한 제어 스텝(50ms 예산)의 구간별 비용을 재서 이 PC 가 녹화를 감당하는지 본다."""
    import numpy as np
    from vlapipe.env import SortingEnv
    from vlapipe import config as C

    print("\n[성능 측정] 텔레옵 녹화는 20Hz — 한 스텝에 50ms 예산")
    env = SortingEnv(layout="tidy", image_size=256,
                     cameras=("agentview", "robot0_eye_in_hand"))
    a = np.zeros(C.ACTION_DIM, dtype=np.float32); a[1] = 0.3; a[6] = -1.0

    def bench(fn, n=40):
        fn()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t) / n * 1000

    t_phys = bench(lambda: env._sim_step(C.SIM_STEPS_PER_ACTION))
    t_ik = bench(lambda: env.ctrl.apply_action(a))
    t_img = bench(lambda: env.render("agentview", 256, 256))
    t_view = bench(lambda: env.render("agentview", 480, 640))
    env.close()

    loop = t_phys + t_ik + 2 * t_img
    print(f"  물리 25스텝     {t_phys:6.2f} ms")
    print(f"  IK + 액션       {t_ik:6.2f} ms")
    print(f"  렌더 256x256    {t_img:6.2f} ms  (카메라 2대 = {2 * t_img:.2f})")
    print(f"  뷰어 640x480    {t_view:6.2f} ms")
    print(f"  ─ 녹화 루프 합계 {loop:6.2f} ms / 50 ms")
    if loop < 25:
        print("  판정: 여유 있음. 녹화 그대로 진행하세요.")
    elif loop < 45:
        print("  판정: 빠듯하지만 가능. 뷰어를 작게 쓰거나 --image-size 128 을 권장합니다.")
    else:
        print("  판정: 20Hz 유지 불가. --image-size 128 로 낮추고, 그래도 안 되면")
        print("        teleop_record.py 를 --no-viewer 로 돌리거나 Colab 에서 녹화하세요.")
    return loop < 45


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="텔레옵 성능 측정까지 수행")
    args = ap.parse_args()
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Platform : {platform.platform()}")
    print(f"MUJOCO_GL: {os.environ.get('MUJOCO_GL', '(자동)')}")
    bad = []
    print("\n[필수 패키지]")
    for m in REQUIRED:
        ok = iu.find_spec(m) is not None
        print(f"  {'OK ' if ok else '없음'} {m}")
        if not ok:
            bad.append(m)
    print("\n[선택 패키지]")
    for m, why in OPTIONAL.items():
        print(f"  {'OK ' if iu.find_spec(m) else '없음'} {m:14s} ({why})")

    # GPU 가 없으면 VLM/VLA 는 CPU 로 돌아가긴 하지만 훨씬 느리다.
    # Colab 에서 런타임 유형을 안 바꾼 채 시작하는 일이 잦아 여기서 미리 알려 준다.
    if iu.find_spec("torch"):
        import torch
        print("\n[연산 장치]")
        print(f"  torch {torch.__version__}, CUDA {'사용 가능' if torch.cuda.is_available() else '없음'}")
        if torch.cuda.is_available():
            print(f"  OK  {torch.cuda.get_device_name(0)}")
        else:
            print("  주의: VLM/VLA 가 CPU 로 돌아갑니다 (SmolVLM2-500M 추론 2초 -> 10초대).")
            print("        Colab 이면 런타임 > 런타임 유형 변경 > T4 GPU 로 바꾸고 처음부터 다시 실행하세요.")
            print("        윈도우 PC 면 README 11절('torch.cuda.is_available() 이 False')을 보세요.")

    print("\n[에셋]")
    for label, path in [("Panda 원본", C.PANDA_DIR / "panda.xml"),
                        ("패치본", C.PANDA_PATCHED_XML)]:
        print(f"  {'OK ' if Path(path).exists() else '없음'} {label}: {path}")
    if not C.PANDA_PATCHED_XML.exists():
        print("  -> python setup_assets.py 를 먼저 실행하세요.")
        bad.append("assets")

    if not bad:
        print("\n[렌더링 테스트]")
        try:
            from vlapipe.env import SortingEnv
            env = SortingEnv(layout="messy", seed=1, image_size=128)
            img = env.render("agentview")
            print(f"  OK  agentview {img.shape}, 평균밝기 {img.mean():.1f}")
            print(f"  OK  물체 {len(C.OBJECT_NAMES)}개, 잘못 놓인 것 {len(env.misplaced_objects())}개")
            env.close()
        except Exception as e:
            print(f"  실패: {type(e).__name__}: {e}")
            bad.append("render")

    if not bad and args.bench:
        try:
            benchmark()
        except Exception as e:
            print(f"  성능 측정 실패: {type(e).__name__}: {e}")

    if bad:
        print(f"\n문제: {bad}")
        print("SETUP_STUDENT.md 8절(문제 해결)을 보세요.")
        return 1
    print("\n모두 정상입니다. 다음은 실제로 도는지 확인하세요:")
    print("  " + _bootstrap.cmd("scripts/smoke_test.py"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
