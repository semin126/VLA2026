r"""게임패드/키보드로 로봇을 몰면서 데모를 녹화한다 (윈도우 로컬에서 실행).

수업 흐름:
  1) 정돈된 씬에서 시작
  2) 녹화 시작 -> 물체를 집어 옮기며 **어지른다** -> 녹화 종료
  3) scripts/build_dataset.py 로 뒤집어서 '정돈하는' 학습 데이터로 만든다

  .\.venv\Scripts\python.exe scripts\teleop_record.py --input keyboard   # 패드 없을 때
  .\.venv\Scripts\python.exe scripts\teleop_record.py --calibrate        # 패드 매핑 만들기
  .\.venv\Scripts\python.exe scripts\teleop_record.py --input gamepad --episodes 5

--calibrate 로 만든 gamepad_map.json 이 폴더에 있으면 --gamepad-map 을 생략해도
자동으로 씁니다.

중요: 물체를 던지거나 떨어뜨리지 마세요. 역재생하면 물체가 저절로 날아오르는
      물리적으로 불가능한 장면이 됩니다. 항상 집어서-옮겨서-내려놓기.
"""
import _bootstrap  # noqa: F401
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from vlapipe import camview
from vlapipe import config as C
from vlapipe import teleop
from vlapipe.env import SortingEnv
from vlapipe.episode import EpisodeWriter
from vlapipe.teleop import (GamepadInput, KeyboardInput, dump_layout, load_gamepad_map,
                            load_layout)

TASK_MESS = "Take the objects out of the storage zones and scatter them on the table."
TASK_TIDY = "Tidy up the table: put every object back into its storage zone."


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="keyboard", choices=["gamepad", "keyboard"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--out", default=str(C.EPISODE_DIR))
    ap.add_argument("--layout", default="tidy",
                    help="'tidy'/'messy'/'random'/'cluttered' 또는 배치 JSON 경로")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=C.IMAGE_SIZE)
    ap.add_argument("--cameras", default="agentview,robot0_eye_in_hand")
    ap.add_argument("--speed", type=float, default=0.55, help="전체 속도")
    # z 는 기본으로 느리다. 한 제어 스텝의 최대 이동이 0.05 m 라, speed 0.55 면 27.5 mm 로
    # 타일 높이(30 mm)와 맞먹어서 한 스텝에 물체를 지나쳐 버린다.
    ap.add_argument("--speed-z", type=float, default=None,
                    help=f"높이(z) 배율. 기본 {teleop.AXIS_GAIN['gain_z']} "
                         f"(작을수록 정밀. 0.2 면 스텝당 약 5.5 mm)")
    ap.add_argument("--joystick", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true",
                    help="시키는 대로 움직이면 축·버튼 매핑을 찾아 JSON 으로 저장한다")
    ap.add_argument("--calibrate-raw", action="store_true",
                    help="축·버튼 값을 그대로 띄운다 (진단용)")
    ap.add_argument("--gamepad-map", default=None,
                    help="축/버튼 번호를 적은 JSON 경로 (--calibrate 결과). "
                         "생략하면 gamepad_map.json 이 있을 때 자동으로 씁니다")
    ap.add_argument("--save-map", default="gamepad_map.json",
                    help="--calibrate 결과를 저장할 경로")
    ap.add_argument("--no-viewer", action="store_true", help="창 없이(헤드리스) 테스트용")
    # 팔이 물체를 가려서 조작이 어려울 때, 고정 카메라 여러 대를 옆 창에 띄운다.
    ap.add_argument("--no-panel", action="store_true",
                    help="카메라 뷰 창을 띄우지 않는다 (기본은 띄움)")
    ap.add_argument("--panel-cameras", default=",".join(camview.DEFAULT_CAMERAS),
                    help=f"뷰 창에 띄울 카메라. 쓸 수 있는 것: {','.join(C.ALL_CAMERAS)}")
    ap.add_argument("--panel-size", type=int, default=240, help="뷰 창의 칸 하나 크기(px)")
    ap.add_argument("--panel-cols", type=int, default=None, help="뷰 창 열 수 (기본: 최대 3)")
    ap.add_argument("--panel-every", type=int, default=1,
                    help="N 프레임마다 갱신 (느리면 2~3 으로)")
    return ap.parse_args()


def reset_env(env, spec, seed):
    # 이름은 scene.LAYOUTS 에서 가져온다. 여기에 손으로 적어 두면 'cluttered' 처럼
    # 나중에 늘어난 배치가 빠져서, 이름을 파일 경로로 오해하고 엉뚱한 오류를 낸다.
    from vlapipe.scene import LAYOUTS
    if spec in LAYOUTS:
        env.reset(layout=spec, seed=seed)
    else:
        env.reset(layout="tidy")
        load_layout(env, Path(spec))


def main() -> int:
    args = parse_args()
    # 캘리브레이션 중에는 기존 매핑을 읽지 않는다. 아직 만들지도 않은 파일을
    # --gamepad-map 으로 함께 넘기는 실수가 잦고, 읽어도 결과에 영향이 없다.
    if args.calibrate_raw:
        GamepadInput(args.joystick).calibrate_raw()
        return 0
    if args.calibrate:
        GamepadInput(args.joystick).calibrate(out_path=args.save_map)
        return 0

    # --gamepad-map 을 생략해도 옆에 있는 캘리브레이션 결과를 쓴다.
    # 플래그를 깜빡하면 기본 매핑(z = 아날로그 축)으로 돌아가는데, 스프링 없는
    # 스로틀에 z 가 걸린 기기에서는 로봇이 멈추지 않아 위험하다.
    map_path = args.gamepad_map
    if map_path is None and args.input == "gamepad" and Path(args.save_map).exists():
        map_path = args.save_map
        print(f"[매핑] {map_path} 를 자동으로 씁니다 (--gamepad-map 으로 바꿀 수 있습니다)")
    mapping = load_gamepad_map(map_path)

    # --speed-z 는 매핑 파일의 gain_z 보다 우선한다 (현장에서 바로 조절하려고 있는 옵션).
    if args.speed_z is not None:
        mapping = {**(mapping or {}), "gain_z": args.speed_z}

    cams = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    env = SortingEnv(layout="tidy", image_size=args.image_size, cameras=cams)
    src = GamepadInput(args.joystick, mapping=mapping, speed=args.speed) \
        if args.input == "gamepad" \
        else KeyboardInput(speed=args.speed,
                           gain={"gain_z": args.speed_z} if args.speed_z is not None else None)
    if args.input == "keyboard":
        print(KeyboardInput.HELP)
    else:
        print(src.describe())
        print(GamepadInput.KEY_HELP)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    viewer = None
    if not args.no_viewer:
        import mujoco.viewer
        # key_callback 을 넘기지 않는다. 뷰어는 W/S/A/D/... 를 자기 렌더링 토글로도
        # 처리하기 때문에(Wireframe, Static Body, ...), 같은 키로 로봇까지 움직이면
        # "조작했더니 화면이 깨졌다"가 된다. 조작 키는 카메라 뷰 창에서만 받는다.
        viewer = mujoco.viewer.launch_passive(
            env.model, env.data, show_left_ui=False, show_right_ui=False)
        # 그래도 뷰어 창에 포커스를 둔 채 키를 누르면 MuJoCo 가 시각화 플래그를
        # 토글해 버린다. 시작 시점 상태를 기억해 두고 매 프레임 되돌려 원상복구한다.
        # (Wireframe/Shadow/Reflection 은 scene 쪽 플래그라 여기서 못 되돌린다)
        view_flags = list(viewer.opt.flags)
        print("\n뷰어 팁: Ctrl + 왼쪽드래그로 물체를 직접 끌어 배치를 바꿀 수 있습니다.")
        print("        배치를 바꾼 뒤 P 를 누르면 JSON 으로 저장됩니다.\n")

    # 카메라 뷰 창 — 팔에 가려 안 보이는 것을 옆에서 확인한다. 보기 전용이라
    # 녹화되는 관측(env.cameras)과는 무관하다. 실패해도 녹화는 계속한다.
    panel = None
    if not args.no_panel and not args.no_viewer:
        try:
            panel = camview.CameraPanel(
                env, cameras=[c.strip() for c in args.panel_cameras.split(",") if c.strip()],
                size=args.panel_size, cols=args.panel_cols, every=args.panel_every)
            print(f"[카메라 뷰] {args.panel_cameras}")
            print(f"           {panel.focus_hint(args.input)}\n")
        except Exception as e:
            print(f"[카메라 뷰] 창을 못 열었습니다({type(e).__name__}: {e}).")
            if args.input == "keyboard":
                print("  키보드 조작은 이 창에서 받습니다. 창이 없으면 조작할 수 없습니다.")
                print("  게임패드를 쓰거나(--input gamepad), 창이 열리지 않는 원인을 먼저 해결하세요.")
                env.close()
                return 1
            print("  게임패드 조작은 창 없이도 됩니다. 그냥 진행합니다.\n")

    if panel is None and args.input == "keyboard" and not args.no_viewer:
        # 뷰어 창의 키는 MuJoCo 자신의 렌더링 토글로 먹힌다(카메라 뷰 창이 있어야 조작 가능).
        print("키보드 조작에는 카메라 뷰 창이 필요합니다. --no-panel 을 빼고 실행하세요.")
        print("(뷰어 창의 W/S/A/D 는 로봇이 아니라 Wireframe·Static Body 같은 표시 옵션을 바꿉니다)")
        env.close()
        return 1

    reset_env(env, args.layout, args.seed)
    writer = None
    saved = 0
    dt = 1.0 / C.CONTROL_HZ
    start = "ENTER" if args.input == "keyboard" else src.record_hint()
    print(f"준비 완료. {start} 으로 녹화를 시작하세요.")

    try:
        while saved < args.episodes:
            t_start = time.perf_counter()

            # 키보드 입력은 뷰어가 아니라 이 창에서 받는다 (camview.py 첫머리 설명 참고).
            if panel is not None:
                panel.pump_input(src)
                if not panel.alive and args.input == "keyboard":
                    # 이 창이 닫히면 키보드로는 더 이상 조작할 방법이 없다.
                    print("\n카메라 뷰 창이 닫혀 조작할 수 없습니다. 종료합니다.")
                    break

            action = src.poll()

            for ev in getattr(src, "events", []):
                if ev == "record":
                    if writer is None:
                        writer = EpisodeWriter(env, task=TASK_MESS, direction="mess",
                                               cameras=cams,
                                               meta={"input": args.input, "operator_seed": args.seed,
                                                     "reverse_task": TASK_TIDY})
                        writer.record_obs()
                        print(f"● 녹화 시작 (에피소드 {saved + 1}/{args.episodes})")
                    else:
                        name = f"ep_{datetime.now():%m%d_%H%M%S}.hdf5"
                        p = writer.save(out_dir / name)
                        saved += 1
                        print(f"■ 저장: {p.name}  ({len(writer)} 프레임, {len(writer) / C.CONTROL_HZ:.1f}초)")
                        writer = None
                elif ev == "discard":
                    if writer is not None:
                        print("✕ 현재 녹화를 버렸습니다.")
                        writer = None
                elif ev == "reset":
                    writer = None
                    reset_env(env, args.layout, args.seed)
                    print("↺ 씬을 초기 상태로 되돌렸습니다.")
                elif ev == "dump_layout":
                    p = dump_layout(env, out_dir.parent / "layouts" /
                                    f"layout_{datetime.now():%m%d_%H%M%S}.json")
                    print(f"배치 저장: {p}")
            src.events = []

            env.step(action)
            if writer is not None:
                writer.record_action(action)
                writer.record_obs()

            if panel is not None:
                panel.update()

            if viewer is not None:
                if not viewer.is_running() or getattr(src, "quit", False):
                    break
                # 뷰어 창에서 실수로 누른 키가 켠 시각화 토글을 되돌린다
                # (D=Static Body 로 테이블이 사라지는 것 등).
                if list(viewer.opt.flags) != view_flags:
                    with viewer.lock():
                        for i, f in enumerate(view_flags):
                            viewer.opt.flags[i] = f
                viewer.sync()

            sleep = dt - (time.perf_counter() - t_start)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n중단됨.")
    finally:
        if panel is not None:
            panel.close()
        if viewer is not None:
            viewer.close()
        env.close()

    print(f"\n총 {saved}개 에피소드를 {out_dir} 에 저장했습니다.")
    print("다음: " + _bootstrap.cmd("scripts/build_dataset.py"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
