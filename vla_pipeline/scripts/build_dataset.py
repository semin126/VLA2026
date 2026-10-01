"""녹화된 '어지르기' 에피소드를 뒤집어 '정돈하기' 학습 데이터로 만든다.

  python scripts/build_dataset.py                       # 역재생 + 검증
  python scripts/build_dataset.py --lerobot --repo-id dshs/vla_sorting   # Colab 에서
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import mujoco

from vlapipe import config as C
from vlapipe.env import SortingEnv
from vlapipe.episode import Episode
from vlapipe.reverse import reverse_file
from vlapipe import to_lerobot

TASK_TIDY = "Tidy up the table: put every object back into its storage zone."


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(C.EPISODE_DIR))
    ap.add_argument("--out-dir", default=str(C.REVERSED_DIR))
    ap.add_argument("--task", default=TASK_TIDY)
    ap.add_argument("--keep-flagged", action="store_true",
                    help="역재생 불가 구간이 있어도 그대로 저장 (기본은 제외)")
    ap.add_argument("--lerobot", action="store_true", help="LeRobot 데이터셋까지 생성 (Colab)")
    ap.add_argument("--repo-id", default="dshs2026/vla_sorting")
    ap.add_argument("--lerobot-root", default=str(C.LEROBOT_DIR))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    raw = sorted(Path(args.raw_dir).glob("*.hdf5"))
    if not raw:
        print(f"{args.raw_dir} 에 에피소드가 없습니다. 먼저 녹화하세요:")
        print("  " + _bootstrap.cmd("scripts/teleop_record.py", "--input", "keyboard"))
        return 1

    # 물체 freejoint 의 qpos 주소 (역재생 가능성 검사에 필요)
    env = SortingEnv(layout="tidy", image_size=64)
    object_adr = {
        n: int(env.model.jnt_qposadr[
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_freejoint")])
        for n in C.OBJECT_NAMES}
    env.close()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports, written = [], []
    print(f"원본 에피소드 {len(raw)}개\n")
    for p in raw:
        dst, rep = reverse_file(p, out_dir, object_adr=object_adr, task=args.task,
                                strict=not args.keep_flagged)
        reports.append(rep)
        rv = rep.get("reversibility", {})
        if dst:
            written.append(dst)
            note = "" if rv.get("ok", True) else f"  (경고: 역재생 위험 프레임 {rv['n_flagged']}개)"
            print(f"  OK   {p.name} -> {dst.name}  {rep['frames']}프레임{note}")
        else:
            print(f"  제외 {p.name}: 역재생 불가 프레임 {rv.get('n_flagged')}개")
            print(f"       {rv.get('hint', '')}")

    print(f"\n역재생 완료: {len(written)}/{len(raw)}개 -> {out_dir}")

    val = to_lerobot.validate(written)
    print(f"\n[데이터 검증] 에피소드 {val['n_episodes']}개, 프레임 {val['n_frames']}개, "
          f"해상도 {val['image_shapes']}")
    for prob in val["problems"]:
        print(f"  ! {prob}")
    if val["ok"]:
        print("  문제 없음.")

    (out_dir / "build_report.json").write_text(
        json.dumps({"reverse": reports, "validate": val}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    if args.lerobot:
        if not written:
            print("\n변환할 에피소드가 없습니다.")
            return 1
        print(f"\nLeRobot 데이터셋 생성 -> {args.lerobot_root}")
        ds = to_lerobot.convert(written, repo_id=args.repo_id, root=Path(args.lerobot_root))
        print(f"완료: {ds}")
    else:
        print("\nLeRobot 변환은 Colab 에서:  python scripts/build_dataset.py --lerobot")
        print("(윈도우 로컬에는 lerobot 이 설치되지 않습니다 — README 4절)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
