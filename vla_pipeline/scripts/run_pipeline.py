"""VLM -> VLA 엔드투엔드 실행.

  이미지 + 자연어 명령
      -> [1단계 VLM]  순차 지시문 목록
      -> [2단계 VLA]  각 지시문을 관절 제어로 실행

예시:
  python scripts/run_pipeline.py --layout messy --seed 3
  python scripts/run_pipeline.py --planner vlm            # SmolVLM2 로 계획
  python scripts/run_pipeline.py --policy smolvla --checkpoint outputs/ckpt/last
  python scripts/run_pipeline.py --objects number         # 색/모양이 아니라 숫자로 물체 구분
  python scripts/run_pipeline.py --objects number --planner vlm
"""
import _bootstrap  # noqa: F401
import argparse
import json
import time
from pathlib import Path

import numpy as np

from vlapipe import config as C
from vlapipe import numbered           # 임포트만으로 숫자용 프롬프트가 등록된다
from vlapipe.env import SortingEnv
from vlapipe.planner import PROMPTS, RulePlanner, VLMPlanner, plan_accuracy
from vlapipe.policy import ScriptedSortingPolicy


def parse_args():
    ap = argparse.ArgumentParser()
    # cluttered = 가림이 있는 배치. 핵심 가설(가림 씬에서 계층 구조가 유리한가)의 실험 조건이다.
    ap.add_argument("--layout", default="messy",
                    choices=["messy", "random", "tidy", "cluttered"])
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--command", default="테이블을 정리하시오")
    # 물체를 무엇으로 구분하는가 = 이 실험의 변수.
    #   color  : 색 + 모양 (기준 조건)
    #   number : 똑같은 흰 정육면체 10개, 면에 찍힌 1~10 숫자만 다르다
    ap.add_argument("--objects", default="color", choices=["color", "number"])
    ap.add_argument("--planner", default="rule", choices=["rule", "vlm"])
    ap.add_argument("--prompt-style", default=None, choices=sorted(PROMPTS),
                    help="생략하면 물체 세트에 맞는 기본값(color->json, number->number)")
    ap.add_argument("--vlm-image-size", type=int, default=0,
                    help="VLM 에 넣는 agentview 이미지 한 변 [px]. "
                         "0 이면 color=--image-size, number=512 (숫자가 너무 작게 찍히지 않게)")
    ap.add_argument("--policy", default="scripted", choices=["scripted", "smolvla"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--passes", type=int, default=2, help="실패한 물체를 다시 시도할 횟수")
    ap.add_argument("--max-steps", type=int, default=500, help="서브태스크당 최대 제어 스텝")
    ap.add_argument("--image-size", type=int, default=C.IMAGE_SIZE)
    ap.add_argument("--video-camera", default="frontview")
    ap.add_argument("--out", default=str(C.OUTPUT_DIR))
    ap.add_argument("--no-video", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    # 물체 세트 교체는 env 를 만들기 전에 해야 한다 (env 가 물체 이름으로 조인트를 잡는다).
    if args.objects == "number":
        numbered.activate()
    tag = "" if args.objects == "color" else f"_{args.objects}"
    out = Path(args.out) / f"run_{args.layout}{args.seed}{tag}_{args.planner}_{args.policy}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    env = SortingEnv(layout=args.layout, seed=args.seed, image_size=args.image_size)
    rule = RulePlanner(env)
    style = args.prompt_style or ("number" if args.objects == "number" else "json")
    # 숫자는 윗면 몇 px 안에 들어가므로 VLM 입력은 관측 해상도보다 크게 렌더한다.
    vlm_px = args.vlm_image_size or (512 if args.objects == "number" else args.image_size)
    vlm = VLMPlanner(prompt_style=style) if args.planner == "vlm" else None
    if vlm is not None:
        print(f"[VLM] 프롬프트 '{style}', 입력 이미지 {vlm_px}x{vlm_px} px "
              f"(물체 윗면 약 {numbered.face_pixels(vlm_px):.0f} px, 계산값)")

    if args.policy == "smolvla":
        from vlapipe.policy import SmolVLAPolicy
        if not args.checkpoint:
            raise SystemExit("--policy smolvla 를 쓰려면 --checkpoint 가 필요합니다.")
        policy = SmolVLAPolicy(args.checkpoint)
    else:
        policy = ScriptedSortingPolicy(env)

    import imageio.v2 as imageio
    frames = []
    from PIL import Image
    Image.fromarray(env.render(args.video_camera, 480, 640)).save(out / "before.png")

    report = {"args": vars(args), "passes": [], "planner_eval": None}
    total_steps = 0

    for p in range(args.passes):
        if env.is_tidy():
            break
        gold = rule.plan(args.command)
        if vlm is not None:
            plan = vlm.plan(env.render("agentview", vlm_px, vlm_px), args.command)
            if p == 0:
                report["planner_eval"] = plan_accuracy(plan, gold)
                report["vlm_plan"] = plan
                report["rule_plan"] = gold
            if not plan:
                print("VLM 이 계획을 못 만들어 규칙 플래너로 대체합니다.")
                plan = gold
        else:
            plan = gold

        print(f"\n=== pass {p + 1}: {len(plan)} 단계 ===")
        pass_log = []
        for st in plan:
            print(f"  [{st['step']}] {st['instruction']}")
            if hasattr(policy, "set_subtask"):
                policy.set_subtask(st)
            obs, n = env.get_obs(), 0
            while n < args.max_steps:
                if hasattr(policy, "set_subtask"):
                    a = policy.select_action(obs)
                    done = policy.done
                else:
                    a = policy.select_action({
                        "observation.image": obs["agentview_image"],
                        "observation.state": SortingEnv.state_vector(obs),
                        "task": st["instruction"]})
                    # 학습된 정책은 '끝났다' 신호를 주지 않는다.
                    # 목표 구역에 들어갔고 그리퍼가 열려 있으면(= 놓았으면) 성공으로 보고 넘어간다.
                    done = (env.zone_of(st["object"]) == st["target_zone"]
                            and float(a[6]) < 0)
                obs, _, _, _ = env.step(a)
                n += 1
                total_steps += 1
                if not args.no_video and total_steps % 3 == 0:
                    frames.append(env.render(args.video_camera, 300, 400))
                if done:
                    break
            good = env.zone_of(st["object"]) == st["target_zone"]
            fail = getattr(policy, "failure", None)
            print(f"        -> {env.zone_of(st['object'])} {'성공' if good else '실패'}"
                  f"{'' if not fail else ' (' + fail + ')'}  {n}스텝")
            pass_log.append({**st, "steps": n, "success": bool(good), "failure": fail,
                             "final_zone": env.zone_of(st["object"])})
        report["passes"].append(pass_log)

    Image.fromarray(env.render(args.video_camera, 480, 640)).save(out / "after.png")
    Image.fromarray(env.render("agentview", 480, 480)).save(out / "after_agentview.png")
    if frames and not args.no_video:
        imageio.mimsave(out / "rollout.gif", frames[::2], fps=14, loop=0)

    subs = [s for pl in report["passes"] for s in pl]
    report["summary"] = {
        "is_tidy": env.is_tidy(),
        "misplaced": env.misplaced_objects(),
        "subtasks": len(subs),
        "subtask_success": sum(s["success"] for s in subs),
        "total_steps": total_steps,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    s = report["summary"]
    print(f"\n{'=' * 58}")
    print(f"정리 완료 여부 : {s['is_tidy']}   남은 물체: {s['misplaced']}")
    print(f"서브태스크     : {s['subtask_success']}/{s['subtasks']} 성공")
    print(f"총 {s['total_steps']} 스텝, {s['wall_seconds']}초")
    if report["planner_eval"]:
        pe = report["planner_eval"]
        print(f"VLM 계획 정확도: 정답 {pe['correct']}/{pe['n_gold']}  "
              f"누락 {pe['missed']}  없는물체 {pe['hallucinated']}  구역오류 {pe['wrong_zone']}")
    print(f"결과 저장      : {out}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
