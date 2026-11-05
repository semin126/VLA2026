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
from datetime import datetime
import uuid
from pathlib import Path

import numpy as np

from vlapipe import config as C
from vlapipe import numbered           # 임포트만으로 숫자용 프롬프트가 등록된다
from vlapipe.env import SortingEnv
from vlapipe.planner import PROMPTS, VLM_MODEL_ID, RulePlanner, VLMPlanner, parse_plan, plan_accuracy
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
    ap.add_argument("--number-count", type=int, default=10)
    ap.add_argument("--planner", default="rule", choices=["rule", "vlm"])
    ap.add_argument("--vlm-model", default=VLM_MODEL_ID, help="Hub ID 또는 파인튜닝한 로컬 VLM 폴더")
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
    ap.add_argument("--allow-oracle-fallback", action="store_true",
                    help="빈 VLM 계획에 정답 계획 사용 (보조 실행이며 순수 VLM 성능에서 제외)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    # 물체 세트 교체는 env 를 만들기 전에 해야 한다 (env 가 물체 이름으로 조인트를 잡는다).
    if args.objects == "number":
        numbered.activate(count=args.number_count)
    tag = "" if args.objects == "color" else f"_{args.objects}"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out = Path(args.out) / f"run_{args.layout}{args.seed}{tag}_{args.planner}_{args.policy}_{run_id}"
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()

    env = SortingEnv(layout=args.layout, seed=args.seed, image_size=args.image_size)
    rule = RulePlanner(env)
    style = args.prompt_style or ("number" if args.objects == "number" else "json")
    # 숫자는 윗면 몇 px 안에 들어가므로 VLM 입력은 관측 해상도보다 크게 렌더한다.
    vlm_px = args.vlm_image_size or (512 if args.objects == "number" else args.image_size)
    vlm = VLMPlanner(model_id=args.vlm_model, prompt_style=style) if args.planner == "vlm" else None
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

    def unplaced_objects():
        return [name for name in C.OBJECT_NAMES
                if abs(float(env.object_pos(name)[2]) - C.TABLE_TOP_Z
                       - C.half_size(C.OBJECT_BY_NAME[name])[2]) > 0.02]

    def scene_completed():
        return (env.is_tidy() and not unplaced_objects()
                and (args.policy == "scripted" or float(np.min(env.data.qpos[env._finger_adr])) > 0.035))

    import imageio.v2 as imageio
    frames = []
    from PIL import Image
    Image.fromarray(env.render(args.video_camera, 480, 640)).save(out / "before.png")

    report = {"protocol_version": 2, "args": vars(args), "passes": [], "planner_eval": None,
              "oracle_fallback_used": False, "execution_errors": [], "planner_calls": [],
              "rule_destination_assignment": args.planner == "vlm" and style == "number_read"}
    report["initial_misplaced"] = env.misplaced_objects()
    total_steps = 0

    for p in range(args.passes):
        if scene_completed():
            break
        gold = rule.plan(args.command)
        if vlm is not None:
            start = time.perf_counter()
            raw = vlm.raw_response(env.render("agentview", vlm_px, vlm_px), args.command)
            plan = parse_plan(raw, allow_default_zone=style == "number_read")
            report["planner_calls"].append({"pass": p + 1, "raw_response": raw,
                                            "parsed_plan": plan,
                                            "seconds": time.perf_counter() - start})
            if p == 0:
                report["planner_eval"] = plan_accuracy(plan, gold)
                report["vlm_plan"] = plan
                report["rule_plan"] = gold
            if not plan:
                if args.allow_oracle_fallback:
                    print("[보조 실행] 빈 VLM 계획을 정답으로 대체합니다. 순수 VLM 점수가 아닙니다.")
                    report["oracle_fallback_used"] = True
                    plan = gold
                else:
                    print("VLM 계획이 비어 있습니다. 정답으로 대체하지 않고 실패를 기록합니다.")
        else:
            plan = gold

        print(f"\n=== pass {p + 1}: {len(plan)} 단계 ===")
        pass_log = []
        for st in plan:
            print(f"  [{st['step']}] {st['instruction']}")
            if st["object"] not in C.OBJECT_BY_NAME or st["target_zone"] not in C.ZONES:
                report["execution_errors"].append({"pass": p + 1, "subtask": st,
                                                    "error": "unknown object or zone"})
                continue
            if hasattr(policy, "reset"):
                policy.reset()
            if hasattr(policy, "set_subtask"):
                policy.set_subtask(st)
            start_pos = env.object_pos(st["object"]).copy()
            already_satisfied = env.zone_of(st["object"]) == st["target_zone"]
            obs, n, stable = env.get_obs(), 0, 0
            while n < args.max_steps:
                if hasattr(policy, "set_subtask"):
                    a = policy.select_action(obs)
                    done = policy.done
                else:
                    a = policy.select_action({**obs, "task": st["instruction"]})
                    # 학습된 정책은 '끝났다' 신호를 주지 않는다.
                    # 목표 구역에 들어갔고 그리퍼가 열려 있으면(= 놓았으면) 성공으로 보고 넘어간다.
                    done = False
                obs, _, _, _ = env.step(a)
                if not hasattr(policy, "set_subtask"):
                    released = (float(np.min(obs["robot0_gripper_qpos"])) > 0.035
                                and env.object_pos(st["object"])[2] < C.TABLE_TOP_Z
                                + C.half_size(C.OBJECT_BY_NAME[st["object"]])[2] + 0.02)
                    stable = stable + 1 if (env.zone_of(st["object"]) == st["target_zone"]
                                           and released) else 0
                    done = stable >= 5
                n += 1
                total_steps += 1
                if not args.no_video and total_steps % 3 == 0:
                    frames.append(env.render(args.video_camera, 300, 400))
                if done:
                    break
            # Clearing/pushing is an intermediate action, not a home-zone placement.
            displacement = float(np.linalg.norm(env.object_pos(st["object"])[:2] - start_pos[:2]))
            criterion = "new_zone_membership" if st.get("action", "transport") == "transport" else "moved_at_least_2cm"
            good = ((not already_satisfied and env.zone_of(st["object"]) == st["target_zone"])
                    if criterion == "new_zone_membership"
                    else displacement >= 0.02)
            if not hasattr(policy, "set_subtask"):
                good = good and stable >= 5
            fail = getattr(policy, "failure", None)
            print(f"        -> {env.zone_of(st['object'])} {'성공' if good else '실패'}"
                  f"{'' if not fail else ' (' + fail + ')'}  {n}스텝")
            pass_log.append({**st, "steps": n, "success": bool(good), "failure": fail,
                             "success_criterion": criterion, "displacement_m": displacement,
                             "already_satisfied_at_start": already_satisfied,
                             "final_zone": env.zone_of(st["object"])})
        report["passes"].append(pass_log)

    Image.fromarray(env.render(args.video_camera, 480, 640)).save(out / "after.png")
    Image.fromarray(env.render("agentview", 480, 480)).save(out / "after_agentview.png")
    if frames and not args.no_video:
        # Recorded every 3 controls, exported every other recorded image.
        # Keep approximate simulation speed rather than an arbitrary fast GIF.
        exported = frames[::2]
        imageio.mimsave(out / "rollout.gif", exported, duration=1000 * 6 / C.CONTROL_HZ, loop=0)
        report["video"] = {"camera": args.video_camera, "frames_before_gif_dedup": len(exported),
                           "control_hz": C.CONTROL_HZ, "controls_per_frame": 6,
                           "frame_duration_seconds": 6 / C.CONTROL_HZ,
                           "note": "Approximate simulation-time preview; use control logs for exact duration."}

    subs = [s for pl in report["passes"] for s in pl]
    report["summary"] = {
        "is_tidy": scene_completed(),
        "misplaced": env.misplaced_objects(),
        "unplaced": unplaced_objects(),
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
