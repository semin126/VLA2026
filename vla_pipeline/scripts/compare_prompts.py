"""여러 VLM 프롬프트 스타일의 계획 정확도를 여러 시드에 걸쳐 비교한다.

run_pipeline.py 와 달리 로봇 제어(정책 실행)는 하지 않고 "계획 단계"만 비교하므로
훨씬 빠르다. 프롬프트를 바꿔 가며 recall/precision 을 실측으로 비교할 때 쓴다.

예시:
  python scripts/compare_prompts.py
  python scripts/compare_prompts.py --styles json semin --seeds 3 7 11 19 23
"""
import _bootstrap  # noqa: F401
import argparse

from vlapipe.env import SortingEnv
from vlapipe.planner import PROMPTS, RulePlanner, VLMPlanner, plan_accuracy


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="messy", choices=["messy", "random", "tidy", "cluttered"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11, 19, 23])
    ap.add_argument("--styles", nargs="+", default=list(PROMPTS), choices=list(PROMPTS))
    ap.add_argument("--command", default="테이블을 정리하시오")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    results = {style: [] for style in args.styles}

    for style in args.styles:
        print(f"\n=== prompt_style={style} 모델 로딩 ===")
        vlm = VLMPlanner(prompt_style=style)
        for seed in args.seeds:
            env = SortingEnv(layout=args.layout, seed=seed)
            gold = RulePlanner(env).plan(args.command)
            pred = vlm.plan(env.render("agentview"), args.command)
            acc = plan_accuracy(pred, gold)
            results[style].append(acc)
            print(f"  seed {seed:>3}: 정답 {acc['correct']}/{acc['n_gold']}  "
                  f"recall={acc['recall']:.2f}  precision={acc['precision']:.2f}  "
                  f"누락={acc['missed']}  없는물체={acc['hallucinated']}  구역오류={acc['wrong_zone']}")
            env.close()
        del vlm

    print(f"\n{'=' * 66}")
    print(f"요약 (layout={args.layout}, seeds={args.seeds})")
    for style, accs in results.items():
        n = len(accs) or 1
        avg_recall = sum(a["recall"] for a in accs) / n
        avg_precision = sum(a["precision"] for a in accs) / n
        total_wrong_zone = sum(len(a["wrong_zone"]) for a in accs)
        total_hallucinated = sum(len(a["hallucinated"]) for a in accs)
        print(f"  {style:8s}  recall={avg_recall:.3f}  precision={avg_precision:.3f}  "
              f"구역오류 합계={total_wrong_zone}  없는물체 합계={total_hallucinated}")
    print("\n⚠ 씬 수가 적으면 신뢰구간이 겹칠 수 있습니다 (AGENTS.md 6절 참고).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
