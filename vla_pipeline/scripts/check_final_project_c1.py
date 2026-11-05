"""C1(실패검사 후 재계획, reactive) 가설의 성공/실패 기준을 사후 로그로 검증한다.

전략 파일: final_project_v2/student_strategy_c1.py (v1)
검증하는 가설(문제 조사.txt 5절 C1):
  "매 운반 직후 실패를 '추정'해 실패일 때만 재계획하면, 전체계획 후 재계획(C0)보다
   미정리 누적이 줄어든다. 단 런타임엔 oracle이 없어 실패 추정 정확도에 성능이 종속된다."

핵심 검증(4-B 나): 루프의 '실패->재계획' 판단이 실제 실패와 얼마나 맞는가.
  · 계획 '도중'(after_action 에 큐가 남은) 결정만 센다(큐 소진 재계획은 강제이므로 제외).
  · oracle 실제실패 = 방금 옮긴 물체가 그 시점에 정답 구역에 없음(trace 로 재계산).
  · 지표: 실패탐지 precision / recall.

성공/실패 판별(이 trial 기준):
  · 실패탐지 유효: precision·recall 이 기준 이상(추정이 쓸 만함).  -> 이게 C1의 전제.
  · C1 가치 성립의 '필요조건': mid_plan_replans > 0 (실제로 도중 재계획을 했는가).
  · 파이프라인 성공/행동선택/접근성/거짓완료: C0 검증과 동일 지표도 함께 본다.
  주의: "C1 > C0"는 같은 씬·시드에서 두 조건을 돌려 비교해야 한다(이 스크립트는 한 trial 측정).

사용법:
  실제 trial 분석:  python scripts/check_final_project_c1.py --trial <runs/.../C1>
  배선 셀프테스트:  python scripts/check_final_project_c1.py --selftest --out <tmp_dir>
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from vlapipe.final_project_v2.hypothesis_eval import evaluate, run_diagnostic_trial

# ---- 성공/실패 기준(임계값) ----
FAILURE_DETECTION_PRECISION_MIN = 0.60  # '실패라 보고 재계획'이 실제 실패였던 비율.
FAILURE_DETECTION_RECALL_MIN = 0.50     # 실제 실패 중 재계획으로 잡아낸 비율.
ACTION_SELECTION_MIN = 0.90
ACCESSIBILITY_F1_MIN = 0.60
ALLOW_FALSE_DONE = 0

STRATEGY = str(Path(__file__).resolve().parents[1] / 'final_project_v2' / 'student_strategy_c1.py')


def verdict(ev: dict) -> dict:
    fd = ev['failure_detection']
    prec, rec = fd['precision'], fd['recall']
    a_sel = ev['action_selection']['correct_zone_rate']
    acc_f1 = ev['accessibility']['f1']
    criteria = {
        'reactive_replans_happened': {
            'value': ev['cost']['mid_plan_replans'], 'threshold': '>0',
            'pass': ev['cost']['mid_plan_replans'] > 0},
        'failure_detection_precision': {
            'value': prec, 'threshold': FAILURE_DETECTION_PRECISION_MIN,
            'pass': (prec is not None and prec >= FAILURE_DETECTION_PRECISION_MIN)},
        'failure_detection_recall': {
            'value': rec, 'threshold': FAILURE_DETECTION_RECALL_MIN,
            'pass': (rec is not None and rec >= FAILURE_DETECTION_RECALL_MIN)},
        'pipeline_success': {
            'value': {'scene_success': ev['scene_success'], 'acc_improvement': ev['acc_improvement']},
            'pass': bool(ev['scene_success'] or (ev['acc_improvement'] or 0) > 0)},
        'action_selection': {
            'value': a_sel, 'threshold': ACTION_SELECTION_MIN,
            'pass': (a_sel is not None and a_sel >= ACTION_SELECTION_MIN)},
        'accessibility_judgment': {
            'value': acc_f1, 'threshold': ACCESSIBILITY_F1_MIN,
            'pass': (acc_f1 is not None and acc_f1 >= ACCESSIBILITY_F1_MIN)},
        'no_false_done': {
            'value': ev['false_done']['false_done'], 'threshold': ALLOW_FALSE_DONE,
            'pass': ev['false_done']['false_done'] <= ALLOW_FALSE_DONE},
    }
    note = ('failure_detection 가 n=0 이면 도중 재계획 표본이 없다는 뜻 — 밀집 씬 또는 '
            '실패가 실제로 나는 체크포인트에서 측정해야 유의미.')
    return {'strategy': 'C1', 'trial': ev['trial'], 'scene_id': ev['scene_id'],
            'metrics': ev, 'criteria': criteria, 'note': note,
            'all_pass': all(c['pass'] for c in criteria.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trial', type=Path)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()

    if args.selftest:
        if not args.out:
            ap.error('--selftest 에는 --out 이 필요합니다')
        # observer_steps/action_steps 를 짧게 주어 after_action 결정이 여러 번 나도록 한다.
        trial_dir, _ = run_diagnostic_trial(args.out, 'C', STRATEGY,
                                            max_calls=4, max_steps=8, action_steps=2)
        ev = evaluate(trial_dir)
        for key in ('failure_detection', 'cost', 'accessibility', 'action_selection'):
            assert key in ev, key
        assert ev['cost']['planner_calls'] >= 1
        print(json.dumps({'selftest': 'ok', 'note': '진단 어댑터 실행 — 배선만 확인, 성능 아님',
                          'failure_detection': ev['failure_detection'],
                          'mid_plan_replans': ev['cost']['mid_plan_replans']},
                         ensure_ascii=False, indent=2, default=float))
        return

    if not args.trial:
        ap.error('--trial 또는 --selftest 중 하나가 필요합니다')
    print(json.dumps(verdict(evaluate(args.trial)), ensure_ascii=False, indent=2, default=float))


if __name__ == '__main__':
    main()
