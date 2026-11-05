"""C2(매 시행마다 재계획, eager) 가설의 성공/실패 기준을 사후 로그로 검증한다.

전략 파일: final_project_v2/student_strategy_c2.py (v2)
검증하는 가설(문제 조사.txt 5절 C2 + 사전예측):
  "동작마다 새 이미지로 다시 계획하면(= 매 운반 실패 즉시 복구) 밀집/가림 씬에서
   성공률이 가장 높다. 단 VLM 호출·시간 비용이 가장 크다."

핵심 검증: '성공률'과 '비용'을 함께 본다(둘을 떼면 불공정).
  · 성공: scene_success / oracle_acc 개선.
  · eager 성립의 필요조건: 거의 매 운반마다 재계획이 일어났는가
    (replans 가 실행 운반 수에 근접 = 호출이 동작 수만큼 쓰임).
  · 비용: planner_calls, planner_seconds, 호출당 운반 효율.
  · 예산 공정성 경고: 호출 상한(max_calls)에 걸려 조기 종료했으면(stop_reason=planner_call_limit)
    C2가 불리하게 묶인 것 — 비교 전 max_calls 를 충분히 키워야 한다.

성공/실패 판별(이 trial 기준):
  · eager 동작 확인: replans >= (운반 수 - 1) 수준.
  · 호출 예산 미소진: stop_reason 이 planner_call_limit 이 아님(= 예산이 충분했음).
  · 파이프라인 성공/행동선택/접근성/거짓완료: 공통 지표도 함께.
  주의: "C2 > C0"는 같은 씬·시드에서 같은(혹은 명시적으로 다른) 예산으로 비교.

사용법:
  실제 trial 분석:  python scripts/check_final_project_c2.py --trial <runs/.../C2>
  배선 셀프테스트:  python scripts/check_final_project_c2.py --selftest --out <tmp_dir>
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from vlapipe.final_project_v2.hypothesis_eval import evaluate, run_diagnostic_trial

# ---- 성공/실패 기준(임계값) ----
ACTION_SELECTION_MIN = 0.90
ACCESSIBILITY_F1_MIN = 0.60
ALLOW_FALSE_DONE = 0

STRATEGY = str(Path(__file__).resolve().parents[1] / 'final_project_v2' / 'student_strategy_c2.py')


def verdict(ev: dict) -> dict:
    cost = ev['cost']
    n_transports = ev['action_selection']['n_transports']
    replans = cost['replans']
    a_sel = ev['action_selection']['correct_zone_rate']
    acc_f1 = ev['accessibility']['f1']
    calls = cost['planner_calls'] or 0
    efficiency = (n_transports / calls) if calls else None  # 호출당 운반 수(낮을수록 eager).
    criteria = {
        'eager_replanning': {
            'value': {'replans': replans, 'transports': n_transports},
            'threshold': 'replans >= transports-1',
            'pass': replans >= max(0, n_transports - 1)},
        'call_budget_not_exhausted': {
            'value': cost['stop_reason'], 'threshold': '!= planner_call_limit',
            'pass': cost['stop_reason'] != 'planner_call_limit'},
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
    cost_report = {'planner_calls': calls, 'planner_seconds': cost['planner_seconds'],
                   'transports_per_call': efficiency, 'wall_seconds': cost['wall_seconds'],
                   'stop_reason': cost['stop_reason']}
    return {'strategy': 'C2', 'trial': ev['trial'], 'scene_id': ev['scene_id'],
            'metrics': ev, 'criteria': criteria, 'cost_report': cost_report,
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
        # eager 는 호출을 많이 쓰므로 max_calls 를 넉넉히 준다.
        trial_dir, _ = run_diagnostic_trial(args.out, 'C', STRATEGY,
                                            max_calls=6, max_steps=8, action_steps=2)
        ev = evaluate(trial_dir)
        for key in ('cost', 'accessibility', 'action_selection', 'false_done'):
            assert key in ev, key
        assert ev['cost']['planner_calls'] >= 1
        print(json.dumps({'selftest': 'ok', 'note': '진단 어댑터 실행 — 배선만 확인, 성능 아님',
                          'replans': ev['cost']['replans'],
                          'planner_calls': ev['cost']['planner_calls']},
                         ensure_ascii=False, indent=2, default=float))
        return

    if not args.trial:
        ap.error('--trial 또는 --selftest 중 하나가 필요합니다')
    print(json.dumps(verdict(evaluate(args.trial)), ensure_ascii=False, indent=2, default=float))


if __name__ == '__main__':
    main()
