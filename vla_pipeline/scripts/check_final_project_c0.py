"""C0(전체계획 후 재계획) 가설의 성공/실패 기준을 사후 로그로 검증한다.

전략 파일: final_project_v2/student_strategy.py
검증하는 가설(문제 조사.txt 3절):
  H1 (계층 > 단일): VLM이 자연어를 단위 운반으로 분해하면 정리 성공률/행동선택 정확도가
     A(단일 VLA)·B(1회 계획)보다 높다.  -> 1차 지표: scene_success, oracle_acc, 행동선택 정확도.
  H2 (접근성 추론 기여): VLM이 가림/접근성을 제대로 읽는가. -> accessibility F1.
  H3 (재계획 기여, 제한적): 계획 소진 후 재관측·재계획이 B보다 낫다. 단, 계획 '도중' 실패는
     복구 못 하므로 밀집 씬에서 이득이 제한적.  -> 비교는 c1/c2 결과와 함께 본다.

성공/실패 판별(이 trial 기준 — 교차조건 '유의성'은 씬·시드를 많이 모아 별도 집계):
  · 파이프라인 성공: scene_success == True, 또는 oracle_acc 가 초기보다 개선.
  · 행동선택 성공: 실행된 운반이 번호별 정답 구역(1~5 오른쪽, 6~10 왼쪽)을 목표.
  · 접근성 판단 성공: VLM accessible 판단의 F1 이 기준 이상.
  · 거짓 완료 없음: 모델 done=true 시점에 oracle 상 실제 정리 완료.

사용법:
  실제 trial 분석:  python scripts/check_final_project_c0.py --trial <runs/.../C>
  배선 셀프테스트:  python scripts/check_final_project_c0.py --selftest --out <tmp_dir>
  (셀프테스트는 모델 없이 진단 어댑터로 돌려 '분석기가 도는지'만 본다. 성능 판정 아님.)
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from vlapipe.final_project_v2.hypothesis_eval import evaluate, run_diagnostic_trial

# ---- 성공/실패 기준(임계값). 연구 진행에 맞춰 예비시험으로 조정하세요. ----
ACTION_SELECTION_MIN = 0.90   # 운반 중 정답 구역을 목표로 한 비율의 하한.
ACCESSIBILITY_F1_MIN = 0.60   # VLM 접근성 판단 F1 하한.
ALLOW_FALSE_DONE = 0          # 허용하는 거짓 완료 횟수.

STRATEGY = str(Path(__file__).resolve().parents[1] / 'final_project_v2' / 'student_strategy.py')


def verdict(ev: dict) -> dict:
    """evaluate() 결과에 C0 기준을 적용해 기준별 pass/fail 과 근거값을 만든다."""
    a_sel = ev['action_selection']['correct_zone_rate']
    acc_f1 = ev['accessibility']['f1']
    fd = ev['false_done']['false_done']
    criteria = {
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
            'value': fd, 'threshold': ALLOW_FALSE_DONE, 'pass': fd <= ALLOW_FALSE_DONE},
    }
    return {'strategy': 'C0', 'trial': ev['trial'], 'scene_id': ev['scene_id'],
            'metrics': ev, 'criteria': criteria,
            'all_pass': all(c['pass'] for c in criteria.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trial', type=Path, help='분석할 trial 출력 폴더')
    ap.add_argument('--selftest', action='store_true', help='모델 없이 배선 확인')
    ap.add_argument('--out', type=Path, help='셀프테스트 작업 폴더')
    args = ap.parse_args()

    if args.selftest:
        if not args.out:
            ap.error('--selftest 에는 --out 이 필요합니다')
        trial_dir, _ = run_diagnostic_trial(args.out, 'C', STRATEGY)
        ev = evaluate(trial_dir)
        # 분석기 배선 확인: 핵심 지표 키가 모두 산출되는가(성능 판정 아님).
        for key in ('scene_success', 'accessibility', 'action_selection', 'false_done', 'cost'):
            assert key in ev, key
        assert ev['accessibility']['n_judged'] >= 0 and ev['cost']['planner_calls'] >= 1
        print(json.dumps({'selftest': 'ok', 'note': '진단 어댑터 실행 — 배선만 확인, 성능 아님',
                          'computed_keys': sorted(ev.keys())}, ensure_ascii=False, indent=2))
        return

    if not args.trial:
        ap.error('--trial 또는 --selftest 중 하나가 필요합니다')
    out = verdict(evaluate(args.trial))
    print(json.dumps(out, ensure_ascii=False, indent=2, default=float))


if __name__ == '__main__':
    main()
