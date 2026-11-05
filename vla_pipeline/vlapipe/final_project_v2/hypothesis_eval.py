"""가설 검증용 '사후(offline)' 평가 엔진 — 문제 조사.txt 4-B의 구현.

왜 사후인가: B/C는 런타임에 oracle(실제 위치·접근성·성공)을 보면 안 된다(공정성).
그러나 실행기는 모든 것을 '로그로' 남긴다. 그래서 trial이 끝난 뒤 로그만으로
"모델이 접근성/성공을 잘 판단했는가"를 정량 비교할 수 있다. 이 파일이 그 비교를 한다.

입력: run_trial 이 만든 trial 출력 폴더 1개.
  report.json  : condition, events[], initial/final outcome(oracle), summary
  scene.json   : 고정된 씬 스펙(환경 복원용)
  plan_XXX.json: 각 VLM 호출의 assessments/plan/done
  events.jsonl : plan_received / action_started / after_action / student_decision ...
  trace.npz    : 스텝별 qpos 등 -> 임의 시점 상태를 복원해 oracle 재계산

출력: evaluate() 가 조건 무관 지표 + 조건별 지표(실패탐지/비용)를 한 dict로 반환.
각 scripts/check_final_project_c{0,1,2}.py 가 이 dict에 '성공/실패 기준(임계값)'을 적용한다.

주의: 이 엔진은 '분석'만 한다. 모델 성능의 좋고 나쁨이 아니라 '가설이 맞는지'를 측정한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import accessibility as access
from .. import config as C
from .metrics import observe_outcome
from .scenes import environment, read_scene


def load_trial(trial_dir) -> dict:
    """trial 폴더에서 분석에 필요한 로그를 모두 읽는다(trace 는 지연 로딩)."""
    d = Path(trial_dir)
    report = json.loads((d / 'report.json').read_text(encoding='utf-8'))
    spec = read_scene(d / 'scene.json')
    plans = {}
    for p in sorted(d.glob('plan_*.json')):
        row = json.loads(p.read_text(encoding='utf-8'))
        plans[row['call']] = row
    return {'dir': d, 'report': report, 'spec': spec, 'plans': plans,
            'events': report.get('events', [])}


def _trace_index(trace, step):
    """event['step'] -> trace 배열 인덱스(방어적 클램프)."""
    n = len(trace['qpos'])
    return int(max(0, min(step, n - 1)))


def _plan_received(events):
    """VLM 호출 결과 이벤트들: {call, step, plan, assessments}."""
    return [e for e in events if e.get('event') == 'plan_received']


def action_selection(report) -> dict:
    """행동 선택 정확도: 실제 실행된 각 운반이 '그 물체의 정해진 구역'을 목표로 했는가.

    규칙(config): 물체 1~5 -> right_zone, 6~10 -> left_zone (number_home_zone).
    [문제 조사.txt 교정] '왼쪽 고정'이 아니라 번호별 목적지 기준으로 채점한다.
    action_started 이벤트(= 실제 집행된 운반)를 기준으로 한다.
    """
    started = [e for e in report.get('events', []) if e.get('event') == 'action_started']
    total, correct, wrong = 0, 0, []
    for e in started:
        act = e.get('action') or {}
        obj, tz = act.get('object'), act.get('target_zone')
        if obj not in C.OBJECT_BY_NAME:
            continue
        gold = C.OBJECT_BY_NAME[obj]['home_zone']
        total += 1
        if tz == gold:
            correct += 1
        else:
            wrong.append({'object': obj, 'planned': tz, 'correct_zone': gold})
    return {'n_transports': total, 'correct_zone': correct,
            'correct_zone_rate': (correct / total) if total else None, 'wrong': wrong}


def accessibility_accuracy(trial, oracle_at) -> dict:
    """VLM 접근성 판단 vs 기하 접근성(oracle). 문제 조사.txt 4-B (가).

    각 VLM 호출 시점의 상태에서 기하 접근성을 '다시 계산'해 라벨로 쓴다.
    초기 계획=초기상태, 재계획=그 시점 상태. assessments[].accessible 과 비교한다.
    양성(positive) = '접근 가능'. precision/recall/F1 산출.
    """
    tp = fp = fn = tn = 0
    n_judged = 0
    for pr in _plan_received(trial['events']):
        _, geo_access = oracle_at(pr['step'])  # {object: bool}
        for a in pr.get('assessments', []):
            obj = a.get('object')
            if obj not in geo_access:
                continue
            pred = bool(a.get('accessible'))
            gold = bool(geo_access[obj])
            n_judged += 1
            if pred and gold:
                tp += 1
            elif pred and not gold:
                fp += 1
            elif not pred and gold:
                fn += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else (0.0 if n_judged else None))
    return {'n_judged': n_judged, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision, 'recall': recall, 'f1': f1}


def false_done(trial, oracle_at) -> dict:
    """거짓 완료: 모델 done=true 시점에 oracle 상 실제로 다 정리됐는가. 4-B (다)."""
    declared = 0
    false_cnt = 0
    details = []
    for call, row in trial['plans'].items():
        resp = row.get('response') or {}
        if resp.get('done') is True:
            declared += 1
            pr = next((e for e in _plan_received(trial['events']) if e.get('call') == call), None)
            out, _ = oracle_at(pr['step'] if pr else 0)
            ok = bool(out['scene_success'])
            if not ok:
                false_cnt += 1
            details.append({'call': call, 'scene_success_at_done': ok})
    return {'declared_done': declared, 'false_done': false_cnt, 'details': details}


def failure_detection(trial, oracle_at) -> dict:
    """C1 전용: 루프의 '실패->재계획' 판단 vs oracle 실제 실패. 4-B (나).

    '계획 도중(after_action 에 큐가 남은)' 결정만 센다(큐 소진 재계획은 강제이므로 제외).
    양성(positive) = '실패로 보고 재계획'. oracle 실제실패 = 방금 옮긴 물체가 정답구역 아님.
    """
    # 이벤트를 순서대로 걸으며 after_action 시점의 '남은 큐 길이'를 복원한다.
    remaining = 0
    pending = None  # (step, object)
    pairs = []  # (step, object, remaining, decision)
    for e in trial['events']:
        name = e.get('event')
        if name == 'plan_received':
            remaining = len(e.get('plan') or [])
        elif name == 'action_started':
            remaining = max(0, remaining - 1)  # 한 동작을 큐에서 꺼내 시작.
        elif name == 'after_action':
            obj = ((e.get('feedback') or {}).get('action') or {}).get('object')
            pending = (e.get('step'), obj, remaining)
        elif name == 'student_decision' and e.get('trigger') == 'after_action' and pending:
            pairs.append((*pending, e.get('decision')))
            pending = None

    tp = fp = fn = tn = 0
    mid = 0
    for step, obj, rem, decision in pairs:
        if rem <= 0 or obj is None:
            continue  # 큐 소진(강제 재계획)은 '실패 추정'이 아니므로 제외.
        mid += 1
        out, _ = oracle_at(step)
        actual_fail = not bool(out['objects'].get(obj, {}).get('correct', False))
        pred_fail = decision == 'replan'
        if pred_fail and actual_fail:
            tp += 1
        elif pred_fail and not actual_fail:
            fp += 1
        elif not pred_fail and actual_fail:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {'n_mid_plan_decisions': mid, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision, 'recall': recall}


def replan_and_cost(trial) -> dict:
    """재계획 횟수 분해 + 비용(호출/시간/스텝). 예산 공정성·효율 비교용."""
    events = trial['events']
    report = trial['report']
    summary = report.get('summary', {})
    # 재계획 결정 분해: after_action 에서 남은 큐가 있었는지로 mid/exhausted 구분.
    remaining = 0
    mid_replan = exhausted_replan = 0
    pending_rem = None
    for e in events:
        name = e.get('event')
        if name == 'plan_received':
            remaining = len(e.get('plan') or [])
        elif name == 'action_started':
            remaining = max(0, remaining - 1)
        elif name == 'after_action':
            pending_rem = remaining
        elif name == 'student_decision' and e.get('trigger') == 'after_action' and pending_rem is not None:
            if e.get('decision') == 'replan':
                if pending_rem > 0:
                    mid_replan += 1
                else:
                    exhausted_replan += 1
            pending_rem = None
    calls = summary.get('planner_calls', 0)
    return {'planner_calls': calls, 'replans': max(0, calls - 1),
            'mid_plan_replans': mid_replan, 'queue_exhausted_replans': exhausted_replan,
            'planner_seconds': summary.get('planner_seconds'),
            'steps': summary.get('steps'), 'wall_seconds': summary.get('total_wall_seconds'),
            'stop_reason': report.get('stop_reason')}


def evaluate(trial_dir) -> dict:
    """한 trial 폴더를 열어 모든 가설 지표를 계산해 반환한다(환경을 1회만 연다)."""
    import mujoco
    trial = load_trial(trial_dir)
    report = trial['report']
    trace = np.load(trial['dir'] / 'trace.npz')

    with environment(trial['spec']) as env:
        def oracle_at(step):
            i = _trace_index(trace, step)
            env.data.qpos[:] = trace['qpos'][i]
            env.data.qvel[:] = trace['qvel'][i]
            if 'actuator_ctrl' in trace:
                env.data.ctrl[:] = trace['actuator_ctrl'][i]
            if 'target_pos' in trace:
                env.ctrl.target_pos = trace['target_pos'][i].copy()
            if 'target_quat' in trace:
                env.ctrl.target_quat = trace['target_quat'][i].copy()
            mujoco.mj_forward(env.model, env.data)
            outcome = observe_outcome(env)
            pos, yaw = env.object_xy_yaw()
            geo = {n: bool(access.is_accessible(n, pos, yaw)[0]) for n in C.OBJECT_NAMES}
            return outcome, geo

        acc_geo = accessibility_accuracy(trial, oracle_at)
        fdone = false_done(trial, oracle_at)
        fdet = failure_detection(trial, oracle_at)

    initial = report.get('initial', {})
    final = report.get('final', {})
    return {
        'trial': str(trial['dir']),
        'condition': report.get('condition'),
        'diagnostic': report.get('diagnostic'),
        'scene_id': report.get('scene_id'),
        'family': report.get('family'),
        'strategy': (report.get('strategy') or {}).get('path'),
        # H1 1차 지표(파이프라인 결과, oracle).
        'scene_success': bool(final.get('scene_success', False)),
        'oracle_acc': final.get('acc'),
        'oracle_macro_f1': final.get('macro_f1'),
        'initial_acc': initial.get('acc'),
        'acc_improvement': (None if final.get('acc') is None or initial.get('acc') is None
                            else final.get('acc') - initial.get('acc')),
        # H2 지표: VLM 접근성 판단 정확도.
        'accessibility': acc_geo,
        # 행동 선택 정확도(번호별 목적지).
        'action_selection': action_selection(report),
        # 거짓 완료.
        'false_done': fdone,
        # C1 전용: 실패 탐지 정확도.
        'failure_detection': fdet,
        # 비용/재계획 분해(예산 공정성·효율).
        'cost': replan_and_cost(trial),
    }


# --------------------------------------------------------------------------- #
# 셀프테스트용 진단 어댑터(모델 없이 실행기를 돌려 분석기 배선을 확인).
# check_final_project_v2.py 의 IdlePolicy/Plans 와 같은 성격이다: 계약/배선만 본다.
# --------------------------------------------------------------------------- #
class IdlePolicy:
    """가짜 VLA. 아무 것도 옮기지 않는다(그리퍼 open)."""
    def reset(self):
        pass

    def select_action(self, observation):
        return np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


class ScriptedPlanner:
    """가짜 VLM. 고정된 plan/assessments/done 를 돌려준다(JSON)."""
    def __init__(self, plan=None, assessments=None, done=False):
        self.calls = []
        self.plan = plan if plan is not None else [
            {'object': 'object_1', 'target_zone': 'right_zone'},
            {'object': 'object_2', 'target_zone': 'right_zone'}]
        self.assessments = assessments or []
        self.done = done

    def generate(self, image, prompt, deadline):
        self.calls.append(prompt)
        return json.dumps({'assessments': self.assessments, 'plan': self.plan, 'done': self.done})


def run_diagnostic_trial(out_dir, condition, strategy_path, *, seed=100,
                         planner=None, max_calls=4, seconds=3, max_steps=6,
                         action_steps=2, observer_steps=1):
    """모델 없이 condition 트라이얼 1개를 만들어 분석 대상 폴더를 생성한다.

    반환: (trial_dir, report). 분석기(evaluate)를 이 폴더에 적용해 배선을 확인한다.
    """
    from .runner import TrialConfig, run_trial
    from .scenes import freeze, template
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = freeze(template('mixed', seed))
    scene_path = out / 'scene.json'
    from .common import dump
    dump(scene_path, spec)
    trial_dir = out / condition
    cfg = TrialConfig(condition=condition, seconds=seconds, max_steps=max_steps,
                      action_steps=action_steps, observer_steps=observer_steps,
                      max_calls=max_calls, strategy=strategy_path, checkpoint='diagnostic',
                      video=False)
    report = run_trial(scene_path, trial_dir, cfg,
                       planner_adapter=planner or ScriptedPlanner(), policy_adapter=IdlePolicy())
    return trial_dir, report
