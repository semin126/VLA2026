"""피드백 루프 전략 v2 (C0 / C1 / C2 공용 틀) — 조교 지원본, 2026-11-06.

학생 원안(Semin fe839ff)의 세 가지 재계획 시점은 그대로 둔다.
  C0: 계획을 끝까지 실행한 뒤에만 재계획한다.
  C1: 운반 직후 실패가 의심될 때만 재계획한다.
  C2: 운반이 끝날 때마다 재계획한다.
세 파일(c0_v2, c1_v2, c2_v2)은 아래 MODE 한 줄만 다르다. 나머지는 글자까지 같다.

원안에서 바뀐 점(자세한 이유와 생각해 볼 질문은 STUDENT_COMMENTS.md):
  1. C1의 실패 의심을 '운반 전후 이미지 비교'로 판정한다. 이미지가 없으면 원안의
     메타데이터 신호를 쓴다(방금 끝난 운반을 '이전 운반'으로 세던 오류는 고쳤다).
  2. 재계획 지침 문단은 VLM이 학습한 문장(C0 원안) 하나로 통일했다.
  3. 호출 예산이 없으면 멈추지 않고 남은 계획을 이어 실행한다(원안 C2는 멈췄다).
  4. 남은 시간이 VLM 호출 1회보다 짧으면 재계획하지 않는다.
  5. VLM이 done=true라고 해도 이미지에 중앙 구역 큐브가 보이면 한 번 더 계획한다.
  6. 다음 운반이 시도 한도(3회)에 걸릴 물체면 그대로 가지 않고 재계획한다.
  7. student_feedback은 기본으로 보내지 않는다(VLM 학습 데이터에 없는 입력이기 때문).

공정성: 이 파일이 쓰는 것은 실행기가 주는 문맥과 카메라 이미지뿐이다.
물체의 실제 좌표, 접근성 정답, 성공 여부는 쓰지 않는다.
"""

import time

import numpy as np

from vlapipe.final_project_v2.prompting import build_strategy_prompt

try:
    import cv2
except ImportError:  # 이미지 판정 없이 메타데이터 신호만 쓴다.
    cv2 = None

MODE = 'C1'  # 'C0' | 'C1' | 'C2'  <- 세 파일의 유일한 차이

# ---- 실험 스위치 (세 전략에 같은 값을 써야 비교가 공정하다) ----
USE_IMAGE_CHECK = True          # C1: 운반 전후 이미지로 실패를 의심한다.
UNKNOWN_IS_SUSPECT = True       # 이미지로 판정할 수 없으면 실패로 의심한다(재현율 우선).
VERIFY_DONE_WITH_IMAGE = True   # done=true를 이미지로 한 번 확인한다.
MAX_DONE_RECHECKS = 1
ATTEMPT_GUARD = True            # 시도 한도에 걸릴 운반을 피한다.
MAX_ATTEMPTS = 3                # 실행기 TrialConfig.max_attempts 기본값과 같아야 한다.
SEND_STUDENT_FEEDBACK = False   # True면 의심 물체·시도 횟수를 다음 VLM 프롬프트에 넣는다.
DEFAULT_CALL_SECONDS = 25.0     # VLM 호출 1회 예상 시간. 첫 호출 뒤에는 실제 측정값으로 바뀐다.
REPLAN_MARGIN_SECONDS = 5.0     # 재계획 뒤 최소한 운반을 시작할 여유.

# VLM 학습 데이터(r2)에 쓰인 문장과 글자까지 같아야 한다. 바꾸면 VLM도 다시 학습해야 한다.
REPLAN_GUIDANCE = (
    "\nRe-planning step (hierarchical VLM->VLA feedback loop):\n"
    "- Earlier transports were already executed. Look at the NEW image FIRST.\n"
    "- The execution context includes your history and feedback.student_feedback\n"
    "  describing what was already attempted; use it, do not rely on memory.\n"
    "- A fixed action horizon ending does NOT prove the object was placed; verify\n"
    "  each object's current zone from the image before planning.\n"
    "- Plan transports ONLY for objects that are not yet in their correct zone.\n"
    "  Do not re-transport an object the image shows already correctly sorted.\n"
    "- If a target is still blocked, transport the accessible blocker to its own\n"
    "  home storage zone first, then the freed target.\n"
    "- Keep the plan short and in execution order. If the image shows everything\n"
    "  sorted, return done=true with an empty plan.\n"
)

# 전략 모듈은 trial마다 새로 로드되므로 이 상태는 한 trial 안에서만 유지된다.
_state = {'prev_frame': None, 'tick_frame': None, 'prompt_time': None,
          'call_seconds': DEFAULT_CALL_SECONDS, 'done_rechecks': 0, 'log': []}


def build_prompt(public_task: dict, context: dict) -> str:
    _state['prompt_time'] = time.perf_counter()
    base = build_strategy_prompt(public_task, context)
    if context.get('event') == 'initial':
        return base  # 초기 호출은 B와 글자까지 같아야 한다(공정 비교).
    return base + REPLAN_GUIDANCE


# --------------------------------------------------------------------------- #
# 이미지 판정: agentview 512px 이미지에서 구역 바닥(채도 높음)과 큐브·로봇(채도 낮음)을 나눈다.
# --------------------------------------------------------------------------- #
ZONE_X = {'left_zone': (50, 187), 'center_zone': (187, 324), 'right_zone': (324, 462)}
ZONE_Y = (163, 420)
SAT_MAX = 60        # 구역 바닥의 채도는 95~115, 큐브·로봇은 45 이하(실측).
CUBE_AREA = 980.0   # 큐브 1개가 가리는 픽셀 수(실측 중앙값 993, 중앙 구역에서는 약 0.8배).


def masks(rgb):
    """(큐브, 로봇 팔, 구역 바닥) 마스크. 구역 사각형 안만 참이다."""
    rgb = np.asarray(rgb)
    if rgb.shape[0] != 512 or rgb.shape[1] != 512:
        rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
    a = rgb[..., :3].astype(np.int16)
    mx, mn = a.max(2), a.min(2)
    sat = (mx - mn) * 255 // np.maximum(mx, 1)
    inside = np.zeros((512, 512), bool)
    inside[ZONE_Y[0]:ZONE_Y[1], 50:462] = True
    occluder = (sat < SAT_MAX) & inside
    floor = (sat >= SAT_MAX) & inside
    n, labels, stats, _ = cv2.connectedComponentsWithStats(occluder.astype(np.uint8), connectivity=4)
    cube = np.zeros_like(occluder)
    arm = np.zeros_like(occluder)
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        from_bottom = y + h >= ZONE_Y[1]  # 로봇은 화면 아래에 있으므로 팔은 아래 가장자리를 지난다.
        if from_bottom or area > 3.3 * CUBE_AREA or max(w, h) > 95:
            arm |= labels == i
        elif area >= 0.25 * CUBE_AREA:
            cube |= labels == i
    return cube, arm, floor


def _zone(zone):
    x0, x1 = ZONE_X[zone]
    return (slice(ZONE_Y[0], ZONE_Y[1]), slice(x0, x1))


def check_transport(before_rgb, after_rgb):
    """운반 전후 두 장을 비교한다. verdict: 'moved' | 'not_moved' | 'unknown'.

    moved     : 중앙 구역에서 큐브 하나가 있던 자리가 비었고, 중앙의 다른 자리에 새로 나타나지 않았다.
    not_moved : 어느 구역에서도 큐브가 자리를 비우지 않았다.
    unknown   : 중앙의 큐브가 팔에 가렸거나, 중앙 밖에서 움직였거나, 중앙 안에서 밀렸다.
    수치의 단위는 큐브 1개 넓이다.
    """
    cube_b, _, floor_b = masks(before_rgb)
    cube_a, arm_a, floor_a = masks(after_rgb)
    c = _zone('center_zone')
    vacated_center = float((cube_b[c] & floor_a[c]).sum()) / CUBE_AREA
    appeared_center = float((floor_b[c] & cube_a[c]).sum()) / CUBE_AREA
    vacated_all = float((cube_b & floor_a).sum()) / CUBE_AREA
    hidden = float((cube_b[c] & arm_a[c]).sum()) / CUBE_AREA
    if vacated_center >= 0.6 and appeared_center < 0.4:
        verdict = 'moved'
    elif hidden >= 0.4 or vacated_all >= 0.6:
        verdict = 'unknown'
    else:
        verdict = 'not_moved'
    return {'verdict': verdict, 'vacated_center': round(vacated_center, 2),
            'appeared_center': round(appeared_center, 2), 'hidden': round(hidden, 2)}


def center_cube_area(rgb):
    """중앙 구역에 보이는 큐브 넓이(큐브 1개 단위). 중앙에 큐브가 있으면 정리가 끝나지 않은 것이다."""
    cube, _, _ = masks(rgb)
    return float(cube[_zone('center_zone')].sum()) / CUBE_AREA


# --------------------------------------------------------------------------- #
# 루프 판단
# --------------------------------------------------------------------------- #
def _frame(context):
    if cv2 is None:
        return None
    image = (context.get('observation') or {}).get('agentview_image')
    return None if image is None else np.asarray(image)


def _measure_call_time(context):
    """build_prompt 호출부터 다음 loop_decision까지 걸린 시간으로 VLM 호출 시간을 추정한다."""
    started = _state['prompt_time']
    if started is None:
        return
    _state['prompt_time'] = None
    steps = (context.get('feedback') or {}).get('steps') or 0
    estimate = time.perf_counter() - started - steps * 0.05  # 실시간 실행에서 한 스텝은 0.05초 이상
    if estimate > 1.0:
        _state['call_seconds'] = estimate


def _can_replan(context):
    return (context.get('remaining_calls', 0) > 0
            and context.get('remaining_seconds', 0) > _state['call_seconds'] + REPLAN_MARGIN_SECONDS)


def _attempts(context):
    counts = {}
    for h in context.get('history') or []:
        if h.get('event') == 'after_action':
            name = (h.get('action') or {}).get('object')
            counts[name] = counts.get(name, 0) + 1
    return counts


def _metadata_suspect(context):
    """원안의 블라인드 신호. 이미지 판정을 쓸 수 없을 때만 쓴다."""
    feedback = context.get('feedback') or {}
    obj = (feedback.get('action') or {}).get('object')
    if not obj:
        return False
    history = context.get('history') or []
    # 실행기는 방금 끝난 운반을 이력 맨 끝에 먼저 넣고 loop_decision을 부른다. 그 항목은 뺀다.
    repeated = any(h.get('event') == 'after_action' and (h.get('action') or {}).get('object') == obj
                   for h in history[:-1])
    last = context.get('last_model_response') or {}
    risky = any(a.get('object') == obj and a.get('accessible') is False
                for a in (last.get('assessments') or []))
    return repeated or risky


def _suspect(context, frame):
    """방금 끝난 운반이 실패했다고 의심하는가. (의심 여부, 근거) 를 돌려준다."""
    before = _state['prev_frame'] if _state['prev_frame'] is not None else _state['tick_frame']
    if USE_IMAGE_CHECK and frame is not None and before is not None:
        verdict = check_transport(before, frame)['verdict']
        if verdict == 'moved':
            return False, 'image:moved'
        if verdict == 'not_moved':
            return True, 'image:not_moved'
        return UNKNOWN_IS_SUSPECT, 'image:unknown'
    return _metadata_suspect(context), 'metadata'


def _replan(context, suspected, reason):
    if not SEND_STUDENT_FEEDBACK:
        return 'replan'
    return {'decision': 'replan',
            'feedback': {'event': context.get('event'), 'reason': reason,
                         'suspected_unplaced': suspected, 'attempts': _attempts(context)}}


def loop_decision(context: dict) -> str | dict:
    _measure_call_time(context)
    event = context.get('event')
    feedback = context.get('feedback') or {}
    last_response = context.get('last_model_response') or {}
    queue = context.get('queued_actions') or []
    frame = _frame(context)

    # (1) 시간 소진 -> 종료.
    if context.get('remaining_seconds', 0) <= 0:
        return 'stop'

    # (2) 동작 진행 중 관측 틱 -> 계속. 첫 운반은 '직전 프레임'이 없으므로 첫 틱 화면을 대신 보관한다.
    if event == 'observation_tick':
        if _state['prev_frame'] is None and _state['tick_frame'] is None and frame is not None:
            _state['tick_frame'] = frame
        return 'continue'

    # (3) 모델이 완료를 선언 -> 이미지에 중앙 큐브가 보이면 한 번 더 계획, 아니면 종료.
    if feedback.get('model_done') or last_response.get('done'):
        if (VERIFY_DONE_WITH_IMAGE and frame is not None
                and _state['done_rechecks'] < MAX_DONE_RECHECKS
                and center_cube_area(frame) >= 0.6 and _can_replan(context)):
            _state['done_rechecks'] += 1
            _state['prev_frame'] = frame
            return _replan(context, [], 'done_contradicted_by_image')
        return 'stop'

    # (4) 운반 하나가 끝났다. 세 전략은 여기서만 갈린다.
    if event == 'after_action':
        obj = (feedback.get('action') or {}).get('object')
        suspect, basis = _suspect(context, frame)
        if frame is not None:
            _state['prev_frame'] = frame
            _state['tick_frame'] = None
        next_hits_limit = (ATTEMPT_GUARD and bool(queue)
                           and _attempts(context).get(queue[0].get('object'), 0) >= MAX_ATTEMPTS)
        if MODE == 'C0':
            want_replan = False           # 계획을 끝까지 실행한다.
        elif MODE == 'C1':
            want_replan = suspect         # 실패가 의심될 때만.
        else:
            want_replan = True            # C2: 매번.
        want_replan = want_replan or next_hits_limit
        _state['log'].append({'object': obj, 'suspect': suspect, 'basis': basis, 'want_replan': want_replan})
        if want_replan and _can_replan(context):
            return _replan(context, [obj] if suspect and obj else [], basis)
        if queue:
            return 'continue'             # 예산이 없으면 남은 계획을 이어 실행한다.
        if _can_replan(context):
            return _replan(context, [obj] if suspect and obj else [], basis)
        return 'stop'

    # (5) 계획 소진/파싱 실패 -> 예산이 있으면 재계획.
    if event in ('plan_exhausted', 'invalid_plan'):
        if frame is not None:
            _state['prev_frame'] = frame
        if _can_replan(context):
            return _replan(context, [], event)
        return 'stop'
    return 'continue'
