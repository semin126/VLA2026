"""정량 평가 지표.

성공률만으로는 이 연구의 주장을 검증할 수 없다.
'물체 접근성 및 공간 제약'을 다루는 능력을 직접 재는 지표가 필요하다.

지표군
  1) 성공  : 씬 정리 완료율, 운반 성공률
  2) 효율  : 씬당 행동 수, 불필요 행동 수, 제어 스텝 수
  3) 접근성: 가림 위반율, 장애물 선처리율   <- 이 연구 고유
  4) 계획  : recall/precision, 순서 거리, 행동 유형 혼동
  5) 비용  : 사람 조작 시간, 역재생 폐기율
"""

from __future__ import annotations

import math
from collections import Counter

from . import accessibility as acc
from . import config as C


# ---------------------------------------------------------------- 3) 접근성
def accessibility_violation(env, subtask: dict) -> dict:
    """이 행동을 시작하는 시점에 대상 물체가 접근 가능했는가.

    가려진 물체를 집으려 드는 것은 '접근성을 못 읽었다'는 직접 증거다.
    밀기(push)는 애초에 가려진 물체를 대상으로 하므로 위반이 아니다.
    """
    name = subtask["object"]
    ok, why = env.is_accessible(name)
    action = subtask.get("action", "transport")
    return {
        "object": name, "action": action,
        "accessible": bool(ok), "reason": why,
        # 잡으려 했는데 가려져 있었다 = 위반
        "violation": bool((not ok) and action in ("transport", "clear")),
    }


def blocked_first_rate(records: list[dict]) -> dict:
    """장애물 선처리율 — 막힌 목표를 만났을 때 장애물부터 치웠는가."""
    clears = sum(1 for r in records if r.get("action") in ("clear", "push"))
    viol = sum(1 for r in records if r.get("violation"))
    denom = clears + viol
    return {
        "n_clear_or_push": clears,
        "n_violation": viol,
        "blocked_first_rate": round(clears / denom, 3) if denom else None,
    }


# ---------------------------------------------------------------- 4) 계획 품질
def order_distance(pred: list[str], gold: list[str]) -> dict:
    """두 행동 순서가 얼마나 다른가.

    Kendall tau 거리(뒤바뀐 쌍의 수)를 공통 원소에 대해 계산한다.
    성공률이 같아도 '순서를 제대로 골랐는가'는 다를 수 있으므로 따로 잰다.
    """
    common = [o for o in pred if o in gold]
    rank = {o: i for i, o in enumerate(gold)}
    n = len(common)
    if n < 2:
        return {"n_common": n, "kendall_tau_distance": None, "normalized": None}
    inv = sum(1 for i in range(n) for j in range(i + 1, n)
              if rank[common[i]] > rank[common[j]])
    total = n * (n - 1) / 2
    return {
        "n_common": n,
        "kendall_tau_distance": inv,
        "normalized": round(inv / total, 3),      # 0 = 완전 일치, 1 = 완전 역순
        "coverage": round(n / len(gold), 3) if gold else None,
    }


def action_type_confusion(pred: list[dict], gold: list[dict]) -> dict:
    """운반/치우기/밀기를 제대로 골랐는가 — 물체 기준으로 맞춰 비교."""
    g = {s["object"]: s.get("action", "transport") for s in gold}
    cm: Counter = Counter()
    for s in pred:
        o = s["object"]
        cm[(g.get(o, "none"), s.get("action", "transport"))] += 1
    return {f"{a}->{b}": n for (a, b), n in sorted(cm.items())}


# ---------------------------------------------------------------- 2) 효율
def minimum_actions(env) -> int:
    """이 씬을 정리하는 데 필요한 최소 행동 수(하한).

    잘못 놓인 물체 수가 하한이다. 가림 때문에 치우기가 더 필요할 수 있으므로
    실제 최적값은 이보다 크거나 같다. '불필요 행동 수'의 기준선으로만 쓴다.
    """
    return len(env.misplaced_objects())


def efficiency(n_actions: int, n_min: int, n_steps: int) -> dict:
    return {
        "n_actions": n_actions,
        "n_min_actions": n_min,
        "excess_actions": n_actions - n_min,
        "action_ratio": round(n_actions / n_min, 3) if n_min else None,
        "n_control_steps": n_steps,
    }


# ---------------------------------------------------------------- 5) 데이터 비용
def episode_cost(ep) -> dict:
    """에피소드 하나의 수집 비용. 사람 조작 시간 = 프레임 수 / 제어 주파수."""
    n = len(ep)
    return {
        "frames": n,
        "seconds": round(n / ep.fps, 1),
        "direction": str(ep.direction),
    }


def collection_cost(episodes: list, n_rejected: int = 0) -> dict:
    tot = sum(len(e) for e in episodes)
    fps = episodes[0].fps if episodes else C.CONTROL_HZ
    n = len(episodes)
    return {
        "n_episodes": n,
        "n_rejected": n_rejected,
        "reject_rate": round(n_rejected / (n + n_rejected), 3) if (n + n_rejected) else None,
        "total_frames": tot,
        "total_minutes": round(tot / fps / 60, 1),
        "minutes_per_episode": round(tot / fps / 60 / n, 2) if n else None,
    }


# ---------------------------------------------------------------- 통계
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """이항 비율의 Wilson 신뢰구간. 씬 수가 적을 때 정규근사보다 정확하다."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 3), round((c + h) / d, 3))


def paired_difference(a: list[float], b: list[float], n_boot: int = 4000,
                      seed: int = 0) -> dict:
    """같은 씬에서 잰 두 조건의 차이 (a - b) 와 부트스트랩 신뢰구간.

    H3 의 '차이의 차이'도 이 함수를 두 번 써서 구한다.
    씬 시드를 조건마다 똑같이 맞춰야 짝지은 비교가 성립한다.
    """
    import random
    assert len(a) == len(b), "짝지은 비교이므로 같은 씬 집합이어야 합니다."
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    mean = sum(d) / n if n else 0.0
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = [d[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    return {
        "n_pairs": n,
        "mean_difference": round(mean, 4),
        "ci95": (round(boots[int(0.025 * n_boot)], 4),
                 round(boots[int(0.975 * n_boot)], 4)),
        "significant": not (boots[int(0.025 * n_boot)] <= 0 <= boots[int(0.975 * n_boot)]),
    }
