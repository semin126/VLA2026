"""물체 접근성 판정 — 이 연구 제목의 '물체 접근성 및 공간 제약'을 코드로 정의한 곳.

위에서 내려와 잡으려면 열린 그리퍼의 두 손가락이 물체 중심에서 개폐축 방향으로
±(최대개폐/2) 떨어진 두 지점을 통과해야 한다. 그 통로에 다른 물체가 있으면 가려진 것이다.

순수 함수로 둔 이유: 시뮬레이터의 현재 상태(SortingEnv)뿐 아니라
플래너가 '이 물체를 옮기고 나면 저 물체를 집을 수 있는가'를 **가상으로** 따질 때도 써야 한다.
"""

from __future__ import annotations

import math

from . import config as C

GRIPPER_HALF_OPEN = 0.040     # Panda 최대 개폐 0.08 m 의 절반
FINGER_RADIUS = 0.016         # 손가락 폭 + 여유
CLEAR_MARGIN = 0.004          # 판정 여유


def object_radius(name: str) -> float:
    """위에서 본 보수적 반경 — 회전과 무관하게 물체를 감싸는 원."""
    hx, hy, _ = C.half_size(C.OBJECT_BY_NAME[name])
    return math.hypot(hx, hy)


def grasp_yaw(name: str, yaw: float) -> float:
    """개폐축 방향. 긴 변이 있으면 짧은 변을 물도록 90도 돌린다 (env.grasp_pose 와 동일)."""
    hx, hy, _ = C.half_size(C.OBJECT_BY_NAME[name])
    return yaw + (math.pi / 2 if hx > hy else 0.0)


def finger_slots(name: str, xy, yaw: float):
    """두 손가락이 내려올 위치 (월드 xy) 두 개."""
    g = grasp_yaw(name, yaw)
    ux, uy = math.cos(g), math.sin(g)
    return ((xy[0] + GRIPPER_HALF_OPEN * ux, xy[1] + GRIPPER_HALF_OPEN * uy),
            (xy[0] - GRIPPER_HALF_OPEN * ux, xy[1] - GRIPPER_HALF_OPEN * uy))


def blockers(name: str, positions: dict, yaws: dict) -> list[str]:
    """name 을 위에서 잡으려 할 때 손가락 통로를 막는 물체들."""
    a, b = finger_slots(name, positions[name], yaws.get(name, 0.0))
    out = []
    for other, q in positions.items():
        if other == name:
            continue
        need = object_radius(other) + FINGER_RADIUS + CLEAR_MARGIN
        da = math.hypot(q[0] - a[0], q[1] - a[1])
        db = math.hypot(q[0] - b[0], q[1] - b[1])
        if min(da, db) < need:
            out.append(other)
    return out


def is_reachable(xy) -> bool:
    d = math.hypot(xy[0] - C.ROBOT_BASE_XY[0], xy[1] - C.ROBOT_BASE_XY[1])
    return C.REACH_MIN <= d <= C.REACH_MAX


def is_accessible(name: str, positions: dict, yaws: dict) -> tuple[bool, str]:
    """(지금 바로 집을 수 있는가, 안 되는 이유)."""
    if not is_reachable(positions[name]):
        return False, "out_of_reach"
    blk = blockers(name, positions, yaws)
    if blk:
        return False, "blocked_by:" + ",".join(blk)
    return True, ""


def free_spot(zone: str, positions: dict, exclude: set[str] | None = None,
              min_gap: float = 0.115) -> tuple[float, float] | None:
    """zone 안에서 다른 물체와 min_gap 이상 떨어진 자리를 찾는다.
    '장애물을 어디로 치울 것인가'(공간 제약)를 푸는 함수."""
    exclude = exclude or set()
    z = C.ZONES[zone]
    others = [p for n, p in positions.items() if n not in exclude]
    best, best_d = None, -1.0
    steps_x, steps_y = 5, 6
    for i in range(steps_x):
        gx = z["x"][0] + 0.055 + (z["x"][1] - z["x"][0] - 0.11) * i / max(steps_x - 1, 1)
        for j in range(steps_y):
            gy = max(z["y"][0] + 0.07, 0.12) + (z["y"][1] - z["y"][0] - 0.13) * j / max(steps_y - 1, 1)
            if not is_reachable((gx, gy)):
                continue
            d = min((math.hypot(gx - p[0], gy - p[1]) for p in others), default=9.9)
            if d > best_d:
                best, best_d = (gx, gy), d
    return best if best_d >= min_gap else best
