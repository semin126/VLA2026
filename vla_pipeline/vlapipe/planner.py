"""VLM 플래너 — 고수준 자연어 명령을 순차적 지시문으로 쪼갠다.

파이프라인 1단계:
    (이미지, "테이블을 정리하시오")  ->  [지시문1, 지시문2, ...]

두 가지 구현:
  RulePlanner : 시뮬레이터의 물체 위치를 직접 보고 계획을 세운다(정답 플래너).
                VLM 이 없어도 파이프라인 전체가 돌아가고, VLM 채점 기준이 된다.
  VLMPlanner  : SmolVLM2 로 이미지를 보고 계획을 세운다(진짜 1단계).

두 플래너 모두 같은 스키마를 낸다:
  {"step": 1, "instruction": "...", "instruction_ko": "...",
   "object": "red_block", "target_zone": "right_zone"}
"""

from __future__ import annotations

import json
import math
import re

from . import accessibility as acc
from . import config as C
from . import scene as scene_mod

VLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

ZONE_ALIASES = {
    "left_zone": ["left storage zone", "left zone", "left", "green zone", "왼쪽", "좌측"],
    "right_zone": ["right storage zone", "right zone", "right", "blue zone", "오른쪽", "우측"],
    "center_zone": ["center workspace", "center zone", "middle", "center", "가운데", "중앙"],
}


def instruction_text(obj_name: str, zone: str) -> tuple[str, str]:
    """물체를 가리키는 말은 config 의 ref_* 에서 온다.

    색 세트면 "red block"/"빨간 물체", 숫자 세트면 "object 7"/"7번 물체".
    지시문 틀(앞뒤 문장)은 두 조건에서 완전히 같아야 비교가 성립한다.
    """
    o = C.OBJECT_BY_NAME[obj_name]
    en = f"Pick up the {o['ref_en']} and place it in the {C.ZONES[zone]['label_en']}."
    ko = f"{o['ref_ko']}를 집어 {C.ZONES[zone]['label_ko']}에 놓으시오."
    return en, ko


_LIST_MARKER = re.compile(r"^\s*\d+\s*[.)]\s*")


def parse_instruction(text: str) -> dict | None:
    """자연어 지시문 -> {'object':..., 'target_zone':...}.

    VLM 이 낸 문장을 scripted VLA 가 실행할 수 있게 잇는 다리 역할.
    SmolVLA 로 바꾸면 이 함수 없이 문장을 그대로 정책에 넣는다.
    """
    low = text.lower()
    obj = None
    # 강한 단서(patterns) 를 먼저 다 돌고, 하나도 안 맞을 때만 폴백(weak_patterns).
    # 숫자 세트에서 "1. move object 10 ..." 같은 줄의 목록 번호 '1.' 을 물체로
    # 오인하지 않게 하려고 두 단계로 나눈다 (폴백 단계에서는 목록 번호를 떼고 본다).
    for tier in ("patterns", "weak_patterns"):
        body = _LIST_MARKER.sub("", text) if tier == "weak_patterns" else text
        for o in C.OBJECTS:
            if any(re.search(p, body, re.IGNORECASE) for p in o.get(tier, ())):
                obj = o["name"]
                break
        if obj is not None:
            break
    zone = None
    for zname, aliases in ZONE_ALIASES.items():
        if any(a in low or a in text for a in aliases):
            zone = zname
            break
    if obj is None:
        return None
    return {"object": obj, "target_zone": zone or C.OBJECT_BY_NAME[obj]["home_zone"]}


def clear_instruction(obj_name: str, mode: str) -> tuple[str, str]:
    o = C.OBJECT_BY_NAME[obj_name]
    if mode == "push":
        return (f"Push the {o['ref_brief_en']} aside to clear the way.",
                f"{o['ref_ko']}를 옆으로 밀어 길을 트시오.")
    return (f"Move the {o['ref_brief_en']} out of the way first.",
            f"{o['ref_ko']}를 먼저 치워 길을 트시오.")


def to_subtask(obj_name: str, zone: str, step: int, action: str = "transport",
               reason: str = "", push_to: tuple[float, float] | None = None) -> dict:
    """action: transport(제자리로 운반) | clear(장애물 치우기) | push(밀어내기)"""
    if action == "transport":
        en, ko = instruction_text(obj_name, zone)
    else:
        en, ko = clear_instruction(obj_name, action)
    st = {"step": step, "instruction": en, "instruction_ko": ko,
          "object": obj_name, "target_zone": zone,
          "action": action, "reason": reason}
    if push_to is not None:
        st["push_to"] = [round(push_to[0], 4), round(push_to[1], 4)]
    return st


# --------------------------------------------------------------------------
class RulePlanner:
    """시뮬레이터 상태를 보고 만드는 정답 계획 — VLM 채점 기준.

    이 연구의 제목 그대로, **물체 접근성과 공간 제약**을 보고 운반 행동을 고른다.
      - 지금 집을 수 있는 물체가 있으면 로봇에서 가까운 것부터 운반한다.
      - 전부 막혀 있으면 장애물을 먼저 치운다.
          · 장애물을 집을 수 있으면  -> clear (집어서 빈 자리로)
          · 장애물도 막혀 있으면(상호 가림) -> push (밀어서 길 트기)
    계획은 가상 상태를 갱신하며 한 수씩 진행하므로, 순서 자체가 계획의 산출물이 된다.
    """

    MAX_STEPS = 16

    def __init__(self, env):
        self.env = env

    def plan(self, command: str = "테이블을 정리하시오") -> list[dict]:
        env = self.env
        pos, yaw = env.object_xy_yaw()
        pos = dict(pos)

        home = {o["name"]: o["home_zone"] for o in C.OBJECTS}
        tidy = scene_mod.tidy_layout()

        def in_home(n):
            z = C.ZONES[home[n]]
            x, y = pos[n]
            return z["x"][0] <= x <= z["x"][1] and z["y"][0] <= y <= z["y"][1]

        def dist_from_base(n):
            return math.hypot(pos[n][0] - C.ROBOT_BASE_XY[0], pos[n][1] - C.ROBOT_BASE_XY[1])

        plan: list[dict] = []
        for _ in range(self.MAX_STEPS):
            todo = [n for n in C.OBJECT_NAMES if not in_home(n)]
            if not todo:
                break

            ready = [n for n in todo if acc.is_accessible(n, pos, yaw)[0]]
            if ready:
                pick = min(ready, key=dist_from_base)
                plan.append(to_subtask(pick, home[pick], len(plan) + 1, "transport"))
                pos[pick] = (tidy[pick][0], tidy[pick][1])      # 가상으로 제자리에 놓는다
                continue

            # 남은 것이 전부 막혔다 -> 장애물을 치운다
            blocked = [(n, acc.blockers(n, pos, yaw)) for n in todo]
            blocked = [(n, b) for n, b in blocked if b]
            if not blocked:
                break                                           # 도달 불가만 남음
            target, blks = min(blocked, key=lambda t: len(t[1]))
            blocker = blks[0]

            if acc.is_accessible(blocker, pos, yaw)[0]:
                spot = acc.free_spot(home[blocker], pos, exclude={blocker, target})
                plan.append(to_subtask(blocker, home[blocker], len(plan) + 1, "clear",
                                       reason=f"{target} 접근 경로를 막고 있음"))
                if spot:
                    pos[blocker] = spot
            else:
                # 서로 막고 있다 -> 집을 수 없으니 밀어낸다
                bx, by = pos[blocker]
                tx, ty = pos[target]
                ux, uy = bx - tx, by - ty
                nrm = math.hypot(ux, uy) or 1.0
                dest = (bx + 0.11 * ux / nrm, by + 0.11 * uy / nrm)
                plan.append(to_subtask(blocker, home[blocker], len(plan) + 1, "push",
                                       reason=f"{target} 와(과) 상호 가림", push_to=dest))
                pos[blocker] = dest
        return plan


# --------------------------------------------------------------------------
_PROMPT = """You are the high-level planner of a robot that tidies a table.

The table has three zones, seen from above in the image:
- left storage zone (green, on the LEFT of the image)
- center workspace (olive, in the MIDDLE) -- this must end up EMPTY
- right storage zone (blue, on the RIGHT of the image)

Rules:
- Long rectangular blocks (red, orange, yellow, lime, cyan) belong in the right storage zone.
- Cubes and small tiles (blue, purple, pink, green, brown) belong in the left storage zone.
- Anything already in its correct zone must NOT be moved.

User command: "{command}"

Look at the image. List ONLY the objects that are in the wrong place, in the order the
robot should move them. Answer with a JSON array and nothing else. Each element must be:
{{"object": "<color> <block|cube|tile>", "target_zone": "left storage zone|right storage zone"}}

JSON:"""


_PROMPT_LINES = """You are the high-level planner of a robot that tidies a table.

The image is a top-down view of a table with three zones:
- LEFT part of the image  = left storage zone (green)
- MIDDLE of the image     = center workspace (olive) -- must end up EMPTY
- RIGHT part of the image = right storage zone (blue)

Where each object belongs:
- long rectangular blocks (red, orange, yellow, lime, cyan) -> right storage zone
- cubes and flat tiles (blue, purple, pink, green, brown)   -> left storage zone

User command: "{command}"

Look at the olive middle area. For EVERY object lying there, write one line:
move the <color> object to the <left|right> storage zone

Do not mention objects that are already inside the green or blue areas.
Write only those lines, nothing else.

Lines:"""


# 실측(SmolVLM2-500M, messy 씬 3개): json recall 0.23 / lines recall 0.00.
# 'lines' 는 모델이 한 줄만 쓰고 멈춰 버려서 기본값은 'json' 으로 둔다.
# 프롬프트 배틀 미니과제에서 학생들이 이 표를 갱신하게 할 것.
PROMPTS = {"json": _PROMPT, "lines": _PROMPT_LINES}


class VLMPlanner:
    """SmolVLM2 로 이미지를 보고 계획을 세운다."""

    def __init__(self, model_id: str = VLM_MODEL_ID, device: str = C.DEFAULT_DEVICE,
                 dtype: str | None = None, prompt_style: str = "json"):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        # GPU 가 없는 환경(런타임 유형을 안 바꾼 Colab 등)에서는 CPU 로 내려간다.
        # CPU 에서는 bfloat16 이 느려서 dtype 도 같이 float32 로 바뀐다.
        device = C.pick_device(device)
        dtype = C.pick_dtype(device, dtype)

        self.torch = torch
        self.device = device
        self.prompt_style = prompt_style
        self.processor = AutoProcessor.from_pretrained(model_id)

        # device_map 으로 가중치를 GPU 로 바로 흘려보낸다.
        # 기본 방식(CPU 에 전부 올린 뒤 .to('cuda'))은 호스트 RAM 피크가 모델 크기의 약 2배라,
        # 2.2B 급을 데스크톱에서 부르면 메모리 압박으로 systemd-oomd 가
        # VSCode 스코프째 죽여 버리는 일이 있다. (README '문제 해결' 참고)
        kwargs = {"dtype": getattr(torch, dtype), "low_cpu_mem_usage": True}
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, device_map=device, **kwargs).eval()
        except (TypeError, ValueError, ImportError):
            # accelerate 가 없거나 구버전 transformers 면 예전 방식으로
            kwargs.pop("low_cpu_mem_usage", None)
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, **kwargs).to(device).eval()

    def raw_response(self, image, command: str, max_new_tokens: int = 400) -> str:
        from PIL import Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPTS[self.prompt_style].format(command=command)},
        ]}]
        # return_tensors 를 그냥 넘기면 transformers 5 가 추론할 때마다
        #   "Kwargs passed to `processor.__call__` have to be in `processor_kwargs`"
        # 를 찍는다. 동작은 정상이지만 노트북이 경고로 도배되어 오류로 오해하기 쉽다.
        # 권장 방식으로 넘기고, 이 인자가 없는 구버전에서는 예전 방식으로 돌아간다.
        kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True)
        inputs = None
        try:
            got = self.processor.apply_chat_template(
                messages, processor_kwargs={"return_tensors": "pt"}, **kw)
            inputs = got if hasattr(got, "to") else None   # 구버전은 그냥 무시하고 list 를 준다
        except (TypeError, ValueError):
            pass
        if inputs is None:
            inputs = self.processor.apply_chat_template(messages, return_tensors="pt", **kw)
        inputs = inputs.to(self.device, dtype=self.model.dtype)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        for marker in ("Lines:", "JSON:", "Assistant:"):
            if marker in text:
                text = text.split(marker)[-1]
        return text.strip()

    def plan(self, image, command: str = "Tidy up the table.") -> list[dict]:
        return parse_plan(self.raw_response(image, command))


def _repair_json_array(blob: str):
    """토큰 한계로 중간에 잘린 JSON 배열을 살려 낸다.

    VLM 은 `[{"object": "red", ...}, {"object": "blue", "target_z` 처럼
    끝이 잘린 채로 멈추는 일이 잦다. 그대로 두면 파싱이 통째로 실패하므로
    마지막으로 온전히 닫힌 원소까지만 남기고 배열을 닫아 준다.
    """
    try:
        parsed = json.loads(blob)
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        pass
    cut = max(blob.rfind("}"), blob.rfind("]", 0, len(blob) - 1))
    if cut <= 0:
        return None
    try:
        parsed = json.loads(blob[:cut + 1] + "]")
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        return None


def _flatten(items):
    """VLM 이 [{"object":..},..] / [["blue","left zone"],..] / ["blue block",..] 등
    제각각으로 내놓으므로 전부 '한 줄 문자열' 리스트로 편다."""
    out = []
    for item in items:
        if isinstance(item, dict):
            out.append(" ".join(str(v) for v in item.values()))
        elif isinstance(item, (list, tuple)):
            out.append(" ".join(str(v) for v in item))
        else:
            out.append(str(item))
    return out


def parse_plan(text: str) -> list[dict]:
    """VLM 출력에서 계획을 뽑아낸다. JSON 실패 시 줄 단위로 재시도."""
    candidates: list[str] = []
    m = re.search(r"\[.*\]", text, re.S)          # 바깥쪽 배열 전체 (greedy)
    blob = m.group(0) if m else (text[text.index("["):] if "[" in text else "")
    if blob:
        parsed = _repair_json_array(blob)      # 잘린 배열도 살려 낸다
        if parsed is not None:
            candidates = _flatten(parsed)
    if not candidates:                            # 폴백: 줄 단위
        candidates = [ln for ln in text.splitlines() if ln.strip()]

    subtasks: list[dict] = []
    seen: set[str] = set()
    for line in candidates:
        got = parse_instruction(line)
        if not got or got["object"] in seen:
            continue
        seen.add(got["object"])
        subtasks.append(to_subtask(got["object"], got["target_zone"], len(subtasks) + 1))
    return subtasks


def plan_accuracy(pred: list[dict], gold: list[dict]) -> dict:
    """VLM 계획을 정답(RulePlanner) 계획과 비교 — 미니과제 채점용."""
    p = {s["object"]: s["target_zone"] for s in pred}
    g = {s["object"]: s["target_zone"] for s in gold}
    tp = sum(1 for k, v in p.items() if g.get(k) == v)
    return {
        "n_pred": len(p), "n_gold": len(g), "correct": tp,
        "precision": tp / len(p) if p else 0.0,
        "recall": tp / len(g) if g else 0.0,
        "missed": sorted(set(g) - set(p)),
        "hallucinated": sorted(set(p) - set(g)),
        "wrong_zone": sorted(k for k in set(p) & set(g) if p[k] != g[k]),
    }
