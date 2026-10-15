"""숫자 조건(--objects number)이 제대로 만들어졌는지 확인한다.

두 부분으로 나뉘어 있다.

  1) 글/파싱 검사 — MuJoCo 가 없어도 돌아간다
     - 물체 세트 표 (어느 숫자가 어느 구역으로 가는가)
     - 숫자 텍스처 10장 생성 + 한 장에 모아 붙인 미리보기
     - 씬 XML 생성 및 문법 검증
     - 지시문 -> 파싱 왕복 검사: "object 7" 이 다시 object_7 로 돌아오는가
       (VLM 출력이 ["7", "3"] / "1. move object 10 to ..." 처럼 와도 맞는지)
     - 규칙 플래너(정답)가 숫자 규칙대로 계획을 세우는가

  2) 렌더 검사 — MuJoCo 필요
     - agentview / frontview 그림 저장 (숫자가 실제로 보이는지는 눈으로 확인)
     - 가운데 구역 확대 그림
     - 짧은 롤아웃: 규칙 플래너 + 스크립트 정책으로 실제 집어 옮겨 보기

사용:
  .\\.venv\\Scripts\\python.exe scripts\\check_numbers.py
  .\\.venv\\Scripts\\python.exe scripts\\check_numbers.py --layout cluttered --seed 7
  .\\.venv\\Scripts\\python.exe scripts\\check_numbers.py --rollout 0      # 렌더만, 롤아웃 생략
"""
import _bootstrap  # noqa: F401
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from vlapipe import config as C
from vlapipe import numbered
from vlapipe import planner as P
from vlapipe import scene as scene_mod

OUT = C.OUTPUT_DIR / "number_check"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="messy",
                    choices=["messy", "random", "tidy", "cluttered"])
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--rollout", type=int, default=2,
                    help="실제로 집어 옮겨 볼 서브태스크 개수 (0 = 생략)")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--no-render", action="store_true", help="MuJoCo 없이 글/파싱 검사만")
    return ap.parse_args()


# --------------------------------------------------------------------------
# 1) 글/파싱 검사
# --------------------------------------------------------------------------
class _StubEnv:
    """규칙 플래너는 물체의 (x, y, yaw) 만 본다. 그래서 MuJoCo 없이도 검사할 수 있다."""

    def __init__(self, layout):
        self._l = layout

    def object_xy_yaw(self):
        return ({n: (v[0], v[1]) for n, v in self._l.items()},
                {n: v[2] for n, v in self._l.items()})


def texture_sheet(paths, path: Path) -> Path:
    """숫자 텍스처 10장을 한 줄로 붙여 저장 — MuJoCo 없이도 숫자를 눈으로 볼 수 있다."""
    from PIL import Image

    tiles = [Image.open(p).resize((128, 128)) for p in paths]
    sheet = Image.new("RGB", (128 * len(tiles), 128), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, (128 * i, 0))
    sheet.save(path)
    return path


def check_text(args) -> int:
    fails = 0
    print("=" * 66)
    print("1) 글/파싱 검사")
    print("=" * 66)

    numbered.activate()                             # 세트 교체 + 텍스처 생성
    paths = numbered.write_number_textures()        # 이미 있으면 그대로 경로만 돌려준다
    print(f"\n[물체 세트] {C.ACTIVE_OBJECT_SET}  물체 {len(C.OBJECT_NAMES)}개"
          f"  (규칙: 1~{C.NUMBER_SPLIT} -> 오른쪽, {C.NUMBER_SPLIT + 1}~{C.NUMBER_COUNT} -> 왼쪽)")
    for o in C.OBJECTS:
        en, ko = P.instruction_text(o["name"], o["home_zone"])
        print(f"   {o['name']:<10} 숫자 {o['number']:>2}  {o['home_zone']:<11} | {en}")
        print(f"   {'':<10}              {'':<11} | {ko}")

    OUT.mkdir(parents=True, exist_ok=True)
    sheet = texture_sheet(paths, OUT / "textures.png")
    print(f"\n[텍스처] {len(paths)}장 -> {paths[0].parent}")
    print(f"         미리보기 한 장: {sheet}")

    # 씬 XML — 텍스처/재질이 들어갔고 문법이 맞는가
    xml = scene_mod.build_scene_xml(scene_mod.messy_layout(args.seed))
    ET.fromstring(xml)
    n_tex = xml.count('type="cube"')
    n_mat = sum(1 for o in C.OBJECTS if f'material="{o["material"]}"' in xml)
    print(f"\n[씬 XML] 문법 OK, 숫자 텍스처 {n_tex}개, 재질이 붙은 물체 {n_mat}/{len(C.OBJECTS)}개")
    if n_tex != C.NUMBER_COUNT or n_mat != len(C.OBJECTS):
        print("   !! 텍스처/재질이 다 안 붙었습니다")
        fails += 1

    # 지시문 -> 파싱 왕복
    print("\n[왕복 검사] 지시문을 다시 파싱해서 같은 물체/구역이 나오는가")
    bad = []
    for o in C.OBJECTS:
        for zone in ("left_zone", "right_zone"):
            for text in P.instruction_text(o["name"], zone):
                got = P.parse_instruction(text)
                if not got or got["object"] != o["name"] or got["target_zone"] != zone:
                    bad.append((text, got))
    print(f"   지시문 {len(C.OBJECTS) * 4}개 중 실패 {len(bad)}개")
    for text, got in bad[:5]:
        print(f"   !! {text!r} -> {got}")
    fails += bool(bad)

    # VLM 이 실제로 내놓는 모양들
    print("\n[VLM 출력 모양] 까다로운 경우들")
    cases = [
        ('[{"object": "cube number 7", "target_zone": "left storage zone"}]', ["object_7"]),
        ('[{"object": "cube number 10", "target_zone": "left storage zone"},'
         ' {"object": "object 1", "target_zone": "right storage zone"}]',
         ["object_10", "object_1"]),
        ("1. move object 10 to the left storage zone\n"
         "2. move object 3 to the right storage zone", ["object_10", "object_3"]),
        ("[3, 7]", ["object_3", "object_7"]),          # number_read 프롬프트의 답
        ("object_7", ["object_7"]),
        ("7번 물체를 집어 왼쪽 보관 구역에 놓으시오", ["object_7"]),
        ('["cube #2", "no. 9"]', ["object_2", "object_9"]),
    ]
    for text, want in cases:
        got = [s["object"] for s in P.parse_plan(text)]
        ok = got == want
        fails += not ok
        flag = "OK " if ok else "!! "
        print(f"   {flag}{text.splitlines()[0][:48]:<50} -> {got}")
        if not ok:
            print(f"       기대한 값: {want}")

    # 숫자만 보고 세운 계획의 목적지가 숫자 규칙과 맞는가
    print("\n[규칙 플래너] 숫자 규칙대로 계획을 세우는가")
    lay = scene_mod.LAYOUTS[args.layout](args.seed) if args.layout != "tidy" \
        else scene_mod.tidy_layout()
    plan = P.RulePlanner(_StubEnv(lay)).plan()
    wrong = [s for s in plan if s["action"] == "transport"
             and s["target_zone"] != C.number_home_zone(C.OBJECT_BY_NAME[s["object"]]["number"])]
    print(f"   {args.layout} seed {args.seed}: 서브태스크 {len(plan)}개, 구역 오류 {len(wrong)}개")
    for s in plan:
        print(f"   [{s['step']}] {s['action']:<9} {s['instruction']}")
    fails += bool(wrong)

    # 색 세트가 그대로인지 (바꾼 코드가 기준 조건을 망가뜨리지 않았는지)
    numbered.deactivate()
    print(f"\n[기준 조건 확인] 세트: {C.ACTIVE_OBJECT_SET}, 물체 {len(C.OBJECT_NAMES)}개")
    en, ko = P.instruction_text("red_block", "right_zone")
    expect_en = "Pick up the red block and place it in the right storage zone."
    print(f"   {en}")
    print(f"   {ko}")
    if en != expect_en:
        print(f"   !! 기존 문장과 달라졌습니다 (기대: {expect_en!r})")
        fails += 1
    if P.parse_instruction(en) != {"object": "red_block", "target_zone": "right_zone"}:
        print("   !! 색 세트 파싱이 깨졌습니다")
        fails += 1
    numbered.activate()
    return fails


# --------------------------------------------------------------------------
# 2) 렌더 검사
# --------------------------------------------------------------------------
def check_render(args) -> int:
    print("\n" + "=" * 66)
    print("2) 렌더 검사 (MuJoCo)")
    print("=" * 66)
    from PIL import Image

    from vlapipe.env import SortingEnv

    print("\n[숫자 크기] agentview 에서 물체 윗면 한 변 — 카메라 기하 계산값(실측 아님)")
    for px in (256, 384, 512, 768):
        f = numbered.face_pixels(px)
        print(f"   {px:>4} px 렌더 -> 윗면 {f:5.1f} px, 숫자 높이 약 {f * 0.7:4.1f} px")
    print("   (숫자를 읽히게 하려면 VLM 입력을 크게: run_pipeline 은 number 조건에서 512 를 쓴다)")

    env = SortingEnv(layout=args.layout, seed=args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    saved = []
    # 896 상한: 씬의 <global offheight="960"> 보다 크게 렌더하면 MuJoCo 가 거부한다.
    for cam, size in (("agentview", 256), ("agentview", 512), ("agentview", 896),
                      ("frontview", 640), ("robot0_eye_in_hand", 256)):
        img = env.render(cam, size, size)
        p = OUT / f"{args.layout}{args.seed}_{cam}_{size}.png"
        Image.fromarray(img).save(p)
        saved.append(p)
        if cam == "agentview" and size == 896:
            # 가운데 구역만 잘라 확대 — 숫자가 읽히는지 눈으로 보기 쉽게
            c = Image.fromarray(img)
            s = c.size[0]
            crop = c.crop((int(s * 0.30), int(s * 0.30), int(s * 0.70), int(s * 0.70)))
            p = OUT / f"{args.layout}{args.seed}_center_zoom.png"
            crop.resize((crop.size[0] * 2, crop.size[1] * 2), Image.NEAREST).save(p)
            saved.append(p)
    print(f"\n[그림] {len(saved)}장 -> {OUT}")
    for p in saved:
        print(f"   {p.name}")
    print("   *** 숫자가 읽히는지, 뒤집혀 보이지 않는지는 사람이 직접 봐야 합니다 ***")

    report = env.accessibility_report()
    print(f"\n[씬] 잘못 놓인 물체 {len(env.misplaced_objects())}개 {env.misplaced_objects()}")
    print(f"     집을 수 있는 물체 {report['n_accessible']}개, 막힌 물체 {report['n_blocked']}개")

    fails = 0
    if args.rollout > 0:
        from vlapipe.policy import ScriptedSortingPolicy
        plan = P.RulePlanner(env).plan()[:args.rollout]
        policy = ScriptedSortingPolicy(env)
        print(f"\n[롤아웃] 서브태스크 {len(plan)}개를 실제로 실행한다")
        for st in plan:
            policy.set_subtask(st)
            obs, n = env.get_obs(), 0
            while n < args.max_steps and not policy.done:
                obs, _, _, _ = env.step(policy.select_action(obs))
                n += 1
            good = env.zone_of(st["object"]) == st["target_zone"]
            fails += not good
            print(f"   [{st['step']}] {st['instruction']}")
            print(f"        -> {env.zone_of(st['object'])} "
                  f"{'성공' if good else '실패'}  {n}스텝"
                  f"{'' if not policy.failure else ' (' + policy.failure + ')'}")
        Image.fromarray(env.render("agentview", 512, 512)).save(OUT / "after_rollout.png")
        print(f"   결과 그림: {OUT / 'after_rollout.png'}")
    env.close()
    return fails


def main() -> int:
    args = parse_args()
    fails = check_text(args)
    if not args.no_render:
        try:
            fails += check_render(args)
        except ImportError as e:
            print(f"\n[렌더 검사 건너뜀] MuJoCo 를 불러올 수 없습니다: {e}")
            print("  설치는 SETUP_STUDENT.md 3절을 보세요.")
    print("\n" + "=" * 66)
    print("실패 없음 — 숫자 조건 준비 완료" if fails == 0 else f"실패 {fails}건 — 위 '!!' 줄을 보세요")
    print("=" * 66)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
