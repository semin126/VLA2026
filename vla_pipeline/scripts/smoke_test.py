"""설치가 끝난 뒤 파이프라인 전 구간이 실제로 도는지 한 번에 확인한다.

check_env.py 가 '설치되어 있는가'를 보는 반면, 이 스크립트는 '실제로 돌아가는가'를 본다.
사람의 조작(뷰어·게임패드) 없이 끝까지 자동으로 진행되므로 AI 에이전트도 그대로 쓸 수 있다.

  python scripts/smoke_test.py            # 기본 5단계 (torch 없이도 돎, 1~2분)
  python scripts/smoke_test.py --vlm      # SmolVLM2 로드+추론까지 (torch 필요, 최초 1GB 다운로드)

검사 항목:
  1) 씬·렌더      MJCF 생성, agentview 렌더, 물체 배치
  2) IK·제어      스크립트 정책으로 서브태스크 1개 실행
  3) 녹화         EpisodeWriter 로 HDF5 저장 후 다시 읽기
  4) 역재생       부호 반전 액션을 적분해 원본 시작점으로 돌아오는지 (mm 단위)
  5) 데이터 검증  to_lerobot.validate 로 학습 데이터 규격 확인
  6) VLM (--vlm)  SmolVLM2 로드 + 이미지 1장 추론

결과물은 outputs/smoke/ 에 남는다. 연구용 데이터(data/)는 건드리지 않는다.
"""
import _bootstrap  # noqa: F401
import argparse
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

from vlapipe import config as C
from vlapipe import to_lerobot
from vlapipe.env import SortingEnv
from vlapipe.episode import Episode, EpisodeWriter
from vlapipe.planner import RulePlanner
from vlapipe.policy import ScriptedSortingPolicy
from vlapipe.reverse import check_reversibility, reverse_episode

# 역재생 적분 오차 허용치. 액션 적분은 원리상 정확히 0 이어야 하고(README 3절),
# 작업공간 클리핑이 걸린 경우에만 커진다. 0.5mm 를 넘으면 뭔가 잘못된 것이다.
REVERSE_TOL_MM = 0.5


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", action="store_true", help="SmolVLM2 로드·추론까지 확인 (torch 필요)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=128, help="스모크 테스트는 작게 (기본 128)")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", default=str(C.OUTPUT_DIR / "smoke"))
    ap.add_argument("--keep", action="store_true", help="끝나고 outputs/smoke 를 지우지 않는다 (기본 유지)")
    return ap.parse_args()


class Checks:
    def __init__(self):
        self.rows: list[tuple[str, bool, str]] = []

    def run(self, name, fn):
        t0 = time.perf_counter()
        try:
            detail = fn() or ""
            ok = True
        except Exception as e:  # 실패해도 나머지 검사는 계속한다
            detail = f"{type(e).__name__}: {e}"
            ok = False
            traceback.print_exc()
        dt = time.perf_counter() - t0
        mark = " OK " if ok else "실패"
        print(f"  [{mark}] {name}  ({dt:.1f}s)")
        if detail:
            print(f"         {detail}")
        self.rows.append((name, ok, detail))
        return ok

    @property
    def failed(self):
        return [n for n, ok, _ in self.rows if not ok]


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"스모크 테스트 시작 — 결과물 {out}\n")
    ck = Checks()
    state: dict = {}

    # ---- 1. 씬 · 렌더 -----------------------------------------------------
    def step_scene():
        env = SortingEnv(layout="messy", seed=args.seed, image_size=args.image_size)
        img = env.render("agentview")
        assert img.shape == (args.image_size, args.image_size, 3), f"렌더 shape {img.shape}"
        assert img.std() > 1.0, "렌더 이미지가 단색입니다 (그래픽 백엔드 문제)"
        misplaced = env.misplaced_objects()
        assert misplaced, "messy 씬인데 잘못 놓인 물체가 없습니다"
        state["env"] = env
        from PIL import Image
        Image.fromarray(env.render("agentview", 256, 256)).save(out / "1_scene.png")
        return f"물체 {len(C.OBJECT_NAMES)}개, 잘못 놓인 것 {len(misplaced)}개, 평균밝기 {img.mean():.1f}"

    if not ck.run("1. 씬 생성 · 렌더링", step_scene):
        print("\n씬을 못 만들면 나머지는 볼 필요가 없습니다. 중단합니다.")
        return 1
    env = state["env"]

    # ---- 2. IK · 제어 + 3. 녹화 ------------------------------------------
    # 한 번의 롤아웃으로 두 가지를 함께 본다: 제어가 되는가, 녹화가 되는가.
    def step_control():
        plan = RulePlanner(env).plan("테이블을 정리하시오")
        assert plan, "규칙 플래너가 빈 계획을 냈습니다"
        st = plan[0]
        policy = ScriptedSortingPolicy(env)
        policy.set_subtask(st)

        writer = EpisodeWriter(env, task=st["instruction"], direction="mess",
                               cameras=C.CAMERA_NAMES, meta={"smoke_test": True})
        writer.record_obs()
        obs, n = env.get_obs(), 0
        while n < args.max_steps and not policy.done:
            a = policy.select_action(obs)
            obs, _, _, _ = env.step(a)
            writer.record_action(a)
            writer.record_obs()
            n += 1

        assert n >= 5, f"제어 스텝이 {n}개뿐입니다"
        state["writer"] = writer
        state["subtask"] = st
        zone = env.zone_of(st["object"])
        return (f"'{st['instruction'][:52]}...' {n}스텝, "
                f"최종 구역 {zone} (목표 {st['target_zone']})")

    ck.run("2. IK · 제어 (스크립트 정책)", step_control)

    def step_record():
        writer = state["writer"]
        p = writer.save(out / "smoke_ep.hdf5")
        ep = Episode(p)                       # 다시 읽어서 왕복 확인
        assert len(ep) == len(writer), f"저장 {len(writer)} / 재읽기 {len(ep)} 프레임 불일치"
        assert len(ep.obs["robot0_joint_pos"]) == len(ep) + 1, "관측이 액션보다 하나 많아야 합니다"
        sv = ep.state_vectors()
        assert sv.shape[1] == C.STATE_DIM, f"state 차원 {sv.shape[1]} != {C.STATE_DIM}"
        assert ep.actions.shape[1] == C.ACTION_DIM, f"action 차원 {ep.actions.shape[1]}"
        state["ep"] = ep
        mb = p.stat().st_size / 1024 / 1024
        return (f"{len(ep)}프레임, state {sv.shape}, action {ep.actions.shape}, "
                f"{mb:.1f}MB, 카메라 {list(ep.cameras)}")

    ck.run("3. 녹화 (HDF5 저장 → 재읽기)", step_record)

    # ---- 4. 역재생 --------------------------------------------------------
    def step_reverse():
        ep = state["ep"]
        rev = reverse_episode(ep, task="Tidy up the table.")
        assert len(rev) == len(ep), "역재생 길이가 다릅니다"
        assert rev.direction == "tidy", f"direction 이 {rev.direction} 입니다"

        # 핵심 성질: 역방향 액션을 끝점에서 적분하면 원본 시작점으로 돌아와야 한다.
        drift = ep.target_pos[-1] + rev.actions[:, :3].sum(axis=0) * C.MAX_DPOS - ep.target_pos[0]
        drift_mm = float(np.linalg.norm(drift)) * 1000.0
        assert drift_mm < REVERSE_TOL_MM, f"적분 오차 {drift_mm:.3f}mm > {REVERSE_TOL_MM}mm"

        # 그리퍼는 부호 반전이 아니라 재생성 — -1/+1 만 나와야 한다.
        assert set(np.unique(rev.actions[:, 6])) <= {-1.0, 1.0}, "그리퍼 액션이 -1/+1 이 아닙니다"

        p = rev.save(out / "smoke_ep_reversed.hdf5")
        state["rev_path"] = p

        # 던지기/낙하 검출기도 호출해 본다 (스크립트 정책은 이송 중 낙하가 있을 수 있으므로
        # 결과로 성패를 가르지는 않고 보고만 한다).
        import mujoco
        adr = {n: int(env.model.jnt_qposadr[
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_freejoint")])
            for n in C.OBJECT_NAMES}
        rv = check_reversibility(ep, adr)
        state["rv"] = rv
        return (f"적분 오차 {drift_mm:.3f}mm (허용 {REVERSE_TOL_MM}mm), "
                f"역재생 위험 프레임 {rv['n_flagged']}개")

    ck.run("4. 역재생 (부호 반전 + 그리퍼 재생성)", step_reverse)

    # ---- 5. 데이터 검증 ---------------------------------------------------
    def step_validate():
        val = to_lerobot.validate([state["rev_path"]])
        frames = to_lerobot.episode_frames(Episode(state["rev_path"]))
        feats = to_lerobot.build_features(Episode(state["rev_path"]))
        assert frames, "LeRobot 프레임이 비었습니다"
        f0 = frames[0]
        for k in (to_lerobot.KEY_STATE, to_lerobot.KEY_ACTION, to_lerobot.KEY_IMAGE,
                  to_lerobot.KEY_WRIST):
            assert k in f0, f"프레임에 {k} 가 없습니다"
        assert not val["problems"], "; ".join(val["problems"])
        return (f"에피소드 {val['n_episodes']}개 / 프레임 {val['n_frames']}개, "
                f"해상도 {val['image_shapes']}, 피처 {len(feats)}종")

    ck.run("5. 학습 데이터 규격 검증", step_validate)

    # ---- 6. VLM (선택) ----------------------------------------------------
    if args.vlm:
        def step_vlm():
            from vlapipe.planner import VLMPlanner, plan_accuracy
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            t0 = time.perf_counter()
            vlm = VLMPlanner(device=dev)
            load_s = time.perf_counter() - t0
            env2 = SortingEnv(layout="messy", seed=args.seed, image_size=C.IMAGE_SIZE)
            t0 = time.perf_counter()
            plan = vlm.plan(env2.render("agentview"), "테이블을 정리하시오")
            infer_s = time.perf_counter() - t0
            gold = RulePlanner(env2).plan("테이블을 정리하시오")
            acc = plan_accuracy(plan, gold)
            env2.close()
            # 정확도는 낮은 것이 정상이다(README 7절). 여기서 보는 것은 '돌아가는가' 뿐.
            assert isinstance(plan, list), "plan 이 리스트가 아닙니다"
            return (f"device={dev}, 로드 {load_s:.1f}s, 추론 {infer_s:.1f}s, "
                    f"단계 {len(plan)}개, 정답 {acc['correct']}/{acc['n_gold']} "
                    f"(정확도가 낮은 것은 정상 — README 7절)")

        ck.run("6. VLM 플래너 (SmolVLM2)", step_vlm)
    else:
        print("  [건너뜀] 6. VLM 플래너 — --vlm 을 붙이면 확인합니다")

    env.close()

    # ---- 요약 -------------------------------------------------------------
    print(f"\n{'=' * 58}")
    n_ok = sum(1 for _, ok, _ in ck.rows if ok)
    print(f"{n_ok}/{len(ck.rows)} 통과")
    if ck.failed:
        print(f"실패: {', '.join(ck.failed)}")
        print("\nREADME.md 9절(윈도우 문제 해결)을 먼저 보세요.")
        return 1
    rv = state.get("rv", {})
    if rv and not rv.get("ok", True):
        print(f"참고: 스크립트 정책 롤아웃에 역재생 위험 프레임 {rv['n_flagged']}개가 있습니다.")
        print("      검출기가 동작한다는 뜻이므로 정상입니다. 실제 텔레옵 녹화에서는")
        print("      던지지 말고 '집어서-옮겨서-내려놓기'로 하세요.")
    print("\n전부 정상입니다. 다음:")
    print("  " + _bootstrap.cmd("scripts/run_pipeline.py", "--layout", "messy", "--seed", "3"))
    print("  " + _bootstrap.cmd("scripts/teleop_record.py", "--input", "keyboard"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
