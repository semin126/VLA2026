"""Held-out single-transfer evaluation. Ground truth is used only for scoring."""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path
import time
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image

from vlapipe.policy import SmolVLAPolicy
from vlapipe import config as C
from vlapipe.planner import instruction_text
from vlapipe.final_project_v2.scenes import read_scene,environment
from vlapipe.final_project_v2.metrics import observe_outcome
from vlapipe.final_project_v2.common import task_text,public_task
from vlapipe.metrics import wilson_interval

def evaluate(checkpoint,scenes,out,system='hier',max_steps=240,overfit=False,cached_actions=None,
             evaluation_split='validation',record_video=False,full_horizon=False):
    if max_steps < 1:
        raise ValueError('max_steps must be positive')
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    policy=SmolVLAPolicy(str(checkpoint),device='cuda')
    if cached_actions is not None:
        if not 1 <= cached_actions <= policy.policy.config.chunk_size:
            raise ValueError('cached_actions must be within the trained chunk size')
        policy.policy.config.n_action_steps=cached_actions
        policy.reset()
    results=[]
    for i,path in enumerate(scenes):
        spec=read_scene(path)
        if (spec['split']!=evaluation_split and not (overfit and spec['split']=='train')) or len(spec['research_targets'])!=1:
            raise ValueError('Use independent validation scenes with one designated transfer target')
        target=spec['research_targets'][0]
        destination=C.OBJECT_BY_NAME[target]['home_zone']
        case=out/spec['id']
        case.mkdir()
        torch.manual_seed(42+i)
        policy.reset()
        started=time.perf_counter()
        import imageio.v2 as imageio
        with environment(spec,image_size=policy.image_size) as env, (
            imageio.get_writer(case/'process.mp4',fps=C.CONTROL_HZ/2) if record_video else nullcontext()) as movie:
            initial=env.object_pos(target).copy()
            initial_score=observe_outcome(env)
            if initial_score['objects'][target]['correct']:
                raise ValueError('Target is already correctly placed; would inflate transfer success')
            Image.fromarray(env.render('agentview',512,512)).save(case/'before.png')
            task=instruction_text(target,destination)[0] if system=='hier' else task_text(public_task(spec['command']))
            lift=False
            transported=False
            stable=0
            wrong_lifted=set()
            other_z={n:float(env.object_pos(n)[2]) for n in C.OBJECT_NAMES if n!=target}
            trajectory=[]
            first_success=None
            for step in range(max_steps):
                policy.set_controller_target(env.ctrl.target_pos)
                action=policy.select_action({**env.get_obs(),'task':task})
                env.step(action)
                if movie is not None and step%2==0:
                    movie.append_data(np.concatenate([env.render('agentview',512,512),env.render('frontview',512,512)],axis=1))
                pos=env.object_pos(target)
                lift=lift or bool(pos[2]>initial[2]+.04)
                transported=bool(transported or (pos[2]>initial[2]+.02 and float(np.linalg.norm(pos[:2]-initial[:2]))>.06))
                wrong_lifted.update(n for n,z in other_z.items() if env.object_pos(n)[2]>z+.04)
                final=observe_outcome(env)
                trajectory.append({'step':step,'action':action.tolist(),'eef':env.ctrl.eef_pos().tolist(),
                    'control_target':env.ctrl.target_pos.tolist(),'object_position':pos.tolist(),
                    'finger_qpos':env.data.qpos[env._finger_adr].tolist()})
                stable=stable+1 if final['objects'][target]['correct'] else 0
                if lift and transported and stable>=5:
                    if first_success is None:
                        first_success=step+1
                    if not full_horizon:
                        break
            placed=bool(stable>=5)
            damaged=[n for n,d in initial_score['objects'].items() if n!=target and d['correct'] and not final['objects'][n]['correct']]
            success=bool(lift and transported and placed)
            detail=final['objects'][target]
            stage=(None if success else 'regressed_after_success' if first_success is not None else
                   'wrong_object' if not lift and wrong_lifted else
                   'pushed_not_lifted' if not lift and placed else 'grasp' if not lift else
                   'transport' if not transported else 'release' if detail['held'] else
                   'dropped_or_misplaced' if detail['zone']!=destination else 'unstable')
            Image.fromarray(env.render('agentview',512,512)).save(case/'after.png')
            row={'scene':spec['id'],'seed':spec['seed'],'object':target,'system':system,
                 'success':success,'lifted':lift,'transported':transported,'placed':placed,
                 'failure_stage':stage,'strict_success':bool(success and not damaged and not wrong_lifted),
                 'wrong_objects_lifted':sorted(wrong_lifted),
                 'initial_scene_acc':initial_score['acc'],'scene_acc_gain':final['acc']-initial_score['acc'],
                 'first_success_step':first_success,'lost_after_success':bool(first_success is not None and not success),
                 'steps':step+1,'wall_seconds':time.perf_counter()-started,'final_scene_acc':final['acc'],
                 'damaged_other_objects':damaged,
                 'note':'Training overfit diagnostic, not generalization' if overfit else 'Unpaced skill diagnostic, not A-E wall-time benchmark'}
        (case/'trajectory.json').write_text(json.dumps(trajectory),encoding='utf-8')
        (case/'report.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        results.append(row)
        summary={'n':len(results),'successes':sum(r['success'] for r in results),
                 'transfer_success_rate':sum(r['success'] for r in results)/len(results),
                 'strict_success_rate':sum(r['strict_success'] for r in results)/len(results),
                 'transfer_wilson_95':list(wilson_interval(sum(r['success'] for r in results),len(results))),
                 'strict_wilson_95':list(wilson_interval(sum(r['strict_success'] for r in results),len(results))),
                 'checkpoint':str(checkpoint),'system':system,'max_steps':max_steps,
                 'cached_actions':policy.policy.config.n_action_steps,'overfit_diagnostic':overfit,'evaluation_split':evaluation_split,'trials':results,
                 'action_representation':policy.action_representation,
                 'full_horizon':full_horizon,
                 'scope':'Motor diagnostic; single central target does not establish language/number grounding'}
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(spec['id'],row['success'],row['failure_stage'],'seconds',round(row['wall_seconds'],1),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--scenes',type=Path,required=True,help='Directory containing frozen validation single-transfer JSONs')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--system',choices=['single','hier'],default='hier')
    p.add_argument('--max-steps',type=int,default=240)
    p.add_argument('--cached-actions',type=int,help='Explicit diagnostic override; record and keep identical in B/C')
    p.add_argument('--scene-pattern',help='Explicit filename glob for a separate motor validation/qualification suite')
    p.add_argument('--evaluation-split',choices=['validation','sft_test'],default='validation')
    p.add_argument('--video',action='store_true')
    p.add_argument('--full-horizon',action='store_true',help='Continue after first placement to test whether the policy undoes its work')
    p.add_argument('--allow-train-overfit',action='store_true')
    p.add_argument('--limit',type=int)
    a=p.parse_args()
    files=sorted(a.scenes.glob(a.scene_pattern or ('release_train_single_*.json' if a.allow_train_overfit else 'release_validation_single_*.json')))
    if a.limit:
        files=files[:a.limit]
    if not files:
        raise SystemExit('No held-out single-transfer scenes found')
    print(json.dumps(evaluate(a.checkpoint,files,a.out,a.system,a.max_steps,a.allow_train_overfit,a.cached_actions,a.evaluation_split,a.video,a.full_horizon),indent=2))
