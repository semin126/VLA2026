"""Held-out single-transfer evaluation. Ground truth is used only for scoring."""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from vlapipe.policy import SmolVLAPolicy
from vlapipe import config as C
from vlapipe.planner import instruction_text
from vlapipe.final_project_v2.scenes import read_scene,environment
from vlapipe.final_project_v2.metrics import observe_outcome
from vlapipe.final_project_v2.common import task_text,public_task

def evaluate(checkpoint,scenes,out,system='hier',max_steps=300,overfit=False):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    policy=SmolVLAPolicy(str(checkpoint),device='cuda')
    results=[]
    for i,path in enumerate(scenes):
        spec=read_scene(path)
        if (spec['split']!='validation' and not (overfit and spec['split']=='train')) or len(spec['research_targets'])!=1:
            raise ValueError('Use independent validation scenes with one designated transfer target')
        target=spec['research_targets'][0]
        destination=C.OBJECT_BY_NAME[target]['home_zone']
        case=out/spec['id']
        case.mkdir()
        torch.manual_seed(42+i)
        policy.reset()
        started=time.perf_counter()
        with environment(spec,image_size=policy.image_size) as env:
            initial=env.object_pos(target).copy()
            initial_score=observe_outcome(env)
            if initial_score['objects'][target]['correct']:
                raise ValueError('Target is already correctly placed; would inflate transfer success')
            Image.fromarray(env.render('agentview',512,512)).save(case/'before.png')
            task=instruction_text(target,destination)[0] if system=='hier' else task_text(public_task(spec['command']))
            lift=False
            transported=False
            stable=0
            for step in range(max_steps):
                action=policy.select_action({**env.get_obs(),'task':task})
                env.step(action)
                pos=env.object_pos(target)
                lift=lift or bool(pos[2]>initial[2]+.04)
                transported=transported or (lift and float(np.linalg.norm(pos[:2]-initial[:2]))>.06)
                final=observe_outcome(env)
                stable=stable+1 if final['objects'][target]['correct'] else 0
                if stable>=5:
                    break
            placed=bool(stable>=5)
            damaged=[n for n,d in initial_score['objects'].items() if n!=target and d['correct'] and not final['objects'][n]['correct']]
            success=bool(lift and transported and placed)
            Image.fromarray(env.render('agentview',512,512)).save(case/'after.png')
            row={'scene':spec['id'],'seed':spec['seed'],'object':target,'system':system,
                 'success':success,'lifted':lift,'transported':transported,'placed':placed,
                 'failure_stage':None if success else 'grasp' if not lift else 'transport' if not transported else 'placement',
                 'steps':step+1,'wall_seconds':time.perf_counter()-started,'final_scene_acc':final['acc'],
                 'damaged_other_objects':damaged,
                 'note':'Training overfit diagnostic, not generalization' if overfit else 'Unpaced skill diagnostic, not A-E wall-time benchmark'}
        (case/'report.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        results.append(row)
        summary={'n':len(results),'successes':sum(r['success'] for r in results),
                 'transfer_success_rate':sum(r['success'] for r in results)/len(results),
                 'checkpoint':str(checkpoint),'system':system,'max_steps':max_steps,'overfit_diagnostic':overfit,'trials':results}
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(spec['id'],row['success'],row['failure_stage'],'seconds',round(row['wall_seconds'],1),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--scenes',type=Path,required=True,help='Directory containing frozen validation single-transfer JSONs')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--system',choices=['single','hier'],default='hier')
    p.add_argument('--max-steps',type=int,default=300)
    p.add_argument('--allow-train-overfit',action='store_true')
    p.add_argument('--limit',type=int)
    a=p.parse_args()
    files=sorted(a.scenes.glob('release_train_single_*.json' if a.allow_train_overfit else 'release_validation_single_*.json'))
    if a.limit:
        files=files[:a.limit]
    if not files:
        raise SystemExit('No held-out single-transfer scenes found')
    print(json.dumps(evaluate(a.checkpoint,files,a.out,a.system,a.max_steps,a.allow_train_overfit),indent=2))
