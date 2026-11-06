"""Actual short diagnostic trials: whole-task A, multi-C aggregation and cohort separation."""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path
import numpy as np
from vlapipe.final_project_v2.common import dump,public_task,task_text
from vlapipe.final_project_v2.runner import TrialConfig,run_trial
from vlapipe.final_project_v2.scenes import freeze,template,DEFAULT_COMMAND
from vlapipe.final_project_v2.results import aggregate

class Idle:
    def __init__(self):self.tasks=[]
    def reset(self):pass
    def select_action(self,obs):
        self.tasks.append(obs['task'])
        return np.array([0,0,0,0,0,0,-1],dtype='float32')
class Planner:
    def __init__(self):self.calls=[]
    def generate(self,image,prompt,deadline):
        self.calls.append(prompt)
        return json.dumps({'assessments':[],'plan':[{'object':'object_1','target_zone':'right_zone'}],'done':False})

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
args.out.mkdir(parents=True,exist_ok=False)
project=Path(__file__).resolve().parents[1]
spec=freeze(template('open',seed=771))
full_instruction=task_text(public_task(DEFAULT_COMMAND))
scene=args.out/'scene.json';dump(scene,spec)
reports=[]
for shared in [False,True]:
    methods=[('A',None),('B',None)]
    if shared:methods += [('C',project/'final_project_v2'/f'student_strategy_c{i}_v2.py') for i in range(3)]
    for condition,strategy in methods:
        name=('shared_' if shared else 'separate_')+(strategy.stem if strategy else condition)
        policy=Idle();planner=Planner()
        cfg=TrialConfig(condition=condition,shared_vla=shared,seconds=3,max_steps=1,action_steps=1,
            observer_steps=1,max_calls=1,video=False,strategy=str(strategy) if strategy else None)
        report=run_trial(scene,args.out/name,cfg,policy_adapter=policy,planner_adapter=planner)
        assert report['status']=='completed' and report['diagnostic']
        assert len(planner.calls)==(0 if condition=='A' else 1)
        assert policy.tasks and (policy.tasks[0]==full_instruction if condition=='A' else policy.tasks[0]!=full_instruction)
        reports.append(report)
shared_prompts={r['initial_prompt_sha256'] for r in reports if r['configuration']['shared_vla'] and r['condition'] in 'BC'}
assert len(shared_prompts)==1
summary=aggregate(args.out,include_diagnostic=True)
assert len(summary['groups'])==7 and not summary['warnings']
assert all(group['n']==1 for group in summary['groups'])
assert sum(group['condition'].startswith('C:') for group in summary['groups'])==3
assert sum(row['comparison'].endswith(' minus B') for row in summary['paired'])==3
assert sum(row['comparison']=='B minus A' for row in summary['paired'])==2
source=args.out/'shared_student_strategy_c0_v2/report.json'
duplicate=args.out/'duplicate/report.json';duplicate.parent.mkdir();duplicate.write_bytes(source.read_bytes())
try:
    try:
        aggregate(args.out,include_diagnostic=True)
        raise AssertionError('Repeated same-strategy execution silently pooled')
    except ValueError as error:
        assert 'Duplicate' in str(error)
finally:
    duplicate.unlink()
aggregate(args.out,include_diagnostic=True)
print('PASS: A keeps full task and zero VLM calls; shared/separate cohorts stay apart; C0/C1/C2 remain separate and pair with B; duplicate same strategy rejected. Diagnostic adapters only.')
