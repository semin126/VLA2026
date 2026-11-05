"""Regression checks for validation-only scenes and the post-plan C budget."""
import _bootstrap  # noqa: F401
import json
from pathlib import Path
import sys
import tempfile

from vlapipe.final_project_v2.notebook_runtime import check_research_budget, prepare_python, select_scene
from vlapipe.final_project_v2.runner import TrialConfig

assert TrialConfig().seconds == 300 and TrialConfig().max_steps == 6000

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    path = root/'validation.json'
    path.write_text('{}', encoding='utf-8')
    registry = {'scenes':[{'split':'validation','path':'validation.json'}]}
    assert select_scene(root, root, registry) == path.resolve()
    assert select_scene(root, root, registry, 'validation.json') == path.resolve()
    try:
        select_scene(root,root,{'scenes':[]})
        raise AssertionError('empty scene list accepted')
    except ValueError:
        pass
    for seconds, steps, calls in [(12,2400,8),(300,2400,8),(300,6000,1)]:
        try:
            check_research_budget(['C'],seconds,steps,240,calls)
            raise AssertionError('C research budget cannot reach replan')
        except ValueError:
            pass
    check_research_budget(['B','C'],300,6000,240,8)
    check_research_budget(['C'],2,4,2,2,smoke=True)
    if (Path(sys.prefix)/'conda-meta').is_dir():
        project=Path(__file__).resolve().parent.parent
        assert Path(prepare_python(project,root)).samefile(sys.executable), 'Selected conda kernel was replaced'
print(json.dumps({'passed':True,'checks':['validation-only/default/custom/empty selection',
    'C post-plan budget and smoke separation','selected conda interpreter respected']},indent=2))
