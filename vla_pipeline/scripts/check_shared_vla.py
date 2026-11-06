"""CPU check: explicit shared-mode selection preserves strict checkpoint roles."""
import _bootstrap  # noqa: F401
import json
from pathlib import Path
import tempfile
from vlapipe.final_project_v2.runner import TrialConfig, resolve_trial_vla
from final_project_v2 import parser, trial_config

project=Path(__file__).resolve().parents[1]
scratch=project/'outputs'
scratch.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='check_shared_vla_',dir=scratch) as directory:
    root=Path(directory).resolve()
    assert root.parent==scratch.resolve()
    for role in ['vla_single','vla_hier']:
        folder=root/role/'pretrained_model';folder.mkdir(parents=True)
        (folder/'config.json').write_text('{}')
        (folder/'model.safetensors').write_bytes(b'role-selection fixture; not a neural model')
        (folder/'model_info.json').write_text(json.dumps({'role':role}))
    original=resolve_trial_vla(TrialConfig(condition='A',models_dir=str(root),model_source='local'))
    assert original.role=='vla_single'
    choices=[resolve_trial_vla(TrialConfig(condition=c,shared_vla=True,models_dir=str(root),model_source='local')) for c in 'ABC']
    assert all(c.role=='vla_hier' and c.location==choices[0].location for c in choices)
    try:
        resolve_trial_vla(TrialConfig(condition='A',checkpoint=choices[0].location,model_source='local'))
        raise AssertionError('Default A silently accepted partial-role weights')
    except ValueError:
        pass
    args=parser().parse_args(['suite','--scenes',str(root),'--out',str(root/'results'),
                             '--conditions','A,B','--shared-vla','--models-dir',str(root)])
    assert all(trial_config(args,c).shared_vla for c in 'AB')
    args.a_checkpoint=original.location
    try:
        trial_config(args,'A')
        raise AssertionError('Shared mode accepted a separate A checkpoint')
    except ValueError:
        pass
print('PASS: shared A/B/C selects one partial VLA, default A role guard preserved, conflicting A override rejected; CPU selection only')
