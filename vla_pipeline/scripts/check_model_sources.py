"""Test local priority, explicit Hub selection and incomplete checkpoint handling."""
import _bootstrap  # noqa: F401
from pathlib import Path
import tempfile
from vlapipe.final_project_v2.model_sources import resolve_model

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    for role in ('vlm','vla_single','vla_hier'):
        assert resolve_model(role,models_dir=root).source=='hub'
        local=root/role/'pretrained_model'
        local.mkdir(parents=True)
        (local/'config.json').write_text('{}')
        try:
            resolve_model(role,models_dir=root)
            raise AssertionError('Incomplete model silently fell back to Hub')
        except ValueError:
            pass
        (local/'model.safetensors').write_bytes(b'format-check-fixture')
        assert resolve_model(role,models_dir=root).source=='local'
        assert resolve_model(role,models_dir=root,source='hub',hub_id='team/compatible_model').location=='team/compatible_model'
        assert resolve_model(role,models_dir=root,source='local').location==str(local.resolve())
        assert resolve_model(role,models_dir=root,reference='team/explicit').source=='hub'
        assert resolve_model(role,models_dir=root,reference=str(local)).source=='local'
    try:
        resolve_model('vlm',models_dir=root/'absent',source='local')
        raise AssertionError('Missing strict-local checkpoint accepted')
    except FileNotFoundError:
        pass
print('PASS: all three roles, local priority, missing/partial models, explicit local/Hub')
