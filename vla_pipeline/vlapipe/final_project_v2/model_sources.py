"""Local-first selection. Invalid local checkpoints never silently switch models."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2]
DEFAULT_HUB={'vlm':'HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
             'vla_single':'lerobot/smolvla_base','vla_hier':'lerobot/smolvla_base'}

@dataclass(frozen=True)
class ModelChoice:
    role: str
    source: str
    location: str
    reason: str
    def to_dict(self):
        return asdict(self)

def resolve_model(role, *, source='auto', reference=None, models_dir=None, hub_id=None):
    if role not in DEFAULT_HUB or source not in ('auto','local','hub'):
        raise ValueError('Choose role vlm/vla_single/vla_hier and source auto/local/hub')
    root=Path(models_dir).expanduser() if models_dir else PROJECT/'models'
    if not root.is_absolute():
        root=PROJECT/root
    registry=root/'model_sources.json'
    settings=json.loads(registry.read_text(encoding='utf-8')) if registry.is_file() else {}
    hub_id=hub_id or settings.get(role,{}).get('hub_id') or DEFAULT_HUB[role]
    local=root/role/'pretrained_model'
    if reference:
        candidate=Path(reference).expanduser()
        if not candidate.is_absolute():
            candidate=PROJECT/candidate
        if candidate.is_dir():
            if source=='hub':
                raise ValueError('Hub mode expects a Hub ID, not a local checkpoint directory')
            local=candidate
        elif source=='local' or Path(reference).is_absolute() or str(reference).startswith(('.', 'models/', 'outputs/')):
            raise FileNotFoundError('Explicit model directory does not exist: '+str(candidate))
        else:
            return ModelChoice(role,'hub',str(reference),'explicit Hub ID')
    has_config=(local/'config.json').is_file()
    has_weights=any(local.glob('*.safetensors')) or any(local.glob('*.safetensors.index.json')) or (local/'pytorch_model.bin').is_file()
    if source!='hub' and has_config and has_weights:
        return ModelChoice(role,'local',str(local.resolve()),'explicit directory' if reference else 'models directory preferred')
    if source!='hub' and (has_config or has_weights):
        raise ValueError('Incomplete local checkpoint; repair it instead of silently using Hub: '+str(local))
    if source=='local':
        raise FileNotFoundError('Local model missing: '+str(local)+'. Download the model release first.')
    return ModelChoice(role,'hub',hub_id,'explicit Hub mode' if source=='hub' else 'local model absent')
