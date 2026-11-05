"""Local-first selection. Invalid local checkpoints never silently switch models."""
from dataclasses import asdict, dataclass
import json
import re
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2]
DEFAULT_HUB={'vlm':'HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
             'vla_single':'lerobot/smolvla_base','vla_hier':'lerobot/smolvla_base'}
HUB_ID=re.compile(r'^[A-Za-z0-9][\w.-]*(?:/[A-Za-z0-9][\w.-]*)?$')


def verify_local_checkpoint(local,role):
    """Cheap load-time completeness/role check. Installation also checks hashes."""
    local=Path(local)
    config=local/'config.json'
    weights=[p for p in (local/'model.safetensors',local/'pytorch_model.bin') if p.is_file()]
    index=local/'model.safetensors.index.json'
    if index.is_file():
        shards=set(json.loads(index.read_text(encoding='utf-8-sig'))['weight_map'].values())
        if not shards:
            raise ValueError('Empty checkpoint shard index: '+str(index))
        weights.extend(local/name for name in shards)
    if not config.is_file() or config.stat().st_size==0 or not weights or any(not p.is_file() or p.stat().st_size==0 for p in weights):
        raise ValueError('Incomplete local checkpoint; repair it instead of silently using Hub: '+str(local))
    info=local/'model_info.json'
    if info.is_file():
        recorded=json.loads(info.read_text(encoding='utf-8-sig')).get('role')
        if recorded and recorded!=role:
            raise ValueError(f'Checkpoint role {recorded} does not match requested {role}')
    manifest=local/'files_manifest.json'
    if manifest.is_file():
        for rel,record in json.loads(manifest.read_text(encoding='utf-8-sig')).items():
            path=(local/rel).resolve()
            if not path.is_relative_to(local.resolve()) or not path.is_file() or path.stat().st_size!=record['bytes']:
                raise ValueError('Checkpoint differs from files_manifest.json: '+rel)

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
    settings=json.loads(registry.read_text(encoding='utf-8-sig')) if registry.is_file() else {}
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
        elif source=='local' or not HUB_ID.fullmatch(str(reference)) or str(reference).startswith(('.', 'models/', 'outputs/')):
            raise FileNotFoundError('Explicit model directory does not exist: '+str(candidate))
        else:
            return ModelChoice(role,'hub',str(reference),'explicit Hub ID')
    if source!='hub' and local.is_dir() and (reference or any(local.iterdir())):
        verify_local_checkpoint(local,role)
        return ModelChoice(role,'local',str(local.resolve()),'explicit directory' if reference else 'models directory preferred')
    if source=='local':
        raise FileNotFoundError('Local model missing: '+str(local)+'. Download the model release first.')
    return ModelChoice(role,'hub',hub_id,'explicit Hub mode' if source=='hub' else 'local model absent')
