"""Build a portable inference-only model directory (no optimizer or training data)."""
import _bootstrap  # noqa: F401
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def bundle(source, destination, role, *, extra=None):
    source,destination=Path(source).resolve(strict=True),Path(destination).absolute()
    if destination.exists():
        raise FileExistsError('Choose a fresh model bundle directory: '+str(destination))
    destination.mkdir(parents=True)
    for path in source.iterdir():
        if path.name in ('train_config.json','training_state','optimizer.pt','scheduler.pt'):
            continue
        if path.is_dir():
            shutil.copytree(path,destination/path.name)
        elif path.suffix in ('.json','.safetensors','.model','.txt','.jinja','.bin'):
            shutil.copy2(path,destination/path.name)
    info={'role':role,'stage':'preview','source_weights_sha256':None,
          'license':'apache-2.0',
          'scope':'Finetuned weights; consult the release evaluation for actual measured performance.',**(extra or {})}
    for name in ('APACHE-2.0.txt','THIRD_PARTY_NOTICES.md'):
        notice=Path(__file__).resolve().parent.parent/'models'/name
        if notice.is_file():
            shutil.copy2(notice,destination/name)
    if role=='vlm':
        import torch
        from safetensors.torch import load_file,save_file
        tensors=load_file(str(destination/'model.safetensors'),device='cpu')
        tensors={k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in tensors.items()}
        save_file(tensors,str(destination/'model.safetensors'),metadata={'format':'pt'})
        info['inference_storage_dtype']='bfloat16'
    if role.startswith('vla'):
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        config=json.loads((source/'config.json').read_text(encoding='utf-8'))
        backbone=config['vlm_model_name']
        files=Path(backbone) if Path(backbone).is_dir() else Path(snapshot_download(backbone,
            allow_patterns=['*.json','*.txt','*.model','*.jinja','tokenizer*','merges*','vocab*'],local_files_only=True))
        target=destination/'backbone_metadata'
        target.mkdir(exist_ok=True)
        for path in files.iterdir():
            if path.is_file() and path.suffix in ('.json','.txt','.model','.jinja'):
                shutil.copy2(path,target/path.name)
        with safe_open(source/'model.safetensors',framework='pt',device='cpu') as tensors:
            key=next(k for k in tensors.keys() if '.vlm.' in k)
            info['backbone_dtype']={'BF16':'bfloat16','F16':'float16','F32':'float32'}[tensors.get_slice(key).get_dtype()]
        config['vlm_model_name']='backbone_metadata'
        config['load_vlm_weights']=False
        config['pretrained_path']=None
        (destination/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    weights=source/'model.safetensors'
    if weights.is_file():
        with weights.open('rb') as f:
            info['source_weights_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    (destination/'model_info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={}
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            with path.open('rb') as f:
                manifest[path.relative_to(destination).as_posix()]={'bytes':path.stat().st_size,'sha256':hashlib.file_digest(f,'sha256').hexdigest()}
    (destination/'files_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return info

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--role',choices=['vlm','vla_single','vla_hier'],required=True)
    a=p.parse_args()
    print(json.dumps(bundle(a.source,a.out,a.role),indent=2))
