"""Install a tiny local release fixture; verify hashes and existing-model preservation."""
import _bootstrap  # noqa: F401
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
from download_models import download

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    content=b'not-a-real-weight: download integrity fixture'
    info=json.dumps({'release':'fixture','role':'vlm'}).encode()
    config=b'{}'
    files={n:{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)} for n,data in [('model.safetensors',content),('model_info.json',info),('config.json',config)]}
    archive=root/'bundle.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('vlm/pretrained_model/model.safetensors',content)
        z.writestr('vlm/pretrained_model/model_info.json',info)
        z.writestr('vlm/pretrained_model/config.json',config)
        z.writestr('vlm/pretrained_model/files_manifest.json',json.dumps(files))
    manifest={'release':'fixture','models':{'vlm':{'url':archive.as_uri(),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}}}
    path=root/'manifest.json'
    path.write_text(json.dumps(manifest))
    download(path,root/'models',['vlm'])
    weight=root/'models/vlm/pretrained_model/model.safetensors'
    # A model set can reference an unchanged asset from an earlier release.
    manifest['release']='fixture-set-2'
    manifest['models']['vlm']['bundle_release']='fixture'
    path.write_text(json.dumps(manifest))
    download(path,root/'models',['vlm'])
    weight.write_bytes(b'y'*len(content))
    download(path,root/'models',['vlm'],upgrade=True)
    assert weight.read_bytes()==content
    backups=list((root/'models/_previous').glob('vlm_*/pretrained_model/model.safetensors'))
    assert len(backups)==1 and backups[0].read_bytes()==b'y'*len(content)
    weight=root/'models/vlm/pretrained_model/model.safetensors'
    weight.write_bytes(b'x'*len(content))
    try:
        download(path,root/'models',['vlm'])
        raise AssertionError('Installed corruption silently accepted')
    except ValueError:
        pass
    weight.write_bytes(content)
    assert (root/'models/vlm/pretrained_model/model.safetensors').read_bytes()==content
    download(path,root/'models',['vlm'])
    manifest['models']['vlm']['sha256']='0'*64
    path.write_text(json.dumps(manifest))
    try:
        download(path,root/'bad_models',['vlm'])
        raise AssertionError('Bad archive digest accepted')
    except ValueError:
        pass
    assert not (root/'bad_models/vlm').exists()
    manifest['models']['vlm']['bundle_release']='fixture-next'
    path.write_text(json.dumps(manifest))
    try:
        download(path,root/'models',['vlm'],upgrade=True)
        raise AssertionError('Corrupt upgrade was installed')
    except ValueError:
        pass
    assert weight.read_bytes()==content
    assert len(list((root/'models/_previous').iterdir()))==1
print('PASS: archive/file hashes, repeat install, corruption rejected, per-asset release, verified upgrade preserves old weights')
