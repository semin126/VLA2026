"""Download release inference bundles into models/ and verify SHA256 before installing."""
import _bootstrap  # noqa: F401
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile
import urllib.request
from urllib.parse import urlparse
import zipfile

PROJECT=Path(__file__).resolve().parent.parent


def verify_bundle(bundle,role,release):
    from vlapipe.final_project_v2.model_sources import verify_local_checkpoint
    verify_local_checkpoint(bundle,role)
    info=json.loads((bundle/'model_info.json').read_text(encoding='utf-8-sig'))
    if info.get('release')!=release or info.get('role')!=role:
        raise ValueError('Release/role does not match the requested bundle')
    files=json.loads((bundle/'files_manifest.json').read_text(encoding='utf-8-sig'))
    actual_files={p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()}
    if actual_files != set(files)|{'files_manifest.json'}:
        raise ValueError('Bundle contains missing or unlisted files')
    for rel,record in files.items():
        path=(bundle/rel).resolve()
        if not path.is_relative_to(bundle.resolve()):
            raise ValueError('Unsafe manifest path')
        with path.open('rb') as f:
            digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=record['sha256']:
            raise ValueError(role+': file mismatch '+rel)

def download(manifest_path,models,roles):
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
    models=Path(models).absolute()
    models.mkdir(parents=True,exist_ok=True)
    for role in roles:
        if role not in ('vlm','vla_single','vla_hier'):
            raise ValueError('Unsupported model role: '+role)
        asset=manifest['models'][role]
        target=models/role
        if target.exists():
            info=target/'pretrained_model/model_info.json'
            if info.is_file() and json.loads(info.read_text(encoding='utf-8-sig')).get('release')==manifest['release']:
                verify_bundle(target/'pretrained_model',role,manifest['release'])
                print(role,'already installed and verified; leaving it unchanged')
                continue
            raise FileExistsError('Existing model is preserved. Choose --models-dir with a new path: '+str(target))
        with tempfile.TemporaryDirectory(prefix='vla_download_',dir=models) as tmp:
            stage=Path(tmp)
            assert stage.resolve().is_relative_to(models.resolve())
            archive=stage/'bundle.zip'
            digest=hashlib.sha256()
            print('Downloading',role,flush=True)
            if urlparse(asset['url']).scheme not in ('https','file'):
                raise ValueError('Use HTTPS for releases or file URLs for local checks')
            with urllib.request.urlopen(asset['url'],timeout=60) as response,archive.open('wb') as f:
                while chunk:=response.read(1024*1024):
                    f.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest()!=asset['sha256']:
                raise ValueError(role+': archive SHA256 mismatch')
            extracted=stage/'unpacked'
            extracted.mkdir()
            with zipfile.ZipFile(archive) as z:
                for item in z.infolist():
                    dest=(extracted/item.filename).resolve()
                    if not dest.is_relative_to(extracted.resolve()) or stat.S_ISLNK(item.external_attr>>16):
                        raise ValueError('Unsafe archive entry: '+item.filename)
                    if item.is_dir():
                        dest.mkdir(parents=True,exist_ok=True)
                    else:
                        dest.parent.mkdir(parents=True,exist_ok=True)
                        with z.open(item) as src,dest.open('wb') as out:
                            shutil.copyfileobj(src,out,length=1024*1024)
            bundle=extracted/role/'pretrained_model'
            verify_bundle(bundle,role,manifest['release'])
            assert target.resolve().is_relative_to(models.resolve())
            (extracted/role).rename(target)
            print('Installed',target,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=PROJECT/'models/release_manifest.json')
    p.add_argument('--models-dir',type=Path,default=PROJECT/'models')
    p.add_argument('--roles',default='vlm,vla_single,vla_hier')
    a=p.parse_args()
    download(a.manifest,a.models_dir,a.roles.split(','))
