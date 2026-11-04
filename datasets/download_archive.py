"""Download, verify and reassemble the VLM/VLA archive using only Python 3."""
import argparse, hashlib, json, urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--join-only', action='store_true', help='Use all already downloaded .001 etc files')
parser.add_argument('--directory', default='.')
args = parser.parse_args()
base = Path(args.directory).resolve()
base.mkdir(parents=True, exist_ok=True)
manifest_path = base / 'archive_manifest.json'
if not manifest_path.exists():
    urllib.request.urlretrieve('https://github.com/semin126/VLA2026/releases/download/dataset-full-20261104/archive_manifest.json', manifest_path)
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
for part in manifest['parts']:
    path = base / part['name']
    if not path.exists():
        if args.join_only:
            raise SystemExit('Missing part: ' + str(path))
        temp = path.with_name(path.name + '.downloading')
        print('Downloading', part['name'], flush=True)
        urllib.request.urlretrieve(part['url'], temp)
        temp.rename(path)
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    if path.stat().st_size != part['bytes'] or digest != part['sha256']:
        raise SystemExit('Part verification failed: ' + str(path))
    print('Verified', part['name'], flush=True)
target = base / manifest['archive_name']
if target.exists():
    with target.open('rb') as f:
        actual = hashlib.file_digest(f, 'sha256').hexdigest()
    if actual == manifest['archive_sha256']:
        print('Archive already restored and verified:', target)
        raise SystemExit(0)
    raise SystemExit('Destination ZIP already exists with different contents; no files overwritten.')
temp = target.with_name(target.name + '.assembling')
digest = hashlib.sha256()
with temp.open('xb') as out:
    for part in manifest['parts']:
        with (base / part['name']).open('rb') as f:
            while chunk := f.read(8 * 1024**2):
                out.write(chunk)
                digest.update(chunk)
if temp.stat().st_size != manifest['archive_bytes'] or digest.hexdigest() != manifest['archive_sha256']:
    raise SystemExit('Restored archive verification failed')
temp.rename(target)
print('Restored and verified:', target)
