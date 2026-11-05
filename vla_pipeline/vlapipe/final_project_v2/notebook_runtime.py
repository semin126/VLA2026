"""Standard-library notebook helpers; simulator/ML run in an explicit Python 3.12 subprocess."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


def prepare_python(project, work, *, install=False, python_override=''):
    project, work = Path(project), Path(work)
    if python_override:
        py = Path(python_override)
    elif platform.system() == 'Windows':
        py = project / '.venv' / 'Scripts' / 'python.exe'
        if not py.is_file():
            if not install:
                raise RuntimeError('Create the local environment first; see SETUP_STUDENT.md')
            subprocess.run(['py', '-3.12', '-m', 'venv', str(project / '.venv')], check=True)
    else:
        py = work / 'python312' / 'bin' / 'python'
        if not py.is_file():
            if not install:
                raise RuntimeError('Select INSTALL=True for the first Colab environment setup')
            if not shutil.which('uv'):
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv==0.13.7'], check=True)
            subprocess.run(['uv', 'python', 'install', '3.12'], check=True)
            subprocess.run(['uv', 'venv', '--python', '3.12', str(py.parent.parent)], check=True)
    if install:
        requirements = project / ('requirements-windows.txt' if platform.system() == 'Windows' else 'requirements-colab-lock.txt')
        if shutil.which('uv'):
            cmd = ['uv', 'pip', 'install', '--python', str(py), '-r', str(requirements)]
            if platform.system() != 'Windows':
                cmd += ['--index-strategy', 'unsafe-best-match', '--extra-index-url', 'https://download.pytorch.org/whl/cu128']
        else:
            cmd = [str(py), '-m', 'pip', 'install', '-r', str(requirements)]
        subprocess.run(cmd, check=True, cwd=project)
    subprocess.run([str(py), '-c', 'import sys,mujoco,numpy; print(sys.version); print("mujoco",mujoco.__version__)'], check=True)
    if not (project/'assets/franka_emika_panda/panda_vla.xml').is_file():
        subprocess.run([str(py), str(project/'setup_assets.py')], cwd=project, check=True)
    # Resolving a POSIX venv's python symlink selects the base interpreter and
    # discards the venv's site-packages in subsequent subprocesses.
    return str(py.absolute())


class NotebookRun:
    def __init__(self, project, work, backup=None, team='team01'):
        self.project = Path(project).resolve()
        ident = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
        self.root = Path(work) / team / ident
        self.root.mkdir(parents=True, exist_ok=False)
        self.backup = Path(backup) / team / ident if backup else None
        if self.backup:
            self.backup.mkdir(parents=True, exist_ok=False)
            probe = self.backup / 'WRITE_TEST.txt'
            probe.write_text(ident, encoding='utf-8')
            assert probe.read_text(encoding='utf-8') == ident
        self.commands = []
        self._hash_cache = {}
        self.python = None
        # check_env imports mujoco before env.py. Select EGL in the parent before
        # launching any child, rather than relying on env.py's later setup.
        from ..render_setup import configure_headless_rendering
        rendering = configure_headless_rendering(self.root/'egl')
        (self.root/'renderer_setup.json').write_text(json.dumps(rendering, indent=2), encoding='utf-8')
        self.snapshot()

    def snapshot(self):
        manifest = {}
        with tarfile.open(self.root / 'source_snapshot.tar.gz', 'w:gz') as archive:
            for directory, dirs, files in os.walk(self.project):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('outputs', 'data', '__pycache__')
                           and not (Path(directory)/d).is_symlink()
                           and not getattr(Path(directory)/d, 'is_junction', lambda: False)()]
                for name in files:
                    path = Path(directory) / name
                    if path.suffix not in ('.py', '.md', '.txt', '.ipynb', '.json', '.xml', '.png', '.stl', '.obj') or path.is_symlink():
                        continue
                    if path.stat().st_size > 20_000_000:
                        continue
                    rel = path.relative_to(self.project).as_posix()
                    manifest[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
                    archive.add(path, arcname=rel, recursive=False)
        (self.root / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        self.sync()

    def sync(self):
        def sha256(path):
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    h.update(chunk)
            return h.hexdigest()
        manifest = {}
        paths = []
        for directory, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if not (Path(directory)/d).is_symlink()
                       and not getattr(Path(directory)/d, 'is_junction', lambda: False)()]
            for name in files:
                path = Path(directory) / name
                if path.is_symlink():
                    continue
                rel = path.relative_to(self.root)
                if rel.as_posix() in ('artifact_manifest.json', 'ARTIFACTS.md'):
                    continue
                signature = (path.stat().st_size, path.stat().st_mtime_ns)
                cached = self._hash_cache.get(rel.as_posix())
                sha = cached[1] if cached and cached[0] == signature else sha256(path)
                manifest[rel.as_posix()] = {'sha256': sha, 'bytes': signature[0]}
                paths.append(path)
                if not self.backup:
                    self._hash_cache[rel.as_posix()] = (signature, sha)
                    continue
                dest = self.backup / rel
                dst_signature = (dest.stat().st_size, dest.stat().st_mtime_ns) if dest.is_file() else None
                cached_dst = cached[2] if cached and len(cached) > 2 else None
                if not cached or cached[0] != signature or dst_signature != cached_dst:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_name(dest.name+'.copying')
                    shutil.copyfile(path, tmp)
                    if sha256(tmp) != sha:
                        raise IOError('Backup hash mismatch: ' + str(rel))
                    tmp.replace(dest)
                self._hash_cache[rel.as_posix()] = (signature, sha, (dest.stat().st_size, dest.stat().st_mtime_ns))
        (self.root/'artifact_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        links = ['# Final project ver2.0 artifacts', '']
        links += [f'- [{rel}]({rel})' for rel in sorted(manifest)]
        (self.root/'ARTIFACTS.md').write_text('\n'.join(links)+'\n', encoding='utf-8')
        if self.backup:
            for name in ('artifact_manifest.json', 'ARTIFACTS.md'):
                shutil.copyfile(self.root/name, self.backup/name)
            (self.backup/'backup_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    def run(self, script, *args):
        if not self.python:
            raise RuntimeError('Set run.python to the prepared Python executable')
        command = [self.python, str(self.project / script), *map(str, args)]
        log = self.root / 'logs' / f'{len(self.commands):03d}_{Path(script).stem}.log'
        log.parent.mkdir(exist_ok=True)
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
        entry = {'command': command, 'log': str(log.relative_to(self.root)), 'returncode': None}
        self.commands.append(entry)
        try:
            with log.open('w', encoding='utf-8') as f:
                process = subprocess.Popen(command, cwd=self.project, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                for line in process.stdout:
                    print(line, end='')
                    f.write(line)
                    f.flush()
                entry['returncode'] = process.wait()
            if entry['returncode']:
                raise subprocess.CalledProcessError(entry['returncode'], command)
        finally:
            (self.root / 'commands.json').write_text(json.dumps(self.commands, ensure_ascii=False, indent=2), encoding='utf-8')
            self.sync()
        return log
