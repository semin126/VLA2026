"""Select installed NVIDIA EGL for headless Colab without changing drivers.

Some Colab GPU images ship NVIDIA EGL libraries but only register Mesa with
GLVND. CUDA then works while MuJoCo silently renders on llvmpipe. A process-local
vendor file selects the existing driver. Explicit user EGL selection is kept.
"""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import sys
import tempfile


def configure_headless_rendering(config_dir: Path | None = None) -> dict:
    backend = os.environ.get('MUJOCO_GL')
    if not backend:
        backend = 'glfw' if os.name == 'nt' or sys.platform == 'darwin' or os.environ.get('DISPLAY') else 'egl'
        os.environ['MUJOCO_GL'] = backend
    result = {'backend': backend, 'vendor_policy': 'system'}
    if not sys.platform.startswith('linux') or backend != 'egl':
        return result
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    if os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES'):
        return {**result, 'vendor_policy': 'explicit', 'vendor_file': os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']}
    if os.environ.get('VLA_EGL_VENDOR', 'auto') == 'system':
        return result
    library = next((p for p in [Path('/usr/lib64-nvidia/libEGL_nvidia.so.0'),
                               Path('/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0')]
                    if p.is_file()), None)
    if library is None:
        return result
    if config_dir is None:
        fd, filename = tempfile.mkstemp(prefix='vlapipe_nvidia_egl_', suffix='.json')
        os.close(fd)
        target = Path(filename)
        atexit.register(lambda: target.unlink(missing_ok=True))
    else:
        config_dir = Path(config_dir)
        config_dir.mkdir(parents=True, exist_ok=True)
        target = config_dir / 'nvidia_egl_vendor.json'
    target.write_text(json.dumps({'file_format_version': '1.0.0', 'ICD': {'library_path': str(library)}}), encoding='utf-8')
    os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = str(target)
    return {**result, 'vendor_policy': 'installed_nvidia', 'vendor_file': str(target), 'library': str(library)}
