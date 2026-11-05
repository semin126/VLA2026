"""Regression for simultaneous texture writers and readers on Windows/Colab."""
import _bootstrap  # noqa: F401
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import tempfile
from PIL import Image
from filelock import FileLock

def writer(directory):
    from vlapipe import numbered
    numbered.TEXTURE_DIR=Path(directory)
    for _ in range(8):
        numbered.write_number_textures(force=True)
    return True

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as tmp:
        reads=0
        with ProcessPoolExecutor(max_workers=3) as pool:
            jobs=[pool.submit(writer,tmp) for _ in range(3)]
            while not all(f.done() for f in jobs):
                with FileLock(str(Path(tmp)/'.write.lock'),timeout=60):
                    for path in Path(tmp).glob('num_[0-9][0-9].png'):
                        with Image.open(path) as im:
                            im.verify()
                        reads+=1
            assert all(f.result() for f in jobs)
        assert reads>0
        print('PASS: concurrent atomic texture writes; verified reads',reads)
