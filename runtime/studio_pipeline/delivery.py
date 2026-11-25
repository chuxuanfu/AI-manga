"""Copy a complete release atomically to the user's chosen volume."""
import hashlib
from pathlib import Path
import shutil
import tempfile

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def export_files(folder, job, files):
    folder = Path(folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {name: digest(path) for name, path in files.items()}
    signature = hashlib.sha256(repr(sorted(hashes.items())).encode()).hexdigest()[:12]
    target = folder / f'{job.name}-{signature}'
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=folder, prefix='.export-') as temp:
            staging = Path(temp) / 'ready'
            staging.mkdir()
            for name, path in files.items():
                shutil.copyfile(path, staging / name)
                if digest(staging / name) != hashes[name]:
                    raise OSError(f'Export verification failed for {name}')
            try:
                staging.rename(target)
            except OSError:
                if not target.is_dir():
                    raise
    if any(not (target / name).is_file() or digest(target / name) != value for name, value in hashes.items()):
        raise FileExistsError(f'Existing export differs: {target}')
    print('输出位置', target, flush=True)
    return str(target)
