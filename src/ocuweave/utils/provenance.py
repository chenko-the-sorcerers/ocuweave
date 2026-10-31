"""Atomic outputs and content-addressed provenance, without dataset content in logs."""
import hashlib
import json
import os
import subprocess
from pathlib import Path


def software_provenance():
    """Resolve this package's checkout, never the caller's unrelated Git directory."""
    from ocuweave import __version__
    root = Path(__file__).resolve().parents[3]
    result = {'software_version': __version__, 'source_commit': 'unavailable', 'source_dirty': None}
    if not (root / 'pyproject.toml').is_file():
        return result
    try:
        args = ['git', '-C', str(root)]
        top = subprocess.check_output(args + ['rev-parse', '--show-toplevel'], text=True, stderr=subprocess.DEVNULL).strip()
        if Path(top).resolve() != root:
            return result
        result['source_commit'] = subprocess.check_output(args + ['rev-parse', 'HEAD'], text=True).strip()
        result['source_dirty'] = bool(subprocess.check_output(args + ['status', '--porcelain', '--untracked-files=normal'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return result


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temporary, path)
