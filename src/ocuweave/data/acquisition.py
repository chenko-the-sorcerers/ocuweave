"""Official releases only; streamed downloads, disk guards, and verified checksums."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
from ocuweave.utils.provenance import write_json, sha256


def download_verified(url, destination, expected_size, algorithm, checksum):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    def digest(path):
        h = hashlib.new(algorithm)
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()
    if destination.exists():
        if destination.stat().st_size == expected_size and digest(destination) == checksum:
            return destination
        raise ValueError(f'Existing file failed verification: {destination.name}; preserve and review it')
    if shutil.disk_usage(destination.parent).free < expected_size + 256 * 1024 ** 2:
        raise OSError('Insufficient disk space: archive size plus 256 MiB headroom required')
    partial = destination.with_suffix(destination.suffix + '.part')
    if partial.exists():
        raise FileExistsError(f'Incomplete download exists: {partial}; review before retrying')
    request = urllib.request.Request(url, headers={'User-Agent': 'OcuWeave-research/0.2.0'})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open('xb') as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        if partial.stat().st_size != expected_size or digest(partial) != checksum:
            raise ValueError('Downloaded bytes failed official size/checksum verification')
        os.replace(partial, destination)
    except Exception:
        # Only our partial file is removed. Never overwrite a previously verified archive.
        partial.unlink(missing_ok=True)
        raise
    return destination


def acquire_otfid(destination):
    destination = Path(destination)
    url = 'https://zenodo.org/api/records/13883485'
    with urllib.request.urlopen(url, timeout=45) as response:
        metadata = json.load(response)
    if metadata['metadata'].get('version') != '3.0.0' or metadata['metadata'].get('license', {}).get('id') != 'cc-by-4.0':
        raise ValueError('Unexpected release version or license; review before download')
    item = next(f for f in metadata['files'] if f['key'] == 'Ocular_Toxoplasmosis_Data_V3.zip')
    algorithm, checksum = item['checksum'].split(':', 1)
    path = download_verified(item['links']['self'], destination / item['key'], item['size'], algorithm, checksum)
    write_json(destination / 'source_metadata.json', metadata)
    write_json(destination / 'download_manifest.json', {'dataset': 'OTFID', 'version': '3.0.0',
        'license': 'CC-BY-4.0', 'source_url': 'https://zenodo.org/records/13883485',
        'filename': path.name, 'bytes': path.stat().st_size, 'official_checksum': item['checksum'],
        'sha256': sha256(path)})
    return path
