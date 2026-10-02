"""Regenerate the file inventory of a Git checkout without changing research data.

Includes tracked and untracked non-ignored files except the manifest itself.
Historical source inventories and freeze records are never modified.
Run this only for a deliberately prepared new repository revision.
"""
from pathlib import Path
import hashlib
import json
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    if not (root / '.git').exists():
        raise RuntimeError('Run in an initialized Git checkout.')
    inventory = subprocess.check_output(
        ['git', '-C', str(root), 'ls-files', '-c', '-o', '--exclude-standard', '-z']
    ).decode('utf-8')
    names = sorted(set(inventory.split('\0')) - {'', 'release_manifest.json'})
    rows = []
    for name in names:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError('Expected an ordinary repository file: ' + name)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        rows.append({'relative_path': name, 'bytes': path.stat().st_size,
                     'sha256': digest.hexdigest()})
    result = {
        'schema': 'file-length-and-sha256-v1',
        'scope': 'All Git-tracked or untracked non-ignored release files except this manifest itself.',
        'purpose': 'Byte identity; not a proof of scientific validity or new pre-test registration.',
        'licensing': 'File-level scope and upstream attributions are described in DATA_LICENSES.md.',
        'files': rows,
    }
    (root / 'release_manifest.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'files': len(rows), 'bytes': sum(r['bytes'] for r in rows),
                      'largest_bytes': max(r['bytes'] for r in rows)}, indent=2))


if __name__ == '__main__':
    main()
