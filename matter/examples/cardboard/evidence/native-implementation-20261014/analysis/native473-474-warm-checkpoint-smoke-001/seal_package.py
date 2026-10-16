#!/usr/bin/env python3
"""Create the deterministic package file-hash inventory."""
from pathlib import Path
import hashlib
import json

PACKAGE = Path(__file__).resolve().parent
MANIFEST = PACKAGE / 'package-manifest.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


files = {}
for path in sorted(PACKAGE.rglob('*')):
    if not path.is_file() or path == MANIFEST or '__pycache__' in path.parts:
        continue
    if path.is_symlink():
        raise SystemExit(f'refusing to inventory symlink: {path}')
    files[path.relative_to(PACKAGE).as_posix()] = sha(path)

record = {
    'schema': 'numi.cardboard.native473-474-package.v1',
    'files_sha256': files,
    'file_count': len(files),
    'note': 'The manifest inventories every package file except itself and generated Python bytecode caches.',
}
MANIFEST.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
print(json.dumps({'manifest': str(MANIFEST), 'file_count': len(files),
                  'manifest_sha256': sha(MANIFEST)}, sort_keys=True))
