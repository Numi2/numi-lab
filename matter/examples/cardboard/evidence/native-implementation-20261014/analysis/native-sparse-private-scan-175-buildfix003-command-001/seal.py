#!/usr/bin/env python3
"""Write a deterministic static-file manifest for this command package."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'package-manifest.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


files = {}
for path in sorted(ROOT.rglob('*')):
    if path.is_symlink():
        raise SystemExit(f'symlink forbidden: {path}')
    if not path.is_file() or path == MANIFEST or '__pycache__' in path.parts:
        continue
    rel = path.relative_to(ROOT).as_posix()
    files[rel] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
payload = {'schema': 'numi.cardboard.native-check-command-package.v1',
           'files': files,
           'sealed_without_self_hash': True}
MANIFEST.write_text(json.dumps(payload, sort_keys=True, indent=2) + '\n')
print(json.dumps({'file_count': len(files), 'manifest': str(MANIFEST),
                  'manifest_sha256': sha(MANIFEST)}, sort_keys=True, indent=2))
