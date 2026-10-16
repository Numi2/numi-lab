#!/usr/bin/env python3
"""Create a deterministic manifest for the static checker package files."""
from pathlib import Path
import hashlib
import json
PACKAGE = Path(__file__).resolve().parent
MANIFEST = PACKAGE / 'package-manifest.json'
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
files = {}
for path in sorted(PACKAGE.rglob('*')):
    if path.is_symlink():
        raise SystemExit(f'refusing symlink: {path.relative_to(PACKAGE)}')
    if not path.is_file() or path == MANIFEST or '__pycache__' in path.parts:
        continue
    relative = path.relative_to(PACKAGE).as_posix()
    if relative.startswith('executions/'):
        continue
    files[relative] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
manifest = {'schema':'numi.cardboard.native-sparse-scan-package.v1', 'files':files}
with MANIFEST.open('x') as stream:
    json.dump(manifest, stream, sort_keys=True, indent=2)
    stream.write('\n')
print(f'sealed_files={len(files)} manifest_sha256={sha(MANIFEST)}')
