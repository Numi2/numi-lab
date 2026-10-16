#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    files = {}
    for path in sorted(HERE.iterdir()):
        if path.is_file() and path.name != 'manifest.json':
            files[path.name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    report = json.loads((HERE / 'independent-review.json').read_text())
    manifest = {
        'schema': 'numi.cardboard.source153_independent_review_manifest.v1',
        'source_snapshot_manifest_sha256': report['source_snapshot_manifest_sha256'],
        'source_runtime_sha256': report['source_runtime_sha256'],
        'parent_runtime_sha256': report['source149_runtime_sha256'],
        'overlay_patch_sha256': report['overlay_patch_sha256'],
        'source_build_receipt_sha256': report['build_receipt_sha256'],
        'review_sha256': sha(HERE / 'independent-review.json'),
        'files': files,
        'native_or_physical_qualification': False,
    }
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2,
                                                    sort_keys=True,
                                                    allow_nan=False) + '\n')
    print(json.dumps({'manifest_sha256': sha(HERE / 'manifest.json'),
                      'review_sha256': manifest['review_sha256'],
                      'files': files}, sort_keys=True, allow_nan=False))

if __name__ == '__main__':
    main()
