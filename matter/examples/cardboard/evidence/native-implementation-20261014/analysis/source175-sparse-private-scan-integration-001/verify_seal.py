#!/usr/bin/env python3
"""Verify the sealed source175 overlay inventory and parent bindings."""
from pathlib import Path
import hashlib
import json
import stat

B = Path('/Users/home/cardboard-implementation-20261014')
OUT = B / 'analysis/source175-sparse-private-scan-integration-001'
MANIFEST = OUT / 'source175-overlay.json'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


doc = json.loads(MANIFEST.read_text())
if doc['schema'] != 'numi.cardboard.source175-integration-overlay.v1':
    raise SystemExit('wrong schema')
inventory = {row['path']: row for row in doc['evidence_inventory']}
actual = {}
for path in OUT.rglob('*'):
    if not path.is_file():
        continue
    if path.is_symlink():
        raise SystemExit(f'symlink in package: {path.relative_to(OUT)}')
    rel = path.relative_to(OUT).as_posix()
    if rel != 'source175-overlay.json':
        actual[rel] = path
if set(actual) != set(inventory):
    raise SystemExit(f'inventory path set mismatch: missing={sorted(set(inventory)-set(actual))[:8]} extra={sorted(set(actual)-set(inventory))[:8]}')
for rel, path in actual.items():
    row = inventory[rel]
    info = path.stat()
    if sha(path) != row['sha256'] or info.st_size != row['size_bytes'] or stat.S_IMODE(info.st_mode) != row['mode']:
        raise SystemExit(f'inventory bytes/mode mismatch: {rel}')
for key, expected in [
    ('common_base_169', 'f5054091c27a7e1add41630aae1b45ee6ecc3e6d4112ee85c931cc4c70f44ea4'),
    ('source173_integration_base', '6d7e76fdb07702b0b0727fc6610ae979c4e0e161ad92f6888036b1dfef3b53b3'),
]:
    if doc['parents'][key]['manifest_sha256'] != expected:
        raise SystemExit(f'parent hash mismatch: {key}')
for row in doc['integration']['changed_sources']:
    path = OUT / 'candidate' / row['path']
    if sha(path) != row['source175_candidate_sha256']:
        raise SystemExit(f'candidate source hash mismatch: {row["path"]}')
if doc['integration']['runtime_three_way_merge_exit_code'] != 0 or doc['integration']['runtime_three_way_merge_conflict_markers'] != 0:
    raise SystemExit('three-way merge is not clean')
if not doc['validation']['cpu_all_passed'] or not doc['validation']['metal_all_passed']:
    raise SystemExit('validation status is not passing')
if doc['gpu_dispatch'] or doc['physical_transactions']:
    raise SystemExit('unexpected GPU or physical transaction claim')
print(json.dumps({
    'verified_inventory_files': len(inventory),
    'changed_source_files': doc['integration']['changed_source_count'],
    'cpu_all_passed': doc['validation']['cpu_all_passed'],
    'metal_all_passed': doc['validation']['metal_all_passed'],
    'gpu_dispatch': doc['gpu_dispatch'],
}, sort_keys=True))
