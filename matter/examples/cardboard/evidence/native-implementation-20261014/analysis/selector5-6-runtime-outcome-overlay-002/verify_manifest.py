#!/usr/bin/env python3
"""Verify every immutable file listed in this overlay manifest."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
seen = set()
for row in manifest['files']:
    rel = row['path']
    if rel in seen or rel == 'manifest.json':
        raise SystemExit(f'invalid inventory path: {rel}')
    seen.add(rel)
    path = root / rel
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f'missing or non-regular package file: {rel}')
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != row['bytes'] or digest != row['sha256']:
        raise SystemExit(f'inventory mismatch: {rel}')
print(f'PASS: verified {len(seen)} regular package files')
