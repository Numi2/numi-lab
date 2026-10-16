#!/usr/bin/env python3
"""Apply source159's patch to a temporary copy of only its source inputs."""
from __future__ import annotations
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

BASE = Path('/Users/home/cardboard-implementation-20261014')
PKG = BASE / 'analysis/source159-inside-surface-role-map-overlay-003'
SNAP = BASE / 'snapshots/source-right-grip-telemetry-157'
FILES = (
    'matter/tools/cardboard_probe.mm',
    'matter/tools/cardboard_shipping_memory.hpp',
    'matter/tools/cardboard_inside_clearance.hpp',
)

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    with tempfile.TemporaryDirectory(prefix='source159-patch-check-') as temporary:
        root = Path(temporary)
        for rel in FILES[:2]:
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SNAP / rel, target)
        patch = PKG / 'source-overlay.patch'
        for dry_run in (True, False):
            args = ['patch', '-p1', '-d', str(root)]
            if dry_run:
                args.append('--dry-run')
            subprocess.run(args, input=patch.read_bytes(), check=True)
        for rel in FILES:
            actual = root / rel
            expected = PKG / 'candidate' / rel
            if sha(actual) != sha(expected):
                raise SystemExit(f'patch result hash mismatch: {rel}')
    print('source159 patch application: PASS (base files were copied to a disposable temp directory)')

if __name__ == '__main__':
    main()
