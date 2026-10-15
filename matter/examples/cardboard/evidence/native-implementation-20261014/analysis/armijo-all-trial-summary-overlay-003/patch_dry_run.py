#!/usr/bin/env python3
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

BASE = Path('/Users/home/cardboard-implementation-20261014/analysis/armijo-all-trial-summary-overlay-002/overlay/matter/src/runtime.mm')
PATCH = Path(__file__).parent / 'overlay.patch'
EXPECTED_BASE = 'feaf5a062138e8dc1a84994090fde27a59a2605e863bb0d5c8ed1280e07b7fc4'
EXPECTED_CANDIDATE = 'ac3b2bc725a6787f342b20c1c11e8adc57ad47a80b78d9281e82cce41b9ed38c'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(BASE) == EXPECTED_BASE, 'sealed overlay-002 base hash changed'
with tempfile.TemporaryDirectory(prefix='armijo-summary-003-') as td:
    root = Path(td)
    target = root / 'matter/src/runtime.mm'
    target.parent.mkdir(parents=True)
    shutil.copyfile(BASE, target)
    dry = subprocess.run(['patch', '--dry-run', '-p1', '-i', str(PATCH)], cwd=root,
                         capture_output=True, text=True)
    assert dry.returncode == 0, dry.stdout + dry.stderr
    applied = subprocess.run(['patch', '-p1', '-i', str(PATCH)], cwd=root,
                             capture_output=True, text=True)
    assert applied.returncode == 0, applied.stdout + applied.stderr
    assert sha(target) == EXPECTED_CANDIDATE, 'patch result does not match candidate'
print('PASS patch base binding, dry-run, apply, and candidate hash')
