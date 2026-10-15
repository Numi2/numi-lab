#!/usr/bin/env python3
"""Verify the isolated center-residual/J-delta overlay against frozen source 095."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path('/Users/home/cardboard-implementation-20261014')
PACKAGE = ROOT / 'analysis/center-residual-jdelta-history-overlay-001'
BASE = ROOT / 'snapshots/source-contact-capture-and-resolved-material-sweep-095'
OVERLAY = PACKAGE / 'overlay'
MANIFEST = BASE / 'frozen-source.json'

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit('FAIL: ' + message)

frozen = json.loads(MANIFEST.read_text())
source_hashes = frozen['source_sha256']
bad = []
for relative, expected in source_hashes.items():
    path = BASE / relative
    actual = sha(path) if path.is_file() else None
    if actual != expected:
        bad.append((relative, expected, actual))
check(not bad, f'{len(bad)} frozen source hash mismatch(es): {bad[:3]}')

changed = [
    'matter/src/runtime.mm',
    'matter/src/metal/fgmres.metalinc',
    'matter/src/metal/mixed_fem.metalinc',
]
for relative in changed:
    source = BASE / relative
    base_copy = PACKAGE / 'base' / relative
    overlay = OVERLAY / relative
    check(source_hashes[relative] == sha(source), f'frozen hash mismatch: {relative}')
    check(sha(base_copy) == source_hashes[relative], f'base copy mismatch: {relative}')
    check(overlay.is_file() and not overlay.is_symlink(),
          f'changed overlay file is not an independent regular file: {relative}')

# No overlay path may write through into the frozen source.
for path in OVERLAY.rglob('*'):
    check(not path.is_symlink(), f'overlay contains a symlink: {path}')

runtime = (OVERLAY / 'matter/src/runtime.mm').read_text()
fgmres = (OVERLAY / 'matter/src/metal/fgmres.metalinc').read_text()
mixed = (OVERLAY / 'matter/src/metal/mixed_fem.metalinc').read_text()

def dispatch_blocks(text: str, name: str):
    token = f'dispatchThreads("{name}"'
    pos = 0
    while True:
        start = text.find(token, pos)
        if start < 0:
            return
        end = text.find('});', start)
        check(end >= 0, f'unterminated dispatch block for {name}')
        yield text[start:end + 3]
        pos = end + 3

mixed_calls = list(dispatch_blocks(runtime, 'nm_mixed_apply_operator_elements'))
check(len(mixed_calls) == 5, f'expected five mixed element dispatches, found {len(mixed_calls)}')
for block in mixed_calls:
    check('atIndex:11u' in block and 'setBytes:' in block,
          'mixed element kernel call lacks buffer-11 binding')
check(mixed.count('allowConvergedState == 0u') == 1,
      'mixed kernel converged-state bypass predicate not uniquely present')
check('constant uint& allowConvergedState [[buffer(11)]]' in mixed,
      'mixed kernel buffer-11 declaration missing')

for name in ('nm_fgmres_apply_free_rigid', 'nm_fgmres_apply_primal_rigid_contacts'):
    calls = list(dispatch_blocks(runtime, name))
    check(len(calls) == 2, f'expected production + diagnostic {name} calls, found {len(calls)}')
    for block in calls:
        check('atIndex:7u' in block and 'setBytes:' in block,
              f'{name} call lacks buffer-7 binding')
    check(f'constant uint& allowConvergedState [[buffer(7)]]' in fgmres,
          f'{name} kernel buffer-7 declaration missing')

check('nonlinearHistories, nonlinearDeformableContactHistories' in runtime,
      'capture call does not pass the selected Newton-center history buffers')
check('center_production_jdelta_action' in runtime and
      'after_fgmres_final_restart_before_armijo' in runtime,
      'same-generation J-delta capture/point metadata missing')
check('NM_MICROSTEP_FGMRES_TRUE_RESIDUAL' in runtime,
      'true-residual microstep bypass absent')

print(json.dumps({
    'status': 'PASS',
    'schema': 'numi.matter.center_residual_jdelta_overlay_check.v1',
    'frozen_source_entries_verified': len(source_hashes),
    'changed_overlay_files_regular': changed,
    'overlay_symlink_count': sum(1 for p in OVERLAY.rglob('*') if p.is_symlink()),
    'mixed_element_call_count': len(mixed_calls),
    'free_rigid_operator_call_count': 2,
    'primal_rigid_contact_operator_call_count': 2,
    'capture_call_binds_newton_selected_histories': True,
}, sort_keys=True, indent=2))
