#!/usr/bin/env python3
"""Write the source166-003 patch and exact package manifests."""
from __future__ import annotations
import difflib
import hashlib
import json
from pathlib import Path

BASE = Path('/Users/home/cardboard-implementation-20261014')
ROOT = BASE / 'analysis/bilinear-capture-layer-audit-source166-003'
PARENT = BASE / 'analysis/bilinear-capture-layer-audit-source166-002'
CANDIDATE = ROOT / 'candidate/matter/tools'
PARENT_TOOLS = PARENT / 'candidate/matter/tools'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def unified(old: Path | None, new: Path, new_rel: str) -> str:
    old_lines = old.read_text().splitlines(keepends=True) if old else []
    new_lines = new.read_text().splitlines(keepends=True)
    return ''.join(difflib.unified_diff(old_lines, new_lines,
        fromfile='a/' + new_rel if old else '/dev/null',
        tofile='b/' + new_rel, n=3))


patch = ''.join((
    unified(PARENT_TOOLS / 'cardboard_bilinear_fp64_reference.py',
            CANDIDATE / 'cardboard_bilinear_fp64_reference.py',
            'matter/tools/cardboard_bilinear_fp64_reference.py'),
    unified(None, CANDIDATE / 'test_reference_fb_jacobian.py',
            'matter/tools/test_reference_fb_jacobian.py'),
))
(ROOT / 'overlay.patch').write_text(patch)

parent_manifest = PARENT / 'manifest.json'
files = []
for name in ('cardboard_bilinear_capture_layers.py', 'cardboard_bilinear_fp64_reference.py',
             'test_cardboard_bilinear_capture_layers.py', 'test_reference_fb_jacobian.py'):
    new = CANDIDATE / name
    old = PARENT_TOOLS / name
    files.append({
        'path': 'matter/tools/' + name,
        'base_source166002_sha256': sha(old) if old.is_file() else None,
        'overlay_sha256': sha(new),
        'changed': not old.is_file() or sha(old) != sha(new),
    })

overlay = {
    'schema': 'numi.cardboard.source-overlay.v1',
    'base_snapshot': str(BASE / 'snapshots/source-right-grip-telemetry-157'),
    'base_overlay': str(PARENT),
    'base_overlay_manifest_sha256': sha(parent_manifest),
    'overlay_id': ROOT.name,
    'patch_sha256': sha(ROOT / 'overlay.patch'),
    'supersedes': {'package': str(PARENT), 'manifest_sha256': sha(parent_manifest)},
    'files': files,
    'native_execution_performed': False,
    'qualification': False,
}
(ROOT / 'source-overlay.json').write_text(json.dumps(overlay, indent=2) + '\n')

summary = json.loads((ROOT / 'validation/summary.json').read_text())
input_paths = {}
for run in summary['runs']:
    case = Path(run['case'])
    for rel in ('prediction.json', 'receipt.json', 'stderr.log', 'output/manifest.json', 'operator-capture'):
        path = case / rel
        if path.is_file(): input_paths[str(path)] = sha(path)
    output = Path(run['output'])
    for p in output.iterdir():
        if p.is_file(): input_paths[str(p)] = sha(p)
for path in (
    parent_manifest,
    PARENT_TOOLS / 'cardboard_bilinear_fp64_reference.py',
    BASE / 'analysis/native469-independent-projection-002/manifest.json',
    BASE / 'analysis/native469-independent-projection-002/run-sealed002/report.json',
    BASE / 'analysis/native469-independent-projection-002/run-sealed002/mp_states.npz',
    BASE / 'snapshots/source-right-grip-telemetry-157/matter/tools/cardboard_cached_operator_audit.py',
):
    input_paths[str(path)] = sha(path)

package_files = {}
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or p.name == 'manifest.json':
        continue
    rel = p.relative_to(ROOT).as_posix()
    package_files[rel] = {'sha256': sha(p), 'bytes': p.stat().st_size}

manifest = {
    'schema': 'numi.cardboard.bilinear-capture-layer-audit-source166-003.v1',
    'classification': 'reference-only FB Jacobian correction; complete 419/469 reader runs are diagnostic, not qualification',
    'source_overlay': str(ROOT / 'source-overlay.json'),
    'source_overlay_sha256': sha(ROOT / 'source-overlay.json'),
    'patch_sha256': sha(ROOT / 'overlay.patch'),
    'base_source166002_manifest_sha256': sha(parent_manifest),
    'validation_summary': str(ROOT / 'validation/summary.json'),
    'validation_summary_sha256': sha(ROOT / 'validation/summary.json'),
    'input_file_hashes': input_paths,
    'changed_files': [x['path'] for x in files if x['changed']],
    'unchanged_reader_and_existing_contract_test': [x['path'] for x in files if not x['changed']],
    'validation': {
        'unittest_count': 10,
        'unittest_returncode': 0,
        'py_compile_returncode': 0,
        'full_reader_runs': 2,
        'native_runs': 0,
        'gpu_runs': 0,
    },
    'package_files': package_files,
    'limits': [
        'The rationalized residual value and all strict residual/correction gates are unchanged.',
        '469 retains four center strict-floor projection failures and state-specific failures; 419 retains side-state failures.',
        'All solver correctness, numerical convergence, calibration, analogue, assembly, and crush-impact qualification axes remain false.',
        'The tangent matrix comparison is an independent FP64 reader check of all nine columns, not a native finite-difference/noise-floor qualification.',
    ],
}
(ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
print(json.dumps({'manifest': str(ROOT / 'manifest.json'), 'manifest_sha256': sha(ROOT / 'manifest.json'),
                  'patch_sha256': sha(ROOT / 'overlay.patch'), 'changed_files': manifest['changed_files'],
                  'files': len(package_files)}, sort_keys=True))
