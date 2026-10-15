#!/usr/bin/env python3
"""Verify an unnumbered diagnostic integration candidate against frozen source 098."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PKG = Path(r'/Users/home/cardboard-implementation-20261014/analysis/center-residual-jdelta-source098-candidate-001')
BASE = Path('/Users/home/cardboard-implementation-20261014/snapshots/source-private-mortar-witness-098')
MERGED = PKG / 'merged'
PATCH = PKG / 'source098-integration.patch'
EXPECTED_CHANGED = {
    'matter/src/runtime.mm',
    'matter/src/metal/fgmres.metalinc',
    'matter/src/metal/mixed_fem.metalinc',
    'matter/include/numi/matter/matter.hpp',
    'matter/tools/cardboard_graph_lu_search_native_check.mm',
    'matter/tools/cardboard_regional_recycle_masked_basis_native_check.mm',
}
PROTECTED_098 = {
    'matter/CMakeLists.txt',
    'matter/src/metal/contact.metalinc',
    'matter/include/numi/matter/mortar_shared_edge_guard.h',
    'matter/tools/cardboard_full_contact_armijo_check.mm',
}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def need(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit('FAIL: ' + message)

frozen_path = BASE / 'frozen-source.json'
frozen = json.loads(frozen_path.read_text())
source = frozen['source_sha256']
base_hash = sha(frozen_path)
need(len(source) == 776, 'unexpected source098 inventory size')
changed = set()
for rel, expected in source.items():
    path = MERGED / rel
    need(path.is_file() and not path.is_symlink(), f'missing/nonregular candidate source {rel}')
    actual = sha(path)
    if actual != expected:
        changed.add(rel)
need(changed == EXPECTED_CHANGED, f'unexpected tracked change set {sorted(changed)}')
for rel in PROTECTED_098:
    need(sha(MERGED / rel) == source[rel], f'source098 contact/mortar witness was modified: {rel}')
need(not any(p.is_symlink() for p in MERGED.rglob('*')), 'candidate source tree contains symlinks')
need(PATCH.is_file(), 'integration patch missing')
result = subprocess.run(['patch', '--dry-run', '-p1', '-d', str(BASE)],
                        input=PATCH.read_bytes(), stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
need(result.returncode == 0, 'candidate patch does not dry-run against frozen098')

runtime = (MERGED / 'matter/src/runtime.mm').read_text()
fgmres = (MERGED / 'matter/src/metal/fgmres.metalinc').read_text()
center_arrays = (
    'fgmres_vector_row_map',
    'center_production_jdelta_action',
    'center_operator_diagnostic_status',
    'contact_history_used_by_newton_center',
    'deformable_history_used_by_newton_center',
)
for name in center_arrays:
    need(name in runtime, f'center capture array missing: {name}')
need('NM_FEM_COUPLED_CAPTURE_SELECTOR' in runtime, 'center selector missing')
need('NM_FEM_COUPLED_TRIAL_CAPTURE_SELECTOR' in runtime, 'private-trial selector missing')
need('center_production_jdelta_action is the same-generation matrix-free production J times fem_solution_correction' in runtime,
     'same-generation J-delta semantics missing')
need('femRegionalRestartAugmentationEnabled' in runtime and
     'environment_major_padded_mode_stride' in runtime,
     'two-environment restart augmentation metadata missing')
need('femRegionalRestartGeneration' in runtime and
     'femRegionalRestartTailPreserved' in runtime,
     'two-environment generation/tail state missing')
mixed = (MERGED / 'matter/src/metal/mixed_fem.metalinc').read_text()
need('constant uint& allowConvergedState [[buffer(11)]]' in mixed,
     'center element operator guard signature missing')

manifest_path = PKG / 'candidate-manifest.json'
need(manifest_path.is_file(), 'candidate manifest missing')
manifest = json.loads(manifest_path.read_text())
need(manifest['base098']['manifest_sha256'] == base_hash, 'candidate manifest has wrong frozen098 hash')
need(manifest['candidate_status'] == 'unfrozen_unversioned_candidate', 'candidate is incorrectly promoted')
need(manifest['center_capture_contract']['new_array_count_max'] == 5,
     'center array bound mismatch')
need(manifest['private_trial_capture_contract']['phase_bounds_unchanged'] ==
     {'pre': 71, 'post': 109, 'both': 136}, 'private-trial reader bounds changed')
need(manifest['private_trial_capture_contract']['array_cap'] == 192 and
     manifest['center_capture_contract']['array_cap'] == 192,
     '192-array cap changed')
need(manifest['private_trial_capture_contract']['payload_cap_bytes'] == 33554432 and
     manifest['center_capture_contract']['payload_cap_bytes'] == 33554432,
     '32 MiB cap changed')

print(json.dumps({
    'status': 'PASS',
    'schema': 'numi.matter.candidate_overlay_integration_check.v1',
    'base098_manifest_sha256': base_hash,
    'verified_source098_entries': len(source),
    'changed_files': sorted(changed),
    'protected_source098_files_unchanged': sorted(PROTECTED_098),
    'candidate_symlink_count': 0,
    'patch_dry_run': 'PASS',
    'center_capture_array_additions_max': len(center_arrays),
    'private_trial_phase_bounds': {'pre': 71, 'post': 109, 'both': 136},
    'array_cap': 192,
    'aggregate_payload_cap_bytes': 33554432,
}, indent=2, sort_keys=True))
