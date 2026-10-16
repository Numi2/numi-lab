#!/usr/bin/env python3
"""Build deterministic source patch and source/package hash manifests."""
from __future__ import annotations
import difflib
import hashlib
import json
from pathlib import Path

BASE = Path('/Users/home/cardboard-implementation-20261014')
PKG = BASE / 'analysis/source159-inside-surface-role-map-overlay-003'
SNAP = BASE / 'snapshots/source-right-grip-telemetry-157'
SOURCE_OVERLAY = SNAP / 'source-overlay.json'
CHANGED = {
    'matter/tools/cardboard_probe.mm': PKG / 'candidate/matter/tools/cardboard_probe.mm',
    'matter/tools/cardboard_shipping_memory.hpp': PKG / 'candidate/matter/tools/cardboard_shipping_memory.hpp',
    'matter/tools/cardboard_inside_clearance.hpp': PKG / 'candidate/matter/tools/cardboard_inside_clearance.hpp',
}

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def patch_text(rel: str, candidate: Path) -> str:
    base = SNAP / rel
    if base.exists():
        old = base.read_text(encoding='utf-8').splitlines(keepends=True)
        old_name = 'a/' + rel
    else:
        old = []
        old_name = '/dev/null'
    new = candidate.read_text(encoding='utf-8').splitlines(keepends=True)
    return ''.join(difflib.unified_diff(old, new, fromfile=old_name,
                                        tofile='b/' + rel, n=3))

def main() -> None:
    patch = ''.join(patch_text(rel, candidate) for rel, candidate in CHANGED.items())
    patch_path = PKG / 'source-overlay.patch'
    patch_path.write_text(patch, encoding='utf-8')
    base_manifest = json.loads(SOURCE_OVERLAY.read_text(encoding='utf-8'))
    changed = {}
    for rel, candidate in CHANGED.items():
        base = SNAP / rel
        changed[rel] = {
            'base_sha256': sha(base) if base.exists() else None,
            'candidate_sha256': sha(candidate),
            'candidate_bytes': candidate.stat().st_size,
            'role': 'modified' if base.exists() else 'new',
        }
    manifest = {
        'schema': 'numi.cardboard.source159_inside_surface_clearance_overlay.v3',
        'overlay_id': 'source159_panel_local_closure_and_bounded_accounting',
        'base_snapshot': str(SNAP),
        'base_frozen_source_sha256': base_manifest['base_frozen_source_sha256'],
        'base_overlay_manifest': str(SOURCE_OVERLAY),
        'base_overlay_manifest_sha256': sha(SOURCE_OVERLAY),
        'base_source_hash_entry_count': base_manifest['source_hash_entry_count'],
        'base_configure_support_entry_count': base_manifest['configure_support_entry_count'],
        'base_probe_sha256': base_manifest['changed_source_files']['matter/tools/cardboard_probe.mm']['sha256'],
        'base_files': {
            rel: changed[rel]['base_sha256'] for rel in CHANGED if changed[rel]['base_sha256']
        },
        'changed_source_files': changed,
        'patch_file': str(patch_path),
        'patch_sha256': sha(patch_path),
        'patch_bytes': patch_path.stat().st_size,
        'only_source_changes': sorted(CHANGED),
        'all_unlisted_source_files_unchanged': True,
        'native_or_gpu_execution': 'not launched',
        'full_source_build': 'not run',
        'validation': {
            'cpu_contracts': 'panel-local geometry/seam/work-cap contracts and shipping memory at O0/O3; PASS',
            'probe': 'source-157 compile-command Objective-C++ syntax only; PASS with one pre-existing unused-function warning',
            'scale': 'synthetic combinatorial topology-only fixture; PASS; no geometry/material evaluated; modeled scratch 1028812212 bytes',
            'cpu_validation_log_sha256': sha(PKG / 'validation/cpu-validation.log'),
            'patch_application': 'source157 files patched in a temporary directory; all three candidate hashes match; PASS',
            'patch_application_log_sha256': sha(PKG / 'validation/patch-apply.log'),
            'scale_validation_log_sha256': sha(PKG / 'validation/topology-scale-preflight.log'),
            'scale_result_sha256': sha(PKG / 'validation/topology-scale-preflight.json'),
        },
        'limits': [
            'No actual accepted shipping geometry or physical assembly was evaluated.',
            'The topology-scale fixture validates indexing and workspace/work caps only.',
            'Existing solver/contact runtime admission caps remain independently enforced.',
            'The clearance workspace is modeled as concurrent with the live world and checkpoints; this is not an observed process peak.',
            'The 1 mm butt-seam allowance is an engineering tolerance, not a measured kerf or manufacturing validation.',
            'The 1% dimension comparison is independent of wall/end local opening coverage.',
            'The FP32 ULP-derived area floor is a numerical bound, and accepted input spacing above 10 micrometres rejects the diagnostic.',
            'Synthetic slit/seam tests do not establish closure or strength of a physical carton.',
        ],
    }
    manifest_path = PKG / 'source-overlay.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    inventory_files = sorted(p for p in PKG.rglob('*') if p.is_file() and p.name not in {
        'package-inventory.json', 'package-inventory.sha256'
    })
    inventory = {
        'schema': 'numi.cardboard.source159_overlay_package_inventory.v1',
        'files': [
            {'path': str(p.relative_to(PKG)), 'bytes': p.stat().st_size,
             'sha256': sha(p)} for p in inventory_files
        ],
    }
    inventory_path = PKG / 'package-inventory.json'
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (PKG / 'package-inventory.sha256').write_text(sha(inventory_path) + '  package-inventory.json\n', encoding='ascii')
    print(json.dumps({
        'source_overlay_sha256': sha(manifest_path),
        'patch_sha256': sha(patch_path),
        'inventory_sha256': sha(inventory_path),
        'file_count': len(inventory_files),
        'changed_source_files': changed,
    }, sort_keys=True, indent=2))

if __name__ == '__main__':
    main()
