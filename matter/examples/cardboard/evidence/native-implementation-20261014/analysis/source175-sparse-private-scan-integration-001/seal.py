#!/usr/bin/env python3
"""Seal the source175 overlay and all retained validation evidence."""
from pathlib import Path
import hashlib
import json
import stat

B = Path('/Users/home/cardboard-implementation-20261014')
OUT = B / 'analysis/source175-sparse-private-scan-integration-001'
P169 = B / 'snapshots/source-warm-cgls-refinement-169'
P173 = B / 'snapshots/source-checkpoint-warm-breakdown-173'
O170 = B / 'analysis/hierarchical-sparse-private-contact-scan-overlay-170-001'
MANIFEST = OUT / 'source175-overlay.json'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def frozen(path: Path):
    doc = json.loads((path / 'frozen-source.json').read_text())
    return {
        'snapshot_path': str(path),
        'manifest_path': str(path / 'frozen-source.json'),
        'manifest_sha256': sha(path / 'frozen-source.json'),
        'file_count': len(doc['source_sha256']),
    }


def main() -> None:
    proof = json.loads((OUT / 'three-way-proof.json').read_text())
    src170 = json.loads((O170 / 'source-overlay.json').read_text())
    changed = []
    for row in src170['changed_files']:
        rel = row['path']
        target = OUT / 'candidate' / rel
        source170 = O170 / 'candidate' / rel
        base173 = P173 / rel
        if target.is_symlink() or not target.is_file():
            raise RuntimeError(f'candidate is not a regular file: {rel}')
        changed.append({
            'path': rel,
            'base173_sha256': sha(base173) if base173.is_file() else None,
            'source170_candidate_sha256': sha(source170),
            'source175_candidate_sha256': sha(target),
            'source170_runtime_replaced_by_three_way_merge': rel == 'matter/src/runtime.mm',
        })

    inventory = []
    for path in sorted(OUT.rglob('*')):
        if path == MANIFEST or not path.is_file():
            continue
        if path.is_symlink():
            raise RuntimeError(f'symlink in sealed overlay: {path.relative_to(OUT)}')
        info = path.stat()
        inventory.append({
            'path': path.relative_to(OUT).as_posix(),
            'size_bytes': info.st_size,
            'mode': stat.S_IMODE(info.st_mode),
            'sha256': sha(path),
        })

    cpu = json.loads((OUT / 'validation/cpu-validation.json').read_text())
    metal = json.loads((OUT / 'validation/metal-validation.json').read_text())
    integration = json.loads((OUT / 'validation/integration-verification.json').read_text())
    shipping = json.loads((OUT / 'shipping-admission-audit-001/report.json').read_text())
    if any(row['exit_code'] != 0 for row in cpu['results']):
        raise RuntimeError('CPU validation contains a nonzero result')
    if any(row['exit_code'] != 0 for row in metal['results']):
        raise RuntimeError('Metal compile/link validation contains a nonzero result')
    if len(changed) != 8 or len(inventory) < 20:
        raise RuntimeError('unexpected changed-file or evidence inventory count')

    doc = {
        'schema': 'numi.cardboard.source175-integration-overlay.v1',
        'overlay_id': 'source175-sparse-private-scan-integration-001',
        'candidate_root': str(OUT / 'candidate'),
        'overlay_only': True,
        'source_snapshots_modified': False,
        'gpu_dispatch': False,
        'physical_transactions': 0,
        'parents': {
            'common_base_169': frozen(P169),
            'source173_integration_base': frozen(P173),
            'source170_sparse_overlay': {
                'overlay_root': str(O170),
                'manifest_path': str(O170 / 'source-overlay.json'),
                'manifest_sha256': sha(O170 / 'source-overlay.json'),
                'patch_path': str(O170 / 'source-overlay.patch'),
                'patch_sha256': sha(O170 / 'source-overlay.patch'),
            },
        },
        'integration': {
            'proof_path': str(OUT / 'three-way-proof.json'),
            'proof_sha256': sha(OUT / 'three-way-proof.json'),
            'patch_path': str(OUT / 'source175-integration.patch'),
            'patch_sha256': sha(OUT / 'source175-integration.patch'),
            'patch_reconstructs_exact_candidate': True,
            'runtime_three_way_merge_exit_code': proof['runtime_three_way_merge']['exit_code'],
            'runtime_three_way_merge_conflict_markers': proof['runtime_three_way_merge']['conflict_markers'],
            'runtime_three_way_merge_candidate_sha256': proof['runtime_three_way_merge']['runtime_candidate_sha256'],
            'changed_source_count': len(changed),
            'changed_sources': changed,
            'merge_summary': {
                'source169_to170_hunks': len(proof['runtime_three_way_merge']['line_hunks_on_base169_coordinates']['169_to_170']),
                'source169_to173_hunks': len(proof['runtime_three_way_merge']['line_hunks_on_base169_coordinates']['169_to_173']),
                'overlapping_runtime_file_count': 1,
                'other_source170_files_copied_exactly': 7,
            },
        },
        'validation': {
            'cpu_report_path': str(OUT / 'validation/cpu-validation.json'),
            'cpu_report_sha256': sha(OUT / 'validation/cpu-validation.json'),
            'cpu_checks': len(cpu['results']),
            'cpu_all_passed': all(row['exit_code'] == 0 for row in cpu['results']),
            'cpu_gpu_dispatch': cpu.get('gpu_dispatch', cpu.get('metal_dispatch', False)),
            'metal_report_path': str(OUT / 'validation/metal-validation.json'),
            'metal_report_sha256': sha(OUT / 'validation/metal-validation.json'),
            'metal_compile_link_steps': len(metal['results']),
            'metal_all_passed': all(row['exit_code'] == 0 for row in metal['results']),
            'metal_optimization_levels': ['O0', 'O3'],
            'objc_syntax_logs': [
                'validation/runtime_objc_syntax.log',
                'validation/full_contact_tool_objc_syntax.log',
                'validation/shipping_memory_tool_syntax.log',
            ],
            'integration_verification_path': str(OUT / 'validation/integration-verification.json'),
            'integration_verification_sha256': sha(OUT / 'validation/integration-verification.json'),
            'shipping_admission_report_path': str(OUT / 'shipping-admission-audit-001/report.json'),
            'shipping_admission_report_sha256': sha(OUT / 'shipping-admission-audit-001/report.json'),
            'shipping_first_scope_gate': shipping['first_explicit_full_contact_armijo_gate']['reason'],
        },
        'limits': [
            'No full CMake source build or native Metal dispatch was performed in this overlay task.',
            'The sparse native checker was compiled and only its --cpu-only route was run; native differential remains pending parent GPU queue.',
            'Metal O0/O3 compile and metallib linking succeeded; two inherited fgmres.metalinc initialization warnings remain.',
            'No shipping admission, 16 GiB peak, performance, or physical qualification is claimed.',
        ],
        'evidence_inventory': inventory,
        'manifest_does_not_self_hash': True,
    }
    MANIFEST.write_text(json.dumps(doc, sort_keys=True, indent=2) + '\n')
    print(json.dumps({
        'manifest': str(MANIFEST),
        'manifest_sha256': sha(MANIFEST),
        'changed_sources': len(changed),
        'inventory_files': len(inventory),
        'gpu_dispatch': False,
    }, sort_keys=True))


if __name__ == '__main__':
    main()
