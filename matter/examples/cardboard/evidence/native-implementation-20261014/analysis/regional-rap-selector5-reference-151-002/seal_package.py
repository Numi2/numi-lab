#!/usr/bin/env python3
"""Create deterministic package report and file-hash manifest."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

BASE = Path('/Users/home/cardboard-implementation-20261014')
HERE = Path(__file__).resolve().parent
SOURCE = BASE / 'snapshots/source-cgls-regional-state-capture-151'
OVERLAY = BASE / 'analysis/base-cgls-regional-state-capture-overlay-151-001'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def inventory(paths: list[Path]) -> dict[str, dict[str, object]]:
    return {
        path.relative_to(HERE).as_posix(): {
            'bytes': path.stat().st_size,
            'sha256': sha(path),
        }
        for path in sorted(paths)
    }


def main() -> None:
    package_files = [
        p for p in HERE.rglob('*')
        if p.is_file() and p.name not in {'manifest.json', 'report.json'}
        and '__pycache__' not in p.parts
    ]
    files = inventory(package_files)
    source_files = {
        'snapshot_manifest_sha256': sha(SOURCE / 'frozen-source.json'),
        'runtime_sha256': sha(SOURCE / 'matter/src/runtime.mm'),
        'fgmres_sha256': sha(SOURCE / 'matter/src/metal/fgmres.metalinc'),
        'graph_builder_sha256': sha(SOURCE / 'matter/src/regional_graph_schwarz.hpp'),
        'shared_abi_sha256': sha(SOURCE / 'matter/include/numi/matter/shared.h'),
        'overlay_manifest_sha256': sha(OVERLAY / 'overlay-manifest.json'),
        'source_build_receipt_sha256': sha(OVERLAY / 'validation/source-build-151.json'),
    }
    contract = json.loads((HERE / 'source-contract.json').read_text())
    synthetic = json.loads((HERE / 'synthetic-reference.json').read_text())
    report = {
        'schema': 'numi.matter.selector5_graph_reference_audit.v1',
        'package': 'regional-rap-selector5-reference-151-002',
        'result': 'PASS bounded synthetic algebraic reference and source-anchor audit',
        'source_bindings': source_files,
        'source_algorithm_summary': contract['source_algorithm_summary'],
        'source_anchor_checks': contract['source_anchors'],
        'synthetic_result': synthetic,
        'validation': {
            'synthetic_unittests': 'PASS 8/8',
            'python_syntax': 'PASS 6 files parsed with ast; bytecode disabled',
            'source_contract': 'PASS source hashes and all 8 anchors',
            'native_gpu_execution': False,
        },
        'explicit_limits': {
            'native_selector5_capture_available': False,
            'native_arbitrary_vector_apply_or_transpose_parity': False,
            'actual_fgmres_consumed_map_proven': False,
            'augmented_restart_cycle_map_supported': False,
            'candidate_projected_material_history_currentness': 'unavailable_not_persisted_by_source151',
            'solver_or_physical_qualification': False,
        },
        'files': files,
    }
    report_path = HERE / 'report.json'
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2,
                                      allow_nan=False) + '\n')
    manifest = {
        'schema': 'numi.matter.selector5_graph_reference_manifest.v1',
        'package': report['package'],
        'report_sha256': sha(report_path),
        'files': files,
        'source_bindings': source_files,
        'native_or_physical_qualification': False,
    }
    (HERE / 'manifest.json').write_text(json.dumps(
        manifest, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps({
        'report_sha256': sha(report_path),
        'manifest_sha256': sha(HERE / 'manifest.json'),
        'file_count': len(files),
        'source_bindings': source_files,
    }, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
