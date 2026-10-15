#!/usr/bin/env python3
"""Verify source lineage and static invariants for overlay 136-001."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

PKG = Path(__file__).resolve().parent
BASE_SNAPSHOT = Path('/Users/home/cardboard-implementation-20261014/snapshots/source-cauchy-positive-mass-admission-135')
RETAINED135 = Path('/Users/home/cardboard-implementation-20261014/analysis/retained-source-snapshot135-001')
SOURCE134 = Path('/Users/home/cardboard-implementation-20261014/snapshots/source-cache-projection-reuse-checker-fixed-134')
EXPECTED_BASE = {
    'matter/src/metal/common.metalinc': '441a3f86f21979a086a0356f5fd2b91f692ba94dbd09a28d07ea9756c1246a5a',
    'matter/src/metal/fem.metalinc': '25b2f3a86999df08b2fa84f14058657274154b4b9671120e5b79248dff351c79',
    'matter/src/runtime.mm': 'f5030219a901ae9acafad1ef9e2d68ad1385d1d5e0b91b893633834aebf9e2b1',
}
FILES = tuple(EXPECTED_BASE)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'expected regular file, got {path}')
    return path.read_bytes()


def function_body(data: bytes, name: str) -> bytes:
    text = data.decode('utf-8')
    marker = name + '(' 
    pos = text.find(marker)
    if pos < 0:
        raise ValueError(f'missing function {name}')
    begin = text.find('{', pos)
    if begin < 0:
        raise ValueError(f'missing function body {name}')
    depth = 0
    for index in range(begin, len(text)):
        if text[index] == '{':
            depth += 1
        elif text[index] == '}':
            depth -= 1
            if depth == 0:
                return text[begin:index + 1].encode('utf-8')
    raise ValueError(f'unclosed function body {name}')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=PKG / 'logs/source-review-report.json')
    args = parser.parse_args()

    snapshot_record = json.loads((RETAINED135 / 'manifest.json').read_text())['snapshots'][0]
    source_gzip = RETAINED135 / snapshot_record['manifest']
    compressed = get_bytes(source_gzip)
    uncompressed = gzip.decompress(compressed)
    snapshot_doc = json.loads(uncompressed)
    require(sha(compressed) == snapshot_record['compressed_sha256'], 'retained source135 compressed hash mismatch')
    require(sha(uncompressed) == snapshot_record['uncompressed_sha256'], 'retained source135 source hash mismatch')
    require(snapshot_doc['number'] == 135, 'retained source bundle is not source135')
    require(Path(snapshot_doc['source_snapshot']) == BASE_SNAPSHOT, 'source135 bundle points at another snapshot')
    source_entries = {entry['path']: entry for entry in snapshot_doc['files']}

    source_hashes = {}
    candidate_hashes = {}
    for rel, expected in EXPECTED_BASE.items():
        base_path = PKG / 'base' / rel
        candidate_path = PKG / 'candidate' / rel
        snapshot_path = BASE_SNAPSHOT / rel
        base_data = get_bytes(base_path)
        candidate_data = get_bytes(candidate_path)
        snapshot_data = get_bytes(snapshot_path)
        require(sha(base_data) == expected, f'overlay base hash mismatch: {rel}')
        require(sha(snapshot_data) == expected, f'frozen source135 hash mismatch: {rel}')
        require(rel in source_entries and source_entries[rel]['sha256'] == expected,
                f'retained source135 manifest binding mismatch: {rel}')
        source_hashes[rel] = expected
        candidate_hashes[rel] = sha(candidate_data)

    common = get_bytes(PKG / 'candidate/matter/src/metal/common.metalinc')
    common134 = get_bytes(SOURCE134 / 'matter/src/metal/common.metalinc')
    require(common == common134, 'common.metalinc is not the source134 helper transplant')

    old_builder = 'nm_fem_build_algorithmic_tangent_cache'
    base_fem = get_bytes(PKG / 'base/matter/src/metal/fem.metalinc')
    candidate_fem = get_bytes(PKG / 'candidate/matter/src/metal/fem.metalinc')
    require(function_body(base_fem, old_builder) == function_body(candidate_fem, old_builder),
            'the production tangent-cache kernel body changed')

    sibling = 'nm_fem_build_algorithmic_tangent_cache_projection_reuse_diagnostic'
    candidate_sibling = function_body(candidate_fem, sibling)
    source134_sibling = function_body(get_bytes(SOURCE134 / 'matter/src/metal/fem.metalinc'), sibling)
    guard = (b'    // Runtime mode 1 is admitted only for this exact coupon material policy.\n'
             b'    // Do not silently turn an explicitly requested reuse build into the\n'
             b'    // ordinary repeated-projection path if an input escapes that admission.\n'
             b'    if (!reuseAccuratePaperProjection) return;\n')
    require(candidate_sibling.count(guard) == 1, 'sibling fail-closed guard missing or duplicated')
    require(candidate_sibling.replace(guard, b'') == source134_sibling,
            'sibling differs from source134 beyond the explicit fail-closed guard')

    runtime = get_bytes(PKG / 'candidate/matter/src/runtime.mm').decode('utf-8')
    checks = {
        'strict_environment_selector': 'NM_FEM_CACHE_PROJECTION_REUSE must be exactly 0 or 1' in runtime,
        'mode_default_off': 'bool femCacheProjectionReuseEnabled = false;' in runtime,
        'unsupported_request_rejected': 'if (!ineligible.empty())' in runtime and 'return diagnostics;' in runtime,
        'single_static_fem_scope': 'exactly_one_authored_FEM_object_required' in runtime and 'environment_count_must_be_1_or_2' in runtime,
        'no_contact_scope': 'rigid_or_contact_state_is_unsupported' in runtime and 'object_must_be_active_immutable_contact_free_FEM' in runtime,
        'material_policy_scope': 'every_active_tetrahedron_requires_compensated_scaled_generic_15_state_material' in runtime,
        'original_pipeline_skipped_when_off': 'if (!candidate->femCacheProjectionReuseEnabled' in runtime,
        'runtime_selects_sibling_only_when_enabled': 'state.femCacheProjectionReuseEnabled\n                        ? "nm_fem_build_algorithmic_tangent_cache_projection_reuse_diagnostic"' in runtime,
    }
    require(all(checks.values()), 'runtime selector/admission invariant missing: ' + ', '.join(k for k, v in checks.items() if not v))
    for name in ('NM_MATTER_IDENTIFICATION', 'NM_MATTER_IPC', 'learnedMaterialCount', 'learnedLayerCount', 'learnedWeightCount'):
        require(name in runtime, f'unsupported-state admission guard missing {name}')

    patch_data = get_bytes(PKG / 'overlay.patch')
    patch_result = subprocess.run(
        ['patch', '--dry-run', '-p1', '-d', str(PKG / 'base')],
        input=patch_data, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    require(patch_result.returncode == 0, 'overlay.patch does not apply cleanly to the bound base')

    symlinks = [str(p.relative_to(PKG)) for p in PKG.rglob('*') if p.is_symlink()]
    require(not symlinks, f'package contains symlinks: {symlinks}')
    report = {
        'schema': 'numi.cardboard.cache_projection_reuse_source_review.v1',
        'base_snapshot': str(BASE_SNAPSHOT),
        'retained_source135': {
            'bundle': str(source_gzip),
            'compressed_sha256': sha(compressed),
            'uncompressed_sha256': sha(uncompressed),
            'source_build_sha256': snapshot_doc['source_build_sha256'],
        },
        'source_file_sha256': source_hashes,
        'candidate_file_sha256': candidate_hashes,
        'source134_common_helper_exact_match': True,
        'original_production_cache_kernel_exact_match': True,
        'sibling_kernel_matches_source134_after_only_admission_guard_removed': True,
        'runtime_invariants': checks,
        'symlinks': symlinks,
        'patch_sha256': sha(patch_data),
        'patch_dry_run': 'PASS',
        'native_history_parity': 'NOT_RUN',
        'physical_qualification': False,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'verification failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
