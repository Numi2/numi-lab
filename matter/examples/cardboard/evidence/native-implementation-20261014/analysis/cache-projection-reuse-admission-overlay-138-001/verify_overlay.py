#!/usr/bin/env python3
"""Verify the narrow source136 cache-reuse admission correction."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PKG = Path(__file__).resolve().parent
BASE = Path('/Users/home/cardboard-implementation-20261014/snapshots/source-cache-projection-reuse-runtime-136')
RETAINED = Path('/Users/home/cardboard-implementation-20261014/analysis/retained-source-snapshot136-001')
SOURCE134 = Path('/Users/home/cardboard-implementation-20261014/snapshots/source-cache-projection-reuse-checker-fixed-134')
CASE = Path('/Users/home/cardboard-implementation-20261014/solver/cache-reuse-full-history-parity-437')
EXPECTED = {
    'matter/src/metal/fem.metalinc': '09c3a806224067777aadd26274581ff46be8c9b14282525a7ff805bbb335d03c',
    'matter/src/runtime.mm': '8c56679c1e547da78f5973382a07ea371b0f3492bebd611cfe73e15f952c959b',
}
FILES = tuple(EXPECTED)
GUARD = (b'    // Runtime mode 1 is admitted only for this exact coupon material policy.\n'
         b'    // Do not silently turn an explicitly requested reuse build into the\n'
         b'    // ordinary repeated-projection path if an input escapes that admission.\n'
         b'    if (!reuseAccuratePaperProjection) return;\n')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'expected regular file: {path}')
    return path.read_bytes()


def function_body(data: bytes, name: str) -> bytes:
    text = data.decode('utf-8')
    pos = text.find(name + '(')
    if pos < 0:
        raise ValueError(f'missing function {name}')
    begin = text.find('{', pos)
    if begin < 0:
        raise ValueError(f'missing function body {name}')
    depth = 0
    for i in range(begin, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[begin:i + 1].encode('utf-8')
    raise ValueError(f'unclosed function {name}')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', type=Path, default=PKG / 'logs/source-review-report.json')
    args = ap.parse_args()

    retained = json.loads(read_regular(RETAINED / 'manifest.json'))
    snapshot_record = retained['snapshots'][0]
    packed = read_regular(RETAINED / snapshot_record['manifest'])
    snapshot_bytes = gzip.decompress(packed)
    snapshot = json.loads(snapshot_bytes)
    require(snapshot_record['number'] == 136 and snapshot['number'] == 136,
            'retained snapshot is not source136')
    require(sha(packed) == snapshot_record['compressed_sha256'], 'source136 gzip hash mismatch')
    require(sha(snapshot_bytes) == snapshot_record['uncompressed_sha256'], 'source136 manifest hash mismatch')
    require(Path(snapshot['source_snapshot']) == BASE, 'source136 points at another frozen snapshot')
    entries = {entry['path']: entry for entry in snapshot['files']}

    base_hashes = {}
    candidate_hashes = {}
    for rel, expected in EXPECTED.items():
        b = read_regular(PKG / 'base' / rel)
        c = read_regular(PKG / 'candidate' / rel)
        frozen = read_regular(BASE / rel)
        require(sha(b) == expected and sha(frozen) == expected, f'base/source136 hash mismatch: {rel}')
        require(entries.get(rel, {}).get('sha256') == expected, f'retained manifest mismatch: {rel}')
        base_hashes[rel] = sha(b)
        candidate_hashes[rel] = sha(c)

    base_fem = read_regular(PKG / 'base/matter/src/metal/fem.metalinc')
    candidate_fem = read_regular(PKG / 'candidate/matter/src/metal/fem.metalinc')
    old_kernel = 'nm_fem_build_algorithmic_tangent_cache'
    sibling = 'nm_fem_build_algorithmic_tangent_cache_projection_reuse_diagnostic'
    require(function_body(base_fem, old_kernel) == function_body(candidate_fem, old_kernel),
            'original production tangent-cache kernel changed')
    candidate_sibling = function_body(candidate_fem, sibling)
    require(candidate_sibling.count(GUARD) == 0, 'old fail-closed 15-state-only shader guard remains')
    require(function_body(read_regular(SOURCE134 / 'matter/src/metal/fem.metalinc'), sibling) ==
            candidate_sibling, 'reuse sibling differs from source134 after admission guard removal')

    runtime = read_regular(PKG / 'candidate/matter/src/runtime.mm').decode('utf-8')
    scope_start = runtime.index('const char* cacheProjectionReuse =')
    scope_end = runtime.index('const TiledCohesiveArmijoOptions tiledCohesiveOptions', scope_start)
    scope = runtime[scope_start:scope_end]
    # Compiler source proves IPC is an unconditional capability bit. All actual
    # contact, adaptive, mixed, multiphysics, MPM and mutation gates remain.
    compiler = read_regular(BASE / 'matter/src/compiler_impl.cpp').decode('utf-8')
    compiler_dispatch = compiler[compiler.index('dispatch.flags ='):compiler.index('dispatch.flags |=', compiler.index('dispatch.flags ='))]
    require('NM_MATTER_IPC |' in compiler_dispatch, 'source136 IPC unconditional-flag evidence changed')
    require('NM_MATTER_IPC' not in scope, 'IPC capability bit still blocks mode 1')
    for token in ('NM_MATTER_ADAPTIVE', 'NM_MATTER_CONTACT', 'NM_MATTER_MIXED_FEM',
                  'NM_MATTER_MULTIPHYSICS', 'NM_MATTER_MUTATION',
                  'world.dispatch.gridNodeCount != 0u', 'world.dispatch.particleCount != 0u',
                  'world.dispatch.mpmGridCount != 0u', 'world.dispatch.mpmBlockCount != 0u'):
        require(token in scope, f'physical-scope rejection was removed: {token}')
    for token in ('world.dispatch.rigidProxyCount != 0u', 'world.dispatch.contactPairCount != 0u',
                  'NM_OBJECT_DISABLE_SELF_CONTACT', 'NM_OBJECT_DISABLE_DEFORMABLE_CONTACT'):
        require(token in scope, f'contact rejection was removed: {token}')

    starch_checks = (
        'NM_MATERIAL_PAPER_COMPENSATED_ARITHMETIC',
        'NM_MATERIAL_HAS_STATE', 'NM_MATERIAL_HAS_IMPLICIT_STATE',
        'NM_MATERIAL_HAS_DISSIPATION', 'NM_MATERIAL_PAPER_HILL_SCALAR_INITIALIZER',
        'NM_MATERIAL_PAPER_HILL_ACCURATE_PROJECTION',
        'NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION',
        'NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION',
        'material.stateCount == 0u', 'material.parameterCount == 3u',
        'NM_CONSTITUTIVE_NEO_HOOKEAN', 'NM_MATERIAL_PROJECTION_GENERIC',
        'material.stateUpdateProgramOffset == NM_INVALID_INDEX',
        'material.dissipationProgram == NM_INVALID_INDEX',
        'material.viscousStressProgramOffset == NM_INVALID_INDEX',
        'material.viscousTangentProgramOffset == NM_INVALID_INDEX',
        'material.implicitResidualProgramOffset == NM_INVALID_INDEX',
        'material.implicitJacobianProgramOffset == NM_INVALID_INDEX',
        'material.implicitDeformationProgramOffset == NM_INVALID_INDEX',
        'material.stressStateDerivativeProgramOffset == NM_INVALID_INDEX')
    require('const bool finiteElasticStarch =' in scope, 'finite elastic starch fallback classification missing')
    for token in starch_checks:
        require(token in scope, f'starch fallback is not tightly bounded: {token}')
    require('if (finiteElasticStarch) continue;' in scope,
            'stateless starch does not stay on the existing repeated-projection path')
    require('reusableTetrahedra == 0u' in scope and 'material.stateCount != 15u' in scope,
            'at least one accurate 15-state paper element is not required')
    require('active materials must be finite-elastic starch or compensated scaled generic 15-state paper' in scope,
            'admission error does not report supported material classes')

    validator = read_regular(BASE / 'matter/src/validation_impl.cpp').decode('utf-8')
    require('const bool finiteElasticStarchArithmetic' in validator and
            'material.constitutiveKind==NM_CONSTITUTIVE_NEO_HOOKEAN' in validator,
            'source136 finite-elastic starch validator definition changed')
    for token in ('parameterCount==3u', 'material.stateCount==0u',
                  'NM_MATERIAL_PAPER_COMPENSATED_ARITHMETIC',
                  'NM_MATERIAL_HAS_IMPLICIT_STATE|NM_MATERIAL_HAS_DISSIPATION'):
        require(token in validator, f'validator starch policy missing {token}')

    case_manifest = json.loads(read_regular(CASE / 'output/manifest.json'))
    glue = case_manifest['materials']
    require(glue['glue_path'].endswith('/starch2007_finite_glue_elastic.nmatter'),
            'native437 case does not identify finite elastic starch')
    cell_names = {cell['name']: cell['tetrahedra'] for cell in glue['cells']}
    require(cell_names.get('starch2007_finite_glue_elastic') == 192 and
            cell_names.get('hajali2009_liner_hill_bilinear_fit') == 1152 and
            cell_names.get('hajali2009_medium_hill_bilinear_fit') == 576,
            'native437 material cell composition changed')
    failed_text = read_regular(CASE / 'output/failure.txt').decode('utf-8')
    require('adaptive_MPM_or_multiphysics_state_is_unsupported' in failed_text and
            'every_active_tetrahedron_requires_compensated_scaled_generic_15_state_material' in failed_text,
            'native437 original admission failure not retained')

    patch_data = read_regular(PKG / 'overlay.patch')
    dry = subprocess.run(['patch', '--dry-run', '-p1', '-d', str(PKG / 'base')],
                         input=patch_data, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    require(dry.returncode == 0, 'patch does not apply to its exact base')
    links = [str(path.relative_to(PKG)) for path in PKG.rglob('*') if path.is_symlink()]
    require(not links, f'symlinks are not permitted: {links}')

    report = {
        'schema': 'numi.cardboard.cache_projection_reuse_admission_review.v1',
        'base_snapshot': str(BASE),
        'retained_snapshot_manifest': str(RETAINED / snapshot_record['manifest']),
        'retained_snapshot_compressed_sha256': sha(packed),
        'retained_snapshot_uncompressed_sha256': sha(snapshot_bytes),
        'base_file_sha256': base_hashes,
        'candidate_file_sha256': candidate_hashes,
        'source134_reuse_sibling_exact_after_guard_removal': True,
        'production_tangent_cache_kernel_unchanged': True,
        'source136_unconditional_ipc_flag_verified': True,
        'only_ipc_capability_rejection_removed': True,
        'adaptive_mpm_contact_multiphysics_and_mutation_rejections_retained': True,
        'finite_elastic_starch_is_limited_to_zero_state_generic_neohookean_three_parameter_no_rate_no_state_path': True,
        'starch_elements_use_legacy_projection_path': True,
        'at_least_one_reusable_accurate_15_state_material_required': True,
        'native437_original_failure_retained': True,
        'native_off_on_history_parity': 'NOT_RUN',
        'physical_qualification': False,
        'patch_sha256': sha(patch_data),
        'patch_dry_run': 'PASS',
        'symlinks': links,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'verification failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
