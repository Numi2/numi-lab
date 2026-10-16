#!/usr/bin/env python3
"""Regenerate the source-bound v2 patch and manifest after CPU checks."""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path('/Users/home/cardboard-implementation-20261014')
PKG = ROOT / 'analysis/base-cgls-warm-start-refinement-overlay-159-002'
PARENT = ROOT / 'analysis/base-cgls-warm-start-refinement-overlay-159-001'
SNAPSHOT = ROOT / 'snapshots/source-right-grip-telemetry-157'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def info(path: Path) -> dict[str, object]:
    return {'bytes': path.stat().st_size, 'sha256': digest(path)}


def main() -> None:
    source_map = {
        'matter/src/metal/base_cgls.metalinc': ('matter/src/metal/base_cgls.metalinc',
            '127d05d6323085ad6abd160ff560b95008981e9421541e9677dddbb3fa253aec'),
        'matter/src/metal/base_cgls_runtime.metalinc': ('matter/src/metal/base_cgls_runtime.metalinc',
            '185d575646d9aef59e60bb70b182caf60ca06cf08b161ca78c32d875e6d6a903'),
        'matter/src/runtime.mm': ('matter/src/runtime.mm',
            '69d012aeaeff6bf7ef45ede8817d9664aea21b756383fe46b38325c75c6ce743'),
    }
    diffs: list[str] = []
    changed_files: dict[str, object] = {}
    for rel, (candidate_rel, expected_base_sha) in source_map.items():
        base = SNAPSHOT / rel
        candidate = PKG / 'candidate' / candidate_rel
        if digest(base) != expected_base_sha:
            raise SystemExit(f'frozen base hash mismatch: {rel}')
        base_text = base.read_text().splitlines(keepends=True)
        candidate_text = candidate.read_text().splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            base_text, candidate_text, fromfile=f'a/{rel}', tofile=f'b/{rel}'))
        diffs.extend(diff)
        changed_files[rel] = {
            'base': info(base), 'candidate': info(candidate),
            'changed_from_base': base.read_bytes() != candidate.read_bytes(),
        }
    patch_path = PKG / 'overlay.patch'
    patch_path.write_text(''.join(diffs))

    v1 = json.loads((PARENT / 'source-overlay.json').read_text())
    frozen = json.loads((SNAPSHOT / 'frozen-source.json').read_text())
    contract = json.loads((PKG / 'tests/source-contract-check.json').read_text())
    runtime_syntax = json.loads((PKG / 'native/runtime-syntax-only.json').read_text())
    compile_log = PKG / 'native/compile-link-v2.log'
    metallib = PKG / 'native/build/WarmStartCheck.metallib'
    air = PKG / 'native/build/warm_start_check.air'
    executable = PKG / 'native/build/warm_start_check'
    manifest = {
        'schema': 'numi.cardboard.base_cgls_warm_start_refinement_overlay.v2',
        'overlay_id': 'base_cgls_warm_start_refinement_159_002',
        'parent_overlay': {
            'overlay_id': v1['overlay_id'],
            'manifest_path': str(PARENT / 'source-overlay.json'),
            'manifest_sha256': digest(PARENT / 'source-overlay.json'),
            'patch_sha256': digest(PARENT / 'overlay.patch'),
            'preserved_immutable': True,
        },
        'base_snapshot': str(SNAPSHOT),
        'base_parent_revision': frozen.get('parent_revision'),
        'base_frozen_source_sha256': digest(SNAPSHOT / 'frozen-source.json'),
        'base_source_overlay_sha256': digest(SNAPSHOT / 'source-overlay.json'),
        'base_declared_source_file_count': v1.get('base_declared_source_file_count'),
        'base_declared_source_hashes_verified': v1.get('base_declared_source_hashes_verified'),
        'cumulative_source_delta_from_157': changed_files,
        'delta_since_159_001': {
            'production_files': ['matter/src/metal/base_cgls_runtime.metalinc'],
            'checker_files': ['native/warm_start_check.metal', 'native/warm_start_check.mm'],
            'support_files': ['native/compile_only.sh', 'native/runtime_syntax_only.py',
                              'tests/source_contract_check.py'],
            'reason': 'allow strict warm residual reduction when total x is zero; preserve legacy xn>0 and original FGMRES compensated accumulation helper; enforce all installer call bindings',
        },
        'environment_contract': {
            'new_registry_key': 'NM_FEM_BASE_CGLS_WARM_START_REFINE',
            'enabled_value': '1', 'default': 'off',
            'checkpoint_compatibility': 'candidate-enabled run must start at control step 0; no digest-changing resume is supported',
        },
        'installer_contract': {
            'slot16': 'existing CGLS B buffer containing the masked fresh baseline residual b-A*x0; bound in runtime and checker even in legacy mode because the kernel declaration reads it unconditionally',
            'slot17': 'warm/legacy selector; 1 only on explicitly enabled selected-policy Newton, otherwise 0',
            'warm_acceptance': 'finite candidate with strictly smaller recomputed unscaled residual norm; no positive solution-norm requirement',
            'legacy_acceptance': 'retains xn>0 and positive-descent validity checks',
            'legacy_accumulation': 'installer uses frozen fgmresCompensatedAdd for pre-existing solution/residual/descent sums; checker helper is exact body match',
        },
        'selected_zero_seed_invariant': {
            'assessment': 'source-supported for valid one-object FEM node ranges; cycle-zero FGMRES accumulation overwrites a converged zero-column solution with zero',
            'limitation': 'accumulation skips NM_INVALID_INDEX node ranges; no proof is claimed for orphan rows',
            'audit': info(PKG / 'selected-zero-x0-audit.md'),
        },
        'validation': {
            'source_contract': info(PKG / 'tests/source-contract-check.json'),
            'source_contract_result': contract['status'],
            'metal_air': info(air), 'metallib': info(metallib),
            'checker_executable_linked_not_run': info(executable),
            'compile_link_log': info(compile_log),
            'compile_script': info(PKG / 'native/compile_only.sh'),
            'runtime_syntax_record': info(PKG / 'native/runtime-syntax-only.json'),
            'runtime_syntax_exit_code': runtime_syntax['exit_code'],
            'runtime_syntax_only': True,
            'native_checker_executed': False,
            'gpu_execution': False,
            'full_solver_trial': False,
        },
        'checker_controls': [
            'fresh residual b-A*x0 seeding', 'x0 plus CGLS correction',
            'strict improved residual install', 'equal residual decline preserving x0 and certificate',
            'negative b dot A*x with residual improvement',
            'zero total x accepted only in warm mode when fresh residual strictly decreases',
            'zero-norm legacy direction rejected while preserving x0 and certificate',
        ],
        'new_persistent_arena_bytes': 0,
        'operator_changed': False,
        'nonlinear_acceptance_changed': False,
        'patch': info(patch_path),
    }
    out = PKG / 'source-overlay.json'
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'manifest': str(out), 'manifest_sha256': digest(out),
                      'patch_sha256': digest(patch_path)}, sort_keys=True))


if __name__ == '__main__':
    main()
