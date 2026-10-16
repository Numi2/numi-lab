#!/usr/bin/env python3
"""Source-derived audit for installer bindings and legacy arithmetic parity.

This checks the actual candidate/source function text and checker dispatches;
it does not simulate Metal behavior. The executable Metal checker is compiled
separately and must be run by the GPU owner for behavioral evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path('/Users/home/cardboard-implementation-20261014')
PKG = ROOT / 'analysis/base-cgls-warm-start-refinement-overlay-159-002'
SNAPSHOT = ROOT / 'snapshots/source-right-grip-telemetry-157'


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def function_body(source: str, signature: str) -> str:
    start = source.find(signature)
    require(start >= 0, f'missing function: {signature}')
    brace = source.find('{', start)
    require(brace >= 0, f'missing body: {signature}')
    depth = 0
    for i in range(brace, len(source)):
        if source[i] == '{':
            depth += 1
        elif source[i] == '}':
            depth -= 1
            if depth == 0:
                return source[brace:i + 1]
    raise AssertionError(f'unterminated body: {signature}')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    kernel_path = PKG / 'candidate/matter/src/metal/base_cgls_runtime.metalinc'
    runtime_path = PKG / 'candidate/matter/src/runtime.mm'
    checker_metal_path = PKG / 'native/warm_start_check.metal'
    checker_host_path = PKG / 'native/warm_start_check.mm'
    fgmres_path = SNAPSHOT / 'matter/src/metal/fgmres.metalinc'
    kernel = kernel_path.read_text()
    runtime = runtime_path.read_text()
    checker_metal = checker_metal_path.read_text()
    checker_host = checker_host_path.read_text()
    fgmres = fgmres_path.read_text()

    installer_signature = 'kernel void nm_fgmres_install_base_cgls_direction('
    installer_start = kernel.find(installer_signature)
    require(installer_start >= 0, 'missing installer declaration')
    installer_open = kernel.find('{', installer_start)
    installer_header = kernel[installer_start:installer_open]
    installer = function_body(kernel, installer_signature)
    require('device const float4* baselineResidual [[buffer(16)]]' in installer_header,
            'installer must declare the baseline residual at buffer 16')
    require('constant uint& requireResidualImprovement [[buffer(17)]]' in installer_header,
            'installer must declare the warm/legacy selector at buffer 17')
    require('const float4 initialResidual = baselineResidual[global];' in installer,
            'baseline residual read is absent')
    require('const bool refinementAccepted = valid && refinementMode &&\n        rn < baselineResidual2;' in installer,
            'warm acceptance must use strict fresh-residual reduction only')
    require('refinementMode ? xn >= 0.0f : xn > 0.0f' in installer,
            'zero-norm acceptance must be limited to the warm branch')
    require('refinementMode ? (isfinite(baselineResidual2) && baselineResidual2 >= 0.0f)' in installer,
            'warm branch must validate the finite baseline residual norm')
    require('fgmresCompensatedAdd(xn, xc' in installer and
            'fgmresCompensatedAdd(rn, rc' in installer and
            'fgmresCompensatedAdd(descent, dc' in installer,
            'installer must retain the original FGMRES accumulation helper')
    require('baseCGLSCompensatedAdd(xn, xc' not in installer and
            'baseCGLSCompensatedAdd(rn, rc' not in installer and
            'baseCGLSCompensatedAdd(descent, dc' not in installer,
            'installer changed legacy accumulation helper')

    original_helper = function_body(fgmres, 'inline void fgmresCompensatedAdd(')
    checker_helper = function_body(checker_metal, 'inline void fgmresCompensatedAdd(')
    require(original_helper == checker_helper,
            'checker helper is not byte-identical to frozen production FGMRES helper')

    runtime_call = 'dispatchThreads("nm_fgmres_install_base_cgls_direction", environments, [&] {'
    pos = runtime.find(runtime_call)
    require(pos >= 0, 'runtime installer dispatch not found')
    runtime_dispatch = function_body(runtime[pos:], runtime_call)
    require('[encoder setBuffer:state.femBaseCGLSB' in runtime_dispatch and
            'offset:0u atIndex:16u' in runtime_dispatch,
            'runtime does not bind fresh baseline residual at 16')
    require('atIndex:17u' in runtime_dispatch and
            'warmStartRefinementThisNewton ? 1u : 0u' in runtime_dispatch,
            'runtime does not bind explicit warm/legacy mode at 17')

    install_host = function_body(checker_host, 'void install(')
    require('bind(e,baseline,16);' in install_host,
            'checker caller does not bind baseline residual at 16')
    require('bytes(e,requireImprovement,17);' in install_host,
            'checker caller does not bind explicit mode at 17')
    mm_callers = []
    for path in sorted((PKG / 'candidate').rglob('*.mm')) + sorted((PKG / 'native').rglob('*.mm')):
        text = path.read_text(errors='replace')
        if 'nm_fgmres_install_base_cgls_direction' in text:
            mm_callers.append(str(path.relative_to(PKG)))
    require(mm_callers == ['candidate/matter/src/runtime.mm', 'native/warm_start_check.mm'],
            f'unexpected or unreviewed .mm installer callers: {mm_callers}')

    require('zero-total strict improvement' in checker_host and
            'legacy zero-norm direction was not rejected' in checker_host,
            'checker lacks separate warm-zero and legacy-zero controls')
    require('zeroCandidateResidual2<poorInitialResidual2' in checker_host,
            'zero-total checker control does not establish strict residual reduction')

    record = {
        'schema': 'numi.cardboard.cgls_warm_start_source_contract.v2',
        'status': 'pass',
        'behavioral_execution': False,
        'checks': {
            'buffer16_baseline_declared_and_runtime_bound': True,
            'buffer17_mode_declared_and_runtime_bound': True,
            'checker_binds_both_installer_inputs': True,
            'all_candidate_mm_callers_enumerated': mm_callers,
            'warm_gate_allows_zero_total_if_fresh_residual_strictly_decreases': True,
            'legacy_zero_total_remains_invalid': True,
            'legacy_accumulation_helper_matches_frozen_fgmres': True,
            'checker_helper_matches_frozen_fgmres': True,
            'quiet_unselected_seed_x0_zero_invariant':
                'conditionally supported by cycle-zero FGMRES accumulation when all FEM node ranges are valid: zero used columns write a zero solution; unreferenced invalid node ranges are not overwritten and remain outside this proof',
        },
        'sha256': {
            'installer': sha(kernel_path),
            'runtime': sha(runtime_path),
            'checker_metal': sha(checker_metal_path),
            'checker_host': sha(checker_host_path),
            'frozen_fgmres': sha(fgmres_path),
        },
    }
    out = PKG / 'tests/source-contract-check.json'
    out.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'status': 'pass', 'mm_callers': mm_callers,
                      'record': str(out)}, sort_keys=True))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'source contract failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
