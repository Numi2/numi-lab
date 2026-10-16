#!/usr/bin/env python3
"""Hash and anchor audit for the source-derived selector-5 map reference."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

BASE = Path('/Users/home/cardboard-implementation-20261014')
S151 = BASE / 'snapshots/source-cgls-regional-state-capture-151'
S149 = BASE / 'snapshots/source-cgls-block-right-preconditioned-149'
OUT = Path(__file__).resolve().parent

EXPECTED = {
    'source149_fgmres': ('snapshots/source-cgls-block-right-preconditioned-149/matter/src/metal/fgmres.metalinc', '8e0b9da3e28e7997eea9ee9db37e89a0bf3b9753d98967c80221e328e5da2af8'),
    'source149_runtime': ('snapshots/source-cgls-block-right-preconditioned-149/matter/src/runtime.mm', 'f859b1ee45390dd844fcbd883356b931ab71d2c36260f53a4b03600b04fb3959'),
    'source149_base_cgls': ('snapshots/source-cgls-block-right-preconditioned-149/matter/src/metal/base_cgls.metalinc', '127d05d6323085ad6abd160ff560b95008981e9421541e9677dddbb3fa253aec'),
    'source151_runtime': ('snapshots/source-cgls-regional-state-capture-151/matter/src/runtime.mm', '1c0d2fd0d376ecc5e7ea9007ab6cc67a77fbd19bbba9f3b22fcbeca4c5c06a08'),
    'source151_fgmres': ('snapshots/source-cgls-regional-state-capture-151/matter/src/metal/fgmres.metalinc', '8e0b9da3e28e7997eea9ee9db37e89a0bf3b9753d98967c80221e328e5da2af8'),
    'source151_shared': ('snapshots/source-cgls-regional-state-capture-151/matter/include/numi/matter/shared.h', None),
    'source151_graph_builder': ('snapshots/source-cgls-regional-state-capture-151/matter/src/regional_graph_schwarz.hpp', None),
}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def kernel_body(text: str, name: str) -> tuple[int, str]:
    match = re.search(r'kernel\s+void\s+' + re.escape(name) + r'\s*\(', text)
    if not match:
        raise RuntimeError(f'missing kernel {name}')
    brace = text.find('{', match.end())
    if brace < 0:
        raise RuntimeError(f'missing body start for {name}')
    depth = 0
    end = None
    for i in range(brace, len(text)):
        if text[i] == '{': depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise RuntimeError(f'unclosed body for {name}')
    return text.count('\n', 0, match.start()) + 1, text[match.start():end]

paths = {name: BASE / rel for name, (rel, _) in EXPECTED.items()}
files = {}
for name, path in paths.items():
    digest = sha(path)
    expected = EXPECTED[name][1]
    if expected is not None and digest != expected:
        raise SystemExit(f'{name} hash mismatch: {digest} != {expected}')
    files[name] = {'path': str(path), 'sha256': digest, 'expected_sha256': expected}

ftext = paths['source151_fgmres'].read_text()
runtime = paths['source151_runtime'].read_text()
graph_source = paths['source151_graph_builder'].read_text()
kernels = {}
for name in ('nm_fgmres_regional_project_basis', 'nm_fgmres_regional_project_rhs',
             'nm_fgmres_regional_solve_rap', 'nm_fgmres_graph_solve_patches',
             'nm_fgmres_graph_apply_patches'):
    line, body = kernel_body(ftext, name)
    kernels[name] = {'line': line, 'body_sha256': hashlib.sha256(body.encode()).hexdigest()}

anchors = {
    'project_fixed_mask': 'state.restAndFixed.w == 1.0f ||\n        state.restAndFixed.w == 3.0f',
    'invalid_rap_zeroes_coefficients': 'if (valid[environment] == 0u)',
    'coarse_rhs_from_PtR': 'sum += dot(p.xyz, basis[basisBase + node].xyz)',
    'graph_local_rhs_subtracts_APc': 'value -= coefficients[coefficientBase + mode] *\n                    regionalAP[modeNodeBase + globalNode][axis]',
    'graph_refinement_uses_original_A': 'originalMatrices[originalMatrixBase + row * n + column]',
    'graph_apply_zeroes_constrained_candidate': 'if (candidate[global].restAndFixed.w != 0.0f)',
    'graph_gather_divides_coverage': 'graphCorrection /= float(coverage)',
    'runtime_layout_fnv': 'checksum = (checksum ^ word) * 16777619u',
}
checks = {}
for key, snippet in anchors.items():
    haystack = runtime if key == 'runtime_layout_fnv' else ftext
    if key == 'runtime_layout_fnv':
        haystack += graph_source
    # C++ graph builder's projection mask is inspected via source file regardless
    # of Metal shader source; use the Metal excerpt for its own check above.
    checks[key] = snippet in haystack
if not all(checks.values()):
    raise SystemExit(f'source anchor failed: {checks}')

report = {
    'schema': 'numi.matter.selector5_graph_reference_source_contract.v1',
    'source_files': files,
    'kernel_bodies': kernels,
    'source_anchors': checks,
    'source151_equals_source149_fgmres': files['source151_fgmres']['sha256'] == files['source149_fgmres']['sha256'],
    'source_algorithm_summary': {
        'base_view': 'modeCount base P/effective_P/AP and base RAP factor only; modeStride is retained for environment-major storage',
        'graph_map': 'H=G D E; local RHS is free-row E(r-AP c); patch D is one original-A residual refinement after pivoted LU; G sums membership-index corrections and divides nodeCoverage',
        'base_selector5_map': 'S=H+(P-H AP)R^-1 P^T',
        'transpose': 'S^T=H^T+P R^-T(P^T-(AP)^T H^T); H^T=E^T D^T G^T; D^T=2F^T-F^T A_p^T F^T in exact arithmetic',
        'invalid_base_rap': 'source writes zero coarse coefficients, so the remaining algebraic map is H; this does not make an invalid graph factor admissible',
        'completed_tail': 'source graph kernels test fgmresStates[env].diagnostics.z > 0.5 and zero graph output; reference models this exact zero latch',
        'augmented_map': 'not supported as an actual selected-view claim; restart-cycle consumption is not bound at the callback. Base view may be analyzed only with explicit reference-only label',
    },
    'native_or_physical_qualification': False,
}
(OUT / 'source-contract.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
print(json.dumps(report, sort_keys=True, allow_nan=False))
