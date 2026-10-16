#!/usr/bin/env python3
"""Independent source/hash audit for the all-controls selector overlay."""
from __future__ import annotations
import difflib
import hashlib
import json
from pathlib import Path
import re

BASE = Path('/Users/home/cardboard-implementation-20261014')
PKG = BASE / 'analysis/cgls-all-controls-selector-overlay-153-001'
SNAP = BASE / 'snapshots/source-cgls-all-controls-153'
PARENT = BASE / 'snapshots/source-cgls-block-right-preconditioned-149'
OUT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def line_of(text: str, needle: str) -> int:
    for line, value in enumerate(text.splitlines(), 1):
        if needle in value:
            return line
    raise AssertionError(f'missing source anchor: {needle}')


def line_of_after(text: str, start_anchor: str, needle: str) -> int:
    start = text.find(start_anchor)
    if start < 0:
        raise AssertionError(f'missing source start anchor: {start_anchor}')
    suffix = text[start:]
    return text.count('\n', 0, start) + line_of(suffix, needle)


def main() -> None:
    candidate = SNAP / 'matter/src/runtime.mm'
    parent = PARENT / 'matter/src/runtime.mm'
    manifest = SNAP / 'frozen-source.json'
    build = PKG / 'validation/build-receipt.json'
    patch = PKG / 'runtime.patch'
    selectors = PKG / 'validation/extracted-selectors.json'
    source = candidate.read_text()
    parent_source = parent.read_text()
    src_manifest = json.loads(manifest.read_text())
    build_receipt = json.loads(build.read_text())
    selector_result = json.loads(selectors.read_text())

    anchors = {
        'true_root_exact_all': 'std::strcmp(selector, "all") == 0',
        'true_root_resolves_current_control': 'request.controlStep : std::strtoul(selector, &end, 10)',
        'cauchy_exact_all_prefix': 'separator - baseCauchySelector == 3',
        'cauchy_all_requires_suffix_cgls_and_primary': 'all_control_steps_requires_cgls_primary_policy_and_all_newton_suffix',
        'same_root_true_residual_gate': 'requires_true_residual_certification_for_the_same_selected_newton_state',
        'cgls_primary_all_iterations_gate': 'requires_cgls_iterations_and_control_step_all_newton_selector',
        'all_newton_trace_cap': '? 512u : 64u',
        'singleton_cgls_snapshot_exclusive': 'O_WRONLY | O_CREAT | O_EXCL',
        'trace_root_numeric_only': 'selector_must_be_control_step_selector',
    }
    checks = {
        'true_root_exact_all': anchors['true_root_exact_all'] in source,
        'true_root_resolves_current_control': anchors['true_root_resolves_current_control'] in source,
        'cauchy_exact_all_prefix': anchors['cauchy_exact_all_prefix'] in source,
        'cauchy_all_requires_suffix_cgls_and_primary': anchors['cauchy_all_requires_suffix_cgls_and_primary'] in source,
        'same_root_true_residual_gate': anchors['same_root_true_residual_gate'] in source,
        'cgls_primary_all_iterations_gate': anchors['cgls_primary_all_iterations_gate'] in source,
        'all_newton_trace_cap': anchors['all_newton_trace_cap'] in source,
        'singleton_cgls_snapshot_exclusive': anchors['singleton_cgls_snapshot_exclusive'] in source,
        'trace_root_numeric_only': 'invalid_control_step_selector' in source and
            'std::strtoul(selector, &end, 10)' in source[
                source.find('NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT'):source.find('NM_FEM_TRUE_RESIDUAL_CERTIFY')],
    }
    changed_source_paths = sorted(
        name for name, digest in src_manifest['source_sha256'].items()
        if digest != json.loads((PARENT / 'frozen-source.json').read_text())['source_sha256'].get(name)
    )
    if changed_source_paths != ['matter/src/runtime.mm']:
        raise SystemExit(f'unexpected source diff set: {changed_source_paths}')
    if not all(checks.values()):
        raise SystemExit(f'source review anchor failed: {checks}')
    if selector_result.get('run_exit_code') != 0 or selector_result.get('stdout') != 'selector_checks=44 passed\n':
        raise SystemExit('extracted-source selector contract result is not 44/44 pass')
    if build_receipt.get('exit_code') != 0 or build_receipt.get('physical_or_native_probe_execution') is not False:
        raise SystemExit('build receipt is not a successful CPU build-only result')

    text = {
        'schema': 'numi.cardboard.source153_independent_review.v2',
        'source_snapshot': str(SNAP),
        'source_snapshot_manifest_sha256': sha(manifest),
        'source_runtime_sha256': sha(candidate),
        'source149_runtime_sha256': sha(parent),
        'overlay_patch_sha256': sha(patch),
        'build_receipt_sha256': sha(build),
        'build_success': True,
        'build_target': 'numi-matter-cardboard-probe',
        'native_or_gpu_execution': False,
        'changed_source_paths': changed_source_paths,
        'extracted_selector_checks': selector_result['stdout'].strip(),
        'source_anchors': checks,
        'line_map': {
            'true_residual_all_parse': line_of_after(
                source, 'NM_FEM_TRUE_RESIDUAL_ROOT',
                'const bool everyControlStep = std::strcmp(selector, "all") == 0;'),
            'cauchy_all_parse': line_of(source, 'separator - baseCauchySelector == 3'),
            'all_scope_gate': line_of(source, 'all_control_steps_requires_cgls_primary_policy_and_all_newton_suffix'),
            'same_root_certification_gate': line_of(source, 'requires_true_residual_certification_for_the_same_selected_newton_state'),
            'per_control_capability_then_selection': line_of(source, 'femBaseCauchyFallbackRoot == request.controlStep'),
            'all_newton_trace_cap': line_of(source, '? 512u : 64u;'),
            'cgls_snapshot_write': line_of(source, 'const int fd = ::open(capturedCGLSPath.c_str(),'),
            'state_capture_write': line_of(source, 'writeBaseCGLSStateCapture(snapshot.contents, payloadBytes,'),
            'numeric_only_armijo_trace_root': line_of(source, 'NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT'),
        },
        'conclusion': 'All-controls selectors preserve the existing per-control admission and solve gates; the all:all CGLS path is opt-in behind bounded CGLS, primary-policy, and same-root true-residual certification requirements.',
        'limits': [
            'No native solver run, default-parity run, or convergence result is established.',
            'The existing CGLS binary and .state path use O_EXCL and omit control-step names; an all-control run with a singleton CGLS/state capture option can only retain the first matching capture. Omit these path-based observers or add control-step namespacing.',
            'NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT remains a numeric single-control selector. The all-control trace cap is 512 where all-Newton tracing is enabled, but this does not supply complete trial traces for all control steps.',
            'The trace-enabled notice still labels 64 as the default record budget; the selected all-Newton branch uses 512. Treat the notice as default-only metadata, not the selected cap.',
        ],
    }
    (OUT / 'independent-review.json').write_text(json.dumps(text, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({
        'source_snapshot_manifest_sha256': text['source_snapshot_manifest_sha256'],
        'source_runtime_sha256': text['source_runtime_sha256'],
        'source149_runtime_sha256': text['source149_runtime_sha256'],
        'overlay_patch_sha256': text['overlay_patch_sha256'],
        'build_receipt_sha256': text['build_receipt_sha256'],
        'changed_source_paths': changed_source_paths,
        'selector_checks': text['extracted_selector_checks'],
        'anchors': checks,
        'line_map': text['line_map'],
    }, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
