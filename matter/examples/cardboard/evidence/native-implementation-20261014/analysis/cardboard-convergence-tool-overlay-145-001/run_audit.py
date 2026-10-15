#!/usr/bin/env python3
"""Run bounded CPU/source-contract checks for the post-release overlay."""
from __future__ import annotations

import difflib
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
B = Path('/Users/home/cardboard-implementation-20261014')
PKG = B / 'analysis/cardboard-convergence-tool-overlay-145-001'
SNAP = B / 'snapshots/source-cgls-primary-policy-143'
SRC = SNAP / 'matter/tools'
OVER = PKG / 'overlay/matter/tools'
SUPPLEMENT = PKG / 'tests/test_convergence_study_overlay.py'
RUNS = [
    B / 'solver/bounded-cgls64-exact-state-450/output',
    B / 'solver/bounded-cgls64-capture-off-parity-452/output',
]
PY = '/Users/home/.pyenv/versions/3.10.12/bin/python3'
PLAN = Path('/Users/home/numi-cardboard-20261006/matter/examples/cardboard/evidence/release-and-resolution/release-resolution-plan.json')
PATCH = PKG / 'post-release-relaxation.patch'
REPORT = PKG / 'report.json'
MANIFEST = PKG / 'manifest.json'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def invoke(label: str, argv: list[str], *, cwd: Path = B) -> dict:
    env = dict(os.environ)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    completed = subprocess.run(argv, cwd=cwd, env=env, capture_output=True,
                               text=True, check=False, timeout=120)
    stdout_path = PKG / f'{label}.stdout.log'
    stderr_path = PKG / f'{label}.stderr.log'
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    transcript = completed.stdout + '\n' + completed.stderr
    match = re.search(r'Ran (\d+) tests? in ', transcript)
    return {
        'argv': argv,
        'cwd': str(cwd),
        'exit_code': completed.returncode,
        'test_count': int(match.group(1)) if match else None,
        'success_marker': 'OK' in transcript,
        'stdout_path': str(stdout_path.relative_to(PKG)),
        'stderr_path': str(stderr_path.relative_to(PKG)),
        'stdout_sha256': sha(stdout_path),
        'stderr_sha256': sha(stderr_path),
    }


def static_contract_checks(study) -> dict:
    probe = (OVER / 'cardboard_probe.mm').read_text()
    convergence = (OVER / 'cardboard_convergence.py').read_text()
    resolution = (OVER / 'cardboard_resolution_study.py').read_text()
    baseline_probe = (SRC / 'cardboard_probe.mm').read_text()
    stage_probe = (SRC / 'cardboard_stage_resolution_study.py').read_text()

    total_steps = probe.split('std::uint32_t totalSteps(const Arguments& a)', 1)[1].split('LoadPoint loadingPoint', 1)[0]
    phase_map = probe.split('LoadPoint loadingPoint(const Arguments& a, std::uint32_t step)', 1)[1].split('struct Metrics', 1)[0]
    targets = probe.split('void fillTargets(', 1)[1].split('double indentationAtEnd', 1)[0]
    checks = {
        'option_default_zero': 'postReleaseRelaxSteps = 0u' in probe,
        'option_parsed': 'option == "--post-release-relax-steps"' in probe,
        'positive_count_requires_release': 'arguments.postReleaseRelaxSteps == 0u || arguments.releaseSteps > 0u' in probe,
        'bounded_count': 'arguments.postReleaseRelaxSteps <= 10000u' in probe,
        'manifest_exports_count': 'post_release_relaxation_steps' in probe and 'arguments.postReleaseRelaxSteps' in probe,
        'total_steps_adds_separate_count': 'a.postReleaseRelaxSteps' in total_steps and 'a.releaseSteps +' in total_steps,
        'phase_order_release_then_post': phase_map.index('if (step < a.releaseSteps) return {0.0, "release"};') < phase_map.index('return {0.0, "post_release_relaxation"};'),
        'both_phases_leave_right_grip_free': 'phase == "release" || phase == "post_release_relaxation"' in targets,
        'shipping_uses_separate_settle_guard': 'shipping box uses --shipping-settle-steps after release' in probe and 'released_assembly_settle' in stage_probe,
        'older_exports_default_missing_post_count_to_zero': 'phase in ("release", "post_release_relaxation")' in convergence,
        'complete_study_requires_all_six_positive_phases': 'for phase in PHASE_ORDER if protocol[phase] <= 0' in resolution and 'post-release relaxation steps' in resolution,
        'all_phase_arguments_are_controlled': '"--post-release-relax-steps"' in resolution and 'for phase in PHASE_ORDER' in resolution,
        'temporal_counts_all_six_phases': 'all six physical phase durations' in resolution and 'for phase in PHASE_ORDER' in resolution,
        'strict_decrease_preserved': 'float(fine) < float(coarse)' in resolution,
        'source143_did_not_gain_new_option': '--post-release-relax-steps' not in baseline_probe,
    }
    if not all(checks.values()):
        raise AssertionError(f'source contract checks failed: {checks}')
    if tuple(study.PHASE_ORDER) != (
            'loading', 'hold', 'unloading', 'relaxation', 'release', 'post_release_relaxation'):
        raise AssertionError(f'unexpected phase order: {study.PHASE_ORDER}')
    return checks


def patch_matches_overlay() -> dict:
    changed = [
        'cardboard_probe.mm', 'cardboard_convergence.py',
        'cardboard_resolution_study.py', 'test_cardboard_convergence.py',
        'test_cardboard_resolution_study.py',
    ]
    with tempfile.TemporaryDirectory(prefix='cardboard-post-release-patch-') as temp_name:
        temp = Path(temp_name)
        for name in changed:
            target = temp / 'matter/tools' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SRC / name, target)
        result = subprocess.run(['patch', '--batch', '--forward', '-p1', '-d', str(temp),
                                 '-i', str(PATCH)], capture_output=True, text=True,
                                check=False, timeout=30)
        if result.returncode != 0:
            raise AssertionError(f'patch apply failed: {result.stdout}\n{result.stderr}')
        mismatches = [name for name in changed
                      if (temp / 'matter/tools' / name).read_bytes() != (OVER / name).read_bytes()]
        if mismatches:
            raise AssertionError(f'patch result differs from overlay copies: {mismatches}')
    dry_run = subprocess.run(['patch', '--dry-run', '--batch', '--forward', '-p1',
                              '-d', str(SNAP), '-i', str(PATCH)],
                             capture_output=True, text=True, check=False, timeout=30)
    if dry_run.returncode != 0:
        raise AssertionError(f'snapshot patch dry-run failed: {dry_run.stdout}\n{dry_run.stderr}')
    return {'changed_paths': [f'matter/tools/{x}' for x in changed],
            'applies_to_source143_snapshot': True,
            'patch_sha256': sha(PATCH),
            'patch_applied_output_matches_overlay': True,
            'snapshot_dry_run_stdout': dry_run.stdout,
            'snapshot_dry_run_stderr': dry_run.stderr}


def main() -> None:
    if REPORT.exists() or MANIFEST.exists():
        raise SystemExit('report.json or manifest.json exists; refusing to overwrite a sealed attempt')

    suites = [
        ('overlay_convergence_tests', OVER / 'test_cardboard_convergence.py'),
        ('overlay_resolution_tests', OVER / 'test_cardboard_resolution_study.py'),
        ('post_release_admission_and_schedule_tests', SUPPLEMENT),
        ('source143_shipping_stage_tests', SRC / 'test_cardboard_stage_resolution_study.py'),
    ]
    results = {name: invoke(name, [PY, str(path)]) for name, path in suites}
    if any(result['exit_code'] != 0 or not result['success_marker'] for result in results.values()):
        raise SystemExit('one or more CPU test suites failed; see retained logs')

    study = load_module('cardboard_resolution_overlay145_audit', OVER / 'cardboard_resolution_study.py')
    contracts = static_contract_checks(study)
    patch_result = patch_matches_overlay()
    if not PLAN.is_file():
        raise SystemExit(f'legacy plan not found: {PLAN}')
    plan = json.loads(PLAN.read_text())

    failed_controls = []
    convergence = load_module('cardboard_convergence_overlay145_audit', OVER / 'cardboard_convergence.py')
    for path in RUNS:
        result = convergence.analyze_run(path)
        raw = json.loads((path / 'result.json').read_text())
        assert raw.get('status') == 'failed' and raw.get('accepted_steps') == 1
        assert result['status'] == 'failed_run' and result['reasons'][0]['code'] == 'native_run_failed'
        assert result['physical_validation'] is False
        failed_controls.append({
            'path': str(path),
            'raw_status': raw['status'],
            'raw_accepted_steps': raw['accepted_steps'],
            'analyzer_status': result['status'],
            'reason_codes': [reason['code'] for reason in result['reasons']],
            'physical_validation': result['physical_validation'],
            'result_sha256': sha(path / 'result.json'),
            'manifest_sha256': sha(path / 'manifest.json'),
            'observations_sha256': sha(path / 'observations.csv'),
        })

    baseline_sources = [
        SRC / 'cardboard_probe.mm', SRC / 'cardboard_convergence.py',
        SRC / 'cardboard_resolution_study.py', SRC / 'cardboard_stage_resolution_study.py',
        SRC / 'test_cardboard_convergence.py', SRC / 'test_cardboard_resolution_study.py',
        SRC / 'test_cardboard_stage_resolution_study.py', SNAP / 'matter/CMakeLists.txt',
    ]
    report = {
        'schema': 'numi.cardboard.post_release_relaxation_overlay_audit.v1',
        'source_snapshot': str(SNAP),
        'overlay_package': str(PKG),
        'audit_pass': True,
        'audit_scope': 'CPU-only tool, schedule, and admission checks; no native build, GPU execution, or physical qualification',
        'source143_sha256': {str(path.relative_to(SNAP)): sha(path) for path in baseline_sources},
        'overlay_sha256': {str(path.relative_to(PKG)): sha(path) for path in sorted(OVER.glob('*')) if path.is_file()},
        'source_contract_checks': contracts,
        'patch_check': patch_result,
        'test_suites': results,
        'retained_failed_run_controls': failed_controls,
        'schedule_contract': {
            'phases': list(study.PHASE_ORDER),
            'default_post_release_steps': 0,
            'default_trajectory_contract': 'the extra segment contributes no steps and leaves old default release schedule unchanged',
            'post_release_requires_release': True,
            'released_nodes': 'right grip remains free in release and post_release_relaxation; left grip remains fixed',
            'phase_boundaries': 'accepted phase snapshots are emitted at the existing loop phase transition for both free phases',
            'shipping_settle': 'separate shipping released_assembly_settle is retained; new end-grip option is rejected for shipping mode',
            'complete_study_requires_positive_steps': True,
            'legacy_analyzer_compatibility': 'missing post_release_relaxation_steps is read as zero; legacy analysis may omit that requested phase',
            'strict_decreasing_differences': True,
        },
        'resolution_plan': {
            'spatial_case_count': len(study._spatial_cases()),
            'spatial_cases': [{'id': case['id'], 'mesh': case['mesh']} for case in study._spatial_cases()],
            'temporal_policy': 'dt, dt/2, dt/4 with every phase step count multiplied uniformly, preserving each physical phase duration and loading rate',
            'slower_load_policy': 'separate load-rate axis changes only loading duration',
            'thresholds': study.THRESHOLDS,
            'legacy_plan_path': str(PLAN),
            'legacy_plan_sha256': sha(PLAN),
            'legacy_plan_trial_count': len(plan.get('trials', [])),
            'force_accuracy_limit': 'tolerance/dt matching and observed force residuals do not establish an absolute force error bound without exported residual normalization scale',
        },
        'limitations': [
            'No native compile or GPU execution was run for this overlay.',
            'Synthetic reader fixtures validate software contracts only.',
            'Numerical convergence, absolute force accuracy, held-out material prediction, and physical coupon/box validation remain unqualified.',
        ],
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    tracked = [path for path in PKG.rglob('*') if path.is_file()
               and '__pycache__' not in path.parts
               and path != MANIFEST]
    manifest = {
        'schema': 'numi.cardboard.post_release_relaxation_overlay_manifest.v1',
        'source_snapshot': str(SNAP),
        'report_sha256': sha(REPORT),
        'entries': [{'path': str(path.relative_to(PKG)), 'sha256': sha(path), 'bytes': path.stat().st_size}
                    for path in sorted(set(tracked)) if path.is_file()],
        'manifest_note': 'This manifest does not hash itself; report and payload hashes are explicit.',
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'report': str(REPORT), 'report_sha256': sha(REPORT),
                      'manifest': str(MANIFEST), 'manifest_sha256': sha(MANIFEST),
                      'test_suites': results, 'failed_controls': failed_controls},
                     indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
