#!/usr/bin/env python3
"""Revalidate source169 frozen-source/build records and the captured CPU test receipts."""
from __future__ import annotations
import hashlib, json, pathlib, re, sys

SNAPSHOT = pathlib.Path('/Users/home/cardboard-implementation-20261014/snapshots/source-warm-cgls-refinement-169')
BUILD = pathlib.Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-169')
PACKAGE = pathlib.Path('/Users/home/cardboard-implementation-20261014/analysis/source169-build-cpu-validation-001')


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

frozen_path = SNAPSHOT / 'frozen-source.json'
build_record_path = BUILD / 'source-build.json'
frozen = json.loads(frozen_path.read_text())
build_record = json.loads(build_record_path.read_text())
need(frozen.get('schema') == 'numi.cardboard.frozen-source.v1', 'unexpected frozen-source schema')
need(frozen.get('scope') == 'Default-off warm-start CGLS refinement candidate plus native kernel checker. No native trajectory or feature-off parity qualification.', 'scope text changed')
source_map = frozen.get('source_sha256', {})
record_map = build_record.get('source_sha256', {})
need(len(source_map) == 804, f'expected 804 frozen sources, got {len(source_map)}')
need(source_map == record_map, 'build source hash map differs from frozen source')
source_failures = []
for rel, expected in sorted(source_map.items()):
    p = pathlib.Path(rel)
    if not p.is_absolute():
        p = SNAPSHOT / p
    if not p.is_file():
        source_failures.append({'path': rel, 'reason': 'missing'})
        continue
    actual = sha(p)
    if actual != expected:
        source_failures.append({'path': rel, 'reason': 'sha256 mismatch', 'actual': actual, 'expected': expected})
need(not source_failures, f'{len(source_failures)} source hash failures')
need(build_record.get('source_snapshot') == str(SNAPSHOT), 'build record points at a different source snapshot')
need(build_record.get('success') is True, 'source build record is not successful')
need(build_record.get('changed_sources') == [], 'build record reports changed sources')
need(build_record.get('targets') == ['numi-matter-cardboard-probe', 'numi-matter-cardboard-warm-start-check'], 'unexpected build target list')

artifact_results = []
for raw, expected in sorted(build_record.get('artifacts_sha256', {}).items()):
    p = pathlib.Path(raw)
    need(p.is_file(), f'missing build artifact {p}')
    actual = sha(p)
    need(actual == expected, f'artifact SHA mismatch: {p}')
    artifact_results.append({'path': str(p), 'sha256': actual, 'bytes': p.stat().st_size})
metadata_results = []
for rel, expected in sorted(build_record.get('build_metadata_sha256', {}).items()):
    p = BUILD / rel
    need(p.is_file(), f'missing build metadata {p}')
    actual = sha(p)
    need(actual == expected, f'build metadata SHA mismatch: {p}')
    metadata_results.append({'path': str(p), 'sha256': actual, 'bytes': p.stat().st_size})

cmake_path = SNAPSHOT / 'matter/CMakeLists.txt'
cmake_text = cmake_path.read_text()
registration_pattern = re.compile(
    r'add_test\(NAME matter\.metal\.cardboard_warm_start_kernel_contract\s+'
    r'COMMAND numi-matter-cardboard-warm-start-check\s*\)\s*'
    r'set_tests_properties\(matter\.metal\.cardboard_warm_start_kernel_contract\s+'
    r'PROPERTIES\s+LABELS\s+"matter;cardboard;solver;metal;synthetic"\s+'
    r'TIMEOUT 60 RUN_SERIAL TRUE\)', re.S)
need(bool(registration_pattern.search(cmake_text)), 'source CMake warm-start test registration mismatch')
ctest_file = BUILD / 'CTestTestfile.cmake'
ctest_text = ctest_file.read_text()
need('matter.metal.cardboard_warm_start_kernel_contract' in ctest_text, 'generated CTest file omits warm-start checker')
need('RUN_SERIAL "TRUE"' in ctest_text[ctest_text.find('matter.metal.cardboard_warm_start_kernel_contract')-200:ctest_text.find('matter.metal.cardboard_warm_start_kernel_contract')+1000], 'generated CTest registration lacks RUN_SERIAL TRUE')

receipts = []
for name in ('cpu-ctest-receipt.json', 'solver-python-ctest-receipt.json', 'solver-environment-receipt.json'):
    p = PACKAGE / name
    r = json.loads(p.read_text())
    need(r.get('exit_code') == 0, f'{name}: failed exit code')
    for key in ('stdout_file', 'stderr_file'):
        log = PACKAGE / r[key]
        need(log.is_file(), f'{name}: missing {key}')
    receipts.append({'receipt': name, 'sha256': sha(p), 'argv': r['argv'], 'exit_code': r['exit_code'], 'elapsed_seconds': r['elapsed_seconds']})

cpu_log = (PACKAGE / 'cpu-ctest.stdout.log').read_text()
solver_log = (PACKAGE / 'solver-python-ctest.stdout.log').read_text()
env_out = (PACKAGE / 'solver-environment.stdout.log').read_text()
env_err = (PACKAGE / 'solver-environment.stderr.log').read_text()
need('100% tests passed, 0 tests failed out of 14' in cpu_log, 'expected 14 analysis CPU tests to pass')
need('100% tests passed, 0 tests failed out of 2' in solver_log, 'expected 2 solver Python CTests to pass')
need('matter.metal.cardboard_warm_start_kernel_contract' not in cpu_log + solver_log, 'Metal kernel checker was executed')
need('Ran 11 tests' in env_err and '\nOK\n' in env_err, 'expected solver environment unittest 11/11 pass')
need('matter.metal.' not in cpu_log + solver_log, 'a Metal CTest appears in CPU execution logs')

report = {
    'schema': 'numi.cardboard.source169.cpu-validation.v1',
    'source_snapshot': str(SNAPSHOT),
    'source_snapshot_sha256': sha(frozen_path),
    'parent_revision': frozen['parent_revision'],
    'base_snapshot': frozen['base_snapshot'],
    'scope': frozen['scope'],
    'frozen_source_count': len(source_map),
    'frozen_source_hash_failures': 0,
    'build_record': str(build_record_path),
    'build_record_sha256': sha(build_record_path),
    'build_record_success': build_record['success'],
    'build_record_targets': build_record['targets'],
    'build_artifacts_verified': artifact_results,
    'build_metadata_verified': metadata_results,
    'cmake_registration': {
        'test': 'matter.metal.cardboard_warm_start_kernel_contract',
        'target': 'numi-matter-cardboard-warm-start-check',
        'labels': ['cardboard', 'matter', 'metal', 'solver', 'synthetic'],
        'run_serial': True,
        'timeout_seconds': 60,
        'source_cmake_sha256': sha(cmake_path),
        'generated_ctest_file_sha256': sha(ctest_file),
        'executed': False,
    },
    'cpu_validation': {
        'receipts': receipts,
        'ctest_analysis_cases_passed': 14,
        'ctest_solver_python_cases_passed': 2,
        'solver_environment_unittest_cases_passed': 11,
        'total_test_cases_as_reported': 27,
        'metal_tests_executed': 0,
    },
    'limitations': [
        'The warm-start kernel checker and its metallib were built and hash-verified, but the Metal test executable was not run.',
        'The build record excludes compiler system headers and OS/driver; it records a successful build of the probe and checker targets.',
        'No native Matter trajectory, feature-off parity, or solver qualification is established by these CPU tests.',
        'The existing source168/root working tree was not edited; validation used the immutable source169 snapshot and its isolated build directory.'
    ]
}
report_path = PACKAGE / 'verification.json'
report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
print(json.dumps({'status': 'PASS', 'report': str(report_path), 'source_count': len(source_map), 'artifacts': len(artifact_results), 'cpu_cases': 27}, indent=2))
