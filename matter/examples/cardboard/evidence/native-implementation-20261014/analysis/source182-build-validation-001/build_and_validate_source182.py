#!/usr/bin/env python3
"""Create frozen source182 from source181 + overlay182-002 and CPU-validate it."""
from __future__ import annotations
import datetime, hashlib, json, os, pathlib, platform, re, shutil, subprocess, sys, time

B = pathlib.Path('/Users/n/cardboard-implementation-20261014-host-macmini')
S181 = B / 'snapshots/source-admission-encoder-181'
S182 = B / 'snapshots/source-admission-encoder-182'
D182 = B / 'build-source182-001'
O = B / 'analysis/source182-build-validation-001'
P = B / 'analysis/coupon-diagnostic-replay-overlay-182-002'
PYTHON = pathlib.Path('/opt/homebrew/Cellar/python@3.13/3.13.14_1/bin/python3.13')
FROZEN_SHA = 'f3ac253b7924dc4cb6ba1d5d9580ceea42150b5e49c6eee569e36b0e204fef7a'
BUILD181_SHA = 'd5ed1e058065435d7faeb48897681306907e61a7b9f765f3c131c38261e1c5be'
COMPILE181_SHA = '050008734269639ebd1b27163f4b4d5e7e099d56de42c73ad89abb25163a34af'
PATCH_SHA = 'c291856a8c52d5887b1f36def014716161a067497565e77c58bcfcf9a9b0cd99'
OVERLAY_MANIFEST_SHA = '6c0ab2856281bc5ff2530786c3e5200293432b750eee49890c3d041b847c50cf'
TARGETS = [
    'numi-matter-cardboard-probe',
    'numi-matter-cardboard-warm-start-check',
    'numi-matter-cardboard-shipping-snapshot-check',
    'numi-matter-cardboard-coupon-resume-check',
]
CTEST_REGEX = r'^matter\.(compiler\.cardboard_(shipping_snapshot|coupon_resume)|cli\.cardboard_argument_admission)$'


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verified_file_map(root: pathlib.Path, mapping: dict[str, str], label: str) -> None:
    for rel, expected in mapping.items():
        path = root / rel
        require(path.is_file() and not path.is_symlink(), f'{label} missing/nonregular {rel}')
        require(sha(path) == expected, f'{label} SHA mismatch {rel}')


def run(command: list[str], env: dict[str, str], timeout: int, cwd: pathlib.Path | None = None) -> dict:
    start = time.monotonic()
    proc = subprocess.run(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout, check=False)
    return {'argv': command, 'cwd': str(cwd) if cwd else None,
            'returncode': proc.returncode, 'wall_seconds': time.monotonic()-start,
            'output': proc.stdout.decode('utf-8', errors='replace'),
            'output_sha256': hashlib.sha256(proc.stdout).hexdigest()}


def main() -> int:
    require(not S182.exists() and not D182.exists() and not O.exists(),
            'source182 snapshot, build, or validation directory already exists')
    require(PYTHON.is_file(), f'expected Python 3.13 executable missing: {PYTHON}')
    overlay_manifest = P / 'overlay-manifest.json'
    require(sha(overlay_manifest) == OVERLAY_MANIFEST_SHA,
            'overlay182-002 manifest SHA mismatch')
    require(sha(P / 'overlay.patch') == PATCH_SHA, 'overlay182-002 patch SHA mismatch')
    overlay_data = json.loads(overlay_manifest.read_text())
    require(overlay_data['native_gpu_run'] == 'NOT_RUN', 'overlay native status changed')
    require(sha(S181 / 'frozen-source.json') == FROZEN_SHA,
            'source181 frozen manifest SHA mismatch')
    source_build181 = B / 'build-source181-001/source-build.json'
    compile_commands181 = B / 'build-source181-001/compile_commands.json'
    require(sha(source_build181) == BUILD181_SHA, 'source181 build provenance mismatch')
    require(sha(compile_commands181) == COMPILE181_SHA,
            'source181 compile_commands mismatch')
    frozen181 = json.loads((S181 / 'frozen-source.json').read_text())
    support181 = json.loads((S181 / 'configure-support-source.json').read_text())
    require(len(frozen181['source_sha256']) == 805, 'source181 selected file count changed')
    require(len(support181['files']) == 1034, 'source181 configure-support file count changed')
    verified_file_map(S181, frozen181['source_sha256'], 'source181 selected source')
    verified_file_map(S181, support181['files'], 'source181 configure support')
    build_meta = json.loads(source_build181.read_text())
    require(build_meta['success'] and build_meta['source_sha256'] == frozen181['source_sha256'],
            'source181 build did not bind the frozen source selection')
    for path, expected in build_meta['artifacts_sha256'].items():
        p = pathlib.Path(path)
        require(p.is_file() and not p.is_symlink() and sha(p) == expected,
                f'source181 build artifact missing or changed: {path}')

    record = {
        'schema': 'numi.cardboard.source182-build-validation.v1',
        'host': 'macmini', 'platform': platform.platform(),
        'source181_snapshot': str(S181), 'source182_snapshot': str(S182),
        'base_frozen_source_sha256': FROZEN_SHA,
        'base_source_build_sha256': BUILD181_SHA,
        'base_compile_commands_sha256': COMPILE181_SHA,
        'overlay_manifest_sha256': OVERLAY_MANIFEST_SHA,
        'overlay_patch_sha256': PATCH_SHA,
        'source181_selected_files_verified': 805,
        'source181_configure_support_files_verified': 1034,
        'targets': TARGETS, 'gpu_dispatch': False,
        'native_trajectory_or_resume_replay': 'NOT_RUN',
        'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    O.mkdir(parents=True)
    (O / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')

    shutil.copytree(S181, S182, symlinks=False)
    patch_proc = subprocess.run(['patch', '-p1', '--directory', str(S182)],
        input=(P / 'overlay.patch').read_text(), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False)
    (O / 'patch-application.log').write_text(patch_proc.stdout)
    require(patch_proc.returncode == 0, f'overlay patch failed: {patch_proc.stdout}')

    frozen182 = json.loads((S182 / 'frozen-source.json').read_text())
    require(frozen182 == frozen181, 'copied source182 frozen manifest drifted before update')
    changed = {
        'matter/tools/cardboard_coupon_resume.hpp': sha(S182 / 'matter/tools/cardboard_coupon_resume.hpp'),
        'matter/tools/cardboard_probe.mm': sha(S182 / 'matter/tools/cardboard_probe.mm'),
    }
    for rel, digest in changed.items():
        frozen182['source_sha256'][rel] = digest
    frozen182['base_snapshot'] = str(S181)
    frozen182['scope'] = ('Source181 plus isolated coupon diagnostic replay overlay182-002; '
        'proof reporting publishes validated current runner digests and explicit proof modes. '
        'No physics/operator/acceptance changes; native replay not run.')
    (S182 / 'frozen-source.json').write_text(json.dumps(frozen182, indent=2) + '\n')
    require(len(frozen182['source_sha256']) == 805, 'source182 selected file count changed')
    verified_file_map(S182, frozen182['source_sha256'], 'source182 selected source')
    require(json.loads((S182 / 'configure-support-source.json').read_text()) == support181,
            'source182 configure-support manifest changed')
    verified_file_map(S182, support181['files'], 'source182 configure support')
    changed_keys = sorted(k for k in frozen181['source_sha256']
                          if frozen181['source_sha256'][k] != frozen182['source_sha256'][k])
    require(changed_keys == sorted(changed), f'unexpected selected source delta: {changed_keys}')
    (O / 'snapshot-verification.json').write_text(json.dumps({
        'schema': 'numi.cardboard.source182-snapshot-verification.v1',
        'source181_frozen_sha256': sha(S181 / 'frozen-source.json'),
        'source182_frozen_sha256': sha(S182 / 'frozen-source.json'),
        'source181_configure_support_sha256': sha(S181 / 'configure-support-source.json'),
        'source182_configure_support_sha256': sha(S182 / 'configure-support-source.json'),
        'selected_file_count': 805, 'configure_support_file_count': 1034,
        'selected_file_hashes_verified': True, 'configure_support_hashes_verified': True,
        'selected_changed_files': changed_keys, 'only_expected_source_delta': True,
        'candidate_hashes': changed,
    }, indent=2) + '\n')

    build_command = [str(PYTHON), '-B', str(S182 / 'matter/tools/cardboard_source_bound_build.py'),
        '--source-snapshot', str(S182), '--build-dir', str(D182), '--jobs', '2']
    for target in TARGETS:
        build_command += ['--target', target]
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith('NM_') or k.startswith('NUMI_MATTER_'))}
    env['PATH'] = '/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'
    (O / 'build-command.json').write_text(json.dumps({'argv': build_command,
        'python_executable': str(PYTHON), 'gpu_dispatch': False}, indent=2) + '\n')
    with (O / 'build.stdout.log').open('wb') as out, (O / 'build.stderr.log').open('wb') as err:
        proc = subprocess.run(build_command, env=env, stdout=out, stderr=err, check=False)
    require(proc.returncode == 0, f'source182 source-bound build failed ({proc.returncode}); inspect retained logs')
    built = json.loads((D182 / 'source-build.json').read_text())
    require(built['success'] and not built.get('changed_sources'),
            'source182 build provenance failed or changed source during build')
    require(built['source_snapshot'] == str(S182) and built['build_dir'] == str(D182) and
            built['targets'] == TARGETS, 'source182 build provenance path/target mismatch')
    artifacts = {}
    for path, expected in built['artifacts_sha256'].items():
        f = pathlib.Path(path)
        require(f.is_file() and not f.is_symlink() and sha(f) == expected,
                f'source182 artifact hash mismatch: {path}')
        artifacts[path] = expected
    verified_file_map(S182, frozen182['source_sha256'], 'source182 selected source post-build')
    verified_file_map(S182, support181['files'], 'source182 configure support post-build')
    (O / 'build-verification.json').write_text(json.dumps({
        'schema': 'numi.cardboard.source182-build-verification.v1',
        'source_build_json_sha256': sha(D182 / 'source-build.json'),
        'compile_commands_sha256': sha(D182 / 'compile_commands.json'),
        'source_file_count': 805, 'configure_support_file_count': 1034,
        'changed_sources': built.get('changed_sources', []),
        'artifacts_sha256': artifacts, 'gpu_dispatch': False,
    }, indent=2) + '\n')

    ctest_command = ['/opt/homebrew/bin/ctest', '--test-dir', str(D182), '-R', CTEST_REGEX,
                     '--output-on-failure', '-j', '1']
    ctest = run(ctest_command, env, 180)
    (O / 'ctest.json').write_text(json.dumps(ctest, indent=2) + '\n')
    (O / 'ctest.log').write_text(ctest['output'])
    require(ctest['returncode'] == 0, f'CTests failed: {ctest["output"]}')
    m = re.search(r'100% tests passed, 0 tests failed out of (\d+)', ctest['output'])
    require(m and int(m.group(1)) == 3, 'expected exactly three passing targeted CTests')

    environment_command = [str(PYTHON), '-B', str(S182 / 'matter/tools/test_cardboard_solver_environment.py')]
    environment_result = run(environment_command, env, 120)
    (O / 'environment-tests.json').write_text(json.dumps(environment_result, indent=2) + '\n')
    (O / 'environment-tests.log').write_text(environment_result['output'])
    require(environment_result['returncode'] == 0, '12-case solver environment test failed')
    require(re.search(r'\b12\s+(?:tests|cases)\b|Ran 12 tests|12 passed', environment_result['output'], re.I),
            'solver environment output does not establish the expected 12 cases')

    contract_test = [
        'clang++', '-std=c++20', '-Wall', '-Wextra', '-Werror',
        '-I', str(P / 'candidate/matter/tools'), str(P / 'tests/coupon_resume_contract_test.cpp'),
        '-o', str(O / 'coupon_resume_contract_test')]
    compile_result = run(contract_test, env, 60)
    (O / 'contract-compile.json').write_text(json.dumps(compile_result, indent=2) + '\n')
    require(compile_result['returncode'] == 0, 'overlay C++ contract compilation failed')
    contract_run = run([str(O / 'coupon_resume_contract_test')], env, 30)
    (O / 'contract-run.json').write_text(json.dumps(contract_run, indent=2) + '\n')
    require(contract_run['returncode'] == 0 and 'contract checks passed' in contract_run['output'],
            'overlay C++ contract test failed')

    report = {
        'schema': 'numi.cardboard.source182-build-validation-report.v1',
        'source_snapshot': str(S182), 'build_dir': str(D182),
        'source182_frozen_sha256': sha(S182 / 'frozen-source.json'),
        'source_build_json_sha256': sha(D182 / 'source-build.json'),
        'source_file_count': 805, 'configure_support_file_count': 1034,
        'selected_source_delta': changed_keys, 'build_success': True,
        'targets': TARGETS, 'cpu_ctest_count': 3,
        'environment_test_cases': 12, 'overlay_contract_test': 'PASS',
        'objcxx_syntax_targets': ['default', 'box-native'],
        'gpu_dispatch': False, 'native_resume_replay': 'NOT_RUN',
        'trajectory_parity': False, 'physical_qualification': False,
        'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (O / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    files = {}
    for f in sorted(O.iterdir()):
        if f.is_file() and f.name != 'manifest.json':
            files[f.name] = {'bytes': f.stat().st_size, 'sha256': sha(f)}
    (O / 'manifest.json').write_text(json.dumps({
        'schema': 'numi.cardboard.source182-build-validation-manifest.v1',
        'files': files, 'report_sha256': sha(O / 'report.json')}, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FAIL: {exc}', file=sys.stderr, flush=True)
        raise
