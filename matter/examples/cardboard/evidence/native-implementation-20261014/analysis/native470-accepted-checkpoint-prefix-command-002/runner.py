#!/usr/bin/env python3
"""Fail-closed source/build/capacity guard for native470; preflight is default."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parent
SPEC_PATH = PACKAGE / 'native-470-command.json'
MANIFEST_PATH = PACKAGE / 'package-manifest.json'
B = Path('/Users/home/cardboard-implementation-20261014')
EXECUTIONS = PACKAGE / 'executions'
MIB = 1024 ** 2
GIB = 1024 * MIB
MINIMUM = 2 * GIB
ABORT_FLOOR = 512 * MIB
RESERVE_BYTES = 128 * MIB
RESERVE_CHUNK = 1024 * 1024
TIMING_KEYS = {
    'NUMI_CARDBOARD_STEP_TIMING',
    'NUMI_CARDBOARD_NATIVE_TIMING',
    'NUMI_MATTER_GPU_TIMING_DENSE45',
}
EXPECTED_CTESTS = [
    'coupon_resume',
    'inside_clearance',
    'shipping_clearance_memory',
    'regional_consumer_capture',
]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def available():
    stats = os.statvfs(B)
    return stats.f_bavail * stats.f_frsize


def write_reserve_file(path, byte_count=RESERVE_BYTES, chunk_size=RESERVE_CHUNK):
    """Materialize and sync a reserve; reject sparse or short allocation."""
    if byte_count <= 0 or chunk_size <= 0:
        raise ValueError('reserve byte and chunk counts must be positive')
    digest = hashlib.sha256()
    chunk = os.urandom(min(chunk_size, byte_count))
    written = 0
    with Path(path).open('xb') as stream:
        while written < byte_count:
            count = min(len(chunk), byte_count - written)
            block = chunk[:count]
            stream.write(block)
            digest.update(block)
            written += count
        stream.flush()
        os.fsync(stream.fileno())
        stats = os.fstat(stream.fileno())
    allocated = stats.st_blocks * 512
    if stats.st_size != byte_count:
        raise RuntimeError(f'emergency reserve size {stats.st_size} != requested {byte_count}')
    if allocated < byte_count:
        raise RuntimeError(f'emergency reserve allocated only {allocated} of {byte_count} bytes')
    return {
        'requested_bytes': byte_count,
        'file_size_bytes': stats.st_size,
        'st_blocks': stats.st_blocks,
        'allocated_bytes': allocated,
        'sha256': digest.hexdigest(),
        'fsynced': True,
    }


def validate_ctest_registration(report, manifest, source_build_sha256, source_validation_sha256,
                                registration_root, packaged_report):
    if report.get('schema') != 'numi.cardboard.ctest-registration-validation.v1':
        raise ValueError('unexpected CTest registration report schema')
    if report.get('source_build_sha256') != source_build_sha256:
        raise ValueError('CTest report is bound to a different source build')
    if report.get('source167_validation_report_sha256') != source_validation_sha256:
        raise ValueError('CTest report is bound to a different source validation report')
    tests = report.get('tests')
    if tests != EXPECTED_CTESTS:
        raise ValueError(f'CTest test inventory mismatch: {tests!r}')
    if type(report.get('passed')) is not int or report['passed'] != len(EXPECTED_CTESTS):
        raise ValueError(f'CTest passed count mismatch: {report.get("passed")!r}')
    expected_files = {'CTestTestfile.cmake', 'LastTest.log', 'report.json', 'seal.py'}
    if set(manifest) != expected_files:
        raise ValueError('CTest registration manifest does not enumerate the exact four sealed files')
    root = Path(registration_root).resolve(strict=True)
    for name in sorted(expected_files):
        entry = manifest[name]
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'CTest registration evidence file is missing or unsafe: {name}')
        if type(entry.get('bytes')) is not int or entry['bytes'] != path.stat().st_size:
            raise ValueError(f'CTest registration evidence byte count mismatch: {name}')
        if entry.get('sha256') != sha(path):
            raise ValueError(f'CTest registration evidence hash mismatch: {name}')
    if manifest['report.json']['sha256'] != sha(packaged_report):
        raise ValueError('CTest manifest does not bind the packaged report bytes')


def verify():
    spec = json.loads(SPEC_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest['schema'] == 'numi.cardboard.native470-package-manifest.v2'
    listed = set(manifest['files_sha256'])
    actual = {
        str(path.relative_to(PACKAGE))
        for path in PACKAGE.rglob('*')
        if path.is_file() and path != MANIFEST_PATH and '__pycache__' not in path.parts
    }
    assert listed == actual, f'package inventory mismatch: missing={sorted(actual-listed)} extra={sorted(listed-actual)}'
    for relative, digest in manifest['files_sha256'].items():
        path = PACKAGE / relative
        assert not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(PACKAGE), f'unsafe package file: {relative}'
        assert sha(path) == digest, f'package file changed: {relative}'

    source = Path(spec['source_snapshot']).resolve(strict=True)
    builddir = Path(spec['runtime_dir']).resolve(strict=True)
    assert source == Path('/Users/home/cardboard-implementation-20261014/snapshots/source-diagnostic-and-resume-167-final').resolve()
    assert builddir == Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-167-final').resolve()
    frozen_path = Path(spec['frozen_source_file']).resolve(strict=True)
    assert sha(frozen_path) == spec['frozen_source_sha256']
    frozen = json.loads(frozen_path.read_text())
    assert frozen.get('schema') == 'numi.cardboard.frozen-source.v1'
    assert len(frozen['source_sha256']) == spec['source_file_count']
    for relative, digest in frozen['source_sha256'].items():
        path = (source / relative).resolve(strict=True)
        assert path.is_relative_to(source) and sha(path) == digest, f'source drift: {relative}'

    provenance = Path(spec['source_build_provenance']).resolve(strict=True)
    assert provenance == builddir / 'source-build.json' and sha(provenance) == spec['source_build_sha256']
    built = json.loads(provenance.read_text())
    assert built.get('schema') == 'numi.cardboard.source-build.v1' and built.get('success') is True
    assert Path(built['source_snapshot']).resolve() == source and Path(built['build_dir']).resolve() == builddir
    assert built['source_sha256'] == frozen['source_sha256']
    assert built['targets'] == spec['build_targets'] and built['artifacts_sha256'] == spec['artifact_sha256']

    native = spec['native_argv']
    assert native[0] == str((builddir / 'numi-matter-cardboard-probe').resolve())
    assert native[native.index('--metallib') + 1] == str((builddir / 'shaders/NumiMatter.metallib').resolve())
    assert native[native.index('--output') + 1] == str(Path(spec['case']) / 'output')
    assert native.count('--coupon-checkpoint-after-control') == 1
    assert native[native.index('--coupon-checkpoint-after-control') + 1] == '19'
    assert not any(value in native for value in ('--coupon-resume-checkpoint', '--coupon-resume-metadata'))
    for filename, digest in built['artifacts_sha256'].items():
        path = Path(filename).resolve(strict=True)
        assert path.is_relative_to(builddir) and sha(path) == digest, f'artifact drift: {filename}'

    env = spec['environment_overrides']
    assert env['NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR'] == '20:0:63:1:0:0'
    assert len(env) == 12 and not any(key.startswith('NM_FEM_OPERATOR_FD') for key in env)
    assert 'NM_FEM_REGIONAL_CONSUMER_CAPTURE_PATH' not in env
    assert spec['physical_arguments_equal_to_469_after_executable_metallib_output_checkpoint_exclusions'] is True
    baseline_path = PACKAGE / 'inputs/source469-prediction.json'
    baseline = json.loads(baseline_path.read_text())
    assert sha(baseline_path) == spec['baseline_469_prediction_sha256']

    def physical_tokens(values):
        output = []
        index = 0
        while index < len(values):
            if index == 0:
                index += 1
                continue
            if values[index] in ('--metallib', '--output', '--coupon-checkpoint-after-control'):
                index += 2
                continue
            output.append(values[index])
            index += 1
        return output

    assert physical_tokens(native) == physical_tokens(baseline['argv']), 'physical argv differs from469 beyond declared path/stop additions'
    expected_env = dict(baseline['environment'])
    expected_env.pop('NM_FEM_OPERATOR_FD_SELECTOR', None)
    expected_env.pop('NM_FEM_OPERATOR_FD_CAPTURE_PATH', None)
    expected_env['NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR'] = '20:0:63:1:0:0'
    assert env == expected_env, 'environment controls differ from declared469 baseline plus regional selector'

    input_by_alias = {item['alias']: item for item in spec['input_bindings']}
    assert len(input_by_alias) == len(spec['input_bindings']) == 12
    source_validation = input_by_alias['source167-build-cpu-validation-report.json']
    assert json.loads((PACKAGE / 'inputs/source167-build-cpu-validation-report.json').read_text())['source_build_sha256'] == spec['source_build_sha256']
    ctest_report = json.loads((PACKAGE / 'inputs/source167-ctest-registration-report.json').read_text())
    ctest_manifest = json.loads((PACKAGE / 'inputs/source167-ctest-registration-manifest.json').read_text())
    ctest_report_binding = input_by_alias['source167-ctest-registration-report.json']
    ctest_manifest_binding = input_by_alias['source167-ctest-registration-manifest.json']
    ctest_report_source = Path(ctest_report_binding['original_path']).resolve(strict=True)
    ctest_manifest_source = Path(ctest_manifest_binding['original_path']).resolve(strict=True)
    assert ctest_report_source.parent == ctest_manifest_source.parent
    validate_ctest_registration(ctest_report, ctest_manifest, spec['source_build_sha256'],
                                source_validation['package_sha256'], ctest_manifest_source.parent,
                                PACKAGE / 'inputs/source167-ctest-registration-report.json')
    assert spec['ctest_registration_contract'] == {
        'schema': 'numi.cardboard.ctest-registration-validation.v1',
        'source_build_sha256': spec['source_build_sha256'],
        'source167_validation_report_sha256': source_validation['package_sha256'],
        'manifest_files': ['CTestTestfile.cmake', 'LastTest.log', 'report.json', 'seal.py'],
        'expected_tests': EXPECTED_CTESTS,
        'expected_passed': 4,
    }

    for flag, alias in (('--liner-material', 'source157-liner-material.nmatter'), ('--medium-material', 'source157-medium-material.nmatter')):
        assert native[native.index(flag) + 1] == input_by_alias[alias]['original_path']
    names = []
    for item in spec['input_bindings']:
        names.append(Path(item['alias']).name)
        copy = PACKAGE / 'inputs' / item['alias']
        assert copy.is_file() and sha(copy) == item['package_sha256']
        original = Path(item['original_path']).resolve(strict=True)
        assert sha(original) == item['original_sha256']
        assert item['package_path'] == str(copy)
    assert len(set(names)) == len(names)

    wrapper = spec['wrapper_argv']
    assert Path(wrapper[1]).resolve() == source / 'matter/tools/cardboard_source_bound_run.py'
    assert wrapper[wrapper.index('--source-snapshot') + 1] == str(source)
    assert wrapper[wrapper.index('--runtime-dir') + 1] == str(builddir)
    assert wrapper[wrapper.index('--build-provenance') + 1] == str(provenance)
    assert wrapper[wrapper.index('--case') + 1] == spec['case'] and '--diagnostic-run' in wrapper
    assert wrapper[wrapper.index('--diagnostic-capture') + 1] == 'NM_FEM_REGIONAL_CONSUMER_CAPTURE_PATH'
    wrapper_inputs = [wrapper[index + 1] for index, value in enumerate(wrapper[:-1]) if value == '--input-file']
    assert sorted(wrapper_inputs) == sorted(str(PACKAGE / 'inputs' / item['alias']) for item in spec['input_bindings'])
    wrapper_env = [wrapper[index + 1].split('=', 1) for index, value in enumerate(wrapper[:-1]) if value == '--env']
    assert dict(wrapper_env) == env
    assert wrapper[wrapper.index('--') + 1:] == native

    case = Path(spec['case'])
    assert case.parent.is_dir() and not case.parent.is_symlink() and case.parent.resolve() == B / 'solver'
    assert not case.exists() and not case.is_symlink(), 'fresh case required'
    assert not EXECUTIONS.exists(), 'fresh execution-record directory required'
    active = []
    for line in subprocess.check_output(['ps', '-axo', 'pid=,stat=,comm='], text=True).splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[2].startswith('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-') and parts[2].endswith('/numi-matter-cardboard-probe') and not parts[1].startswith('T'):
            active.append({'pid': int(parts[0]), 'stat': parts[1], 'executable': parts[2]})
    assert not active, f'one-GPU-worker guard found active probes: {active}'
    free = available()
    assert free >= MINIMUM, f'launch needs 2 GiB free; observed {free}'
    return spec, {
        'available_bytes': free,
        'active_probe_processes': active,
        'verified_source_files': len(frozen['source_sha256']),
        'verified_build_artifacts': len(built['artifacts_sha256']),
        'verified_inputs': len(names),
        'source_build_sha256': sha(provenance),
        'frozen_source_sha256': sha(frozen_path),
        'ctest_registration_tests': EXPECTED_CTESTS,
        'ctest_registration_passed': len(EXPECTED_CTESTS),
    }


def _process_rows():
    rows = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,pgid=,comm='], text=True).splitlines()
    parsed = []
    for row in rows:
        fields = row.split(None, 3)
        if len(fields) != 4:
            continue
        try:
            parsed.append((int(fields[0]), int(fields[1]), int(fields[2]), fields[3]))
        except ValueError:
            continue
    return parsed


def _is_expected_executable(command, expected):
    try:
        return Path(command).resolve() == Path(expected).resolve()
    except (OSError, RuntimeError):
        return False


def _verified_direct_native_groups(wrapper_pid, expected_native):
    return sorted({pid for pid, ppid, pgid, command in _process_rows()
                   if ppid == wrapper_pid and pid == pgid and _is_expected_executable(command, expected_native)})


def _verified_native_still_in_group(pgid, expected_native):
    return any(pid == pgid and current_pgid == pgid and _is_expected_executable(command, expected_native)
               for pid, _ppid, current_pgid, command in _process_rows())


def terminate_owned_native(proc, expected_native, term_grace=20):
    """Stop/reap only the wrapper we started and exact native child groups we verified."""
    outcome = {'verified_native_groups': [], 'signals': [], 'wrapper_terminated': False,
               'wrapper_reaped': False, 'native_groups_remaining': [],
               'native_child_status': 'not_checked', 'cleanup_errors': []}
    if proc is None:
        return outcome
    try:
        groups = _verified_direct_native_groups(proc.pid, expected_native)
        outcome['verified_native_groups'] = groups
        for pgid in groups:
            try:
                os.killpg(pgid, signal.SIGTERM)
                outcome['signals'].append({'pgid': pgid, 'signal': 'SIGTERM', 'identity_verified': True})
            except ProcessLookupError:
                pass
            except OSError as error:
                outcome['cleanup_errors'].append(f'native SIGTERM pgid {pgid}: {type(error).__name__}: {error}')

        try:
            proc.wait(timeout=term_grace)
        except subprocess.TimeoutExpired:
            for pgid in groups:
                try:
                    if _verified_native_still_in_group(pgid, expected_native):
                        os.killpg(pgid, signal.SIGKILL)
                        outcome['signals'].append({'pgid': pgid, 'signal': 'SIGKILL', 'identity_verified': True})
                except ProcessLookupError:
                    pass
                except OSError as error:
                    outcome['cleanup_errors'].append(f'native SIGKILL pgid {pgid}: {type(error).__name__}: {error}')
            if proc.poll() is None:
                try:
                    proc.terminate()
                    outcome['wrapper_terminated'] = True
                except OSError as error:
                    outcome['cleanup_errors'].append(f'wrapper terminate: {type(error).__name__}: {error}')
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if proc.poll() is None:
                    try:
                        proc.kill()
                        outcome['wrapper_terminated'] = True
                        proc.wait(timeout=5)
                    except (OSError, subprocess.TimeoutExpired) as error:
                        outcome['cleanup_errors'].append(f'wrapper kill/reap: {type(error).__name__}: {error}')
        except OSError as error:
            outcome['cleanup_errors'].append(f'wrapper wait: {type(error).__name__}: {error}')

        if proc.poll() is None:
            try:
                proc.terminate()
                outcome['wrapper_terminated'] = True
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    outcome['wrapper_terminated'] = True
                    proc.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired) as error:
                    outcome['cleanup_errors'].append(f'wrapper final reap: {type(error).__name__}: {error}')
            except OSError as error:
                outcome['cleanup_errors'].append(f'wrapper final terminate/reap: {type(error).__name__}: {error}')

        for pgid in groups:
            try:
                if _verified_native_still_in_group(pgid, expected_native):
                    os.killpg(pgid, signal.SIGKILL)
                    outcome['signals'].append({'pgid': pgid, 'signal': 'SIGKILL', 'identity_verified': True})
            except ProcessLookupError:
                pass
            except OSError as error:
                outcome['cleanup_errors'].append(f'final native SIGKILL pgid {pgid}: {type(error).__name__}: {error}')
        outcome['native_groups_remaining'] = [pgid for pgid in groups
                                                if _verified_native_still_in_group(pgid, expected_native)]
        if outcome['native_groups_remaining']:
            outcome['native_child_status'] = 'verified_native_group_still_observed'
        elif groups:
            outcome['native_child_status'] = 'verified_groups_signalled_and_absent'
        else:
            outcome['native_child_status'] = 'no_verified_native_child_observed'
        outcome['wrapper_reaped'] = proc.poll() is not None
    except BaseException as error:
        outcome['cleanup_errors'].append(f'cleanup inspection: {type(error).__name__}: {error}')
        try:
            if proc.poll() is None:
                proc.terminate()
                outcome['wrapper_terminated'] = True
                proc.wait(timeout=5)
            outcome['wrapper_reaped'] = proc.poll() is not None
            outcome['native_child_status'] = 'native_identity_unverified_cleanup_incomplete'
        except BaseException as wait_error:
            outcome['cleanup_errors'].append(f'wrapper fallback reap: {type(wait_error).__name__}: {wait_error}')
    return outcome


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    spec, checks = verify()
    print(json.dumps({'status': 'preflight_pass_no_launch', 'checks': checks,
                      'execute_requested': args.execute}, sort_keys=True))
    if not args.execute:
        return 0

    EXECUTIONS.mkdir(parents=True, exist_ok=False)
    reserve = EXECUTIONS / 'emergency-reserve.bin'
    proc = None
    code = None
    space_abort = False
    signalled = []
    minfree = available()
    start = time.monotonic()
    errors = []
    exception_cleanup = None
    parent = {key: value for key, value in os.environ.items()
              if not key.startswith('NM_FEM_') and key not in TIMING_KEYS}
    removed = sorted(set(os.environ) - set(parent))
    launch = {
        'schema': 'numi.cardboard.native470-capacity-launch.v2',
        'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'command_sha256': sha(SPEC_PATH),
        'package_manifest_sha256': sha(MANIFEST_PATH),
        'source_build_sha256': checks['source_build_sha256'],
        'frozen_source_sha256': checks['frozen_source_sha256'],
        'parent_environment_removed_keys': removed,
        'parent_NM_FEM_values_recorded': False,
        'minimum_launch_available_bytes': MINIMUM,
        'abort_floor_bytes': ABORT_FLOOR,
        'emergency_reserve_bytes': RESERVE_BYTES,
        'available_before_reserve': checks['available_bytes'],
        'diagnostic_only': True,
        'checkpoint_control_index': 19,
    }
    try:
        reserve_info = write_reserve_file(reserve)
        launch['emergency_reserve'] = reserve_info
        launch['available_after_reserve'] = available()
        (EXECUTIONS / 'launch.json').write_text(json.dumps(launch, indent=2, sort_keys=True) + '\n')
        minfree = available()
        with (EXECUTIONS / 'wrapper.log').open('x') as log:
            proc = subprocess.Popen(spec['wrapper_argv'], stdout=log, stderr=subprocess.STDOUT,
                                    env=parent, start_new_session=True)
            while proc.poll() is None:
                free = available()
                minfree = min(minfree, free)
                if free < ABORT_FLOOR and not space_abort:
                    space_abort = True
                    reserve.unlink(missing_ok=True)
                    groups = _verified_direct_native_groups(proc.pid, spec['native_argv'][0])
                    for pgid in groups:
                        os.killpg(pgid, signal.SIGTERM)
                        signalled.append({'pgid': pgid, 'signal': 'SIGTERM', 'identity_verified': True})
                time.sleep(2)
            code = proc.returncode
    except BaseException as exc:
        errors.append(type(exc).__name__ + ': ' + str(exc))
        if proc is not None and proc.poll() is None:
            exception_cleanup = terminate_owned_native(proc, spec['native_argv'][0])
            errors.extend(exception_cleanup['cleanup_errors'])
        code = proc.returncode if proc is not None else None
    finally:
        reserve.unlink(missing_ok=True)
        receipt = {
            'schema': 'numi.cardboard.native470-capacity-guard-receipt.v2',
            'execution_started': proc is not None,
            'wrapper_exit_code': proc.returncode if proc else None,
            'space_abort': space_abort,
            'signalled_native_child_process_groups': signalled,
            'exception_cleanup': exception_cleanup,
            'exception_child_reaped': (exception_cleanup or {}).get('wrapper_reaped') if exception_cleanup else None,
            'errors': errors,
            'minimum_observed_available_bytes': minfree,
            'available_after_run': available(),
            'wall_seconds': time.monotonic() - start,
            'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'case_receipt_path': str(Path(spec['case']) / 'receipt.json'),
            'diagnostic_only': True,
            'full_protocol_complete': False,
            'checkpoint_restore_parity_established': False,
        }
        (EXECUTIONS / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt, sort_keys=True))
    return code if type(code) is int else (0 if not errors else 1)


if __name__ == '__main__':
    raise SystemExit(main())
