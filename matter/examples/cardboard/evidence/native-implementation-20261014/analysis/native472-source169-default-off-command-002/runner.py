#!/usr/bin/env python3
"""Fail-closed source/build/capacity guard for native472; preflight is default."""
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
SPEC_PATH = PACKAGE / 'native-472-command.json'
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
    assert manifest['schema'] == 'numi.cardboard.native472-package-manifest.v1'
    actual = {str(p.relative_to(PACKAGE)) for p in PACKAGE.rglob('*')
              if p.is_file() and p != MANIFEST_PATH and '__pycache__' not in p.parts}
    assert actual == set(manifest['files_sha256'])
    for name, digest in manifest['files_sha256'].items():
        assert sha(PACKAGE / name) == digest, name
    source = Path(spec['source_snapshot'])
    build = Path(spec['runtime_dir'])
    assert sha(source / 'frozen-source.json') == spec['frozen_source_sha256']
    frozen = json.loads((source / 'frozen-source.json').read_text())
    assert len(frozen['source_sha256']) == 804
    for name, digest in frozen['source_sha256'].items():
        assert sha(source / name) == digest, name
    assert sha(build / 'source-build.json') == spec['source_build_sha256']
    built = json.loads((build / 'source-build.json').read_text())
    assert built['success'] is True and built['source_sha256'] == frozen['source_sha256']
    assert built['artifacts_sha256'] == spec['artifact_sha256']
    for name, digest in built['artifacts_sha256'].items():
        assert Path(name).resolve().is_relative_to(build.resolve()) and sha(name) == digest
    for row in spec['input_bindings']:
        assert sha(row['original']) == row['sha256'] == sha(row['package'])
    cpu = json.loads((PACKAGE / 'inputs/source169-cpu-verification.json').read_text())
    assert cpu['source_snapshot_sha256'] == spec['frozen_source_sha256']
    assert cpu['build_record_sha256'] == spec['source_build_sha256']
    assert cpu['cpu_validation']['total_test_cases_as_reported'] == 27
    kernel = json.loads((PACKAGE / 'inputs/warm-kernel-receipt.json').read_text())
    assert kernel['returncode'] == 0 and kernel['timed_out'] is False and kernel['physical_transactions'] == 0
    baseline = json.loads(Path(spec['baseline470_command']).read_text())
    assert sha(spec['baseline470_command']) == spec['baseline470_command_sha256']
    def physical(values):
        result = []; i = 1
        while i < len(values):
            if values[i] in ('--metallib', '--output', '--coupon-checkpoint-after-control'):
                i += 2
            else:
                result.append(values[i]); i += 1
        return result
    assert physical(spec['native_argv']) == physical(baseline['native_argv'])
    expected_env = dict(baseline['environment_overrides'])
    del expected_env['NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR']
    assert spec['environment_overrides'] == expected_env and len(expected_env) == 11
    assert 'NM_FEM_BASE_CGLS_WARM_START_REFINE' not in expected_env
    native = spec['native_argv']; wrapper = spec['wrapper_argv']; split = wrapper.index('--')
    assert native[0] == str(build / 'numi-matter-cardboard-probe')
    assert native[native.index('--metallib') + 1] == str(build / 'shaders/NumiMatter.metallib')
    assert native[native.index('--output') + 1] == str(Path(spec['case']) / 'output')
    assert '--coupon-checkpoint-after-control' not in native and '--coupon-resume-checkpoint' not in native
    for flag, alias in [('--liner-material', 'liner-material.nmatter'), ('--medium-material', 'medium-material.nmatter')]:
        binding = next(x for x in spec['input_bindings'] if x['alias'] == alias)
        assert native[native.index(flag) + 1] == binding['original']
        assert sha(native[native.index(flag) + 1]) == binding['sha256']
        assert binding['original'] in wrapper[:split]
    actual_inputs = [wrapper[i+1] for i, token in enumerate(wrapper[:split]) if token == '--input-file']
    expected_inputs = [row['original'] if row['alias'].endswith('.nmatter') else row['package'] for row in spec['input_bindings']]
    assert actual_inputs == expected_inputs
    assert wrapper[split + 1:] == native
    assert wrapper[1] == '-B' and wrapper[2] == str(source / 'matter/tools/cardboard_source_bound_run.py')
    assert wrapper[wrapper.index('--timeout') + 1] == '3600'
    wrapper_env = {wrapper[i+1].split('=', 1)[0]: wrapper[i+1].split('=', 1)[1]
                   for i, token in enumerate(wrapper[:split]) if token == '--env'}
    assert wrapper_env == expected_env
    case = Path(spec['case'])
    assert case.parent.resolve() == B / 'solver' and not case.exists() and not EXECUTIONS.exists()
    active = []; stopped = []
    for line in subprocess.check_output(['ps', '-axo', 'pid=,ppid=,stat=,comm='], text=True).splitlines():
        f = line.split(None, 3)
        if len(f) != 4 or f[2].startswith('Z'): continue
        if 'numi-matter-' not in Path(f[3]).name and Path(f[3]).name not in ('warm_start_check','sparse_scan_check'): continue
        if f[:3] == ['77441', '76106', 'T'] and f[3] == '/Users/home/numi-cardboard-20261006/build-cardboard-world/matter/numi-matter-cardboard-paper-material-check':
            parent = subprocess.check_output(['ps', '-p', '76106', '-o', 'comm='], text=True).strip()
            assert Path(parent).name == 'lldb'; stopped.append(line)
        else: active.append(line)
    assert not active, active
    free = available(); assert free >= MINIMUM
    return spec, {'available_bytes': free, 'active_gpu_workers': active,
                  'preserved_stopped_debugger_targets': stopped, 'verified_source_files': 804,
                  'verified_build_artifacts': len(built['artifacts_sha256']),
                  'verified_inputs': len(spec['input_bindings']),
                  'source_build_sha256': spec['source_build_sha256'],
                  'frozen_source_sha256': spec['frozen_source_sha256'], 'cpu_test_cases': 27}


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
        'schema': 'numi.cardboard.native472-capacity-launch.v2',
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
        'checkpoint_control_index': None,
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
            'schema': 'numi.cardboard.native472-capacity-guard-receipt.v2',
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
