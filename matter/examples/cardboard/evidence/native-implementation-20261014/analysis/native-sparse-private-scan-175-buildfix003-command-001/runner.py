#!/usr/bin/env python3
"""Fail-closed remote preflight/runner for the Source175 scan checker.

No Metal dispatch occurs unless --execute is supplied. Intended for macmini.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import signal
import socket
import subprocess
import time

ROOT = Path('/Users/n/cardboard-implementation-20261014-host-macmini')
PACKAGE = ROOT / 'analysis/native-sparse-private-scan-175-buildfix003-command-001'
SPEC_PATH = PACKAGE / 'command.json'
PACKAGE_MANIFEST = PACKAGE / 'package-manifest.json'
BUILD_DIR = ROOT / 'build-source175-buildfix-003'
SNAPSHOT = ROOT / 'snapshots/source-sparse-private-scan-175-buildfix-003'
EXECUTABLE = BUILD_DIR / 'numi-matter-cardboard-sparse-private-scan-native-check'
METALLIB = BUILD_DIR / 'shaders/NumiMatter.metallib'
RUN_ROOT = ROOT / 'native/sparse-private-scan-175-buildfix003-command-001'
MIB = 1024 * 1024
GIB = 1024 * MIB
MINIMUM_FREE = 2 * GIB
ABORT_FLOOR = 512 * MIB
RESERVE_BYTES = 128 * MIB
TIMEOUT_SECONDS = 900


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def free_bytes():
    st = os.statvfs(ROOT)
    return st.f_bavail * st.f_frsize


def _regular(path):
    path = Path(path)
    return not path.is_symlink() and path.is_file()


def verify_inputs():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('runner must execute on the macmini Apple Silicon host')
    if not _regular(SPEC_PATH) or not _regular(PACKAGE_MANIFEST):
        raise RuntimeError('command specification or package manifest missing/unsafe')
    manifest = json.loads(PACKAGE_MANIFEST.read_text())
    if manifest.get('schema') != 'numi.cardboard.native-check-command-package.v1':
        raise RuntimeError('unexpected package manifest schema')
    expected_files = manifest.get('files')
    if not isinstance(expected_files, dict):
        raise RuntimeError('package file inventory missing')
    actual_files = set()
    for path in PACKAGE.rglob('*'):
        if path.is_symlink():
            raise RuntimeError(f'package symlink forbidden: {path}')
        if not path.is_file() or path == PACKAGE_MANIFEST:
            continue
        if 'executions' in path.relative_to(PACKAGE).parts:
            continue
        rel = path.relative_to(PACKAGE).as_posix()
        actual_files.add(rel)
        row = expected_files.get(rel)
        if not isinstance(row, dict) or path.stat().st_size != row.get('bytes') or sha(path) != row.get('sha256'):
            raise RuntimeError(f'package file hash/size mismatch: {rel}')
    if actual_files != set(expected_files):
        raise RuntimeError('package static file inventory differs from sealed manifest')
    spec = json.loads(SPEC_PATH.read_text())
    expected = {
        'source_frozen_manifest_sha256': '6e4f8dad1377af79b1dc2ddeeb0f07e6a273598d291cb030e6222e8aebb5a007',
        'source_build_metadata_sha256': '75c24e646fccb250fe0f8bd7d812491d5aac3892f99e863578982780f1194a25',
        'checker_source_sha256': 'bf18e81d70b3ca120d0afa479af0bcb5c4c1b435c1dbd55b8df6db820a7b8b5d',
        'checker_executable_sha256': 'e5a2ee1edfa6d2395b6d1c512bfc34f94e270fd27aeb6a0276994ce264ffff70',
        'metallib_sha256': '59a7df8c8fd4d336fbcb9e28b7fe3942d86063a4a1fbafe36963e41a0c007cc9',
    }
    if spec.get('schema') != 'numi.cardboard.native-sparse-private-scan-command.v1':
        raise RuntimeError('unexpected command schema')
    if spec.get('host') != 'macmini' or spec.get('native_argv') != [str(EXECUTABLE)]:
        raise RuntimeError('host or command binding changed')
    if spec.get('native_case_count') != 49 or spec.get('environment_pair_slots') != 6366845:
        raise RuntimeError('native checker case inventory changed')
    inventory = json.loads((PACKAGE / 'case-inventory.json').read_text())
    if (inventory.get('native_case_count') != 49 or
            inventory.get('total_environment_pair_slots') != 6366845 or
            inventory.get('source_checker') != 'source/cardboard_sparse_private_contact_scan_native_check.mm'):
        raise RuntimeError('case inventory summary differs from command specification')
    frozen = SNAPSHOT / 'frozen-source.json'
    source_build = BUILD_DIR / 'source-build.json'
    checker_source = SNAPSHOT / 'matter/tools/cardboard_sparse_private_contact_scan_native_check.mm'
    actual = {
        'source_frozen_manifest_sha256': sha(frozen),
        'source_build_metadata_sha256': sha(source_build),
        'checker_source_sha256': sha(checker_source),
        'checker_executable_sha256': sha(EXECUTABLE),
        'metallib_sha256': sha(METALLIB),
    }
    for key, value in expected.items():
        if spec.get(key) != value or actual[key] != value:
            raise RuntimeError(f'{key} mismatch: expected={value} actual={actual[key]}')

    frozen_data = json.loads(frozen.read_text())
    source_hashes = frozen_data.get('source_sha256')
    if not isinstance(source_hashes, dict) or len(source_hashes) != 806:
        raise RuntimeError('source manifest must describe all 806 frozen files')
    for rel, expected_sha in source_hashes.items():
        path = SNAPSHOT / rel
        if not _regular(path) or sha(path) != expected_sha:
            raise RuntimeError(f'frozen source mismatch: {rel}')

    build_data = json.loads(source_build.read_text())
    required_targets = {'numi-matter-cardboard-full-contact-armijo-check',
                        'numi-matter-cardboard-sparse-private-scan-native-check'}
    if (build_data.get('build_exit_code') != 0 or
            build_data.get('source_snapshot') != str(SNAPSHOT) or
            not required_targets.issubset(set(build_data.get('targets', [])))):
        raise RuntimeError('build metadata does not certify this successful source snapshot')
    artifacts = build_data.get('artifacts', {})
    for path, expected_sha in ((str(EXECUTABLE), expected['checker_executable_sha256']),
                               (str(METALLIB), expected['metallib_sha256'])):
        row = artifacts.get(path)
        if not isinstance(row, dict) or row.get('sha256') != expected_sha:
            raise RuntimeError(f'build metadata artifact binding missing: {path}')
    full_contact = spec['related_full_contact_armijo_checker']
    related_source = SNAPSHOT / 'matter/tools/cardboard_full_contact_armijo_check.mm'
    related_binary = BUILD_DIR / full_contact['executable'].rsplit('/', 1)[-1]
    if (sha(related_source) != full_contact['source_sha256'] or
            sha(related_binary) != full_contact['executable_sha256']):
        raise RuntimeError('related full-contact Armijo checker hash mismatch')
    return spec


def _parse_process_rows():
    output = subprocess.check_output(
        ['ps', '-axo', 'pid=,ppid=,pgid=,stat=,comm='], text=True)
    rows = []
    for line in output.splitlines():
        fields = line.split(None, 4)
        if len(fields) != 5:
            continue
        try:
            pid, ppid, pgid = map(int, fields[:3])
        except ValueError:
            continue
        rows.append({'pid': pid, 'ppid': ppid, 'pgid': pgid,
                     'state': fields[3], 'command': fields[4]})
    return rows


def active_workers():
    result = []
    for row in _parse_process_rows():
        name = Path(row['command']).name
        if row['state'].startswith('Z'):
            continue
        if name.startswith('numi-matter-') or name == EXECUTABLE.name:
            result.append(row)
    return result


def parse_checker_output(text):
    match = re.search(r'^hierarchical_sparse_scan_native_checks=(\d+) kernels_executed=1 physical_transactions=0\s*$',
                      text, re.MULTILINE)
    if match is None:
        return {'passed': False, 'checks': None,
                'reason': 'required native checker completion marker is absent'}
    checks = int(match.group(1))
    if checks <= 0:
        return {'passed': False, 'checks': checks,
                'reason': 'checker reported no assertions'}
    return {'passed': True, 'checks': checks, 'reason': None}


def preflight():
    spec = verify_inputs()
    workers = active_workers()
    if workers:
        raise RuntimeError('active Matter/native checker process blocks launch: ' +
                           json.dumps(workers, sort_keys=True))
    available = free_bytes()
    if available < MINIMUM_FREE:
        raise RuntimeError(f'available disk {available} is below 2 GiB launch floor')
    return {'schema': 'numi.cardboard.native-sparse-private-scan-preflight.v1',
            'host': socket.gethostname(), 'gpu_dispatch_occurred': False,
            'available_bytes': available, 'active_workers': workers,
            'native_cases': spec['native_case_count'],
            'environment_pair_slots': spec['environment_pair_slots'],
            'checker_sha256': spec['checker_executable_sha256'],
            'metallib_sha256': spec['metallib_sha256'],
            'source_manifest_sha256': spec['source_frozen_manifest_sha256']}


def allocate_reserve(path):
    chunk = os.urandom(1024 * 1024)
    written = 0
    with Path(path).open('xb') as stream:
        while written < RESERVE_BYTES:
            block = chunk[:min(len(chunk), RESERVE_BYTES - written)]
            stream.write(block)
            written += len(block)
        stream.flush()
        os.fsync(stream.fileno())
        st = os.fstat(stream.fileno())
    allocated = st.st_blocks * 512
    if st.st_size != RESERVE_BYTES or allocated < RESERVE_BYTES:
        raise RuntimeError('emergency reserve was not fully allocated')
    return {'size_bytes': st.st_size, 'allocated_bytes': allocated}


def _owns_group(proc):
    try:
        row = next(r for r in _parse_process_rows() if r['pid'] == proc.pid)
        return row['ppid'] == os.getpid() and row['pgid'] == proc.pid and \
            Path(row['command']).resolve() == EXECUTABLE.resolve()
    except (StopIteration, OSError, RuntimeError):
        return False


def terminate_owned(proc):
    result = {'verified_owned_group': False, 'signals': [], 'reaped': False, 'error': None}
    try:
        if _owns_group(proc):
            result['verified_owned_group'] = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                result['signals'].append('SIGTERM')
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            if _owns_group(proc):
                os.killpg(proc.pid, signal.SIGKILL)
                result['signals'].append('SIGKILL')
            proc.wait(timeout=20)
        result['reaped'] = proc.poll() is not None
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result


def fresh_attempt():
    exec_dir = RUN_ROOT / 'executions'
    exec_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, 10000):
        path = exec_dir / f'attempt-{i:04d}'
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError('no unused fresh receipt directory')


def execute():
    pre = preflight()
    attempt = fresh_attempt()
    reserve = attempt / 'emergency-reserve.bin'
    receipt_path = attempt / 'receipt.json'
    stdout_path = attempt / 'stdout.log'
    stderr_path = attempt / 'stderr.log'
    receipt = {'schema': 'numi.cardboard.native-sparse-private-scan-execution.v1',
               'preflight': pre, 'argv': [str(EXECUTABLE)],
               'start_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'timeout_seconds': TIMEOUT_SECONDS, 'gpu_dispatch_requested': True,
               'timed_out': False, 'space_abort': False}
    proc = None
    start = time.monotonic()
    try:
        receipt['reserve'] = allocate_reserve(reserve)
        with stdout_path.open('xb') as out, stderr_path.open('xb') as err:
            proc = subprocess.Popen([str(EXECUTABLE)], cwd=RUN_ROOT,
                                    stdout=out, stderr=err, start_new_session=True,
                                    close_fds=True)
            receipt['pid'] = proc.pid
            while proc.poll() is None:
                free = free_bytes()
                if free < ABORT_FLOOR:
                    receipt['space_abort'] = True
                    receipt['available_at_abort_bytes'] = free
                    reserve.unlink(missing_ok=True)
                    receipt['termination'] = terminate_owned(proc)
                    break
                if time.monotonic() - start >= TIMEOUT_SECONDS:
                    receipt['timed_out'] = True
                    reserve.unlink(missing_ok=True)
                    receipt['termination'] = terminate_owned(proc)
                    break
                time.sleep(2)
            receipt['exit_code'] = proc.poll()
            receipt['wall_seconds'] = time.monotonic() - start
            receipt['stdout_bytes'] = stdout_path.stat().st_size
            receipt['stderr_bytes'] = stderr_path.stat().st_size
            receipt['stdout_sha256'] = sha(stdout_path)
            receipt['stderr_sha256'] = sha(stderr_path)
            receipt['checker_output_contract'] = parse_checker_output(stdout_path.read_text(errors='replace'))
        receipt['end_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        reserve.unlink(missing_ok=True)
        with receipt_path.open('x') as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        return receipt
    except Exception as exc:
        receipt['exception'] = f'{type(exc).__name__}: {exc}'
        if proc is not None and proc.poll() is None:
            receipt['termination'] = terminate_owned(proc)
        reserve.unlink(missing_ok=True)
        receipt['end_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with receipt_path.open('x') as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true',
                        help='explicitly dispatch the 49-case native Metal checker')
    args = parser.parse_args(argv)
    if args.execute:
        result = execute()
        print(json.dumps(result, sort_keys=True, indent=2))
        return int(result.get('exit_code') != 0 or result.get('timed_out') or
                   result.get('space_abort') or not result.get('checker_output_contract', {}).get('passed', False))
    print(json.dumps(preflight(), sort_keys=True, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
