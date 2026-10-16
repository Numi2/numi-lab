#!/usr/bin/env python3
"""Source-bound, fail-closed preflight runner for native checker 170.

Default invocation only verifies inputs and reports capacity/process state. Metal
kernels are dispatched only with the explicit --execute switch.
"""
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
B = Path('/Users/home/cardboard-implementation-20261014')
OVERLAY = B / 'analysis/hierarchical-sparse-private-contact-scan-overlay-170-001'
BASE = B / 'snapshots/source-warm-cgls-refinement-169'
STATIC_MANIFEST = PACKAGE / 'package-manifest.json'
SPEC = PACKAGE / 'native-170-command.json'
EXECUTABLE = PACKAGE / 'validation/sparse-scan-checker'
METALLIB = PACKAGE / 'NumiMatter-O3.metallib'
EXECUTIONS = PACKAGE / 'executions'
MIB = 1024 ** 2
GIB = 1024 * MIB
MINIMUM_FREE = 2 * GIB
ABORT_FLOOR = 512 * MIB
RESERVE_BYTES = 128 * MIB
RESERVE_CHUNK = 1024 * 1024
TIMEOUT_SECONDS = 900
EXPECTED_STOPPED = {
    'pid': 77441,
    'ppid': 76106,
    'state': 'T',
    'command': '/Users/home/numi-cardboard-20261006/build-cardboard-world/matter/numi-matter-cardboard-paper-material-check',
    'parent_name': 'lldb',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def available_bytes():
    stats = os.statvfs(B)
    return stats.f_bavail * stats.f_frsize


def _safe_regular(path):
    path = Path(path)
    return not path.is_symlink() and path.is_file()


def _package_files():
    result = set()
    for path in PACKAGE.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'package contains symlink: {path.relative_to(PACKAGE)}')
        if not path.is_file():
            continue
        rel = path.relative_to(PACKAGE)
        if rel == Path('package-manifest.json') or rel.parts[0] == 'executions' or '__pycache__' in rel.parts:
            continue
        result.add(rel.as_posix())
    return result


def verify_package():
    if not _safe_regular(STATIC_MANIFEST) or not _safe_regular(SPEC):
        raise ValueError('manifest or command specification is missing/unsafe')
    manifest = json.loads(STATIC_MANIFEST.read_text())
    spec = json.loads(SPEC.read_text())
    if manifest.get('schema') != 'numi.cardboard.native-sparse-scan-package.v1':
        raise ValueError('unexpected package manifest schema')
    if spec.get('schema') != 'numi.cardboard.native-sparse-scan-command.v1':
        raise ValueError('unexpected command spec schema')
    files = manifest.get('files', {})
    if _package_files() != set(files):
        raise ValueError('package static-file inventory differs from sealed manifest')
    for relative, row in files.items():
        path = PACKAGE / relative
        if not _safe_regular(path):
            raise ValueError(f'package file missing/unsafe: {relative}')
        if type(row.get('bytes')) is not int or path.stat().st_size != row['bytes']:
            raise ValueError(f'package file size changed: {relative}')
        if sha(path) != row.get('sha256'):
            raise ValueError(f'package file hash changed: {relative}')

    overlay_json = OVERLAY / 'source-overlay.json'
    overlay_patch = OVERLAY / 'source-overlay.patch'
    overlay = json.loads(overlay_json.read_text())
    if sha(overlay_json) != spec['source_overlay_json_sha256']:
        raise ValueError('source170 overlay manifest hash mismatch')
    if sha(overlay_patch) != spec['source_overlay_patch_sha256']:
        raise ValueError('source170 patch hash mismatch')
    if sha(BASE / 'frozen-source.json') != overlay['base_frozen_source_json_sha256']:
        raise ValueError('source169 base manifest hash mismatch')
    for row in overlay['changed_files']:
        relative = row['path']
        candidate = OVERLAY / 'candidate' / relative
        if not _safe_regular(candidate) or sha(candidate) != row['candidate_sha256']:
            raise ValueError(f'source170 candidate file changed: {relative}')
        if row['base_present']:
            base_file = BASE / relative
            if not _safe_regular(base_file) or sha(base_file) != row['base_sha256']:
                raise ValueError(f'source169 base file changed: {relative}')
    checker_rel = 'matter/tools/cardboard_sparse_private_contact_scan_native_check.mm'
    checker_row = next(row for row in overlay['changed_files'] if row['path'] == checker_rel)
    if sha(PACKAGE / 'source/cardboard_sparse_private_contact_scan_native_check.mm') != checker_row['candidate_sha256']:
        raise ValueError('packaged checker source does not match source170 candidate')
    artifact = next(row for row in overlay['validation_artifacts']
                    if row['path'] == 'validation/NumiMatter-O3-deps.metallib')
    if artifact['sha256'] != spec['source_metallib_sha256'] or sha(METALLIB) != artifact['sha256']:
        raise ValueError('packaged metallib does not match source170 O3-linked artifact')
    if not _safe_regular(EXECUTABLE) or sha(EXECUTABLE) != spec['checker_binary_sha256']:
        raise ValueError('compiled checker binary hash mismatch')
    if spec.get('execution') != 'explicit --execute only; default is preflight and never dispatches Metal':
        raise ValueError('execution mode contract changed')
    inventory = json.loads((PACKAGE / 'inputs/gpu-case-inventory.json').read_text())
    if inventory.get('native_case_count') != 49 or inventory.get('total_environment_pair_slots') != 6366845:
        raise ValueError('native case inventory arithmetic changed')
    return spec, manifest, inventory


def _parse_process_line(line):
    fields = line.split(None, 4)
    if len(fields) != 5:
        return None
    try:
        return {'pid': int(fields[0]), 'ppid': int(fields[1]), 'pgid': int(fields[2]),
                'state': fields[3], 'command': fields[4]}
    except ValueError:
        return None


def _worker_relevant(command):
    name = Path(command).name
    return ('numi-matter-' in name or name in {
        'sparse-scan-checker', 'warm_start_check', 'sparse_scan_check',
        'numi-matter-cardboard-probe', 'numi-matter-cardboard-paper-material-check'})


def classify_process_rows(lines, parent_name_for_pid):
    active = []
    preserved = []
    for line in lines:
        row = _parse_process_line(line)
        if row is None or row['state'].startswith('Z') or not _worker_relevant(row['command']):
            continue
        exact = (row['pid'] == EXPECTED_STOPPED['pid'] and
                 row['ppid'] == EXPECTED_STOPPED['ppid'] and
                 row['state'] == EXPECTED_STOPPED['state'] and
                 row['command'] == EXPECTED_STOPPED['command'] and
                 parent_name_for_pid(row['ppid']) == EXPECTED_STOPPED['parent_name'])
        if exact:
            preserved.append(row)
        else:
            active.append(row)
    return active, preserved


def process_snapshot():
    lines = subprocess.check_output(
        ['ps', '-axo', 'pid=,ppid=,pgid=,stat=,comm='], text=True).splitlines()
    def parent_name(pid):
        return Path(subprocess.check_output(['ps', '-p', str(pid), '-o', 'comm='], text=True).strip()).name
    return classify_process_rows(lines, parent_name)


def preflight():
    spec, manifest, inventory = verify_package()
    active, preserved = process_snapshot()
    if active:
        raise RuntimeError('another Matter/GPU worker is present: ' + json.dumps(active, sort_keys=True))
    free = available_bytes()
    if free < MINIMUM_FREE:
        raise RuntimeError(f'free disk {free} is below launch minimum {MINIMUM_FREE}')
    return {'schema': 'numi.cardboard.native-sparse-scan-preflight.v1',
            'launch_allowed_if_root_queue_is_clear': True,
            'gpu_dispatch_occurred': False,
            'available_bytes': free,
            'preserved_stopped_debugger_targets': preserved,
            'native_gpu_cases': inventory['native_case_count'],
            'environment_pair_slots': inventory['total_environment_pair_slots'],
            'checker_binary_sha256': spec['checker_binary_sha256'],
            'source_metallib_sha256': spec['source_metallib_sha256'],
            'package_file_count': len(manifest['files'])}


def write_reserve(path, byte_count=RESERVE_BYTES, chunk_size=RESERVE_CHUNK):
    if byte_count <= 0 or chunk_size <= 0:
        raise ValueError('reserve sizes must be positive')
    chunk = os.urandom(min(byte_count, chunk_size))
    digest = hashlib.sha256()
    written = 0
    with Path(path).open('xb') as stream:
        while written < byte_count:
            block = chunk[:min(len(chunk), byte_count - written)]
            stream.write(block)
            digest.update(block)
            written += len(block)
        stream.flush()
        os.fsync(stream.fileno())
        st = os.fstat(stream.fileno())
    allocated = st.st_blocks * 512
    if st.st_size != byte_count or allocated < byte_count:
        raise RuntimeError(f'reserve allocation short: size={st.st_size} allocated={allocated} requested={byte_count}')
    return {'requested_bytes': byte_count, 'file_bytes': st.st_size,
            'allocated_bytes': allocated, 'sha256': digest.hexdigest(), 'fsynced': True}


def _process_rows():
    lines = subprocess.check_output(
        ['ps', '-axo', 'pid=,ppid=,pgid=,comm='], text=True).splitlines()
    rows = []
    for line in lines:
        fields = line.split(None, 3)
        if len(fields) != 4:
            continue
        try:
            rows.append((int(fields[0]), int(fields[1]), int(fields[2]), fields[3]))
        except ValueError:
            continue
    return rows


def _verified_owned_group(proc):
    for pid, ppid, pgid, command in _process_rows():
        if pid == proc.pid and ppid == os.getpid() and pgid == proc.pid:
            try:
                if Path(command).resolve() == EXECUTABLE.resolve():
                    return pgid
            except (OSError, RuntimeError):
                pass
    return None


def terminate_owned(proc):
    result = {'group_verified': False, 'signals': [], 'reaped': False, 'error': None}
    try:
        group = _verified_owned_group(proc)
        if group is not None:
            result['group_verified'] = True
            try:
                os.killpg(group, signal.SIGTERM)
                result['signals'].append('SIGTERM')
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            group = _verified_owned_group(proc)
            if group is not None:
                os.killpg(group, signal.SIGKILL)
                result['signals'].append('SIGKILL')
            proc.wait(timeout=20)
        result['reaped'] = proc.poll() is not None
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    return result


def _attempt_directory():
    EXECUTIONS.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10000):
        path = EXECUTIONS / f'attempt-{index:03d}'
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError('no unused attempt directory available')


def execute():
    pre = preflight()
    attempt = _attempt_directory()
    reserve = attempt / 'emergency-reserve.bin'
    stdout_path = attempt / 'stdout.log'
    stderr_path = attempt / 'stderr.log'
    receipt_path = attempt / 'receipt.json'
    receipt = {'schema': 'numi.cardboard.native-sparse-scan-execution-receipt.v1',
               'source': pre, 'start_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'command': [str(EXECUTABLE)], 'timeout_seconds': TIMEOUT_SECONDS,
               'gpu_dispatch_requested': True}
    reserve_info = write_reserve(reserve)
    receipt['reserve'] = reserve_info
    proc = None
    start = time.monotonic()
    try:
        with stdout_path.open('xb') as out, stderr_path.open('xb') as err:
            proc = subprocess.Popen([str(EXECUTABLE)], cwd=PACKAGE, stdout=out, stderr=err,
                                    start_new_session=True, close_fds=True)
            receipt['pid'] = proc.pid
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                free = available_bytes()
                if free < ABORT_FLOOR:
                    receipt['space_abort'] = True
                    receipt['available_at_abort_bytes'] = free
                    reserve.unlink(missing_ok=True)
                    receipt['termination'] = terminate_owned(proc)
                    break
                if elapsed > TIMEOUT_SECONDS:
                    receipt['timed_out'] = True
                    reserve.unlink(missing_ok=True)
                    receipt['termination'] = terminate_owned(proc)
                    break
                time.sleep(2)
            if proc.poll() is None:
                receipt['exit_code'] = None
            else:
                receipt['exit_code'] = proc.returncode
            receipt['wall_seconds'] = time.monotonic() - start
            receipt['stdout_sha256'] = sha(stdout_path)
            receipt['stderr_sha256'] = sha(stderr_path)
            receipt['stdout_bytes'] = stdout_path.stat().st_size
            receipt['stderr_bytes'] = stderr_path.stat().st_size
            receipt['end_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            reserve.unlink(missing_ok=True)
            with receipt_path.open('x') as stream:
                json.dump(receipt, stream, sort_keys=True, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            return receipt
    except Exception as error:
        receipt['exception'] = f'{type(error).__name__}: {error}'
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
    parser.add_argument('--execute', action='store_true', help='explicitly dispatch the native Metal checker')
    args = parser.parse_args(argv)
    if args.execute:
        result = execute()
        print(json.dumps(result, sort_keys=True, indent=2))
        return int(result.get('exit_code', 1) != 0 or result.get('timed_out', False) or result.get('space_abort', False))
    result = preflight()
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
