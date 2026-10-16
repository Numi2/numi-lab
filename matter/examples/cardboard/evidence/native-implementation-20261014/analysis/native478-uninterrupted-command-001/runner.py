from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

B = Path('/Users/n/cardboard-implementation-20261014-host-macmini')
P = B / 'analysis/native478-uninterrupted-command-001'
S = B / 'snapshots/source-checkpoint-warm-breakdown-173'
D = B / 'build-source173-001'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def parse_properties(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        k, sep, v = line.partition('=')
        assert sep and k not in values, line
        values[k] = v
    return values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['478'], required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    spec = json.loads((P / 'command-spec.json').read_text())
    assert sha(P / 'capacity-helper.py') == '8e4a395c27d0dda8f43351760da7d5b6d0c42dbad0d16680777383d94364cc03'
    binding = json.loads((P / 'package-manifest.json').read_text())
    for name, digest in binding['files_sha256'].items():
        assert sha(P / name) == digest, name
    assert sha(S / 'frozen-source.json') == spec['source_frozen_sha256']
    assert sha(D / 'source-build.json') == spec['source_build_sha256']
    frozen = json.loads((S / 'frozen-source.json').read_text())
    build = json.loads((D / 'source-build.json').read_text())
    assert build['success'] and build['source_sha256'] == frozen['source_sha256']
    for name, digest in frozen['source_sha256'].items():
        assert sha(S / name) == digest, name
    for name, digest in build['artifacts_sha256'].items():
        assert sha(name) == digest, name
    validation = json.loads((P / 'inputs/host-validation-report.json').read_text())
    validation_manifest = json.loads((P / 'inputs/host-validation-manifest.json').read_text())
    assert validation_manifest['files_sha256']['report.json'] == sha(P / 'inputs/host-validation-report.json')
    for name, digest in validation_manifest['files_sha256'].items():
        assert sha(B / 'analysis/source173-host-validation-002' / name) == digest, name
    assert validation['source_build_sha256'] == spec['source_build_sha256']
    assert validation['artifact_sha256'] == build['artifacts_sha256']
    assert validation['native_helper_checks'] == 264 and validation['cpu_test_cases'] == 13
    assert all(r['returncode'] == 0 for r in validation['receipts'])
    workers = [line for line in subprocess.check_output(['ps', '-axo', 'pid=,stat=,comm='], text=True).splitlines()
               if any(k in line for k in ('numi-matter-', 'sparse-scan-checker', 'source469-material-replay-check'))
               and not line.split()[1].startswith('Z')]
    assert not workers, workers
    helper_spec = importlib.util.spec_from_file_location('guard472', P / 'capacity-helper.py')
    guard = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(guard)
    guard.B = B
    assert guard.available() >= guard.MINIMUM
    case = Path(spec['cases'][args.phase]['case'])
    execution = P / ('execution-' + args.phase)
    assert not case.exists() and not execution.exists()
    inputs = [P / 'command-spec.json', S / 'frozen-source.json',
              P / 'inputs/host-validation-report.json', P / 'inputs/host-validation-manifest.json']
    prefix_report = None
    names = [p.name for p in inputs] + ['source-build.json']
    assert len(names) == len(set(names))
    command = [sys.executable, '-B', str(S / 'matter/tools/cardboard_source_bound_run.py'),
        '--source-snapshot', str(S), '--runtime-dir', str(D), '--build-provenance', str(D / 'source-build.json'),
        '--case', str(case), '--timeout', '14400', '--question',
        'Does source173 complete all60uninterrupted loading, withdrawal, release and relaxation controls with unchanged strict certificates and no trace readback?',
        '--prediction', spec['qualification']]
    for key, value in sorted(spec['environment'].items()):
        command += ['--env', key + '=' + value]
    for path in inputs:
        command += ['--input-file', str(path)]
    command += ['--'] + spec['cases'][args.phase]['native_argv']
    preflight = {'source_files_verified': len(frozen['source_sha256']), 'host_tag': 'macmini',
        'build_artifacts_verified': len(build['artifacts_sha256']), 'available_bytes': guard.available(),
        'prefix': prefix_report, 'phase': args.phase, 'argv': command, 'execute_requested': args.execute}
    print(json.dumps(preflight), flush=True)
    if not args.execute:
        return 0
    execution.mkdir()
    reserve = execution / 'emergency-reserve.bin'
    proc = cleanup = None
    errors = []
    code = None
    space_abort = timed_out = False
    minfree = guard.available()
    start = time.monotonic()
    environment = {k: v for k, v in os.environ.items() if not k.startswith('NM_FEM_') and k not in guard.TIMING_KEYS}
    try:
        reserve_info = guard.write_reserve_file(reserve)
        (execution / 'launch.json').write_text(json.dumps(dict(preflight,
            started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            reserve=reserve_info, package_manifest_sha256=sha(P / 'package-manifest.json'),
            parent_environment_removed_keys=sorted(set(os.environ) - set(environment))), indent=2) + '\n')
        with (execution / 'wrapper.log').open('x') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
            while proc.poll() is None:
                free = guard.available()
                minfree = min(minfree, free)
                space_abort = free < guard.ABORT_FLOOR
                timed_out = time.monotonic() - start > 14520
                if space_abort or timed_out:
                    reserve.unlink(missing_ok=True)
                    cleanup = guard.terminate_owned_native(proc, spec['cases'][args.phase]['native_argv'][0])
                    errors.extend(cleanup['cleanup_errors'])
                    break
                time.sleep(2)
            code = proc.poll()
    except BaseException as error:
        errors.append(type(error).__name__ + ': ' + str(error))
        if proc is not None and proc.poll() is None:
            cleanup = guard.terminate_owned_native(proc, spec['cases'][args.phase]['native_argv'][0])
            errors.extend(cleanup['cleanup_errors'])
        code = proc.poll() if proc else None
    finally:
        reserve.unlink(missing_ok=True)
        receipt = {'schema': 'numi.cardboard.macmini-native478-guard.v1', 'host_tag': 'macmini',
            'phase': args.phase, 'execution_started': proc is not None, 'wrapper_exit_code': code,
            'space_abort': space_abort, 'timed_out': timed_out, 'errors': errors, 'cleanup': cleanup,
            'minimum_available_bytes': minfree, 'available_after_run': guard.available(),
            'wall_seconds': time.monotonic() - start,
            'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'numerical_convergence_qualified': False, 'uninterrupted_trajectory_parity': False}
        (execution / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)
    return code if type(code) is int and code != 0 else int(bool(errors) or space_abort or timed_out or code is None)

if __name__ == '__main__':
    raise SystemExit(main())
