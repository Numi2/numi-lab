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
P = B / 'analysis/native497498-diagnostic-replay-command-002'
S = B / 'snapshots/source-admission-encoder-182-003'
D = B / 'build-source182-003'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def parse_properties(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        k, sep, v = line.partition('=')
        assert sep and k not in values, line
        values[k] = v
    return values

def prefix():
    C = B / 'solver/warm-cgls-control19-checkpoint-480'
    receipt = json.loads((C / 'receipt.json').read_text())
    assert receipt['exit_code'] == 0 and receipt['source_inputs_unchanged'] and not receipt['timed_out']
    result = json.loads((C / 'output/result.json').read_text())
    assert result['status'] == 'accepted_checkpoint_prefix_saved' and result['accepted_steps'] == 20
    cp = C / 'output/checkpoints/coupon-control-000019.nmcb'
    meta = parse_properties(cp.with_suffix('.resume'))
    pm = cp.with_suffix('.prefix')
    assert meta['checkpoint_sha256'] == sha(cp) and meta['checkpoint_bytes'] == str(cp.stat().st_size)
    assert meta['environment_count'] == '2' and meta['both_environments_accepted'] == 'true'
    assert meta['accepted_control_index'] == '19' and meta['next_control_index'] == '20'
    assert meta['source_run_directory'] == str(C / 'output') and meta['prefix_manifest_sha256'] == sha(pm)
    assert meta['protocol_manifest_sha256'] == sha(C / 'output/manifest.json')
    entries = []
    for line in pm.read_text().splitlines():
        if not line.startswith('file='):
            continue
        rel, size, digest, observation = line[5:].split('|')
        p = Path(rel)
        assert not p.is_absolute() and '..' not in p.parts
        target = C / 'output' / p
        assert target.stat().st_size == int(size) and sha(target) == digest, rel
        entries.append(rel)
    assert 'observations.csv' in entries and 'manifest.json' in entries
    return {'checkpoint_sha256': sha(cp), 'checkpoint_bytes': cp.stat().st_size,
            'metadata_sha256': sha(cp.with_suffix('.resume')), 'prefix_sha256': sha(pm),
            'prefix_files_verified': len(entries), 'complete_protocol': False}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['497', '498'], required=True)
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
    assert validation_manifest['report_sha256'] == sha(P / 'inputs/host-validation-report.json')
    for name, info in validation_manifest['files'].items():
        assert sha(B / 'analysis/source182-build-validation-003-followup-001' / name) == info['sha256'], name
    assert validation['source_build_json_sha256'] == spec['source_build_sha256']
    assert validation['source_frozen_sha256'] == spec['source_frozen_sha256']
    assert validation['artifacts_sha256'] == build['artifacts_sha256']
    assert validation['ctest_count'] == 3
    assert validation['environment_cases'] == 12 and validation['standalone_contract'] == 'PASS'
    assert not validation['gpu_dispatch']
    for material in spec['material_path_bindings']:
        assert sha(material['original']) == material['sha256']
        assert sha(material['current']) == material['sha256']
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
    if args.phase in ('497', '498'):
        prefix_report = prefix()
        C = B / 'solver/warm-cgls-control19-checkpoint-480'
        inputs += [C / rel for rel in ('prediction.json', 'native-environment.json', 'receipt.json',
            'diagnostic-captures.json', 'output/manifest.json', 'output/result.json',
            'output/observations.csv', 'output/coupon-checkpoint-prefix.json',
            'output/coupon-checkpoint-saved.txt', 'output/checkpoints/coupon-control-000019.nmcb',
            'output/checkpoints/coupon-control-000019.resume', 'output/checkpoints/coupon-control-000019.prefix')]
    inputs += [Path(material['original']) for material in spec['material_path_bindings']]
    names = [p.name for p in inputs] + ['source-build.json']
    assert len(names) == len(set(names))
    command = [sys.executable, '-B', str(S / 'matter/tools/cardboard_source_bound_run.py'),
        '--source-snapshot', str(S), '--runtime-dir', str(D), '--build-provenance', str(D / 'source-build.json'),
        '--case', str(case), '--timeout', '14400', '--question',
        'Does source182003 admit strict diagnostic replay of accepted checkpoint19, and do bounded read-only captures preserve the failing control20 production history?',
        '--prediction', spec['qualification'], '--diagnostic-run']
    native_environment = dict(spec['environment'])
    if args.phase == '497':
        native_environment.update(spec['capture_environment'])
        for key in spec['capture_keys']:
            command += ['--diagnostic-capture', key]
    for key, value in sorted(native_environment.items()):
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
        receipt = {'schema': 'numi.cardboard.macmini-native497498-guard.v1', 'host_tag': 'macmini',
            'phase': args.phase, 'execution_started': proc is not None, 'wrapper_exit_code': code,
            'space_abort': space_abort, 'timed_out': timed_out, 'errors': errors, 'cleanup': cleanup,
            'minimum_available_bytes': minfree, 'available_after_run': guard.available(),
            'wall_seconds': time.monotonic() - start,
            'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'full_protocol_qualification': False, 'uninterrupted_trajectory_parity': False}
        (execution / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)
    return code if type(code) is int and code != 0 else int(bool(errors) or space_abort or timed_out or code is None)

if __name__ == '__main__':
    raise SystemExit(main())
