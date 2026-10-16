from pathlib import Path
import datetime, hashlib, json, os, signal, subprocess, time

B = Path('/Users/home/cardboard-implementation-20261014')
command_path = B / 'analysis/native469-source157-final-newton-fd-command-002/native-469-command.json'
command = json.loads(command_path.read_text())
case = Path(command[command.index('--case') + 1])
guard = B / 'analysis/native469-launch-guard-001'

def available():
    s = os.statvfs(B)
    return s.f_bavail * s.f_frsize

assert hashlib.sha256(command_path.read_bytes()).hexdigest() == 'bd06928508eff13b08359b44f32e0ff4dc447c8df31d6f8ecbda558c25c47674'
assert json.loads((B/'analysis/source157-root-integration-001/report.json').read_text())['selected_source_hashes_verified'] == 787
assert hashlib.sha256((B/'analysis/native469-source157-final-newton-fd-command-002/manifest.json').read_bytes()).hexdigest() == 'ceb7ff97c221c1896c418496abec735740ddd7e35b9c8424bebbf554ec9f545e'

contract_path = command_path.parent / 'initial-effective-environment.json'
contract = json.loads(contract_path.read_text())
assert contract['mode'] == 'diagnostic_only' and not contract['positive_qualification_allowed']
parent_environment = {k:v for k,v in os.environ.items() if not k.startswith('NM_FEM_') and k not in ('NUMI_CARDBOARD_STEP_TIMING','NUMI_CARDBOARD_NATIVE_TIMING')}
overrides = dict(command[i+1].split('=',1) for i,v in enumerate(command) if v == '--env')
effective = dict(parent_environment)
effective.update(overrides)
assert set(overrides) <= set(contract['controls'])
assert {k:({'state':'set','value':effective[k]} if k in effective else {'state':'absent'}) for k in contract['controls']} == contract['controls']
assert sorted(k for k in effective if k.startswith('NM_FEM_')) == sorted(k for k in overrides if k.startswith('NM_FEM_'))

# Read-only build/input admission before creating a case or invoking its wrapper.
build_path = Path(command[command.index('--build-provenance') + 1])
build = json.loads(build_path.read_text())
separator = command.index('--')
executable = str(Path(command[separator + 1]).resolve())
assert build['success'] and executable in build['artifacts_sha256'], 'executable/build mismatch'
assert str(Path(command[command.index('--runtime-dir') + 1]).resolve()) == build['build_dir']
assert str(Path(command[command.index('--source-snapshot') + 1]).resolve()) == build['source_snapshot']
for filename, expected in build['artifacts_sha256'].items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
inputs = [command[i+1] for i,v in enumerate(command) if v == '--input-file']
assert len({Path(v).name for v in inputs}) == len(inputs), 'duplicate input basename'
assert all(Path(v).is_file() for v in inputs)
assert json.loads((B/'native/cgls-right-scale-kernel-check-460/receipt.json').read_text())['result'] == 'pass'

# Serialize only this task's probe workers; preserve paused/unrelated work.
active_task_probes = []
for row in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines():
    fields = row.split(None,2)
    if len(fields)==3 and fields[2].startswith('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-') and fields[2].endswith('/numi-matter-cardboard-probe') and not fields[1].startswith('T'):
        active_task_probes.append({'pid':int(fields[0]),'stat':fields[1],'executable':fields[2]})
assert not active_task_probes, ('another task GPU worker is still active',active_task_probes)
assert not case.exists(), 'fresh case required'
assert available() >= 2048 * 1024**2, 'need at least 2 GiB available before this bounded run'
guard.mkdir(exist_ok=False)
reserve = guard / 'emergency-evidence-reserve'
with reserve.open('xb') as f:
    block = bytes(1024**2)
    for _ in range(128):
        f.write(block)
    f.flush()
    os.fsync(f.fileno())
record = {
    'schema': 'numi.cardboard.native-launch-capacity-guard.v1',
    'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'command_sha256': hashlib.sha256(command_path.read_bytes()).hexdigest(),
    'case': str(case), 'available_before_launch': available(),
    'emergency_reserve_allocated_bytes': reserve.stat().st_blocks * 512,
    'abort_floor_bytes': 512 * 1024**2, 'space_abort': False,
    'minimum_launch_available_bytes': 2048 * 1024**2,
    'nominal_storage_recovery_target_bytes': 2560 * 1024**2,
    'preflight_scope': 'Recovery reached nominal2.5GiB; live launch requires2GiB to allow housekeeping drift. This exceeds original1.5GiB preflight; 512MiB abort floor is unchanged.',
    'native_child_process_groups_signalled': [],
    'initial_effective_environment_sha256': hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    'parent_environment_sanitized_before_wrapper_launch': True,
    'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'diagnostic_only': True,
}
(guard / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
minimum = record['available_before_launch']
started = time.monotonic()
try:
    with (B / 'native469-runner.log').open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=parent_environment)
        while process.poll() is None:
            if case.is_dir() and not (case / 'diagnostics').exists():
                (case / 'diagnostics').mkdir(exist_ok=False)
                record['case_local_capture_directory_created'] = True
            free = available()
            minimum = min(minimum, free)
            if free < record['abort_floor_bytes'] and not record['space_abort']:
                record['space_abort'] = True
                reserve.unlink(missing_ok=True)
                # The source-bound wrapper starts its direct native child in a
                # new session. Signal only that exact owned executable group;
                # let the wrapper retain the termination and normal receipt.
                rows = subprocess.check_output(
                    ['ps', '-axo', 'pid=,ppid=,pgid=,comm='], text=True)
                executable = command[command.index('--') + 1]
                for row in rows.splitlines():
                    fields = row.split(None, 3)
                    if len(fields) == 4 and int(fields[1]) == process.pid and fields[3] == executable:
                        child, group = int(fields[0]), int(fields[2])
                        if child != group:
                            raise RuntimeError('native child lacks its expected private process group')
                        os.killpg(group, signal.SIGTERM)
                        record['native_child_process_groups_signalled'].append(group)
                if not record['native_child_process_groups_signalled']:
                    record['space_abort_before_native_child_observed'] = True
                    process.terminate()
            time.sleep(2)
        record['wrapper_exit_code'] = process.returncode
finally:
    reserve.unlink(missing_ok=True)
    record['minimum_observed_available_bytes'] = minimum
    record['available_after_run'] = available()
    record['wall_seconds'] = time.monotonic() - started
    record['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (guard / 'receipt.json').write_text(json.dumps(record, indent=2) + '\n')
    (guard / 'runner.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(record), flush=True)
raise SystemExit(record.get('wrapper_exit_code', 1))
