from pathlib import Path
import datetime, hashlib, json, os, signal, subprocess, time

B = Path('/Users/home/cardboard-implementation-20261014')
command_path = B / 'native-459-command.json'
command = json.loads(command_path.read_text())
case = Path(command[command.index('--case') + 1])
guard = B / 'analysis/native459-launch-guard-001'

def available():
    s = os.statvfs(B)
    return s.f_bavail * s.f_frsize

assert not case.exists(), 'fresh case required'
assert available() >= 1536 * 1024**2, 'need at least 1.5 GiB available before this bounded run'
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
    'native_child_process_groups_signalled': [],
}
(guard / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
minimum = record['available_before_launch']
started = time.monotonic()
try:
    with (B / 'native459-runner.log').open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
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
