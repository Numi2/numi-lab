from pathlib import Path
import csv
import datetime
import hashlib
import json

B = Path('/Users/home/cardboard-implementation-20261014')
C = B / 'solver/accepted-checkpoint-prefix-470'
F = B / 'solver/final-newton-fd-capture-469'
O = B / 'analysis/native470-checkpoint-write-failure-001'
O.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
receipt = json.loads((C / 'receipt.json').read_text())
guard_path = B / 'analysis/native470-accepted-checkpoint-prefix-command-002/executions/receipt.json'
guard = json.loads(guard_path.read_text())
assert receipt['exit_code'] == 1 and receipt['timed_out'] is False
assert receipt['source_inputs_unchanged'] is True and receipt['changed_inputs'] == []
assert guard['wrapper_exit_code'] == 3 and guard['space_abort'] is False
assert not guard['errors'] and not guard['signalled_native_child_process_groups']
error = (C / 'output/failure.txt').read_text()
assert 'shipping checkpoint is incomplete: cohesive interface history' in error
raw = (C / 'output/observations.csv').read_bytes()
baseline = (F / 'output/observations.csv').read_bytes().splitlines(keepends=True)
assert raw == b''.join(baseline[:41])
with (C / 'output/observations.csv').open() as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 40
assert {(int(x['step']), int(x['environment'])) for x in rows} == {(i, e) for i in range(20) for e in range(2)}
assert all(x['status_code'] == '0' and x['step_accepted'] == '1' and x['certificate_raw_accepted_flag'] == '1' for x in rows)
frames = sorted((C / 'output').glob('accepted_step_*.obj'))
assert len(frames) == 40
for frame in frames:
    assert sha(frame) == sha(F / 'output' / frame.name), frame.name
assert not list((C / 'output').glob('**/*.nmcb'))
assert not (C / 'output/coupon-checkpoint-prefix.json').exists()
report = {
    'schema': 'numi.cardboard.native470-checkpoint-write-failure.v1',
    'verdict': 'FAILED_CHECKPOINT_SERIALIZATION; accepted prefix comparison only',
    'failure': error.strip(), 'native_exit_code': 1, 'wrapper_exit_code': 3,
    'timed_out': False, 'capacity_aborted': False,
    'accepted_controls': 20, 'accepted_environment_rows': len(rows),
    'accepted_rows_byte_identical_to_469_prefix': True,
    'accepted_obj_frames_byte_identical_to_469_prefix': len(frames),
    'checkpoint_file_created': False, 'checkpoint_resume_metadata_created': False,
    'registered_future_capture_was_not_reached': True,
    'root_cause': 'Shipping validator expected logical zero cohesive records; Runtime::snapshot retains the exact typed physical allocation sentinel required by Runtime::restore. Isolated source171 corrects that validator contract; no native validation yet.',
    'qualification': {'complete_protocol': False, 'checkpoint_valid': False,
                      'restore_parity': False, 'solver_correctness': False,
                      'mesh_or_time_convergence': False, 'physical_validation': False},
    'case_file_sha256': {str(p.relative_to(C)): sha(p) for p in sorted(C.rglob('*')) if p.is_file()},
    'guard_receipt_sha256': sha(guard_path),
    'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
(O / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
(O / 'analyze.py').write_bytes(Path(__file__).read_bytes())
(O / 'manifest.json').write_text(json.dumps({p.name: sha(p) for p in O.iterdir() if p.is_file()}, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ['verdict', 'accepted_controls', 'accepted_rows_byte_identical_to_469_prefix', 'checkpoint_file_created']}))
