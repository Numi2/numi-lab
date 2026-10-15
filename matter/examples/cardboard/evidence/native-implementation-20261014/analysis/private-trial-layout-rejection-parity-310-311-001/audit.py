from pathlib import Path
import argparse
import hashlib
import json

parser = argparse.ArgumentParser()
parser.add_argument('capture_on', type=Path)
parser.add_argument('capture_off', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
args.output.mkdir(exist_ok=False)
sha = lambda b: hashlib.sha256(b).hexdigest()
rows = []
left, right = [p / 'diagnostic-state/dense' for p in (args.capture_on, args.capture_off)]
names = sorted(p.name for p in left.iterdir() if p.is_file())
assert names == sorted(p.name for p in right.iterdir() if p.is_file())
for name in names:
    a, b = [(p / name).read_bytes() for p in (left, right)]
    row = dict(name=name, bytes_a=len(a), bytes_b=len(b), sha256_a=sha(a),
               sha256_b=sha(b), byte_identical=a == b)
    if not name.endswith('.log'):
        assert a == b, name
    rows.append(row)
def records(run):
    prefix = 'fem_residual_armijo_iteration='
    return [json.loads(line[len(prefix):]) for line in (run / 'stderr.log').read_text().splitlines()
            if line.startswith(prefix)]
a, b = [records(p) for p in (args.capture_on, args.capture_off)]
assert len(a) == len(b) == 32 and a == b
predictions = [json.loads((p / 'prediction.json').read_text()) for p in (args.capture_on, args.capture_off)]
receipts = [json.loads((p / 'receipt.json').read_text()) for p in (args.capture_on, args.capture_off)]
assert all(r['exit_code'] == 1 and r['source_inputs_unchanged'] and not r['timed_out'] for r in receipts)
report = dict(schema='numi.cardboard.private-trial-parity.v1',
              capture_on=str(args.capture_on), capture_off=str(args.capture_off),
              source_snapshots=[p.get('source_snapshot') for p in predictions],
              receipts=receipts, diagnostic_state_files=rows,
              armijo_records_identical=True, armijo_record_count=len(a),
              nonlog_production_files_byte_identical=True,
              instrumentation_notes=[line for line in (args.capture_on / 'stderr.log').read_text().splitlines()
                  if line.startswith('fem_coupled_trial_capture_')],
              limitations=['Both runs reject. This is failed-prefix instrument parity, not nonlinear convergence.',
                           'Fixture logs contain instrument messages and are not claimed byte-identical.'])
(args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
(args.output / 'audit.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(files=len(rows), armijo_records=len(a),
    nonlog_production_files_byte_identical=True), indent=2))
