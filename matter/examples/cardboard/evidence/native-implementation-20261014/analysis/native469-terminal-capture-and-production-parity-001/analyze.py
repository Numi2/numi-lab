from pathlib import Path
from collections import Counter
import csv
import datetime
import hashlib
import json
import re

B = Path('/Users/home/cardboard-implementation-20261014')
C = B / 'solver/final-newton-fd-capture-469'
BASELINE = B / 'solver/cgls-dt-half-cycle-468'
O = B / 'analysis/native469-terminal-capture-and-production-parity-001'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_text())

def events(path):
    data = {}
    invalid = []
    pattern = re.compile(r'^([A-Za-z0-9_]+)=(\{.*\})\s*$')
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            match = pattern.match(line)
            if not match:
                continue
            try:
                event = json.loads(match[2])
            except Exception as error:
                invalid.append(dict(line=line_number, label=match[1], error=str(error)))
                continue
            data.setdefault(match[1], []).append(event)
    return data, invalid

def main():
    prediction, receipt, result = [read(C / name) for name in
                                   ('prediction.json', 'receipt.json', 'output/result.json')]
    guard = read(B / 'analysis/native469-launch-guard-001/receipt.json')
    assert receipt['execution_started'] and receipt['source_inputs_unchanged']
    assert not receipt['changed_inputs'] and not receipt['termination_errors']
    assert not receipt['timed_out'] and not guard['space_abort']
    assert guard['native_child_process_groups_signalled'] == []
    for name, expected in prediction['inputs_sha256'].items():
        assert sha(C / 'inputs' / name) == expected, name
    contract_path = B / 'analysis/native469-source157-final-newton-fd-command-002/initial-effective-environment.json'
    assert sha(contract_path) == guard['initial_effective_environment_sha256']
    contract = read(contract_path)
    assert contract['mode'] == 'diagnostic_only' and not contract['positive_qualification_allowed']
    native, errors = events(C / 'stderr.log')
    base, base_errors = events(BASELINE / 'stderr.log')
    assert not errors and not base_errors
    observations = list(csv.DictReader((C / 'output/observations.csv').open()))
    baseline_observations = list(csv.DictReader((BASELINE / 'output/observations.csv').open()))
    observation_differences = []
    for index in range(max(len(observations), len(baseline_observations))):
        a = observations[index] if index < len(observations) else None
        b = baseline_observations[index] if index < len(baseline_observations) else None
        if a != b:
            observation_differences.append(dict(row=index, replay=a, baseline=b))
    # All common structured production event types are compared. Timing is the
    # sole excluded common label; new diagnostic event types remain inventoried.
    event_parity = {}
    for label in sorted(set(native) & set(base)):
        if label == 'cardboard_step_timing':
            continue
        a, b = native[label], base[label]
        differences = [i for i in range(max(len(a), len(b)))
                       if (a[i] if i < len(a) else None) != (b[i] if i < len(b) else None)]
        event_parity[label] = dict(replay_count=len(a), baseline_count=len(b),
                                  identical=not differences, differing_indices=differences[:100],
                                  differing_count=len(differences))
    geometry = []
    for path in sorted((BASELINE / 'output').glob('accepted_step_*.obj')):
        other = C / 'output' / path.name
        geometry.append(dict(name=path.name, baseline_sha256=sha(path),
                             replay_sha256=sha(other) if other.is_file() else None))
    accepted_geometry_parity = all(r['baseline_sha256'] == r['replay_sha256'] for r in geometry)
    prefix_parity = not observation_differences and accepted_geometry_parity
    trace_parity = all(r['identical'] for r in event_parity.values())
    all_outputs = {str(p.relative_to(C)): dict(bytes=p.stat().st_size, sha256=sha(p))
                   for p in sorted(C.rglob('*')) if p.is_file() and 'inputs' not in p.relative_to(C).parts}
    arms = {}
    for environment in (0, 1):
        rows = [r for r in observations if int(r['environment']) == environment]
        accepted = [r for r in rows if r['step_accepted'] == '1']
        rejected = [r for r in rows if r['step_accepted'] != '1']
        arms[str(environment)] = dict(attempted=len(rows), accepted=len(accepted), rejected=len(rejected),
            last_accepted=accepted[-1] if accepted else None,
            rejected_observations=rejected,
            phases_seen=sorted(set(r['phase'] for r in rows)))
    report = dict(schema='numi.cardboard.native469-terminal-capture-and-production-parity.v1',
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        classification='Terminal diagnostic attempt; no complete-cycle or resolution qualification',
        case=str(C), baseline=str(BASELINE), receipt=receipt, guard=guard, result=result,
        bound_input_files_verified=len(prediction['inputs_sha256']),
        initial_effective_environment_contract_sha256=sha(contract_path), arms=arms,
        production_comparison=dict(observation_rows=len(observations), baseline_rows=len(baseline_observations),
            observation_differences=observation_differences,
            accepted_geometry=geometry, accepted_geometry_identical=accepted_geometry_parity,
            observations_and_accepted_geometry_identical=prefix_parity,
            common_structured_event_comparisons=event_parity,
            common_non_timing_structured_events_identical=trace_parity,
            positive_qualification_pair=False,
            limitation='Native468 did not seal the complete initial effective environment. This exact production-output comparison cannot repair that provenance or qualify an incomplete physical cycle.'),
        replay_event_counts={k:len(v) for k,v in sorted(native.items())},
        baseline_event_counts={k:len(v) for k,v in sorted(base.items())},
        added_event_types=sorted(set(native)-set(base)), missing_event_types=sorted(set(base)-set(native)),
        diagnostic_events={k:v for k,v in sorted(native.items()) if k not in base or 'operator_fd' in k},
        terminal_files=all_outputs,
        qualification={k:False for k in ('solver_correctness', 'numerical_convergence',
            'component_calibration', 'analogue_validation', 'physical_assembly_in_simulation', 'crush_impact_prediction')},
        limitations=['The selected control20/Newton63 terminal state differs from original031 control19/Newton3.',
            'Accepted loading prefixes do not establish withdrawal, release or relaxation behavior.',
            'Rejected geometry/history observations can be rolled back while the residual certificate describes the rejected candidate.',
            'Capture integrity and independent FP64 replay require separate reports; producing a capture alone is not a derivative result.'])
    O.mkdir(exist_ok=False)
    (O / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (O / 'analyze.py').write_bytes(Path(__file__).read_bytes())
    (O / 'manifest.json').write_text(json.dumps(dict(
        schema='numi.cardboard.native469-terminal-capture-files.v1',
        inputs={str(p):sha(p) for p in [C/'prediction.json',C/'receipt.json',C/'stderr.log',
                C/'output/result.json',C/'output/observations.csv',BASELINE/'receipt.json',
                BASELINE/'stderr.log',BASELINE/'output/observations.csv',contract_path]},
        files={p.name:sha(p) for p in sorted(O.iterdir()) if p.is_file()}),indent=2)+'\n')
    print(json.dumps(dict(report_sha256=sha(O/'report.json'),manifest_sha256=sha(O/'manifest.json'),
        accepted_geometry_parity=accepted_geometry_parity,observations_identical=not observation_differences,
        structured_trace_parity=trace_parity,arms={k:{q:v[q] for q in ('attempted','accepted','rejected')} for k,v in arms.items()})))

if __name__ == '__main__':
    main()
