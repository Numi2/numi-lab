from pathlib import Path
import csv
import datetime
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
B = Path('/Users/home/cardboard-implementation-20261014')
C = B / 'solver/source169-default-off-parity-472'
BASE = B / 'solver/cgls-dt-half-cycle-468'
O = B / 'analysis/native472-default-off-production-parity-001'
PACKAGE = B / 'analysis/native472-source169-default-off-command-003'
HELPER = B / 'analysis/native469-terminal-capture-and-production-parity-001/analyze.py'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not O.exists()
    receipt = json.loads((C / 'receipt.json').read_text())
    guard = json.loads((PACKAGE / 'executions/receipt.json').read_text())
    prediction = json.loads((C / 'prediction.json').read_text())
    assert receipt['execution_started'] and receipt['source_inputs_unchanged']
    assert not receipt['changed_inputs'] and not receipt['termination_errors']
    assert not receipt['timed_out'] and not guard['space_abort']
    assert not guard['signalled_native_child_process_groups'] and not guard['errors']
    assert 'NM_FEM_BASE_CGLS_WARM_START_REFINE' not in prediction['environment']
    for name, expected in prediction['inputs_sha256'].items():
        assert sha(C / 'inputs' / name) == expected, name
    spec = importlib.util.spec_from_file_location('parity469', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    native, errors = helper.events(C / 'stderr.log')
    baseline, baseline_errors = helper.events(BASE / 'stderr.log')
    assert not errors and not baseline_errors
    observations = list(csv.DictReader((C / 'output/observations.csv').open()))
    base_observations = list(csv.DictReader((BASE / 'output/observations.csv').open()))
    differences = []
    for index in range(max(len(observations), len(base_observations))):
        a = observations[index] if index < len(observations) else None
        b = base_observations[index] if index < len(base_observations) else None
        if a != b:
            differences.append({'row': index, 'candidate': a, 'baseline': b})
    event_comparisons = {}
    for label in sorted(set(native) & set(baseline)):
        if label == 'cardboard_step_timing':
            continue
        a, b = native[label], baseline[label]
        indices = [i for i in range(max(len(a), len(b)))
                   if (a[i] if i < len(a) else None) != (b[i] if i < len(b) else None)]
        event_comparisons[label] = {'candidate_count': len(a), 'baseline_count': len(b),
            'identical': not indices, 'differing_count': len(indices), 'first_differing_indices': indices[:100]}
    frame_names = sorted({p.name for root in (C, BASE)
                          for p in (root / 'output').glob('accepted_step_*.obj')})
    frames = []
    for name in frame_names:
        a, b = C / 'output' / name, BASE / 'output' / name
        frames.append({'name': name, 'candidate_sha256': sha(a) if a.is_file() else None,
                       'baseline_sha256': sha(b) if b.is_file() else None})
    geometry_same = all(x['candidate_sha256'] == x['baseline_sha256'] for x in frames)
    traces_same = all(x['identical'] for x in event_comparisons.values())
    result = json.loads((C / 'output/result.json').read_text())
    baseline_result = json.loads((BASE / 'output/result.json').read_text())
    production_same = not differences and geometry_same and traces_same and result == baseline_result
    report = {'schema': 'numi.cardboard.native472-default-off-production-parity.v1',
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'case': str(C), 'baseline': str(BASE), 'receipt': receipt, 'guard': guard,
        'feature_off_complete_attempt_output_parity': production_same,
        'source169_warm_refinement_enabled': False,
        'source_inputs_verified': len(prediction['inputs_sha256']),
        'native_result': result, 'baseline_result': baseline_result,
        'production_comparison': {'observations_count': len(observations),
            'baseline_observations_count': len(base_observations),
            'observations_byte_identical': (C / 'output/observations.csv').read_bytes() == (BASE / 'output/observations.csv').read_bytes(),
            'row_differences': differences, 'accepted_geometry_identical': geometry_same,
            'frames': frames, 'common_non_timing_structured_events_identical': traces_same,
            'common_structured_event_comparisons': event_comparisons,
            'added_event_types': sorted(set(native) - set(baseline)),
            'missing_event_types': sorted(set(baseline) - set(native))},
        'qualification': {key: False for key in ('solver_correctness', 'numerical_convergence',
            'component_calibration', 'analogue_validation', 'physical_assembly_in_simulation', 'crush_impact_prediction')},
        'limitations': [
            'Compatibility compares the complete terminal attempt. A failed physical schedule cannot establish complete-cycle qualification.',
            'The historical468 initial effective environment was not completely sealed; output equality cannot repair that provenance.',
            'Warm-enabled behavior requires an independent native run, and source173 includes additional changes not exercised here.',
            'The original031 control19/Newton3 failure is distinct from the later control20/Newton63 capture.'],
        'case_file_sha256': {str(p.relative_to(C)): sha(p) for p in sorted(C.rglob('*')) if p.is_file()}}
    O.mkdir()
    (O / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (O / 'analyze.py').write_bytes(Path(__file__).read_bytes())
    (O / 'manifest.json').write_text(json.dumps({'schema': 'numi.cardboard.native472-parity-manifest.v1',
        'inputs': {str(p): sha(p) for p in [HELPER, PACKAGE / 'package-manifest.json',
            PACKAGE / 'executions/receipt.json', BASE / 'receipt.json', BASE / 'output/result.json',
            BASE / 'stderr.log', BASE / 'output/observations.csv']},
        'files': {p.name: sha(p) for p in O.iterdir() if p.is_file()}}, indent=2) + '\n')
    print(json.dumps({'report_sha256': sha(O / 'report.json'), 'manifest_sha256': sha(O / 'manifest.json'),
        'feature_off_complete_attempt_output_parity': production_same,
        'observation_differences': len(differences), 'geometry_same': geometry_same,
        'traces_same': traces_same, 'native_exit_code': receipt['exit_code']}))

if __name__ == '__main__':
    main()
