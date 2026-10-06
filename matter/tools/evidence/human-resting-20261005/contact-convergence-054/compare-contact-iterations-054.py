from pathlib import Path
import csv, hashlib, json
import numpy as np

E = Path('/Users/n/numi-human-resting-evidence-20261005')
names = {16: 'joint-costal-conditioned-native-049',
         32: 'contact-iterations32-native-050', 64: 'contact-iterations64-native-052'}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
traces = {i: list(csv.DictReader((E/n/'resting-coupled.csv').open())) for i,n in names.items()}
base = traces[16]
contact_columns = {'min_contact_gap_m', 'peak_penetration_m', 'normal_impulse_ns'}
physiology_columns = sorted(set(base[0]) - contact_columns)
assert len(physiology_columns) == 44
report = {'scope': 'Controlled engineering contact-iteration diagnostic on one source023 build and unchanged full-chain003 assets. No body equilibrium, final anatomy, real-time or intervention qualification.',
          'analysis_sha256': sha(__file__), 'runs': {},
          'physiological_columns': physiology_columns,
          'changed_contact_columns': sorted(contact_columns),
          'caution': 'Changing iterations changes the finite-iteration body trajectory. CPU anatomy work ran concurrently on the same Mac mini, so wall timings are contention-affected. Late COM slope is descriptive; it is not a convergence certificate or a stationary-rest gate.'}
for iterations, name in names.items():
    root = E/name
    a = json.loads((root/'engineering-stability-analysis.json').read_text())
    inv = json.loads((root/'invocation.json').read_text())
    assert len(traces[iterations]) == len(base)
    differences = [k for k in base[0] if any(x[k] != y[k] for x,y in zip(base,traces[iterations]))]
    assert set(differences) <= contact_columns
    assert not a['input_hash_changes']
    slopes = np.array(a['center_of_mass_windows']['18-24']['linear_slope_m_per_s'])
    report['runs'][str(iterations)] = {
        'directory': str(root), 'run': a['run'],
        'invocation_sha256': sha(root/'invocation.json'),
        'execution_sha256': sha(root/'execution.json'),
        'stability_analysis_sha256': sha(root/'engineering-stability-analysis.json'),
        'coupled_trace_sha256': sha(root/'resting-coupled.csv'),
        'surface_trace_sha256': sha(root/'resting-surface-audit.csv'),
        'changed_csv_columns_vs_16': differences,
        'physiological_columns_bitwise_equal_to_16': True,
        'late_COM_slope_mm_per_s': (1000*slopes).tolist(),
        'late_COM_slope_norm_mm_per_s': float(1000*np.linalg.norm(slopes)),
        'root_assistance_observed': a['body_consistency']['root_assistance_observed'],
        'maximum_contact_penetration_m': a['body_consistency']['maximum_contact_penetration_m'],
        'maximum_blood_error_ml': a['maximum_abs_blood_volume_error_ml'],
        'input_hash_changes': a['input_hash_changes'],
        'source_file_sha256': inv['source_file_sha256']}
assert all(report['runs'][str(i)]['source_file_sha256'] == report['runs']['16']['source_file_sha256'] for i in names)
report['all_source_file_hashes_equal'] = True
p = E/'contact-iterations-comparison-054.json'
assert not p.exists()
p.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'report_sha256': sha(p), 'runs': {i: {k:r[k] for k in ('late_COM_slope_mm_per_s','late_COM_slope_norm_mm_per_s','changed_csv_columns_vs_16')} for i,r in report['runs'].items()}},indent=2))
