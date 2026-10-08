from pathlib import Path
import json,csv,hashlib
E=Path('/Users/n/numi-human-resting-evidence-20261005')
A=E/'native-terminal-candidate-930';B=E/'native-terminal-production-audits-932'
OUT=E/'native-terminal-production-review-932'
def sha(p):
    with p.open('rb') as f:
        h=hashlib.sha256()
        while x:=f.read(8*1024*1024):h.update(x)
    return h.hexdigest()
m=json.loads((B/'run-metadata.json').read_text())
assert m['exit_code']==0 and m['loaded_metal_runtime']['verified'] and not m['source_files_changed_during_run']
assert m['environment']['NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT']=='0'
assert m['environment']['NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS']=='8'
captures={}
for step in [0,63,64]:
    ra=json.loads((A/f'accepted-geometry/step-{step}.receipt.json').read_text())
    rb=json.loads((B/f'accepted-geometry/step-{step}.receipt.json').read_text())
    for key in ['accepted_body_state_sha256','accepted_respiration_state_sha256','accepted_registered_body_poses','accepted_respiratory_motion','accepted_time_s','accepted_step','pack_file_sha256','captured_vertex_buffer_sha256']:
        assert ra[key]==rb[key],(step,key)
    assert sha(B/f'accepted-geometry/step-{step}.mrvpack')==rb['pack_file_sha256']
    captures[step]={'pack_sha256':rb['pack_file_sha256'],'receipt_sha256':sha(B/f'accepted-geometry/step-{step}.receipt.json'),'accepted_time_s':rb['accepted_time_s']}
a=list(csv.DictReader((A/'resting-coupled.csv').open()))
b=list(csv.DictReader((B/'resting-coupled.csv').open()))
assert len(a)==64 and len(b)==8
aggregated=['min_contact_gap_m','peak_penetration_m','pre_projection_contact_residual_m_s',
 'pre_projection_limit_residual_generalized_s','pre_projection_equality_residual_generalized_s',
 'post_projection_contact_residual_m_s','post_projection_limit_residual_generalized_s',
 'post_projection_equality_residual_generalized_s','equality_position_projection_max_generalized',
 'equality_velocity_projection_max_generalized_s']
for row in b:
    step=int(row['step']);window=a[step-8:step];endpoint=a[step-1]
    for key,value in row.items():
        expected=(min if key=='min_contact_gap_m' else max)(float(x[key]) for x in window) if key in aggregated else float(endpoint[key])
        assert float(value)==expected,(step,key,value,expected)
rows=list(csv.DictReader((B/'resting-com-momentum-diagnostic.csv').open()))
assert len(rows)==8 and [int(r['accepted_step']) for r in rows]==list(range(8,65,8))
qpath=B/'resting-com-q-integration.csv'
assert not qpath.exists() or len(list(csv.DictReader(qpath.open())))==0
report={'scope':'Terminal presentation under final-study diagnostics cadence; not anatomical clearance or endurance',
 'pass':True,'source_revision':'b091d7dcead509a325194563ed38261319118a88',
 'runtime_metadata_sha256':sha(B/'run-metadata.json'),'wall_seconds':m['wall_seconds'],
 'q_integration_audit':0,'com_segment_steps':8,'com_observations':8,
 'captures_byte_identical_to_q1_com1_run930':captures,
 'reference_trace_rows':64,'candidate_trace_rows':8,'identical_endpoint_columns':50,
 'exact_window_aggregated_columns':aggregated,'window_aggregation':'minimum for min_contact_gap_m; maximum for the other nine diagnostic columns',
 'physiology_sha256':sha(B/'resting-coupled.csv'),'movie_sha256':sha(B/'native-viewer.mov'),
 'external_root':str(B),'driver_sha256':sha(E/'native-terminal-production-audits-932.py'),
 'initial_comparison_correction':'An initial whole-CSV identity assertion failed because q0/COM8 emits eight observations and aggregates ten diagnostics over each eight-root window. Explicit endpoint and window-extrema comparison passes all 60 fields; no physical state changed.'}
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS',sha(OUT/'verification.json'))
