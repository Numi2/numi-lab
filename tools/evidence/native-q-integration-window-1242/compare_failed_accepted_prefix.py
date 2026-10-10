from pathlib import Path
import hashlib,json,importlib.util
BASE=Path('/Users/n/numi-human-retained-delivery-20261009/q-integration-audit-window-1240')
OWNER=BASE/'long-comparison-001/compare_closed_arms.py'
assert hashlib.sha256(OWNER.read_bytes()).hexdigest()=='0bb979a5cc8c6f39238e8146c49a4089ead338fbaa35089e9c80def4258a1a08'
spec=importlib.util.spec_from_file_location('compare_owner',OWNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
control=Path('/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218/native-baseline-310s-preparation/native-run')
window=BASE/'late-window-310s-001/window/native-run'
pins={str(OWNER):m.sha(OWNER)}
receipts={}
for name,path,exit_code in [('control',control,0),('window',window,1)]:
 p=m.pin(path.parent/'execution.json',pins);d=json.loads(p.read_text())
 assert d['returncode']==exit_code and not d['changed_inputs'];receipts[name]=d
log=m.pin(window/'native.log',pins).read_text()
assert 'myosim_articulated_visual=failed error="persistent Human borrowed tendon-load snapshot disagreed with publication"' in log
assert 'human_execution_stage=native_horizon_end' in log and 'stage_step=155000' in log
reference=m.pin(BASE/'root-comparison-001/control-vs-window.json',pins)
assert m.sha(reference)=='eac675c8c7fae9d5d342f1b7443b8bf3835784c0086d6ac86b1a2ce8129ae54a'
ref=json.loads(reference.read_text());allowed={k:list(v['differing_columns']) for k,v in ref['csv_comparisons'].items()}
reports={n:m.compare_csv(control/n,window/n,cols,152501,155000,pins) for n,cols in allowed.items()}
grids={}
for arm,path in [('control',control),('failed_window',window)]:
 for name,key in [('resting-coupled.csv','step'),('resting-com-momentum-diagnostic.csv','accepted_step')]:
  grids[arm+'/'+name]=m.grid(path/name,key,range(8,155001,8),pins)
 grids[arm+'/support']=m.grid(path/'resting-com-support-impulses.csv','accepted_step',range(8,155001,8),pins,True)
grids['q']=m.grid(window/'resting-com-q-integration.csv','accepted_step',range(152501,155001),pins)
grids['q_slip']=m.grid(window/'resting-com-q-support-slip.csv','accepted_step',range(152501,155001),pins,True)
fp=['stand_q_fingerprint_fnv64','stand_v_fingerprint_fnv64','root_translation_fingerprint_fnv64'];com=reports['resting-com-momentum-diagnostic.csv']
fp_equal=com['headers_equal'] and com['control_rows']==com['window_rows'] and com['fingerprint_columns_present'] and not(set(fp)&set(com['differing_columns']))
capt={}
for s in [0,155000]:
 p=window/('accepted-geometry/step-%d.mrvpack'%s);c=control/p.relative_to(window)
 capt[str(s)]={'failed_window_present':p.exists(),'control_sha256':m.sha(m.pin(c,pins))}
 if p.exists():capt[str(s)]['failed_window_sha256']=m.sha(m.pin(p,pins))
report={'scope':'Diagnostic comparison of retained accepted rows from a terminally failed run. This is not successful run completion, terminal publication validation, anatomy admission, or force closure.',
 'script_sha256':m.sha(Path(__file__)),'failed_run_exit_code':1,'failed_execution_sha256':m.sha(window.parent/'execution.json'),
 'csv_comparisons':reports,'grids':grids,'captures':capt,
 'q_v_compensated_root_fingerprints_equal_at_all_normal_rows':fp_equal,
 'all_non_submission_diagnostic_cells_equal':all(v['all_non_submission_diagnostic_cells_equal'] for v in reports.values()),
 'different_cells_outside_sampling_intervals_overlapping_window':sum(v['different_cells_outside_sampling_intervals_overlapping_window'] for v in reports.values()),
 'input_sha256':pins,'input_pins_unchanged':all(m.sha(Path(p))==s for p,s in pins.items()),
 'successful_run_completion':False}
dst=Path(__file__).with_name('failed-accepted-prefix-comparison.json')
with dst.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'output':str(dst),'sha256':m.sha(dst),'full_q_v_root_fingerprints_equal':fp_equal,
 'non_submission_fields_equal':report['all_non_submission_diagnostic_cells_equal'],'outside_window_differences':report['different_cells_outside_sampling_intervals_overlapping_window'],
 'grids':{k:v['complete'] for k,v in grids.items()},'differing_columns':{k:list(v['differing_columns']) for k,v in reports.items()},'captures':capt,'successful_run_completion':False}))
