from pathlib import Path
import importlib.util,json,csv,hashlib
import numpy as np
root=Path('/Users/n/numi-human-resting-evidence-20261005/rigid-digits-stability-native-040')
helper=Path('/Users/n/numi-human-resting-liver-parser-source-012/matter/tools/resting_intervention_study.py')
spec=importlib.util.spec_from_file_location('study',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=m.read_trace(root/'resting-coupled.csv')
surfaces=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((root/'resting-surface-audit.csv').open())]
report={
 'qualification':'Completed contact-driven release engineering diagnostic. Whole-skin exact audits at 10,18,24 seconds have zero forbidden intersections. Known lung-rib and ventricular-wall deformation defects remain; this is not anatomical acceptance. Not a paired intervention study or clinical validation.',
 'analysis_script_sha256':sha(__file__),'existing_analysis_owner_sha256':sha(helper),
 'run':m.native_scene_summary((root/'native.log').read_text()),
 'initialization_excluded_s':10.,'continuous_post_initialization_s':14.,
 'body_consistency':m.native_body_trace_consistency(root/'resting-coupled.csv',12000,.002),
 'surface_consistency':m.native_surface_trace_consistency(root/'resting-surface-audit.csv',12000,.002),
 'retained_trace_rows':len(rows),'late_window':m.window_metrics(rows,10,24),
 'late_complete_breaths':m.complete_breath_metrics(rows,10,24),
 'late_reference_comparison':m.resting_reference_comparison(rows,10,24),
 'final_total_breaths':rows[-1]['breaths'],
 'final_total_complete_filling_ejection_cycles':rows[-1]['complete_filling_ejection_cycles'],
 'maximum_abs_blood_volume_error_ml':max(abs(r['blood_error_ml']) for r in rows),
 'maximum_abs_oxygen_balance_error_stpd_ml':max(abs(r['oxygen_balance_error_stpd_ml']) for r in rows),
 'maximum_abs_co2_balance_error_stpd_ml':max(abs(r['co2_balance_error_stpd_ml']) for r in rows),
 'finite_all_trace_values':all(np.isfinite(v) for r in rows for v in r.values()),
 'input_hash_changes':json.loads((root/'execution.json').read_text())['changed_inputs'],
 'artifact_sha256':{p.name:sha(p) for p in root.iterdir() if p.is_file() and p.name in ['native-viewer.mov','native.log','resting-coupled.csv','resting-surface-audit.csv','resting-human.mrvpack','invocation.json','execution.json','source.diff','movie-inspection.txt']},
 'center_of_mass_windows':{},
}
for start,end in [(0,10),(10,18),(18,24)]:
 rs=[r for r in surfaces if start<=r['time_s']<=end]
 p=np.asarray([[r['body_com_'+axis+'_m'] for axis in 'xyz'] for r in rs])
 t=np.asarray([r['time_s'] for r in rs])
 slope=np.sum((t-t.mean())[:,None]*(p-p.mean(axis=0)),axis=0)/np.sum((t-t.mean())**2)
 report['center_of_mass_windows'][str(start)+'-'+str(end)]={'samples':len(rs),'first_m':p[0].tolist(),'last_m':p[-1].tolist(),'range_m':np.ptp(p,axis=0).tolist(),'linear_slope_m_per_s':slope.tolist(),'qualification':'Descriptive bed-supported displacement; no automatic stationary-rest gate is inferred.'}
(root/'engineering-stability-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['run','final_total_breaths','final_total_complete_filling_ejection_cycles','maximum_abs_blood_volume_error_ml','maximum_abs_oxygen_balance_error_stpd_ml','maximum_abs_co2_balance_error_stpd_ml','center_of_mass_windows']},indent=2))
