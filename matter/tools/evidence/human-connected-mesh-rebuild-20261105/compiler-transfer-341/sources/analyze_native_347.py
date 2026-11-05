"""Compare native347 using retained, tested force/strain/field analysis owners."""
from pathlib import Path
import json,time,hashlib,resource
import compare_remeshed_native_310 as owner
from compare_matched_deformation_339 import compare_matched,gate_counts
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');N=W/'reference-screened-conditioned-native-347';O=W/'native-screened-347-comparison.json'
assert not O.exists(),'do not overwrite an analyzed outcome'
start=time.monotonic()
cache={};original=owner.capture
def cached(p):
 k=str(p.resolve())
 if k not in cache:cache[k]=original(p)
 return cache[k]
owner.capture=cached
ref=W/'matched-ccd-baseline-335/install-motion.stdout.log.gz';cand=N/'install-motion.stdout.log.gz'
r=owner.compare(ref,cand)
mapping=compare_matched(cached(ref),cached(cand))
(W/'native-screened-347-matched-field.json').write_text(json.dumps(mapping,indent=2,allow_nan=False)+'\n')
strain=gate_counts(r['regional_strain'])
force=[{'region':k,'channel':c,'relative_resultant_difference_over_reference_nodal_l1':v['relative_resultant_difference_over_reference_nodal_l1'],'difference_N':v['resultant_difference_N']}
       for k,d in r['regional_force_channels'].items() for c,v in d.items()
       if v['relative_resultant_difference_over_reference_nodal_l1'] is None or v['relative_resultant_difference_over_reference_nodal_l1']>.01]
gas=[{'channel':k,**v} for k,v in r['coupled_trace'].items() if v['relative_max_abs_change'] is None or v['relative_max_abs_change']>.01]
r['qualification_gates']={'regional_strain_metrics':strain,'integrated_force_channels':sum(len(v) for v in r['regional_force_channels'].values()),'integrated_force_failures':force,'coupled_field_failures':gas,'measured_gates_pass':strain['over_5percent']==0 and not force and not gas,'material_energy_admission':False,'canonical_pointwise_admission':False,'sustained_motion_admission':False}
a=json.loads((ref.parent/'install-motion.execution.json').read_text());b=json.loads((cand.parent/'install-motion.execution.json').read_text())
assert a['exit_code']==b['exit_code']==0 and a['resource_stop_reason'] is None and b['resource_stop_reason'] is None
duration=float(r['candidate']['header']['endpoint_time_s'])
r['short_run_timing']={'reference_launcher_wall_s':a['elapsed_s'],'candidate_launcher_wall_s':b['elapsed_s'],'reference_launcher_real_time_factor':duration/a['elapsed_s'],'candidate_launcher_real_time_factor':duration/b['elapsed_s'],'reference_body_real_time_factor':duration/float(r['reference']['body']['wall_s']),'candidate_body_real_time_factor':duration/float(r['candidate']['body']['wall_s']),'scope':'One instrumented initialization pair; includes no presentation and proves no sustained throughput gain.'}
r['analysis_execution']={'wall_s':time.monotonic()-start,'maximum_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'source_sha256':hashlib.file_digest(Path(__file__).open('rb'),'sha256').hexdigest(),'comparison_owner_sha256':hashlib.file_digest(Path(owner.__file__).open('rb'),'sha256').hexdigest(),'native_condition':'Reference335 endpoint was used in mesh screening; this is numerical replay qualification, not held-out physical validation.'}
O.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:r[k] for k in ['body_wall_reduction_fraction','body_throughput_ratio','qualification_gates','short_run_timing','analysis_execution']},indent=2),flush=True)
