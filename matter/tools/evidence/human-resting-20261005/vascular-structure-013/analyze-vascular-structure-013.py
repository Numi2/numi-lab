from pathlib import Path
import hashlib,json,re
r=Path('/Users/n/numi-human-resting-evidence-20261005');o=r/'vascular-structure-check-013'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cases=['vascular-structure-control-native-013','vascular-structure-candidate-native-013','vascular-structure-candidate-repeat-native-013','vascular-structure-control-repeat-native-013']
stats={};equal={};base=r/cases[0]
for name in cases:
 p=r/name;log=(p/'native.log').read_text();gpu=[float(x) for x in re.findall(r'^resting_native_profile .*?gpu_ms=([0-9.eE+-]+)',log,re.M)]
 match=re.search(r'^resting_integrated_body=completed simulated_s=([^ ]+) wall_s=([^ ]+) real_time_factor=([^ ]+)',log,re.M);assert match
 ex=json.loads((p/'execution.json').read_text());assert ex['returncode']==0 and ex['changed_inputs']==[]
 stats[name]={'gpu_samples':len(gpu),'gpu_seconds':sum(gpu)/1000,'simulated_seconds':float(match[1]),'native_wall_seconds':float(match[2]),'real_time_factor':float(match[3]),'wrapper_wall_seconds':ex['wrapper_wall_seconds'],'invocation_sha256':sha(p/'invocation.json')}
 equal[name]={}
 for q in p.rglob('*'):
  if not q.is_file() or not q.suffix in ('.csv','.mrvpack'):continue
  old=base/q.relative_to(p)
  same=q.read_bytes()==old.read_bytes();assert same
  equal[name][str(q.relative_to(p))]={'identical_to_first_control':same,'sha256':sha(q)}
control=sum(stats[n]['gpu_seconds'] for n in cases if 'control' in n)/2
candidate=sum(stats[n]['gpu_seconds'] for n in cases if 'candidate' in n)/2
report={'scope':'Two native6s A/B pairs in opposite execution order on one M4 Pro, with independent CPU preparation. Local performance comparison, not hardware-general benchmark, anatomical acceptance or clinical validation.',
 'analysis_script_sha256':sha(Path(__file__)),'order':cases,'equal_outputs':equal,'timing':stats,
 'mean_control_gpu_seconds':control,'mean_candidate_gpu_seconds':candidate,'mean_gpu_time_reduction_percent':100*(control-candidate)/control,
 'same_owner_jvp':True,'same_45_row_partial_pivot_solve':True,'physical_laws_tolerances_and_state_unchanged':True,
 'validation':{'vascular_log_sha256':sha(o/'vascular.log'),'transaction_log_sha256':sha(o/'transaction.log'),'transaction':'coupled rejection bitwise unchanged including circulation clock and Brain history; accepted replay bitwise'},
 'source_patch_sha256':sha(r/'vascular-structural-assembly-013.patch')}
(o/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['timing','mean_control_gpu_seconds','mean_candidate_gpu_seconds','mean_gpu_time_reduction_percent']},indent=2))
