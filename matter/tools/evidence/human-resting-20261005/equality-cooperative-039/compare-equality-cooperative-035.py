from pathlib import Path
import hashlib,json,re
E=Path('/Users/n/numi-human-resting-evidence-20261005');A=E/'rigid-digits-native-032';B=E/'rigid-digits-equality-native-035'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=['resting-coupled.csv','resting-surface-audit.csv','resting-human.mrvpack']+[f'accepted-geometry/step-{i}.mrvpack' for i in [639,2783,2999]]
equal={p:{'control_sha256':sha(A/p),'candidate_sha256':sha(B/p)} for p in paths}
for v in equal.values():v['identical']=v['control_sha256']==v['candidate_sha256']
stats={}
for root in [A,B]:
 log=(root/'native.log').read_text();ex=json.loads((root/'execution.json').read_text())
 assert ex['returncode']==0 and ex['changed_inputs']==[]
 m=re.search(r'resting_integrated_body=completed simulated_s=(\S+) wall_s=(\S+) real_time_factor=(\S+)',log)
 gpu=[float(x) for x in re.findall(r'^resting_native_profile .*?gpu_ms=([0-9.eE+-]+)',log,re.M)]
 terminal=json.loads(next(l.split('=',1)[1] for l in log.splitlines() if l.startswith('stand_terminal_state=')))
 stats[root.name]={'simulated_s':float(m[1]),'wall_s':float(m[2]),'rtf':float(m[3]),'gpu_s':sum(gpu)/1000,'gpu_samples':len(gpu),'terminal':terminal,'rejection_pass':'resting_integrated_rejection=pass' in log}
report={'scope':'Same-source integrated native6s paired engineering comparison; independent CPU preparation during both runs. Not final anatomical or real-time acceptance.','analysis_sha256':sha(__file__),'equal':equal,'terminal_equal':stats[A.name]['terminal']==stats[B.name]['terminal'],'runs':stats}
report['gpu_time_reduction_percent']=100*(1-stats[B.name]['gpu_s']/stats[A.name]['gpu_s'])
(E/'equality-cooperative-comparison-035.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'all_files_identical':all(v['identical'] for v in equal.values()),'terminal_equal':report['terminal_equal'],'gpu_reduction_percent':report['gpu_time_reduction_percent'],'candidate_wall_s':stats[B.name]['wall_s'],'candidate_rtf':stats[B.name]['rtf']}))
assert all(v['identical'] for v in equal.values()) and report['terminal_equal']

