from pathlib import Path
import hashlib,json,re,csv,importlib.util
E=Path('/Users/n/numi-human-resting-evidence-20261005')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(name):
 r=E/name;log=(r/'native.log').read_text();ex=json.loads((r/'execution.json').read_text())
 assert ex['returncode']==0 and ex['changed_inputs']==[]
 m=re.search(r'resting_integrated_body=completed simulated_s=(\S+) wall_s=(\S+) real_time_factor=(\S+)',log)
 gpu=[float(x) for x in re.findall(r'^resting_native_profile .*?gpu_ms=([0-9.eE+-]+)',log,re.M)]
 terminal=json.loads(next(l.split('=',1)[1] for l in log.splitlines() if l.startswith('stand_terminal_state=')))
 return dict(simulated_s=float(m[1]),wall_s=float(m[2]),rtf=float(m[3]),gpu_s=sum(gpu)/1000,gpu_samples=len(gpu),terminal=terminal,rejection_pass='resting_integrated_rejection=pass' in log)
A='rigid-digits-equality-finish-native-037';B='rigid-digits-projected-native-043'
a,b=stats(A),stats(B)
report={'qualification':'Integrated native6s engineering comparison. Existing projected-contact solver now handles91 equality rows; its finite-sweep trajectory differs from the former interleaved fallback. Not a bitwise-equivalent optimization; not final anatomy,300s,physiology or real-time acceptance.','analysis_sha256':sha(__file__),'control':a,'candidate':b,'trace_sha256':{r:{f:sha(E/r/f) for f in ['resting-coupled.csv','resting-surface-audit.csv','invocation.json','execution.json','native.log']} for r in [A,B]},'gpu_time_reduction_percent':100*(1-b['gpu_s']/a['gpu_s']),'terminal_max_abs_delta':{k:max(abs(x-y) for x,y in zip(a['terminal'][k],b['terminal'][k])) for k in ['q','v']}}
report['complete_trace_identical']=sha(E/A/'resting-coupled.csv')==sha(E/B/'resting-coupled.csv')
ra=list(csv.DictReader((E/A/'resting-coupled.csv').open()));rb=list(csv.DictReader((E/B/'resting-coupled.csv').open()))
assert len(ra)==len(rb)
delta={k:max(abs(float(x[k])-float(y[k])) for x,y in zip(ra,rb)) for k in ra[0]}
report['nonzero_trace_column_max_abs_delta']={k:v for k,v in delta.items() if v!=0}
body_fields={'min_contact_gap_m','peak_penetration_m','normal_impulse_ns','root_assistance_n','root_assistance_nm'}
report['physiological_columns_identical']=all(v==0 for k,v in delta.items() if k not in body_fields)
digits=list(range(43,63))+list(range(81,101))
report['candidate_rigid_digits']={'max_q_error':max(abs(b['terminal']['q'][i]-b['terminal']['initial_q'][i]) for i in digits),'max_abs_v':max(abs(b['terminal']['v'][i-1]) for i in digits),'root_assistance':b['terminal']['root_assistance']}
helper=Path('/Users/n/numi-human-resting-equality-finish-source-020/matter/tools/resting_intervention_study.py')
spec=importlib.util.spec_from_file_location('study',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
report['body_consistency']=m.native_body_trace_consistency(E/B/'resting-coupled.csv',3000,.002)
report['surface_consistency']=m.native_surface_trace_consistency(E/B/'resting-surface-audit.csv',3000,.002)
(E/'projected-equality-comparison-043.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['control','candidate','trace_sha256']},indent=2))

