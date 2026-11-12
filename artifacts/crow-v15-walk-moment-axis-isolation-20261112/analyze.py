from pathlib import Path
import collections,gzip,json
A=Path('artifacts/crow-v15-walk-moment-axis-isolation-20261112')
paths={'no_direct_moment':Path('artifacts/crow-v15-uniform-gait-launch-handoff-20261112/candidate-20261127-rebound-paired4/run.stderr.gz'),'direct_moment_diagnostic':A/'candidate-20261127-rebound-axis/run.stderr.gz'}
def read(path):
 state={};contacts={}
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step']
    if t<=300:state[e,t]=x
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);
    if t<=300:contacts[e,t,k]=z[4:]
 return state,contacts
(s,c),(S,C)=map(read,paths.values());rows=[]
for e in range(8):
 assert all((e,t) in s and (e,t) in S for t in range(1,301))
 assert all((e,t,k) in c and (e,t,k) in C for t in range(1,301) for k in range(16))
 def unload(d):return [(t,k) for t in range(201,301) for k in range(16) if float(d[e,t,k][4])+float(d[e,t,k][5])<=1e-7]
 rows.append({'lane':e,'matching_before_4s_state_samples':sum(s[e,t]==S[e,t] for t in range(1,201)),'matching_before_4s_contact_substeps':sum(c[e,t,k]==C[e,t,k] for t in range(1,201) for k in range(16)),'without_moment_unloaded':unload(c),'with_moment_unloaded':unload(C)})
out={'rows':rows,'scope':'Direct body moments enabled only as diagnostic control in 4–6 s. Policy action outputs alone do not establish applied moments; intervention, geometry and controller inputs are otherwise matched. No causal claim about physical gait quality or suitability.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for x in rows:print(x['lane'],'matching',x['matching_before_4s_state_samples'],x['matching_before_4s_contact_substeps'],'unloaded',len(x['without_moment_unloaded']),len(x['with_moment_unloaded']))
