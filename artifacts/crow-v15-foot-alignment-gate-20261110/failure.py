import gzip,json
from pathlib import Path
A=Path('artifacts/crow-v15-foot-alignment-gate-20261110');out={}
for label in ['gate','zero_bias']:
 pre=A/('development-reserved-final-j-'+label);d=json.loads(pre.with_suffix('.stdout').read_text())['first_episode_navigation'];bad={i:t for i,(t,x) in enumerate(zip(d['steps_by_environment'],d['physical_failure_by_environment'])) if x};out[label]={str(i):{'end':t,'reason':d['termination_reason_by_environment'][i],'terminal_contact_traces':[]} for i,t in bad.items()}
 with gzip.open(str(pre)+'.stderr.gz','rt') as f:
  for l in f:
   if not l.startswith('CROW_CONTACT_TRACE '):continue
   z=json.loads(l.split(' ',1)[1]);i=z['lane']
   if i in bad and bad[i]-3<=z['step']<=bad[i]+1:out[label][str(i)]['terminal_contact_traces'].append({'step':z['step'],'pre_q':z['pre_q'],'pre_v':z['pre_v'],'contacts':z['contacts']})
(A/'failure.json').write_text(json.dumps(out,indent=2)+'\n')
for c,rows in out.items():
 for i,z in rows.items():print(c,i,'end',z['end'],'contacts',[[[q['a_name'],q['b_name']] for q in x['contacts']] for x in z['terminal_contact_traces']])
