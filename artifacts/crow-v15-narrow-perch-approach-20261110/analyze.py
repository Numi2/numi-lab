from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-narrow-perch-approach-20261110');out={}
for c in ['i','j']:
 name='development-reserved-final-'+c;d=json.loads((A/(name+'.stdout')).read_text());e=d['first_episode_navigation'];s={i:{'samples':0,'loaded_samples':0,'streak':0,'longest_streak':0,'peak_force_n':0.0,'phase2_samples':0,'last':None} for i in range(8)}
 with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
  for l in f:
   if not l.startswith('CROW_TASK_STATE '):continue
   v=l.split();i=int(v[1]);t=int(v[2]);
   if t>e['steps_by_environment'][i] or float(v[4])<5:continue
   a,b=float(v[5]),float(v[6]);phase=float(v[7]);ok=a>.1 and b>.1 and a+b>2.2;z=s[i];z['samples']+=1;z['loaded_samples']+=ok;z['streak']=z['streak']+1 if ok else 0;z['longest_streak']=max(z['longest_streak'],z['streak']);z['peak_force_n']=max(z['peak_force_n'],a+b);z['phase2_samples']+=phase>=1.5;z['last']={'step':t,'phase':phase,'foot_loads_n':[a,b]}
 out[c]={'first_episode':e,'failed_environment_steps':d['failed_environment_steps'],'lanes':s}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for c,x in out.items():print(c,'survived',sum(not k for k in x['first_episode']['physical_failure_by_environment']),'longest loading',max(x['lanes'][i]['longest_streak'] for i in range(8)))
