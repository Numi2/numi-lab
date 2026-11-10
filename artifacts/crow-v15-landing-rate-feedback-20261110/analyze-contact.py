from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-landing-rate-feedback-20261110');out={}
for name in ['crow-v15-landing-rate-feedback-20261110']:
 base=Path('artifacts')/name;out[name]={}
 for c in ['j-gain03','j-gain06']:
  file='development-reserved-final-'+c;d=json.loads((base/(file+'.stdout')).read_text());e=d['first_episode_navigation'];lanes={i:{'samples':0,'any_perch_foot_impulse_steps':0,'front_rear_opposing_steps':0,'opposing_run':0,'longest_opposing_run':0,'bilateral_opposing_steps':0,'bilateral_run':0,'longest_bilateral_run':0,'nonfoot_perch_steps':0} for i in range(8)};last={};closed=set();nav={i:0. for i in range(8)}
  with gzip.open(base/(file+'.stderr.gz'),'rt') as f:
   for l in f:
    if l.startswith('CROW_TASK_STATE '):
     v=l.split();i=int(v[1]);t=int(v[2]);key=('state',i)
     if key in last and t<last[key]:closed.add(key)
     last[key]=t
     if key not in closed and t<=e['steps_by_environment'][i]:nav[i]=float(v[4])
    elif l.startswith('CROW_SUPPORT_SAMPLE '):
     x=json.loads(l.split(' ',1)[1]);i=x['lane'];t=x['step'];key=('support',i)
     if key in last and t<last[key]:closed.add(key)
     last[key]=t
     if key in closed or t>e['steps_by_environment'][i] or nav[i]<5:continue
     z=lanes[i];z['samples']+=1;op={s:{'front':0.,'rear':0.} for s in ['left','right']};impulse=other=0.
     for cc in x['contacts']:
      an,bn=cc['a_name'],cc['b_name']
      if 'crow_course_perch' not in (an,bn):continue
      body=an if bn=='crow_course_perch' else bn;sgn=-1 if body==an else 1;force=cc['impulses'][0];side=next((s for s in op if '_'+s+'_' in body),None)
      if side and ('_foot' in body or '_toe' in body):
       impulse+=force;nx=-sgn*cc['normal'][0]
       if 'front_toe' in body and nx<-.5:op[side]['front']+=force
       if 'rear_toe' in body and nx>.5:op[side]['rear']+=force
      else:other+=force
     single=any(op[s]['front']>.00005 and op[s]['rear']>.00005 for s in op);both=all(op[s][d]>.00005 for s in op for d in ('front','rear'))
     z['any_perch_foot_impulse_steps']+=impulse>1e-8;z['nonfoot_perch_steps']+=other>1e-8;z['front_rear_opposing_steps']+=single;z['bilateral_opposing_steps']+=both;z['opposing_run']=z['opposing_run']+1 if single else 0;z['longest_opposing_run']=max(z['longest_opposing_run'],z['opposing_run']);z['bilateral_run']=z['bilateral_run']+1 if both else 0;z['longest_bilateral_run']=max(z['longest_bilateral_run'],z['bilateral_run'])
  out[name][c]={'first_episode_steps':e['steps_by_environment'],'physical_failure':e['physical_failure_by_environment'],'lanes':lanes}
(A/'contact-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for n,v in out.items():
 print(n,[(c,sum(z['front_rear_opposing_steps'] for z in x['lanes'].values()),sum(z['bilateral_opposing_steps'] for z in x['lanes'].values()),max(z['longest_bilateral_run'] for z in x['lanes'].values())) for c,x in v.items()])
