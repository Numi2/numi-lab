from pathlib import Path
import gzip,json,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-flight-toe-takeoff-audit-20261110';out={}
for side in ('j','i'):
 p=R/f'artifacts/crow-v15-student-landing-eval-20261110/{side}-student-landing.stderr.gz';bins={}
 with gzip.open(p,'rt') as f:
  for line in f:
   if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
   x=json.loads(line.split(' ',1)[1]);t=x['step']
   if not 650<=t<850:continue
   k=(t-650)//25;z=bins.setdefault(k,{'first_step':650+25*k,'max_absolute_toe_angle_rad':0.,'samples':0,'total_foot_contact_impulse_Ns':0.})
   z['max_absolute_toe_angle_rad']=max(z['max_absolute_toe_angle_rad'],*map(abs,x['q'][23:31]));z['samples']+=1;z['total_foot_contact_impulse_Ns']+=sum(c['impulses'][0] for c in x['contacts'] if any(('_foot' in n or '_toe' in n) and n.startswith('crow_') for n in (c['a_name'],c['b_name'])))
 assert len(bins)==8 and all(z['samples']==25 for z in bins.values())
 out[side]={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'windows':list(bins.values()),'limitations':'Recorded control endpoints; not every substep or biological toe kinematics. The first window includes leg push-off and contact.'}
(A/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
