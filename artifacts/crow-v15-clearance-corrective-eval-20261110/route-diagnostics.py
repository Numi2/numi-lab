from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-clearance-corrective-eval-20261110');out={}
for name in ('j-hard','i-hard','j-original','i-original'):
 path=A/(name+'.stderr.gz')
 if not path.exists() or not (A/(name+'-run.log')).exists() or 'exit 0' not in (A/(name+'-run.log')).read_text():continue
 first=None;clear=None
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_CONTACT_TRACE ') and first is None:
    x=json.loads(line.split(' ',1)[1]);first={'step':x['step'],'position_m':x['pre_q'][:3],'orientation_xyzw':x['pre_q'][3:7],'linear_velocity_m_s':x['pre_v'][:3],'angular_velocity_rad_s':x['pre_v'][3:6],'policy_action':x['policy_action'],'contacts':x['contacts']}
   elif line.startswith('CROW_CLEARANCE_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);
    if clear is None or x['end_step']>clear['end_step']:clear=x
 out[name]={'first_fatal_contact':first,'minimum_accepted_step_obstacle_clearance':clear}
(A/'route-diagnostics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
