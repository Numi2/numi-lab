from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-v1-route-diagnostic-20261111');summary=json.loads((A/'route.stdout').read_text());contact=None;clear=[]
with gzip.open(A/'route.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_CONTACT_TRACE ') and contact is None:contact=json.loads(line.split(' ',1)[1])
  if line.startswith('CROW_CLEARANCE_SAMPLE '):clear.append(json.loads(line.split(' ',1)[1]))
assert contact and summary['first_episode_navigation']['physical_failure_count']==1
out={'first_episode':summary['first_episode_navigation'],'first_fatal_contact':{'step':contact['step'],'position_m':contact['pre_q'][:3],'orientation_xyzw':contact['pre_q'][3:7],'linear_velocity_m_s':contact['pre_v'][:3],'angular_velocity_rad_s':contact['pre_v'][3:6],'policy_action':contact['policy_action'],'contacts':contact['contacts']},'clearance_samples':clear,'overall_termination_count':summary['termination_count'],'scope':'development J only; source actor V2 adapter, execution V1 adapter; diagnostic transfer; post-failure resets do not count as continuous navigation'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
