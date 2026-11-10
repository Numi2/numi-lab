from pathlib import Path
import gzip,json,hashlib
A=Path('artifacts/crow-v15-head-contact-fix-20261110');old=Path('artifacts/crow-v15-latched-balance-20261110')
with gzip.open(old/'layout-j-positive.replay.json.gz','rt') as f:ref=json.load(f)['payload']['frames']
with gzip.open(A/'replay.json.gz','rt') as f:got=json.load(f)['payload']['frames']
first=next((i for i,(a,b) in enumerate(zip(ref,got)) if a['q']!=b['q'] or a['v']!=b['v']),None);episode=json.loads((A/'stdout').read_text())['first_episode_navigation'];event=episode['steps_by_environment'][0]+1;contact=None
with gzip.open(A/'stderr.gz','rt') as f:
 for line in f:
  if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
  d=json.loads(line.split(' ',1)[1]);
  if d['lane']!=0 or d['step']!=event:continue
  contact=[{'a':c['a_name'],'b':c['b_name'],'normal_impulse_Ns':c['impulses'][0]} for c in d['contacts'] if 'crow_lower_beak' in (c['a_name'],c['b_name']) and 'locomotion_ground' in (c['a_name'],c['b_name']) and c['impulses'][0]>1e-5]
assert episode['physical_failure_by_environment']==[True] and episode['termination_reason_by_environment']==[3] and event==2431 and first==2430 and contact
result={'first_qv_difference_frame_zero_based':first,'matching_prior_accepted_frames':first,'termination_step_one_based':event,'native_failure':True,'termination_reason':'contact','contact':contact,'scope':'one exposed seed on experimental toe body; no complete landing claim'};(A/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
