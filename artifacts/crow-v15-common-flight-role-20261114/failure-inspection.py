from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-common-flight-role-20261114');P=A/'straight-20261126-rebound-commonrole';meta=json.loads((P/'run.stdout').read_text())['first_episode_navigation'];states={};targets={};first_nonfoot={}
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step']
   if e not in (0,1):continue
   if 360<=t<=405 or abs(t-meta['steps_by_environment'][e])<=3:states[e,t]=x
   if t>=371 and e not in first_nonfoot:
    hits=[{'a':c.get('a_name'),'b':c.get('b_name'),'normal_impulse':c['impulses'][0]} for c in x['contacts'] if c['impulses'][0]>0 and not any(y in (c.get('a_name','')+' '+c.get('b_name','')) for y in ('foot','toe'))]
    if hits:first_nonfoot[e]={'first_observed_step':t,'contacts':hits}
  elif line.startswith('CROW_DRIVE_TARGET '):
   x=line.split();t,e=map(int,x[1:3])
   if e in (0,1) and (360<=t<=405 or abs(t-meta['steps_by_environment'][e])<=3):targets[e,t]=list(map(float,x[3:]))
rows=[]
for e in (0,1):
 end=meta['steps_by_environment'][e];samples=[]
 for t in sorted(set([370,375,378,380,400,end-1,end,end+1])):
  x=states.get((e,t))
  if not x:continue
  q,v=x['q'],x['v'];pos=[{'pair':[c.get('a_name'),c.get('b_name')],'normal_impulse':c['impulses'][0]} for c in x['contacts'] if c['impulses'][0]>0];samples.append({'step':t,'position_m':q[:3],'quaternion_xyzw':q[3:7],'tilt_rad':math.acos(max(-1,min(1,1-2*(q[3]**2+q[4]**2)))),'linear_velocity_m_s':v[:3],'angular_velocity_rad_s':v[3:6],'positive_contact_pairs':pos,'policy_action':x['policy_action'],'logged_joint_targets':targets.get((e,t))})
 rows.append({'lane':e,'metadata_first_episode_end':end,'termination_reason_code':meta['termination_reason_by_environment'][e],'first_observed_positive_nonfoot_contact':first_nonfoot.get(e),'samples':samples})
out={'rows':rows,'scope':'observation-only diagnostics; metadata, state and contact epochs are not independently synchronized and contact is not assumed to cause termination'};(A/'failure-inspection.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'summary':[{'lane':x['lane'],'end':x['metadata_first_episode_end'],'first_nonfoot':x['first_observed_positive_nonfoot_contact']} for x in rows]},indent=2))
