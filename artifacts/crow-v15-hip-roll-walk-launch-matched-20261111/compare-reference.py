from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-hip-roll-walk-launch-matched-20261111');P=Path('artifacts/crow-v15-walk-leg-launch-085-screen-20261111/trial-on-20261126');old={};new={};sub={}
for p in sorted(P.glob('stderr-*.gz')):
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);old[x['lane'],x['step']]=x
with gzip.open(A/'run.stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);new[x['lane'],x['step']]=x
  elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
   a=l.split();t,k,e=map(int,a[1:4]);sub[e,t,k]=list(map(float,a[6:]))
rows=[]
for e in range(8):
 common=sorted(t for lane,t in old.keys()&new.keys() if lane==e and t<=500);first={};maxima={'root_position_m':0.,'root_velocity_m_s':0.,'old_joint_q_rad':0.}
 for t in common:
  a=old[e,t];b=new[e,t];dp=math.dist(a['q'][:3],b['q'][:3]);dv=math.dist(a['v'][:3],b['v'][:3]);dq=max(abs(x-y) for x,y in zip(a['q'][7:39],b['q'][7:39]));maxima['root_position_m']=max(maxima['root_position_m'],dp);maxima['root_velocity_m_s']=max(maxima['root_velocity_m_s'],dv);maxima['old_joint_q_rad']=max(maxima['old_joint_q_rad'],dq)
  for n,v,lim in [('root_position',dp,.01),('root_velocity',dv,.1),('joint_q',dq,.1)]:
   if v>lim and n not in first:first[n]={'step':t,'difference':v,'reference_root_q':a['q'][:7],'candidate_root_q':b['q'][:7],'reference_root_v':a['v'][:6],'candidate_root_v':b['v'][:6],'candidate_hip_roll_q':b['q'][39:41]}
 rows.append({'lane':e,'compared_steps':len(common),'first_threshold_crossings':first,'maximum_differences':maxima,'candidate_step_500':new.get((e,500)),'reference_step_500':old.get((e,500)),'recorded_contact_substeps':sum((e,t,k) in sub for t in range(1,501) for k in range(16))})
out={'rows':rows,'limitations':'Shared seed and scheduled commands on different body topologies; metadata/observation changes were rebound diagnostically and exact policy-input equivalence is not established. Position thresholds describe divergence, not its physical cause. Teacher-guided development trial.'};(A/'comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([{'lane':r['lane'],'compared':r['compared_steps'],'first':{k:v['step'] for k,v in r['first_threshold_crossings'].items()},'candidate_z500':r['candidate_step_500']['q'][2] if r['candidate_step_500'] else None} for r in rows],indent=2))
