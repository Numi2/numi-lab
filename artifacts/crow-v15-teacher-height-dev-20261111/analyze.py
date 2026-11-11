from pathlib import Path
import gzip,json,collections
A=Path('artifacts/crow-v15-teacher-height-dev-20261111');out={}
for name in ('height',):
 P=A/name;summary=json.loads((P/'stdout').read_text());states={};sub=collections.Counter();ground=collections.defaultdict(float)
 with gzip.open(P/'stderr.gz','rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);states[x['lane'],x['step']]=x
   elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
    a=l.split();t,k,e=map(int,a[1:4]);sub[e,t,k]+=1;ground[e,t]+=float(a[8])+float(a[9])+float(a[10])
 rows=[]
 for e in range(8):
  first=None
  for t in range(356,1301):
   x=states.get((e,t));prev=states.get((e,t-1))
   if not x:continue
   hits=[c for c in x['contacts'] if c.get('impulses',[0])[0]>1e-7 and 'locomotion_ground' not in (c.get('a_name'),c.get('b_name'))]
   if hits:
    first={'step':t,'pairs':[[c.get('a_name'),c.get('b_name')] for c in hits],'position_m':x['q'][:3],'orientation_xyzw':x['q'][3:7],'linear_and_angular_velocity':x['v'][:6],'previous_policy_action':prev['policy_action'] if prev else None};break
  last=states.get((e,1300));rows.append({'lane':e,'waypoints':summary['maximum_navigation_waypoints_by_environment'][e],'missing_substeps':sum(sub[e,t,k]!=1 for t in range(1,1301) for k in range(16)),'first_ground_impulse_after_launch':next((t for t in range(356,1301) if ground[e,t]>1e-7),None),'first_endpoint_non_ground_contact':first,'last_position_m':last['q'][:3] if last else None,'last_velocity':last['v'][:6] if last else None})
 out[name]={'teacher':summary['birdflow_journey_teacher'],'student_authority':summary['birdflow_journey_student_authority'],'termination_count':summary['termination_count'],'failed_environment_steps':summary['failed_environment_steps'],'host_requested_resets':summary['host_requested_resets'],'lanes':rows}
out['limitations']='Matched development teacher-guided trial only. Non-ground contacts are sampled at control-step endpoints and are not a complete substep first-collider or clearance audit. No landing or autonomous qualification.'
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print([(name,[x['waypoints'] for x in out[name]['lanes']],sum(x['first_endpoint_non_ground_contact'] is not None for x in out[name]['lanes'])) for name in ('height',)])
