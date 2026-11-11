from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-teacher-height-long-dev-20261111');P=A/'height';R=Path('artifacts/crow-v15-teacher-height-dev-20261111/height');states={};ref={};sub={};subref={};forces=collections.defaultdict(lambda:[0.]*37);errors=[]
def scan(p,S,C):
 with gzip.open(p/'stderr.gz','rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);S[x['lane'],x['step']]=x
   elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
    a=l.split();t,k,e=map(int,a[1:4]);C[e,t,k]=a[4:]
    if p==P:
     for i,v in enumerate(map(float,a[6:])):forces[e,t][i]+=v
   elif p==P and ('CROW_SUBSTEP_ERROR' in l or 'failed' in l.lower()):errors.append(l.strip())
scan(P,states,sub);scan(R,ref,subref);summ=json.loads((P/'stdout').read_text());rows=[]
for e in range(8):
 first=None
 for t in range(356,3201):
  x=states.get((e,t));prev=states.get((e,t-1))
  if not x:continue
  hits=[c for c in x['contacts'] if c.get('impulses',[0])[0]>1e-7 and 'locomotion_ground' not in (c.get('a_name'),c.get('b_name'))]
  if hits:first={'step':t,'pairs':[[c.get('a_name'),c.get('b_name')] for c in hits],'position_m':x['q'][:3],'orientation_xyzw':x['q'][3:7],'velocity':x['v'][:6],'previous_action':prev['policy_action'] if prev else None};break
 term=[]
 for t in range(3101,3201):
  x=states.get((e,t));v=forces[e,t]
  if not x:term.append(False);continue
  q=x['q'];speed=math.sqrt(sum(z*z for z in x['v'][:3]));tilt=math.acos(max(-1.,min(1.,1.-2*(q[3]**2+q[4]**2))));leftopp=sum(v[21+i] for i in range(6))>=.0002 and sum(v[21+i] for i in (6,7))<=-.0002;rightopp=sum(v[29+i] for i in range(6))>=.0002 and sum(v[29+i] for i in (6,7))<=-.0002;term.append(min(v[0],v[1])>.0002 and v[0]+v[1]>=.5*.45004*9.81*.02 and leftopp and rightopp and abs(v[4])<=1e-7 and tilt<=.35 and speed<=.12)
 last=states.get((e,3200));rows.append({'lane':e,'waypoints':summ['maximum_navigation_waypoints_by_environment'][e],'prefix_different_states':sum(states.get((e,t))!=ref.get((e,t)) for t in range(1,1301)),'prefix_different_contact_substeps':sum(sub.get((e,t,k))!=subref.get((e,t,k)) for t in range(1,1301) for k in range(16)),'missing_substeps':sum((e,t,k) not in sub for t in range(1,3201) for k in range(16)),'first_endpoint_nonground_contact':first,'terminal_100_bilateral_opposing_grip':all(term) and len(term)==100,'last_position':last['q'][:3] if last else None,'last_velocity':last['v'][:6] if last else None})
r={'teacher':summ['birdflow_journey_teacher'],'student_authority':summ['birdflow_journey_student_authority'],'termination_count':summ['termination_count'],'failed_environment_steps':summ['failed_environment_steps'],'host_requested_resets':summ['host_requested_resets'],'first_episode_navigation':summ['first_episode_navigation'],'lanes':rows,'errors':errors[-5:],'limitations':'Teacher on, development training layout. Endpoint non-ground contacts do not constitute a full first-collider or continuous-clearance audit. Terminal grip test assumes the target remains the instrumented locomotion support and requires the opposing impulses recorded by the observer; this is not an autonomous landing or toe-release test.'}
(A/'analysis.json').write_text(json.dumps(r,indent=2)+'\n');print([(x['lane'],x['waypoints'],x['prefix_different_states'],x['missing_substeps'],x['terminal_100_bilateral_opposing_grip']) for x in rows])
