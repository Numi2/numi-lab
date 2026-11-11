from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-landing-forward-envelope-dev-20261111');P=A/'trial';R=Path('artifacts/crow-v15-teacher-clearance-route-dev-20261111/height');summ=json.loads((P/'stdout').read_text());names=json.load(gzip.open(P/'replay.json.gz','rt'))['payload']['body_names'];states={};ref={};contacts=collections.defaultdict(list);terminal=collections.defaultdict(list);clearance=collections.defaultdict(list);coverage=collections.Counter();errors=[]
def scan(p,S):
 with __import__('contextlib').ExitStack() as stack:
  f=__import__('itertools').chain.from_iterable(stack.enter_context(gzip.open(z,'rt')) for z in sorted(p.glob('stderr-*.gz'))) if p==P else stack.enter_context(gzip.open(p/'stderr.gz','rt'))
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);S[x['lane'],x['step']]=x
   elif p==P and l.startswith('CROW_TWO_TOE_SUBSTEP '):
    a=l.split();t,k,e=map(int,a[1:4]);coverage[e,t,k]+=1
   elif p==P and l.startswith('CROW_SCENE_CONTACT '):
    a=l.split();t,k,e=map(int,a[1:4]);bodya,bodyb=map(int,a[4:6]);contacts[e].append({'step':t,'substep':k,'bodies':[names[bodya] if bodya<len(names) else str(bodya),names[bodyb] if bodyb<len(names) else str(bodyb)],'normal_impulse':float(a[6]),'separation':float(a[12])})
   elif p==P and l.startswith('CROW_CONTACT_TRACE '):
    x=json.loads(l.split(' ',1)[1]);terminal[x['lane']].append(x)
   elif p==P and l.startswith('CROW_CLEARANCE_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);clearance[x['lane']].append(x)
   elif p==P and ('CROW_SUBSTEP_ERROR' in l or 'failed' in l.lower()):errors.append(l.strip())
scan(P,states);scan(R,ref);ep=summ['first_episode_navigation'];rows=[]
for e in range(8):
 end=ep['steps_by_environment'][e];actual=[x for x in contacts[e] if x['step']<=end+1];first=actual[0] if actual else None;traces=[x for x in terminal[e] if x['step']<=end+1];fatal=traces[0] if traces else None;before=states.get((e,first['step']-1)) if first else None;validclear=[x for x in clearance[e] if x['end_step']<=end];last=states.get((e,end));prefix=min(end,2990);diff=next((t for t in range(1,prefix+1) if any((states.get((e,t)) or {}).get(k)!=(ref.get((e,t)) or {}).get(k) for k in ('q','v','policy_action'))),None)
 rows.append({'lane':e,'first_episode_end_step':end,'first_episode_waypoints':ep['maximum_waypoints_by_environment'][e],'termination_reason':ep['termination_reason_by_environment'][e],'physical_failure':ep['physical_failure_by_environment'][e],'first_scene_contact':first,'preceding_position_m':before['q'][:3] if before else None,'preceding_orientation_xyzw':before['q'][3:7] if before else None,'preceding_linear_angular_velocity':before['v'][:6] if before else None,'preceding_policy_actions':before['policy_action'] if before else None,'first_contact_termination_trace':fatal,'first_difference_from_prior_collision_enabled_teacher_step':diff,'sampled_minimum_clearance':validclear[-1] if validclear else None,'missing_first_episode_substeps':sum(coverage[e,t,k]!=1 for t in range(1,end+1) for k in range(16)),'last_position':last['q'][:3] if last else None})
r={'teacher':summ['birdflow_journey_teacher'],'student_authority':summ['birdflow_journey_student_authority'],'host_requested_resets':summ['host_requested_resets'],'termination_counts':summ['termination_reason_counts'],'first_episode_navigation':ep,'lanes':rows,'errors':errors[-8:],'limitations':'Teacher-guided development layout only. First scene contact need not be fatal; termination trace reports the simulator contact condition. Contact trace steps are one greater than the first-episode counter at a terminating interval; include end+1 for first contact attribution. Clearance is sampled at accepted control-step endpoints and is not a continuous bound. Comparison to prior collision-enabled clearance route uses q, v and policy_action only. Perch contact is not by itself a grip or supported landing.'};(A/'analysis.json').write_text(json.dumps(r,indent=2)+'\n');print([(x['lane'],x['first_episode_waypoints'],x['termination_reason'],x['first_scene_contact']) for x in rows])
