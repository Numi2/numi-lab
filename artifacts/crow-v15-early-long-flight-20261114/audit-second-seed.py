from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-early-long-flight-20261114');B=Path('artifacts/crow-v15-early-phase-handoff-20261114');D=A/'push-20261127-rebound-long';meta=json.loads((D/'run.stdout').read_text())['first_episode_navigation'];state={};target={};contact={};toe={};errors=[]
with gzip.open(D/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   a=json.loads(line.split(' ',1)[1]);key=(a['lane'],a['step']);
   if key in state:errors.append('duplicate state '+str(key))
   state[key]=a;continue
  z=line.split()
  if not z:continue
  typ=z[0]
  if typ=='CROW_SUBSTEP_ERROR' or 'OBSERVER_ERROR' in typ:errors.append(line.strip());continue
  if typ=='CROW_DRIVE_TARGET':
   t,e=map(int,z[1:3]);key=(e,t)
   if key in target:errors.append('duplicate target '+str(key))
   target[key]=list(map(float,z[3:]));continue
  if typ in ('CROW_TWO_TOE_SUBSTEP','CROW_TOE_V_SUBSTEP'):
   t,k,e=map(int,z[1:4]);key=(e,t);table=contact if typ=='CROW_TWO_TOE_SUBSTEP' else toe;bucket=table.setdefault(key,{})
   if k in bucket or k not in range(16):errors.append('duplicate or invalid substep '+str((typ,e,t,k)))
   if typ=='CROW_TWO_TOE_SUBSTEP':
    if int(z[4])>int(z[5]):errors.append('contact overflow '+str((e,t,k)))
    bucket[k]=list(map(float,z[6:11]))
   else:bucket[k]=list(map(float,z[4:]))
rows=[]
for e in range(8):
 end=min(1800,meta['steps_by_environment'][e]);missing_state=[t for t in range(1,end+1) if (e,t) not in state];missing_target=[t for t in range(335,end+1) if (e,t) not in target];missing_sub=[t for t in range(335,end+1) if set(contact.get((e,t),{}))!=set(range(16)) or set(toe.get((e,t),{}))!=set(range(16))]
 def unloaded(t):return (e,t) in contact and set(contact[e,t])==set(range(16)) and all(sum(v[2:4])<=1e-7 for v in contact[e,t].values())
 release=next((t for t in range(371,end-1) if all(unloaded(j) for j in range(t,t+3))),None);recontact=next((t for t in range((release or end)+3,end+1) if not unloaded(t)),None)
 nonfoot=next(((t,k,v[4]) for t in range(release or 371,end+1) for k,v in sorted(contact.get((e,t),{}).items()) if abs(v[4])>1e-7),None);cap=next(((t,k,max(map(abs,v))) for t in range(release or 371,end+1) for k,v in sorted(toe.get((e,t),{}).items()) if max(map(abs,v))>=11.99),None)
 finite=all(all(math.isfinite(y) for y in v['q']+v['v']+v['policy_action']) for (ee,t),v in state.items() if ee==e and t<=end) and all(all(math.isfinite(y) for y in v) for (ee,t),v in target.items() if ee==e and t<=end) and all(all(math.isfinite(y) for vals in v.values() for y in vals) for tab in (contact,toe) for (ee,t),v in tab.items() if ee==e and t<=end)
 def yaw(q):x,y,z,w=q[3:7];return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
 def tilt(q):return math.acos(max(-1.,min(1.,1.-2*(q[3]*q[3]+q[4]*q[4]))))
 ref=state.get((e,450)); samples=[(t,state[e,t]) for t in range(450,end+1) if (e,t) in state];speed=[(t,state[e,t]['v'][0]) for t in range(550,end+1) if (e,t) in state]
 def yr(s):delta=yaw(s['q'])-yaw(ref['q']);return abs(math.atan2(math.sin(delta),math.cos(delta)))
 metrics={'height_min':min((s['q'][2] for _,s in samples),default=None),'height_max':max((s['q'][2] for _,s in samples),default=None),'tilt_max':max((tilt(s['q']) for _,s in samples),default=None),'yaw_deviation_max':max((yr(s) for _,s in samples),default=None) if ref else None,'lateral_deviation_max':max((abs(s['q'][1]-ref['q'][1]) for _,s in samples),default=None) if ref else None,'forward_speed_min':min((v for _,v in speed),default=None),'forward_speed_max':max((v for _,v in speed),default=None)}
 first_bound=next(({'step':t,'height':s['q'][2],'tilt':tilt(s['q']),'yaw_deviation':yr(s),'lateral_deviation':abs(s['q'][1]-ref['q'][1]),'world_x_speed':s['v'][0],'q':s['q'],'v':s['v'],'policy_action':s['policy_action']} for t,s in samples if s['q'][2]<.25 or s['q'][2]>1.30 or tilt(s['q'])>.35 or yr(s)>.35 or abs(s['q'][1]-ref['q'][1])>.50 or (t>=550 and not 0.<s['v'][0]<=1.0)),None) if ref else None
 checks={'complete_first_episode':end==1800 and not meta['physical_failure_by_environment'][e],'complete_observations':not missing_state and not missing_target and not missing_sub and not errors,'finite':finite,'three_interval_release':release is not None,'no_foot_recontact':release is not None and recontact is None,'no_nonfoot_impulse':nonfoot is None,'no_toe_speed_cap':cap is None,'height_tilt_heading_lateral_speed_bounds':len(samples)==1351 and len(speed)==1251 and first_bound is None}
 rows.append({'lane':e,'reported_end':end,'reason':meta['termination_reason_by_environment'][e],'checks':checks,'pass':all(checks.values()),'first_release':release,'first_recontact':recontact,'first_nonfoot_impulse':nonfoot,'first_toe_speed_cap':cap,'missing_state_count':len(missing_state),'missing_target_count':len(missing_target),'missing_substep_count':len(missing_sub),'metrics':metrics,'first_bound_violation':first_bound})
out={'rows':rows,'observer_errors':errors,'scope':'Full-start used-seed engineering screen with teacher off. Contact and metadata epochs not independently aligned; no inference of fatal contact or biological, maneuver, landing, or unseen-layout qualification.'};(A/'second-seed-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(rows,indent=2))
