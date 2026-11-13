from pathlib import Path
import gzip,json,math,sys,collections
A=Path('artifacts/crow-v15-full-pronation-observer-20261113');P=A/sys.argv[1];refpath=Path(sys.argv[2]);
def read(p,extended):
 o={'state':{},'target':{},'pronation':collections.defaultdict(list),'foot':collections.defaultdict(list),'toe':collections.defaultdict(list),'wing':collections.defaultdict(list),'ground':set(),'errors':[]}
 with gzip.open(p,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);k=x['lane'],x['step'];assert k not in o['state'];o['state'][k]=x;continue
   z=line.split()
   if not z:continue
   typ=z[0]
   if 'OBSERVER_ERROR' in typ or typ=='CROW_SUBSTEP_ERROR':o['errors'].append(line.strip());continue
   if typ=='CROW_DRIVE_TARGET':
    t,e=map(int,z[1:3]);o['target'][e,t]=list(map(float,z[3:]));continue
   if not extended:continue
   if typ in ('CROW_PRONATION_EFFORT','CROW_WING_EFFORT','CROW_TWO_TOE_SUBSTEP','CROW_TOE_V_SUBSTEP','CROW_GROUND_CONTACT'):
    t,sub,e=map(int,z[1:4]);k=e,t
    if typ=='CROW_PRONATION_EFFORT':o['pronation'][k].append((sub,list(map(float,z[4:]))))
    elif typ=='CROW_WING_EFFORT':o['wing'][k].append((sub,list(map(float,z[4:]))))
    elif typ=='CROW_TWO_TOE_SUBSTEP':o['foot'][k].append((sub,list(map(float,z[6:]))))
    elif typ=='CROW_TOE_V_SUBSTEP':o['toe'][k].append((sub,list(map(float,z[4:]))))
    else:o['ground'].add(k)
 return o
new=read(P/'run.stderr.gz',True);ref=read(refpath,False);stdout=json.loads((P/'run.stdout').read_text());meta=stdout['first_episode_navigation'];inputs=json.loads((P/'inputs.json').read_text());rinputs=json.loads((A.parent/'crow-v15-mirrored-leg-wing-longflight-20261113/signed-turn-20261126-rebound-longflight/inputs.json').read_text())
def yaw(q):
 x,y,z,w=q[3:7];return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
def wrap(x):return (x+math.pi)%(2*math.pi)-math.pi
def finite(x):return all(math.isfinite(y) for y in x)
parity=[]
for e in range(8):
 stop=600 if e>=6 else min(meta['steps_by_environment'][e],1800);bad=[]
 for t in range(1,stop+1):
  a=new['state'].get((e,t));b=ref['state'].get((e,t))
  if a is None or b is None or any(a[k]!=b[k] for k in ('q','v','policy_action','contacts')):bad.append(t)
 parity.append({'lane':e,'through':stop,'mismatch_count':len(bad),'first_mismatches':bad[:10]})
rows=[]
for e in (6,7):
 end=min(meta['steps_by_environment'][e],1800);s=new['state'];r=ref['state'];yawdiff=[];ydiff=[]
 for t in range(600,min(end,1200)+1):
  if (e,t) in s and (e,t) in r:yawdiff.append((t,wrap(yaw(s[e,t]['q'])-yaw(r[e,t]['q']))));ydiff.append((t,s[e,t]['q'][1]-r[e,t]['q'][1]))
 release=next((t for t in range(371,end-1) if all(len(new['foot'][e,u])==16 and sum(sum(v[2:4]) for _,v in new['foot'][e,u])<=1e-7 for u in range(t,t+3))),None)
 foot_after=[sum(sum(v[2:4]) for _,v in new['foot'][e,t]) for t in range((release or end)+3,end+1)];nonfoot=sum(sum(abs(v[4]) for _,v in new['foot'][e,t]) for t in range(release or end,end+1));ground=next((t for t in range((release or end)+1,end+1) if (e,t) in new['ground']),None)
 pron=[v for t in range(600,min(end,1200)+1) for _,v in new['pronation'][e,t]];eff=[abs(v[j]) for v in pron if len(v)==14 for j in (0,7)];target=[v[j] for v in pron if len(v)==14 for j in (1,8)];pos=[v[j] for v in pron if len(v)==14 for j in (2,3,9,10)];speed=[abs(v[j]) for v in pron if len(v)==14 for j in (4,5,6,11,12,13)];track=[abs(v[j]-v[j+1]) for v in pron if len(v)==14 for j in (1,8)];
 wing=[v for t in range(348,end+1) for _,v in new['wing'][e,t]];toe=[v for t in range(348,end+1) for _,v in new['toe'][e,t]];heights=[s[e,t]['q'][2] for t in range(450,end+1) if (e,t) in s];forward=[s[e,t]['v'][0] for t in range(550,end+1) if (e,t) in s];delta=s[e,end]['q'][0]-s[e,release]['q'][0] if release and (e,end) in s else None
 signed=(min(yawdiff,key=lambda v:v[1]) if e==6 else max(yawdiff,key=lambda v:v[1])) if yawdiff else None;maxy=max((abs(x[1]) for x in ydiff),default=None)
 checks={'pre_onset_and_other_lane_parity':all(p['mismatch_count']==0 for p in parity),'shared_inputs':all(inputs[k]==rinputs[k] for k in ('landing_policy_sha256','route_policy_sha256')),'teacher_off':stdout['birdflow_journey_teacher'] is False,'first_episode':end==1800 and not meta['physical_failure_by_environment'][e], 'states_complete':all((e,t) in s for t in range(1,end+1)),'release':release is not None,'no_later_ground':release is not None and not any(x>1e-7 for x in foot_after) and nonfoot<=1e-7 and ground is None,'substep_coverage':all(len(new['foot'][e,t])==16 and len(new['toe'][e,t])==16 and len(new['wing'][e,t])==16 and (e,t) in new['target'] for t in range(348,end+1)),'no_toe_speed_cap':all(len(v)==16 and finite(v) and max(map(abs,v))<11.99 for v in toe),'wing_effort':bool(wing) and all(len(v)==14 and finite(v) and max(abs(v[0]),abs(v[7]))<=3.00001 for v in wing),'forward':delta is not None and delta>=.1 and len(forward)==1251 and min(forward)>0 and max(forward)<=1.,'height':len(heights)==1351 and min(heights)>=.25 and max(heights)<=1.5,'pronation_full_window_bounds':len(pron)==16*(min(end,1200)-600+1) and bool(eff) and max(eff)<=.1501 and max(speed)<=25. and max(map(abs,pos))<=.350001 and max(map(abs,target))<=.350001,'turn_effort_full_window':all(len(new['pronation'][e,t])==16 for t in range(600,min(end,1200)+1)),'signed_heading':signed is not None and (signed[1]<=-.1 if e==6 else signed[1]>=.1),'lateral_deviation':maxy is not None and maxy<=2,'no_observer_errors':not new['errors']}
 rows.append({'lane':e,'checks':checks,'passed':all(checks.values()),'release_tick':release,'first_later_ground':ground,'relative_yaw_extreme_tick_and_rad':signed,'max_abs_lateral_deviation_m':maxy,'pronation_observed_substeps':len(pron),'pronation_effort_max_Nm':max(eff,default=None),'pronation_effort_saturated_fraction':sum(x>=.1499 for x in eff)/len(eff) if eff else None,'pronation_abs_target_max_rad':max(map(abs,target),default=None),'pronation_abs_position_max_rad':max(map(abs,pos),default=None),'pronation_abs_speed_max_rad_s':max(speed,default=None),'pronation_abs_tracking_error_max_rad':max(track,default=None),'root_forward_displacement_m':delta,'height_range_m':[min(heights),max(heights)] if heights else None,'forward_speed_range_m_s':[min(forward),max(forward)] if forward else None})
out={'scope':'used seed 20261126, all-wrench-ablated leg-launch development identification, not qualification','parity':parity,'rows':rows,'observer_errors':new['errors'],'passed_both':all(x['passed'] for x in rows)};(P/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
