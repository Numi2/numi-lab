from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-disturbance-observer-20261113')
def read(name):
 d={'state':{},'force':{},'foot':collections.defaultdict(list),'toe':collections.defaultdict(list),'wing':collections.defaultdict(list),'errors':[]};P=A/name
 with gzip.open(P/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);k=x['lane'],x['step'];assert k not in d['state'];d['state'][k]=x;continue
   z=line.split()
   if not z:continue
   typ=z[0]
   if 'OBSERVER_ERROR' in typ or typ=='CROW_SUBSTEP_ERROR':d['errors'].append(line.strip());continue
   if typ in ('CROW_PRE_AERO_WRENCH','CROW_TWO_TOE_SUBSTEP','CROW_TOE_V_SUBSTEP','CROW_WING_EFFORT'):
    t,sub,e=map(int,z[1:4]);k=e,t
    if typ=='CROW_PRE_AERO_WRENCH':
     assert (e,t,sub) not in d['force'];d['force'][e,t,sub]=list(map(float,z[4:]));continue
    if typ=='CROW_TWO_TOE_SUBSTEP':
     if int(z[4])>int(z[5]):d['errors'].append('contact overflow '+str((e,t,sub)))
     d['foot'][k].append((sub,list(map(float,z[6:]))))
    elif typ=='CROW_TOE_V_SUBSTEP':d['toe'][k].append((sub,list(map(float,z[4:]))))
    else:d['wing'][k].append((sub,list(map(float,z[4:]))))
 d['stdout']=json.loads((P/'run.stdout').read_text());d['inputs']=json.loads((P/'inputs.json').read_text());return d
base=read('control-20261127-rebound-observer-control');gust=read('gust-20261127-rebound-observer-gust');orig=json.loads((A/'control-analysis.json').read_text());bm=base['stdout']['first_episode_navigation'];gm=gust['stdout']['first_episode_navigation'];parity=[]
for e in range(8):
 end=min(bm['steps_by_environment'][e],gm['steps_by_environment'][e]);stop=min(end,900) if e>=6 else end;bad=[t for t in range(1,stop+1) if (e,t) not in base['state'] or (e,t) not in gust['state'] or any(base['state'][e,t][k]!=gust['state'][e,t][k] for k in ('q','v','policy_action','contacts'))];parity.append({'lane':e,'through':stop,'mismatch_count':len(bad),'first_mismatches':bad[:5]})
def pitch(q):
 x,y,z,w=q[3:7];return math.asin(max(-1,min(1,2*(w*y-z*x))))
def angle_delta(a,b):return (a-b+math.pi)%(2*math.pi)-math.pi
def check_wrench(e):
 samples=[(t,k,gust['force'].get((e,t,k)),base['force'].get((e,t,k))) for t in range(895,916) for k in range(16)];missing=sum(a is None or b is None for _,_,a,b in samples);res=[]
 for t,k,a,b in samples:
  if a is not None and b is not None and len(a)==len(b)==6 and all(math.isfinite(x) for x in a+b):res.append((t,k,[x-y for x,y in zip(a,b)]))
 active=[(t,k,v) for t,k,v in res if max(map(abs,v))>1e-8];steps=sorted(set(t for t,_,_ in active));outside=[(t,k) for t,k,v in active if not 899<=t<=908];forces=max((max(map(abs,v[:3])) for _,_,v in res),default=None);maxtorque=max((math.sqrt(sum(x*x for x in v[3:])) for _,_,v in res),default=None);mean_y=sum(v[4] for _,_,v in active)/len(active) if active else 0
 good=missing==0 and len(res)==21*16 and len(active)==80 and len(steps)==5 and steps==list(range(steps[0],steps[0]+5)) and not outside and forces is not None and forces<1e-8 and maxtorque is not None and maxtorque<=.020001 and (mean_y>1e-5 if e==6 else mean_y< -1e-5)
 return {'pass':good,'missing':missing,'active_substeps':len(active),'active_steps':steps,'max_force_N':forces,'max_torque_Nm':maxtorque,'mean_torque_y_Nm':mean_y,'outside_window':outside[:10]}
other_forces=[gust['force'].get((e,t,k)) for e in range(6) for t in range(895,916) for k in range(16)]
other_wrenches_zero=all(v is not None and len(v)==6 and all(math.isfinite(x) and abs(x)<=1e-9 for x in v) for v in other_forces)
rows=[]
for e in (6,7):
 end=gm['steps_by_environment'][e];s=gust['state'];b=base['state'];foot=gust['foot'];first=next((t for t in range(371,end+1) if len(foot[e,t])==16 and sum(sum(v[2:4]) for _,v in foot[e,t])<=1e-7),None);continuous=first is not None and all(len(foot[e,t])==16 and sum(sum(v[2:4]) for _,v in foot[e,t])<=1e-7 and sum(sum(abs(x) for x in v[4:5]) for _,v in foot[e,t])<=1e-7 for t in range(first,end+1));coverage=all(sorted(i for i,_ in gust[k][e,t])==list(range(16)) for k in ('foot','toe','wing') for t in range(348,end+1));toe=[v for t in range(348,end+1) for _,v in gust['toe'][e,t]];wing=[v for t in range(348,end+1) for _,v in gust['wing'][e,t]];height=[s[e,t]['q'][2] for t in range(450,end+1) if (e,t) in s];speed=[s[e,t]['v'][0] for t in range(550,end+1) if (e,t) in s];wave=[(t,angle_delta(pitch(s[e,t]['q']),pitch(b[e,t]['q'])),s[e,t]['v'][4]-b[e,t]['v'][4]) for t in range(901,min(end,950)+1) if (e,t) in s and (e,t) in b];at=1100;dp=angle_delta(pitch(s[e,at]['q']),pitch(b[e,at]['q'])) if (e,at) in s and (e,at) in b else None;dv=s[e,at]['v'][4]-b[e,at]['v'][4] if (e,at) in s and (e,at) in b else None;w=check_wrench(e);peakp=max((abs(x[1]) for x in wave),default=None);peakv=max((abs(x[2]) for x in wave),default=None)
 checks={'control_observer_pass':orig['pass'],'expected_shader':gust['inputs']['metallib_sha256']==json.loads((A/'gust-hashes.json').read_text())['metallib'],'shared_inputs':all(gust['inputs'][k]==base['inputs'][k] for k in ('host_sha256','landing_policy_sha256','route_policy_sha256','frozen_flight_sha256')),'parity':all(x['mismatch_count']==0 for x in parity),'teacher_off':gust['stdout']['birdflow_journey_teacher'] is False,'no_failure':end==1800 and not gm['physical_failure_by_environment'][e],'continuous_flight':continuous,'complete_unique_substeps':coverage,'toe_speed_below_limit':bool(toe) and all(len(v)==16 and all(math.isfinite(x) and abs(x)<11.99 for x in v) for v in toe),'wing_effort_bounded':bool(wing) and all(len(v)==14 and all(map(math.isfinite,v)) and max(abs(v[0]),abs(v[7]))<=3.00001 for v in wing),'height_bounded':len(height)==1351 and min(height)>=.25 and max(height)<=1.5,'forward_speed_bounded':len(speed)==1251 and min(speed)>0 and max(speed)<=1.,'measured_external_disturbance':w['pass'],'other_lane_wrenches_zero':other_wrenches_zero,'measurable_response':len(wave)==50 and (peakp>=.04 or peakv>=.10),'recovery_at_1100':dp is not None and dv is not None and abs(dp)<.06 and abs(dv)<.15,'no_observer_errors':not gust['errors']}
 rows.append({'lane':e,'checks':checks,'pass':all(checks.values()),'wrench':w,'first_unloaded':first,'peak_pitch_delta_rad':peakp,'peak_world_y_angular_rate_delta_rad_s':peakv,'pitch_delta_at_1100_rad':dp,'world_y_angular_rate_delta_at_1100_rad_s':dv,'response_window':wave})
out={'rows':rows,'parity':parity,'pass_both':all(x['pass'] for x in rows),'errors':gust['errors'],'scope':'used-seed physical disturbance screen, not qualification of perturbed starts or navigation'};(A/'gust-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
