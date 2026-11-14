from pathlib import Path
import collections,gzip,json,math,sys
A=Path('artifacts/crow-v15-long-leg-wing-transition-20261114');S=Path('artifacts/crow-v15-intermediate-wing-amplitude-20261114');D=A/sys.argv[1];seed=int(sys.argv[2]);state={};targets=set();foot=collections.defaultdict(dict);toe=collections.defaultdict(dict);wing=collections.defaultdict(dict);ground=set();errors=[]
with gzip.open(D/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);k=x['lane'],x['step'];
   if k in state:errors.append('duplicate state '+str(k))
   state[k]=x;continue
  x=line.split()
  if not x:continue
  typ=x[0]
  if 'OBSERVER_ERROR' in typ or typ=='CROW_SUBSTEP_ERROR':errors.append(line.strip());continue
  if typ=='CROW_DRIVE_TARGET':
   t,e=map(int,x[1:3]);targets.add((e,t));continue
  if typ in ('CROW_TWO_TOE_SUBSTEP','CROW_TOE_V_SUBSTEP','CROW_WING_EFFORT','CROW_GROUND_CONTACT'):
   t,k,e=map(int,x[1:4]);key=(e,t)
   if e not in (6,7) or t<348:continue
   if typ=='CROW_GROUND_CONTACT':ground.add(key);continue
   d={'CROW_TWO_TOE_SUBSTEP':foot,'CROW_TOE_V_SUBSTEP':toe,'CROW_WING_EFFORT':wing}[typ]
   if k in d[key]:errors.append('duplicate substep '+str((typ,e,t,k)))
   if typ=='CROW_TWO_TOE_SUBSTEP':
    if int(x[4])>int(x[5]):errors.append('contact capacity '+str((e,t,k)))
    d[key][k]=list(map(float,x[6:]))
   else:d[key][k]=list(map(float,x[4:]))
outmeta=json.loads((D/'run.stdout').read_text());meta=outmeta['first_episode_navigation'];inp=json.loads((D/'inputs.json').read_text());expected=json.loads((S/'push-hashes.json').read_text())['metallib'];prefix=[]
if seed==20261126:
 ref=json.load(gzip.open(S/'push-window.json.gz','rt'));r={(x['lane'],x['step']):x for x in ref['state']}
 for e in range(8):
  bad=[t for t in range(1,501) if (e,t) not in state or (e,t) not in r or any(state[e,t][k]!=r[e,t][k] for k in ('q','v','policy_action','contacts'))];prefix.append({'lane':e,'mismatch_count':len(bad),'first':bad[:10]})
rows=[]
for e in (6,7):
 end=min(meta['steps_by_environment'][e],1800);coverage=all(sorted(foot[e,t])==list(range(16)) and sorted(toe[e,t])==list(range(16)) and sorted(wing[e,t])==list(range(16)) and (e,t) in targets for t in range(348,end+1));f=lambda t:sum(sum(v[2:4]) for v in foot[e,t].values());release=next((t for t in range(371,end+1) if len(foot[e,t])==16 and f(t)<=1e-7),None);recontact=next((t for t in range((release or end)+1,end+1) if f(t)>1e-7),None);nonfoot=sum(abs(v[4]) for t in range(release or end,end+1) for v in foot[e,t].values());alltoe=[v for t in range(348,end+1) for v in toe[e,t].values()];allwing=[v for t in range(348,end+1) for v in wing[e,t].values()];heights=[state[e,t]['q'][2] for t in range(450,end+1) if (e,t) in state];speed=[state[e,t]['v'][0] for t in range(550,end+1) if (e,t) in state];displacement=state[e,end]['q'][0]-state[e,release]['q'][0] if release and (e,end) in state else None;finite=lambda v:all(math.isfinite(a) for a in v)
 checks={'shader_hash':inp['metallib_sha256']==expected,'teacher_off':outmeta['birdflow_journey_teacher'] is False,'first_episode':end==1800 and not meta['physical_failure_by_environment'][e],'states_complete':all((e,t) in state for t in range(1,1801)),'continuous_unloaded':release is not None and recontact is None,'no_nonfoot_or_ground':nonfoot<=1e-7 and release is not None and not any((e,t) in ground for t in range(release+1,end+1)),'coverage':coverage,'toe_speed':bool(alltoe) and all(len(v)==16 and finite(v) and max(map(abs,v))<11.99 for v in alltoe),'wing_effort':bool(allwing) and all(len(v)==14 and finite(v) and max(abs(v[0]),abs(v[7]))<=3.00001 for v in allwing),'forward':displacement is not None and displacement>=.1 and len(speed)==1251 and min(speed)>0 and max(speed)<=1.,'height':len(heights)==1351 and min(heights)>=.25 and max(heights)<=1.5,'no_observer_errors':not errors}
 if seed==20261126:checks['matched_500_prefix']=len(prefix)==8 and all(not p['mismatch_count'] for p in prefix)
 rows.append({'lane':e,'checks':checks,'pass':all(checks.values()),'release':release,'first_recontact':recontact,'nonfoot_abs_Ns':nonfoot,'forward_displacement_m':displacement,'forward_speed_range_m_s':[min(speed),max(speed)] if speed else None,'height_range_m':[min(heights),max(heights)] if heights else None})
out={'seed':seed,'prefix_comparison':('checked' if seed==20261126 else 'not performed: no same-seed short reference'), 'prefix':prefix,'rows':rows,'errors':errors,'scope':'used seed straight-flight development screen; candidate observer state/contact epochs not independently checked'};(D/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
