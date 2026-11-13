from pathlib import Path
import gzip,json,math,hashlib,subprocess,os
R=Path.cwd();A=R/'artifacts/crow-v15-pronation-forward-flight-20261113';D={}
for arm in ('push','hold'):
 p=A/(arm+'-wing-20261126-rebound-pronation');d={'state':{},'contact':{},'toe':{},'aero':{},'pair':{},'target':{},'errors':[],'meta':json.loads((p/'run.stdout').read_text())['first_episode_navigation'],'exit':json.loads((p/'run-result.json').read_text())['exit_code']};D[arm]=d
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);key=x['lane'],x['step'];assert key not in d['state'];d['state'][key]=x;continue
   z=line.split()
   if not z:continue
   typ=z[0]
   if typ in ('CROW_TWO_TOE_SUBSTEP','CROW_TOE_V_SUBSTEP','CROW_AERO_FORCE'):
    t,k,e=map(int,z[1:4]);key=e,t,k
    if typ=='CROW_TWO_TOE_SUBSTEP':target=d['contact'];val=list(map(float,z[6:11]));assert int(z[4])<=int(z[5])
    elif typ=='CROW_TOE_V_SUBSTEP':target=d['toe'];val=max(abs(float(x)) for x in z[4:])
    else:target=d['aero'];val=list(map(float,z[5:]));assert len(val)==12
    assert key not in target;target[key]=val
   elif typ in ('CROW_WING_TARGET_PAIR','CROW_DRIVE_TARGET'):
    t,e=map(int,z[1:3]);target=d['pair'] if typ=='CROW_WING_TARGET_PAIR' else d['target'];assert (e,t) not in target;target[e,t]=list(map(float,z[3:]))
   elif typ in ('CROW_SUBSTEP_ERROR','CROW_AERO_OBSERVER_ERROR'):d['errors'].append(line.strip())
 if d['exit']!=0:d['errors'].append(['exit',d['exit']])
 for e in range(8):
  end=min(500,d['meta']['steps_by_environment'][e]);
  if end<500:d['errors'].append(['early_first_episode_end',e,end])
  for key,start in [('state',1),('contact',335),('toe',335),('aero',348)]:
   expected=max(0,end-start+1)*(1 if key=='state' else 16);n=sum(1 for k in d[key] if k[0]==e and start<=k[1]<=end)
   if n!=expected:d['errors'].append(['coverage_first_episode',e,key,n,expected])
matched=[];targetcheck=[]
for e in range(8):
 x,y=D['push'],D['hold'];sm=sum((e,t) not in x['state'] or (e,t) not in y['state'] or any(x['state'][e,t][k]!=y['state'][e,t][k] for k in ('q','v','policy_action','contacts')) for t in range(1,346));cm=sum((e,t,k) not in x['contact'] or (e,t,k) not in y['contact'] or x['contact'][e,t,k]!=y['contact'][e,t,k] for t in range(335,346) for k in range(16));matched.append({'lane':e,'state_mismatches':sm,'contact_mismatches':cm})
 for arm in ('push','hold'):
  d=D[arm];end=min(500,d['meta']['steps_by_environment'][e]);missing=[t for t in range(335,end+1) if (e,t) not in d['pair']];bad=[]
  if not missing:
   for t in range(336,end+1):
    x,y=d['pair'][e,t],d['pair'][e,t-1]
    if x[1]!=y[0] or x[3]!=y[2]:bad.append(t)
  targetcheck.append({'lane':e,'arm':arm,'missing':missing,'mismatches':bad})
inputs=[];ids={}
for arm,d in D.items():
 for e in range(8):
  end=min(500,d['meta']['steps_by_environment'][e])
  for t in range(350,end+1):
   if (e,t) not in d['state']:continue
   x=d['state'][e,t];ids[len(inputs)]=(arm,e,t);inputs.append(str(len(inputs))+' 0 '+' '.join(map(str,x['q']+x['v'])))
exe=R/'build/crow-v15-com-momentum-audit-20261113/kinematics';lib=R/'build/crow-v15-presolve-contact-audit-20261113/libmetalrobo.dylib';run=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True,check=True,env={**os.environ,'DYLD_LIBRARY_PATH':str(lib.parent)});com={}
for line in run.stdout.splitlines():
 z=line.split();assert z[0]=='C';v=list(map(float,z[3:]));assert len(v)==7 and all(map(math.isfinite,v));com[ids[int(z[1])]]=v
assert len(com)==len(inputs);rows=[]
for arm,d in D.items():
 for e in range(8):
  end=min(500,d['meta']['steps_by_environment'][e]);support=lambda t:any(sum(d['contact'].get((e,t,k),[0]*5)[2:4])>1e-7 for k in range(16));release=next((t for t in range(351,max(351,end-1)) if all(not support(u) for u in range(t,t+3))),None);recontact=next((t for t in range(release+3,end+1) if support(t)),None) if release else None
  toe=sum(d['toe'].get((e,t,k),0)>=11.99 for t in range(348,end+1) for k in range(16));nonfoot=sum(d['contact'].get((e,t,k),[0]*5)[4] for t in range(348,end+1) for k in range(16));c0=com.get((arm,e,350));peak=max((com[arm,e,t][3]-c0[3] for t in range(351,end+1) if (arm,e,t) in com),default=None) if c0 else None
  state=d['state'];dx=state[e,500]['q'][0]-state[e,400]['q'][0] if end==500 else None;vx=sum(state[e,t]['v'][0] for t in range(401,501))/100 if end==500 else None
  tar=[d['target'][e,t][j] for t in range(348,end+1) if (e,t) in d['target'] for j in (10,11)];actual=[state[e,t]['q'][j+1] for t in range(348,end+1) if (e,t) in state for j in (10,11)]
  row={'arm':arm,'lane':e,'first_episode_end':end,'release':release,'recontact':recontact,'unloaded_through_end':release is not None and recontact is None,'peak_com_rise_m':peak,'root_dx_400_500_m':dx,'mean_root_vx_401_500_m_s':vx,'toe_threshold_substeps':toe,'nonfoot_normal_impulse_ns':nonfoot,'pronation_target_range_rad':[min(tar),max(tar)] if tar else None,'pronation_actual_range_rad':[min(actual),max(actual)] if actual else None}
  force=[d['aero'][e,t,k][:3] for t in range(401,end+1) for k in range(16) if (e,t,k) in d['aero']];row['mean_aero_force_xyz_401_to_end_n']=[sum(x[j] for x in force)/len(force) for j in range(3)] if force else None
  row['screen_pass']=end==500 and row['unloaded_through_end'] and nonfoot<=1e-7 and toe<=1 and peak is not None and peak>=.05 and dx is not None and dx>.15 and vx is not None and .15<=vx<=1.0
  rows.append(row)
valid=not any(d['errors'] for d in D.values()) and not any(x['state_mismatches'] or x['contact_mismatches'] for x in matched) and not any(x['missing'] or x['mismatches'] for x in targetcheck if x['lane']>=2)
out={'valid_batch':valid,'errors':{a:d['errors'] for a,d in D.items()},'matched_start_parity':matched,'wing_target_history':targetcheck,'rows':rows,'paired_push_screen_pass':{str(e)+'-'+str(e+1):all(next(x for x in rows if x['arm']=='push' and x['lane']==k)['screen_pass'] for k in (e,e+1)) for e in (2,4,6)},'source_sha256':{a:hashlib.sha256((A/(a+'-wing-20261126-rebound-pronation/run.stderr.gz')).read_bytes()).hexdigest() for a in D},'scope':'Development layout and seed, correlated settings and first episodes only. A batch with any early termination is not valid for promotion; descriptive uncensored lanes are not independent qualification.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('valid batch',valid,'errors',out['errors']);
for row in rows:print(row)
