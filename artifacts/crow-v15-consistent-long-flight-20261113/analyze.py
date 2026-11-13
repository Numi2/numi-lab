from pathlib import Path
import gzip,json,math,hashlib,subprocess,os
R=Path.cwd();A=R/'artifacts/crow-v15-consistent-long-flight-20261113';prev=R/'artifacts/crow-v15-consistent-wing-drive-20261113';D={}
for arm in ('push','hold'):
 for ver in ('short','long'):
  p=(prev/(arm+'-wing-20261126-rebound-consistent-rebound')) if ver=='short' else (A/(arm+'-wing-20261126-rebound-long'))
  d={'state':{},'contact':{},'toe':{},'aero':{},'pair':{},'wing':{},'errors':[],'meta':json.loads((p/'run.stdout').read_text())['first_episode_navigation'],'exit':json.loads((p/'run-result.json').read_text())['exit_code'],'path':str(p)};D[arm,ver]=d
  with gzip.open(p/'run.stderr.gz','rt') as f:
   for line in f:
    if line.startswith('CROW_SUPPORT_SAMPLE '):
     x=json.loads(line.split(' ',1)[1]);key=x['lane'],x['step'];assert key not in d['state'];d['state'][key]=x;continue
    z=line.split()
    if not z:continue
    typ=z[0]
    if typ in ('CROW_TWO_TOE_SUBSTEP','CROW_TOE_V_SUBSTEP','CROW_AERO_FORCE','CROW_WING_EFFORT'):
     t,k,e=map(int,z[1:4]);key=e,t,k
     if typ=='CROW_TWO_TOE_SUBSTEP':target=d['contact'];val=list(map(float,z[6:11]));assert int(z[4])<=int(z[5])
     elif typ=='CROW_TOE_V_SUBSTEP':target=d['toe'];val=max(abs(float(x)) for x in z[4:])
     elif typ=='CROW_AERO_FORCE':target=d['aero'];val=list(map(float,z[5:]));assert len(val)==12
     else:target=d['wing'];val=list(map(float,z[4:]));assert len(val)==14
     assert key not in target;target[key]=val
    elif typ=='CROW_WING_TARGET_PAIR':
     t,e=map(int,z[1:3]);assert (e,t) not in d['pair'];d['pair'][e,t]=list(map(float,z[3:]));assert len(d['pair'][e,t])==4
    elif typ in ('CROW_SUBSTEP_ERROR','CROW_AERO_OBSERVER_ERROR'):d['errors'].append(line.strip())
  if d['exit']!=0:d['errors'].append(['exit',d['exit']])
  if ver=='long':
   for e in range(8):
    if d['meta']['steps_by_environment'][e]<500:d['errors'].append(['early_first_episode_end',e,d['meta']['steps_by_environment'][e]])
    for key,expected in [('state',500),('contact',166*16),('toe',166*16),('aero',153*16),('wing',153*16)]:
     n=sum(1 for k in d[key] if k[0]==e and ((1<=k[1]<=500) if key=='state' else ((335<=k[1]<=500) if key in ('contact','toe') else (348<=k[1]<=500))))
     if n!=expected:d['errors'].append(['coverage',e,key,n,expected])
parity=[];target=[]
for arm in ('push','hold'):
 a,b=D[arm,'short'],D[arm,'long']
 for e in range(8):
  sm=sum(any(a['state'].get((e,t),{}).get(k)!=b['state'].get((e,t),{}).get(k) for k in ('q','v','policy_action','contacts')) or (e,t) not in a['state'] or (e,t) not in b['state'] for t in range(1,401));cm=sum(a['contact'].get((e,t,k))!=b['contact'].get((e,t,k)) or (e,t,k) not in a['contact'] or (e,t,k) not in b['contact'] for t in range(335,401) for k in range(16));parity.append({'arm':arm,'lane':e,'state_mismatches':sm,'contact_mismatches':cm})
  if e>=2:
   missing=[t for t in range(335,501) if (e,t) not in b['pair']];bad=[]
   if not missing:
    for t in range(336,501):
     x,y=b['pair'][e,t],b['pair'][e,t-1]
     if x[1]!=y[0] or x[3]!=y[2]:bad.append(t)
   target.append({'arm':arm,'lane':e,'missing':missing,'mismatches':bad})
matched=[]
for e in range(8):
 a,b=D['push','long'],D['hold','long'];sm=sum(any(a['state'].get((e,t),{}).get(k)!=b['state'].get((e,t),{}).get(k) for k in ('q','v','policy_action','contacts')) or (e,t) not in a['state'] or (e,t) not in b['state'] for t in range(1,346));cm=sum(a['contact'].get((e,t,k))!=b['contact'].get((e,t,k)) or (e,t,k) not in a['contact'] or (e,t,k) not in b['contact'] for t in range(335,346) for k in range(16));matched.append({'lane':e,'state_mismatches':sm,'contact_mismatches':cm})
valid=not any(d['errors'] for (a,v),d in D.items() if v=='long') and not any(x['state_mismatches'] or x['contact_mismatches'] for x in parity+matched) and not any(x['missing'] or x['mismatches'] for x in target)
inputs=[];ids={}
for arm in ('push','hold'):
 d=D[arm,'long']
 for e in range(8):
  for t in range(350,min(500,d['meta']['steps_by_environment'][e])+1):
   if (e,t) not in d['state']:continue
   x=d['state'][e,t];ids[len(inputs)]=(arm,e,t);inputs.append(str(len(inputs))+' 0 '+' '.join(map(str,x['q']+x['v'])))
exe=R/'build/crow-v15-com-momentum-audit-20261113/kinematics';lib=R/'build/crow-v15-presolve-contact-audit-20261113/libmetalrobo.dylib';run=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True,check=True,env={**os.environ,'DYLD_LIBRARY_PATH':str(lib.parent)});com={}
for line in run.stdout.splitlines():
 z=line.split();assert z[0]=='C';v=list(map(float,z[3:]));assert len(v)==7 and all(math.isfinite(x) for x in v);com[ids[int(z[1])]]=v
assert len(com)==len(inputs)
rows=[]
for arm in ('push','hold'):
 d=D[arm,'long']
 for e in range(8):
  end=min(500,d['meta']['steps_by_environment'][e]);support=lambda t:any(sum(d['contact'].get((e,t,k),[0]*5)[2:4])>1e-7 for k in range(16));release=next((t for t in range(351,max(351,end-1)) if all(not support(u) for u in range(t,t+3))),None);recontact=next((t for t in range(release+3,end+1) if support(t)),None) if release else None
  toe=sum(d['toe'].get((e,t,k),0)>=11.99 for t in range(348,end+1) for k in range(16));nonfoot=sum(d['contact'].get((e,t,k),[0]*5)[4] for t in range(348,end+1) for k in range(16));c0=com.get((arm,e,350));cend=com.get((arm,e,end));peak=max((com[arm,e,t][3]-c0[3] for t in range(351,end+1) if (arm,e,t) in com),default=None) if c0 else None
  q=d['state'].get((e,end),{}).get('q');v=d['state'].get((e,end),{}).get('v');angle=2*math.acos(min(1,abs(q[6]))) if q else None;force=[d['aero'][e,t,k][2] for t in range(401,end+1) for k in range(16) if (e,t,k) in d['aero']]
  rows.append({'arm':arm,'lane':e,'first_episode_end':end,'release':release,'first_recontact':recontact,'unloaded_through_end':release is not None and recontact is None,'peak_com_rise_m':peak,'final_com_z_m':cend[3] if cend else None,'final_com_vz_m_s':cend[6] if cend else None,'final_root_orientation_angle_rad':angle,'final_root_speed_m_s':math.sqrt(sum(x*x for x in v[:3])) if v else None,'toe_threshold_substeps':toe,'nonfoot_normal_impulse_ns':nonfoot,'late_mean_vertical_aero_n':sum(force)/len(force) if force else None})
out={'valid_comparison':valid,'errors':{str(k):v['errors'] for k,v in D.items()},'short_to_long_parity':parity,'matched_start_parity':matched,'target_history':target,'rows':rows,'scope':'Development seed and layout, correlated lanes; no qualification, first episode only; COM from model FK and force from the simulated aero observer.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('comparison valid',valid,'errors',out['errors']);
for x in rows[2:8]+rows[10:]:print(x)
if not valid:raise SystemExit('long comparison invalid; inspect analysis')
