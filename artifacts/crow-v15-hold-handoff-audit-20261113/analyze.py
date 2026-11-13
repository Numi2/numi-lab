from pathlib import Path
import gzip,json,hashlib,subprocess,os,math
R=Path.cwd();A=R/'artifacts/crow-v15-hold-handoff-audit-20261113';B=R/'artifacts/crow-v15-consistent-wingheld-leg-control-20261113';paths={'baseline':B/'hold-wing-20261126-rebound-wingheld','extended':A/'hold-wing-20261126-rebound-extended'};D={}
for arm,p in paths.items():
 d={'state':{},'contact':{},'toe':{},'aero':{},'target':{},'meta':json.loads((p/'run.stdout').read_text())['first_episode_navigation'],'exit':json.loads((p/'run-result.json').read_text())['exit_code'],'errors':[]};D[arm]=d
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);key=x['lane'],x['step'];assert key not in d['state'];d['state'][key]=x;continue
   z=l.split()
   if not z:continue
   if z[0] in ('CROW_TWO_TOE_SUBSTEP','CROW_TOE_V_SUBSTEP','CROW_AERO_FORCE'):
    t,k,e=map(int,z[1:4]);key=e,t,k
    if z[0]=='CROW_TWO_TOE_SUBSTEP':q=d['contact'];v=list(map(float,z[6:11]));assert int(z[4])<=int(z[5])
    elif z[0]=='CROW_TOE_V_SUBSTEP':q=d['toe'];v=max(abs(float(x)) for x in z[4:])
    else:q=d['aero'];v=float(z[7])
    assert key not in q;q[key]=v
   elif z[0]=='CROW_DRIVE_TARGET':
    t,e=map(int,z[1:3]);assert (e,t) not in d['target'];d['target'][e,t]=list(map(float,z[3:]))
   elif z[0] in ('CROW_SUBSTEP_ERROR','CROW_AERO_OBSERVER_ERROR'):d['errors'].append(l.strip())
 if d['exit'] or any(x<400 for x in d['meta']['steps_by_environment']):d['errors'].append('incomplete first episodes')
 for e in range(8):
  for kind,expected in [('state',400),('contact',66*16),('toe',66*16),('aero',53*16),('target',66)]:
   n=sum(1 for k in d[kind] if k[0]==e and ((1<=k[1]<=400) if kind=='state' else ((348<=k[1]<=400) if kind=='aero' else (335<=k[1]<=400))))
   if n!=expected:d['errors'].append([e,kind,n,expected])
parity=[]
for e in range(8):
 x,y=D['baseline'],D['extended'];first=next((t for t in range(335,401) if x['target'].get((e,t))!=y['target'].get((e,t))),None);assert first is not None
 sm=[t for t in range(1,first) if any(x['state'][e,t][k]!=y['state'][e,t][k] for k in ('q','v','policy_action','contacts'))];cm=[(t,k) for t in range(335,first) for k in range(16) if x['contact'][e,t,k]!=y['contact'][e,t,k]]
 parity.append({'lane':e,'first_target_difference':first,'prior_state_mismatches':sm,'prior_contact_mismatches':cm})
inputs=[];ids={}
for arm,d in D.items():
 for e in range(8):
  for t in range(350,401):
   x=d['state'][e,t];ids[len(inputs)]=(arm,e,t);inputs.append(str(len(inputs))+' 0 '+' '.join(map(str,x['q']+x['v'])))
exe=R/'build/crow-v15-com-momentum-audit-20261113/kinematics';lib=R/'build/crow-v15-presolve-contact-audit-20261113/libmetalrobo.dylib';run=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True,check=True,env={**os.environ,'DYLD_LIBRARY_PATH':str(lib.parent)});com={}
for line in run.stdout.splitlines():
 z=line.split();assert z[0]=='C';v=list(map(float,z[3:]));assert len(v)==7 and all(map(math.isfinite,v));com[ids[int(z[1])]]=v
assert len(com)==len(inputs);rows=[]
for arm,d in D.items():
 for e in range(2,8):
  support=lambda t:any(sum(d['contact'][e,t,k][2:4])>1e-7 for k in range(16));release=next((t for t in range(351,399) if not any(support(u) for u in range(t,t+3))),None)
  rows.append({'arm':arm,'lane':e,'first_unloaded_three_ticks':release,'com_vz_at_release_m_s':com[arm,e,release][6] if release else None,'peak_com_rise_m':max(com[arm,e,t][3]-com[arm,e,350][3] for t in range(351,401)),'foot_normal_impulse_371_380_ns':sum(sum(d['contact'][e,t,k][:2]) for t in range(371,381) for k in range(16)),'toe_speed_threshold_substeps_348_400':sum(d['toe'][e,t,k]>=11.99 for t in range(348,401) for k in range(16)),'vertical_aero_impulse_348_400_ns':sum(d['aero'][e,t,k] for t in range(348,401) for k in range(16))*.00125,'leg_targets_370_375':{str(t):d['target'][e,t][13:19] for t in range(370,376)}})
valid=not any(d['errors'] for d in D.values()) and not any(x['prior_state_mismatches'] or x['prior_contact_mismatches'] for x in parity)
out={'valid':valid,'parity':parity,'errors':{a:d['errors'] for a,d in D.items()},'rows':rows,'source_logs_sha256':{a:hashlib.sha256((p/'run.stderr.gz').read_bytes()).hexdigest() for a,p in paths.items()},'scope':'Matched development hold, not a launch attempt; toe and status handoff are unchanged, and causal effects may involve subsequent coupled dynamics.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('valid',valid,'parity',parity,'errors',out['errors']);
for r in rows:print(r)
if not valid:raise SystemExit('invalid comparison')
