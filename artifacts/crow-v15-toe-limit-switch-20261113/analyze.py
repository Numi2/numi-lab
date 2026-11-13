from pathlib import Path
import gzip,json,math
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-toe-limit-switch-20261113';arms={'modified':A/'push-nowings-20261127-rebound-switch','baseline':R/'artifacts/crow-v15-stop-walk-toes-on-launch-20261113/push-nowings-20261127-rebound-no-walk-toes'}
def load(p):
 d={'state':{},'contact':{},'toe':{},'targets':{},'limits':{},'errors':[]};d['status']=json.loads((p/'run-result.json').read_text());
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step'];assert (e,t) not in d['state'];d['state'][e,t]=x
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);d['contact'][e,t,k]=list(map(float,z[6:11]));
    if int(z[4])>int(z[5]):d['errors'].append(['overflow',e,t,k])
   elif line.startswith('CROW_TOE_V_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);v=list(map(float,z[4:]));assert len(v)==16 and all(map(math.isfinite,v));d['toe'][e,t,k]=v
   elif line.startswith('CROW_DRIVE_TARGET '):
    z=line.split();t,e=map(int,z[1:3]);d['targets'][e,t]=list(map(float,z[3:]))
   elif line.startswith('CROW_LIMIT_SWITCH '):
    z=line.split();step=int(z[1]);v=list(map(float,z[2:]));assert len(v)==32 and step not in d['limits'];d['limits'][step]=v
   elif line.startswith('CROW_LIMIT_ERROR'):d['errors'].append(line.strip())
   elif line.startswith('CROW_SUBSTEP_ERROR'):d['errors'].append(line.strip())
 return d
D={a:load(p) for a,p in arms.items()};parity=[];rows=[]
for e in range(8):
 differences=[];missing=[]
 for t in range(1,346):
  a=D['modified']['state'].get((e,t));b=D['baseline']['state'].get((e,t))
  if a is None or b is None:missing.append(t)
  elif any(a[k]!=b[k] for k in ('q','v','policy_action','contacts')):differences.append(t)
 c=[]
 for t in range(335,346):
  for k in range(16):
   if D['modified']['contact'].get((e,t,k))!=D['baseline']['contact'].get((e,t,k)):c.append([t,k])
 parity.append({'lane':e,'endpoint_mismatches':len(differences),'first_endpoint':differences[:1],'missing':len(missing),'contact_mismatches':len(c),'first_contact':c[:1]})
 for label,d in D.items():
  enough=all((e,t) in d['state'] for t in range(350,401)) and all((e,t,k) in d['contact'] and (e,t,k) in d['toe'] for t in range(348,381) for k in range(16))
  if not enough:rows.append({'lane':e,'arm':label,'complete':False});continue
  impulse=sum(sum(d['contact'][e,t,k][:2]) for t in range(348,376) for k in range(16));early=sum(sum(d['contact'][e,t,k][:2]) for t in range(348,366) for k in range(16));normal=[any(sum(d['contact'][e,t,k][2:4])>1e-7 for k in range(16)) for t in range(351,381)];release=next((351+i for i in range(len(normal)-2) if not any(normal[i:i+3])),None);rise=max(d['state'][e,t]['q'][2]-d['state'][e,350]['q'][2] for t in range(351,381));speeds=[abs(x) for t in range(348,381) for k in range(16) for x in d['toe'][e,t,k]];rows.append({'lane':e,'arm':label,'complete':True,'foot_impulse_ns':impulse,'early_foot_impulse_ns':early,'root_rise_m':rise,'first_three_unloaded_intervals':release,'max_toe_speed_rad_s':max(speeds),'toe_events_1199':sum(x>=11.99 for x in speeds),'toe_events_2399':sum(x>=23.99 for x in speeds)})
valid=all(x['endpoint_mismatches']==0 and x['missing']==0 and x['contact_mismatches']==0 for x in parity) and all(not d['errors'] and d['status']['exit_code']==0 and all((e,t) in d['state'] for e in range(8) for t in range(1,401)) and len(d['targets'])==368 for d in D.values()) and all(x['complete'] for x in rows)
limits=D['modified']['limits'];limit_valid=set(limits)=={348,349,360,375} and all(abs(x-12.)<1e-6 for x in limits.get(348,[])[:16]) and all(abs(x-24.)<1e-6 for x in limits.get(348,[])[16:]) and all(len(limits[t])==32 and all(abs(x-24.)<1e-6 for x in limits[t]) for t in (349,360,375) if t in limits)
valid=valid and limit_valid
out={'gpu_limit_readbacks':limits,'limit_switch_valid':limit_valid,'parity':parity,'errors':{a:d['errors'] for a,d in D.items()},'state_counts':{a:len(d['state']) for a,d in D.items()},'target_counts':{a:len(d['targets']) for a,d in D.items()},'valid_paired_push_comparison':valid,'rows':rows,'scope':'Uncalibrated 24 rad/s process-local runtime override after world fingerprint compilation; fingerprint intentionally does not reflect it. Even with matched recorded prefix and identical weights/layout, this is not a biological body or takeoff qualification. Late window includes crouch transition.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('valid paired comparison',valid,'parity',parity)
for x in rows:
 if x['lane']>=2:print(x)
