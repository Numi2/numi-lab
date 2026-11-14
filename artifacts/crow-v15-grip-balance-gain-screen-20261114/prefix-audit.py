from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-grip-balance-gain-screen-20261114');events={};state={};target={};contacts={};errors=[]
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  typ=z[0]
  if typ=='event':events[int(z[1]),int(z[2])]=z[3:]
  elif typ=='step':
   e,t=int(z[1]),int(z[2]);i=z.index('q');j=z.index('v');state[e,t]=list(map(float,z[i+1:j]))+list(map(float,z[j+1:]))
  elif typ in ('toeTarget','hipcommand','rightHipCommand','jointtarget','stancetarget'):target.setdefault((int(z[1]),int(z[2])),[]).append([typ]+z[3:])
with gzip.open(A/'substeps.log.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  if 'ERROR' in z[0]:errors.append(line.strip())
  if z[0]=='CROW_TWO_TOE_SUBSTEP':
   t,k,e=map(int,z[1:4]);key=e,t,k
   if key in contacts:errors.append('duplicate '+str(key))
   if int(z[4])>int(z[5]):errors.append('overflow '+str(key))
   contacts[key]=list(map(float,z[6:]))
rows=[]
for e in range(1,8):
 trigger=next((t for t in range(450) if (0,t) in target and (e,t) in target and target[0,t]!=target[e,t]),None)
 stop=trigger if trigger is not None else 450
 state_bad=[t for t in range(stop) if state.get((0,t))!=state.get((e,t)) or (0,t) not in state or (e,t) not in state]
 target_bad=[t for t in range(stop) if target.get((0,t))!=target.get((e,t)) or (0,t) not in target or (e,t) not in target]
 contact_bad=[(t,k) for t in range(1,stop+1) for k in range(32) if (0,t,k) not in contacts or (e,t,k) not in contacts or contacts[0,t,k]!=contacts[e,t,k]]
 rows.append({'lane':e,'first_logged_target_difference_zero_based':trigger,'state_prefix_mismatch_count':len(state_bad),'target_prefix_mismatch_count':len(target_bad),'contact_substep_prefix_mismatch_count':len(contact_bad),'first_state_mismatches':state_bad[:5],'first_contact_mismatches':contact_bad[:5],'prefix_pass':(trigger is not None if e>=2 else trigger is None) and not state_bad and not target_bad and not contact_bad and not errors})
out={'rows':rows,'observer_errors':errors,'scope':'same-run recorded prefix only; contact/state epoch synchronization is not independently established'};(A/'prefix-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
