from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-earlier-foot-return-20261114');events={};state={};target={};contacts={};errors=[]
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  typ=z[0]
  if typ=='event':events[int(z[1]),int(z[2])]=z[3:]
  elif typ=='step':
   e,t=int(z[1]),int(z[2]);i=z.index('q');j=z.index('v');state[e,t]=list(map(float,z[i+1:j]))+list(map(float,z[j+1:]))
  elif typ=='toeTarget':target.setdefault((int(z[1]),int(z[2])),[]).append(z[3:])
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
trigger=next((t for t in range(450) if (1,t) in events and bool(int(events[1,t][8])) and ((0,t) not in events or not bool(int(events[0,t][8])))),None)
# event fields begin after lane/t: field 8 is finished; time t is zero based.
stop=trigger if trigger is not None else 450
state_bad=[t for t in range(stop) if state.get((0,t))!=state.get((1,t)) or (0,t) not in state or (1,t) not in state]
target_bad=[t for t in range(stop) if target.get((0,t))!=target.get((1,t)) or (0,t) not in target or (1,t) not in target]
contact_bad=[(t,k) for t in range(1,stop+1) for k in range(32) if (0,t,k) not in contacts or (1,t,k) not in contacts or contacts[0,t,k]!=contacts[1,t,k]]
out={'candidate_finished_first_zero_based':trigger,'state_prefix_mismatch_count':len(state_bad),'target_prefix_mismatch_count':len(target_bad),'contact_substep_prefix_mismatch_count':len(contact_bad),'first_state_mismatches':state_bad[:10],'first_target_mismatches':target_bad[:10],'first_contact_mismatches':contact_bad[:10],'observer_errors':errors,'prefix_pass':trigger is not None and not state_bad and not target_bad and not contact_bad and not errors,'scope':'same-run logged prefix only; contact/state epoch synchronization is not independently established'};(A/'prefix-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
