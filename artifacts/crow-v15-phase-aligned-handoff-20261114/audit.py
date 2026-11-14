from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-phase-aligned-handoff-20261114');O=Path('artifacts/crow-v15-role6-wing-handoff-20261114');x=json.load(gzip.open(A/'push-window.json.gz','rt'));old=json.load(gzip.open(O/'push-window.json.gz','rt'));state={(v['lane'],v['step']):v for v in x['state']};prev={(v['lane'],v['step']):v for v in old['state']};target={tuple(v[:2]):v[2:] for v in x['target']};oldtarget={tuple(v[:2]):v[2:] for v in old['target']};contact={tuple(v[:3]):v[3:] for v in x['contact']};toes={tuple(v[:3]):v[3:] for v in x['toe']};meta=json.loads((A/'push-20261126-rebound-phase/run.stdout').read_text())['first_episode_navigation'];rows=[]
for e in range(8):
 mismatch=[t for t in range(1,501) if state.get((e,t))!=prev.get((e,t))]; target_mismatch=[t for t in range(335,501) if (e,t) not in oldtarget or (e,t) not in target or target[e,t]!=oldtarget[e,t]];end=min(800,meta['steps_by_environment'][e]);missing=[(t,k) for t in range(335,end+1) for k in range(16) if (e,t,k) not in contact or (e,t,k) not in toes];
 def unloaded(t):return all((e,t,k) in contact and sum(contact[e,t,k][2:4])<=1e-7 for k in range(16))
 release=next((t for t in range(371,end-1) if all(unloaded(j) for j in range(t,t+3))),None);re=next((t for t in range((release or end)+3,end+1) if not unloaded(t)),None);nonfoot=next(((t,k,contact[e,t,k][4]) for t in range(release or 371,end+1) for k in range(16) if (e,t,k) in contact and abs(contact[e,t,k][4])>1e-7),None);cap=next(((t,k,max(map(abs,toes[e,t,k]))) for t in range(release or 371,end+1) for k in range(16) if (e,t,k) in toes and max(map(abs,toes[e,t,k]))>=11.99),None)
 firstcontact=[]
 if re is not None:
  for t in range(max(1,re-1),min(end,re+1)+1):
   s=state.get((e,t),{})
   for c in s.get('contacts',[]):
    if c.get('impulses',[0])[0]>1e-7:firstcontact.append({'step':t,'a':c.get('a_name'),'b':c.get('b_name'),'impulses':c['impulses']})
 def short(t):
  s=state.get((e,t));return None if not s else {'step':t,'q':s['q'],'v':s['v'],'policy_action':s['policy_action']}
 diffs={}
 for t in (499,500,501,502,503):
  if (e,t) in target:diffs[str(t)]={'wing_targets':target[e,t][6:12],'leg_targets':target[e,t][13:19]}
 finite=all(all(math.isfinite(y) for y in v['q']+v['v']+v['policy_action']) for (ee,t),v in state.items() if ee==e and t<=end)
 missing_states=[t for t in range(1,end+1) if (e,t) not in state]; missing_targets=[t for t in range(335,end+1) if (e,t) not in target]; finite_targets=all(all(math.isfinite(y) for y in a) for (ee,t),a in target.items() if ee==e and t<=end)
 passed=not mismatch and not target_mismatch and not missing and end==800 and not meta['physical_failure_by_environment'][e] and release is not None and re is None and nonfoot is None and cap is None and finite and finite_targets and not missing_states and not missing_targets and not x['errors']
 rows.append({'lane':e,'end':end,'physical_failure':meta['physical_failure_by_environment'][e],'prefix_mismatch_count':len(mismatch),'first_prefix_mismatch':next(iter(mismatch),None),'target_prefix_mismatch_count':len(target_mismatch),'first_target_prefix_mismatch':next(iter(target_mismatch),None),'missing_substeps':len(missing),'first_three_unloaded':release,'first_recontact':re,'first_nonfoot_impulse':nonfoot,'first_toe_speed_threshold':cap,'contact_observations_near_recontact':firstcontact,'preceding_recorded_state':short(re-1) if re else None,'recorded_state_at_contact':short(re) if re else None,'state_before_reported_end':short(end-1),'state_at_reported_end':short(end),'handoff_targets':diffs,'first_post500_target_difference':next((t for t in range(501,min(end,800)+1) if (e,t) in target and (e,t) in oldtarget and target[e,t]!=oldtarget[e,t]),None),'blend_window': [{'step':t,'candidate_wing_targets':target[e,t][6:12],'previous_wing_targets':oldtarget[e,t][6:12],'state':short(t)} for t in range(501,min(end,530)+1) if (e,t) in target and (e,t) in oldtarget],'finite_state':finite,'finite_targets':finite_targets,'missing_states':len(missing_states),'missing_targets':len(missing_targets),'continuity_screen_pass':passed})
out={'rows':rows,'observer_errors':x['errors'],'scope':'Development seed, native first episode without reset; contact observations and metadata clocks are not independently aligned. First positive endpoint contact near recontact is not necessarily the fatal collider. No flight-envelope, landing or mission qualification.'};(A/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
