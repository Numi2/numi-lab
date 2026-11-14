from pathlib import Path
import collections,gzip,json,math,sys
A=Path('artifacts/crow-v15-boosted-launch-paired-long-20261114');B=Path('artifacts/crow-v15-early-flight-boost-20261114');arm=sys.argv[1];assert arm in ('push','hold');P=A/(arm+'-20261126-rebound-longpaired');short=json.load(gzip.open(B/(arm+'-window.json.gz'),'rt'));ref={(s['lane'],s['step']):s for s in short['state']};meta=json.loads((P/'run.stdout').read_text());first=meta['first_episode_navigation'];states={};foot=collections.defaultdict(list);toe=collections.defaultdict(list);ground=collections.defaultdict(list);target={};errors=[];duplicates=[]
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);k=(x['lane'],x['step']);
   if k in states:duplicates.append(k)
   states[k]=x;continue
  x=line.split()
  if not x:continue
  typ=x[0]
  if 'OBSERVER_ERROR' in typ or typ=='CROW_SUBSTEP_ERROR':errors.append(line.strip());continue
  if typ=='CROW_DRIVE_TARGET':
   t,e=map(int,x[1:3]);target[e,t]=list(map(float,x[3:]));continue
  if typ not in ('CROW_TWO_TOE_SUBSTEP','CROW_TOE_V_SUBSTEP','CROW_GROUND_CONTACT'):continue
  t,k,e=map(int,x[1:4])
  if typ=='CROW_TWO_TOE_SUBSTEP':
   if int(x[4])>int(x[5]):errors.append('contact capacity exceeded '+str((e,t,k)))
   foot[e,t].append((k,list(map(float,x[6:]))))
  elif typ=='CROW_TOE_V_SUBSTEP':toe[e,t].append((k,list(map(float,x[4:]))))
  else:ground[e].append((t,k,*x[4:]))
inputs=json.loads((P/'inputs.json').read_text());short_inputs=json.loads((B/(arm+'-20261126-rebound-phase/inputs.json')).read_text());rows=[]
for e in range(8):
 end=min(first['steps_by_environment'][e],1800);bad=[t for t in range(1,min(500,end)+1) if (e,t) not in states or (e,t) not in ref or any(states[e,t][k]!=ref[e,t][k] for k in ('q','v','policy_action','contacts'))];row={'lane':e,'first_episode_steps':end,'physical_failure':first['physical_failure_by_environment'][e],'short_prefix_mismatch_count':len(bad),'short_prefix_first_mismatch':bad[:5]}
 if e in (6,7):
  loaded=lambda t:any(sum(v[2:4])>1e-7 for _,v in foot[e,t]);unload=next((t for t in range(371,end+1) if len(foot[e,t])==16 and not loaded(t)),None);later=next((t for t in range(unload+1,end+1) if loaded(t)),None) if unload else None
  missing_contact=[t for t in range(348,end+1) if sorted(k for k,_ in foot[e,t])!=list(range(16))];missing_toe=[t for t in range(348,end+1) if sorted(k for k,_ in toe[e,t])!=list(range(16))];cap=[(t,k) for t in range(371,end+1) for k,v in toe[e,t] if v and max(map(abs,v))>=11.99];nonfoot=[(t,k,v[4]) for t in range(371,end+1) for k,v in foot[e,t] if len(v)>4 and abs(v[4])>1e-7];g=next((x for x in ground[e] if unload and x[0]>unload),None);heights=[states[e,t]['q'][2] for t in range(450,end+1) if (e,t) in states];valid=sum((e,t) in states and all(math.isfinite(v) for v in states[e,t]['q']+states[e,t]['v']) for t in range(1,end+1));row.update({'first_unloaded':unload,'first_later_foot_load':later,'first_later_ground_contact':g,'missing_or_duplicate_contact_intervals':missing_contact[:10],'missing_or_duplicate_contact_count':len(missing_contact),'missing_or_duplicate_toe_intervals':missing_toe[:10],'missing_or_duplicate_toe_count':len(missing_toe),'toe_speed_threshold_count':len(cap),'first_toe_speed_threshold':cap[:1],'nonfoot_substep_count':len(nonfoot),'first_nonfoot':nonfoot[:1],'height_range_450_to_end_m':[min(heights),max(heights)] if heights else None,'finite_states':valid})
 rows.append(row)
out={'arm':arm,'rows':rows,'teacher_off':meta['birdflow_journey_teacher'] is False,'runtime_input_hashes_match_500_step_trial':inputs==short_inputs,'observer_errors':errors,'duplicate_state_keys':duplicates[:10],'duplicate_state_count':len(duplicates),'scope':'used-seed scripted flight; contact/state epochs not independently aligned; no landing or navigation qualification'};(A/(arm+'-analysis.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
