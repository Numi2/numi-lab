from pathlib import Path
import collections,gzip,json,math,sys
p=Path('artifacts/crow-v15-actor-command-flight-20261114')/sys.argv[1];states={};targets={}
with gzip.open(p/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);states[x['lane'],x['step']]=x['policy_action']
  elif line.startswith('CROW_DRIVE_TARGET '):
   x=line.split();targets[int(x[2]),int(x[1])]=list(map(float,x[3:]))
def smooth(a,b,x):
 u=max(0.,min(1.,(x-a)/(b-a)));return u*u*(3.-2.*u)
rows=[]
for offset in (0,4,8,12):
 for lag in range(-2,4):
  errors=[]
  for (lane,t),v in targets.items():
   if not 540<=t<=1800 or len(v)<12:continue
   actions=states.get((lane,t-lag));
   if actions is None or len(actions)<=offset+2:continue
   sec=(t-2)*.02;phase=2*math.pi*6.4*(sec-7.42);gate=smooth(-.2,.2,.55*math.sin(phase)-.83*math.cos(phase))
   if gate<.7:continue
   inferred=(v[10]-v[11])/(2*gate);pred=max(-.07,min(.07,-.15*max(-1.,min(1.,actions[offset+2]))));errors.append(abs(inferred-pred))
  if errors:
   errors.sort();rows.append({'action_offset':offset,'candidate_state_lag_intervals':lag,'samples':len(errors),'median_abs_difference_rad':errors[len(errors)//2],'mean_abs_difference_rad':sum(errors)/len(errors),'max_abs_difference_rad':max(errors),'fraction_within_0_001_rad':sum(x<=.001 for x in errors)/len(errors)})
rows.sort(key=lambda x:x['mean_abs_difference_rad']);out={'ranked_fixed_offset_and_lag_hypotheses':rows,'scope':'exploratory reconstruction from recorded policy actions and differential target; actor head can switch with waypoint and observer epochs are not independently aligned, so a mismatch does not by itself establish non-consumption and a match is not full provenance'}
(p/'command-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
