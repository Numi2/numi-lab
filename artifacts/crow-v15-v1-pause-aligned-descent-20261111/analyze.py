from pathlib import Path
import gzip,json,collections,math
A=Path('artifacts/crow-v15-v1-pause-aligned-descent-20261111');B=Path('artifacts/crow-v15-v1-teacher-launch-parity-20261111');j=json.loads((A/'route.stdout').read_text());first=j['first_episode_navigation'];episodeEnd=min(first['steps_by_environment'][0]+1,3000) if first['physical_failure_by_environment'][0] else 3000
def read(p):
 state={};perch={};fatal=[];clear=[];errors=[]
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);state[x['step']]=x
   elif l.startswith('CROW_PERCH_SUBSTEP '):
    x=l.split();perch[tuple(map(int,x[1:4]))]={'required':int(x[4]),'capacity':int(x[5]),'values':list(map(float,x[6:]))}
   elif l.startswith('CROW_CONTACT_TRACE '):fatal.append(json.loads(l.split(' ',1)[1]))
   elif l.startswith('CROW_CLEARANCE_SAMPLE '):clear.append(json.loads(l.split(' ',1)[1]))
   elif l.startswith('CROW_SUBSTEP_ERROR '):errors.append(l.strip())
 return state,perch,fatal,clear,errors
s,p,f,c,e=read(A/'route.stderr.gz');ref,_,_,_,_=read(B/'route.stderr.gz');diff=[t for t in range(1,1306) if t not in s or t not in ref or any(s[t][k]!=ref[t][k] for k in ('q','v'))];miss=[(t,sub) for t in range(1,episodeEnd+1) for sub in range(16) if (t,sub,0) not in p];bad=[k for k,x in p.items() if len(x['values'])!=21 or x['required']>x['capacity']];by=collections.defaultdict(lambda:[0.0]*21)
for (t,sub,lane),x in p.items():
 if lane==0 and t<=episodeEnd:
  for i,v in enumerate(x['values']):by[t][i]+=v
supported=[];opposed=[]
for t in range(1,episodeEnd+1):
 v=by[t];support=min(v[0],v[1])>.0002 and v[0]+v[1]>=.5*.45004*9.81*.02 and v[4]<=1e-7;supported.append(support);opposed.append(support and all(sum(v[13+base+k] for k in (0,1,2))>=.0002 and v[13+base+3]<=-.0002 for base in (0,4)))
def longest(flags):
 n=best=0
 for x in flags:n=n+1 if x else 0;best=max(n,best)
 return best
firstFoot=next((t for t in sorted(by) if by[t][2]+by[t][3]>1e-7),None);firstOther=next((t for t in sorted(by) if by[t][4]>1e-7),None);out={'first_episode':first,'observation_valid':not (miss or bad or e),'missing_count':len(miss),'overflow_or_shape_count':len(bad),'observer_errors':e,'q_v_differences_through_1305_count':len(diff),'first_difference':diff[:1],'first_perch_foot_impulse_step':firstFoot,'first_perch_other_body_impulse_step':firstOther,'bilateral_half_weight_supported_intervals':sum(supported),'longest_consecutive_supported_intervals':longest(supported),'opposed_supported_intervals':sum(opposed),'longest_consecutive_opposed_intervals':longest(opposed),'first_fatal_contact':f[0] if f else None,'clearance_samples':c,'foot_support_step_count':j['foot_support_step_count'],'failed_environment_steps':j['failed_environment_steps'],'limits':'privileged development route; substep impulses alone do not establish safe toe separation, release or realistic biological landing; exclude post-failure reset'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
