from pathlib import Path
import gzip,json,collections
A=Path('artifacts/crow-v15-perch-substep-observer-20261111');B=Path('artifacts/crow-v15-v1-slow-descent-20261111')
def read(p):
 state={};ground={};perch={};fatal=[];errors=[]
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);state[x['step']]=x
   elif l.startswith('CROW_WALK_SUBSTEP '):
    x=l.split();ground[tuple(map(int,x[1:4]))]=x[4:]
   elif l.startswith('CROW_PERCH_SUBSTEP '):
    x=l.split();k=tuple(map(int,x[1:4]));perch[k]={'required':int(x[4]),'capacity':int(x[5]),'values':list(map(float,x[6:]))}
   elif l.startswith('CROW_CONTACT_TRACE '):fatal.append(json.loads(l.split(' ',1)[1]))
   elif l.startswith('CROW_SUBSTEP_ERROR '):errors.append(l.strip())
 return state,ground,perch,fatal,errors
s,g,p,f,e=read(A/'route.stderr.gz');ref,rg,_,rf,re=read(B/'route.stderr.gz');j=json.loads((A/'route.stdout').read_text());end=2053;stateDiff=[t for t in range(1,end+1) if t not in s or t not in ref or any(s[t][k]!=ref[t][k] for k in ('q','v'))];groundDiff=[k for k,v in rg.items() if k[0]<=505 and g.get(k)!=v];missing=[(t,sub) for t in range(1,end+1) for sub in range(16) if (t,sub,0) not in p];badshape=[k for k,x in p.items() if len(x['values'])!=21 or x['required']>x['capacity']];fatalMatch=bool(f and rf and f[0]['step']==rf[0]['step'] and f[0]['pre_q']==rf[0]['pre_q'] and f[0]['pre_v']==rf[0]['pre_v'] and f[0]['contacts']==rf[0]['contacts']);valid=not(stateDiff or groundDiff or missing or badshape or e or re) and fatalMatch
by=collections.defaultdict(lambda:[0.0]*21)
for (t,sub,lane),x in p.items():
 if lane==0 and t<=end:
  for i,v in enumerate(x['values']):by[t][i]+=v
firstFoot=next((t for t in sorted(by) if by[t][2]+by[t][3]>1e-7),None);firstOther=next((t for t in sorted(by) if by[t][4]>1e-7),None);supported=[t for t,v in by.items() if min(v[0],v[1])>.0002 and v[0]+v[1]>=.5*.45004*9.81*.02 and v[4]<=1e-7]
out={'valid':valid,'first_episode_q_v_difference_count':len(stateDiff),'first_q_v_differences':stateDiff[:10],'ground_substep_difference_count':len(groundDiff),'first_ground_differences':groundDiff[:10],'perch_missing_count':len(missing),'first_perch_missing':missing[:10],'perch_overflow_or_shape_count':len(badshape),'observer_errors':e,'fatal_contact_matches':fatalMatch,'first_perch_foot_impulse_step':firstFoot,'first_perch_other_body_impulse_step':firstOther,'bilateral_half_weight_supported_intervals':len(supported),'first_supported_intervals':supported[:10],'native_first_episode':j['first_episode_navigation'],'boundary':'passive observer parity on one known failed privileged route, not landing qualification'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
