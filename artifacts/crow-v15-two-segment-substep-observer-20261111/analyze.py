from pathlib import Path
import collections,json
A=Path('artifacts/crow-v15-two-segment-substep-observer-20261111');R=Path('artifacts/crow-v15-two-segment-contact-admission-20261111')
def read(p):
 state={};foot=collections.defaultdict(lambda:[0.,0.])
 for l in p.read_text().splitlines():
  a=l.split()
  if not a:continue
  if a[0]=='step':state[int(a[1])]=l
  elif a[0]=='contact':
   t=int(a[1])+1;names=(a[2],a[3]);
   if 'locomotion_ground' in names:
    bird=names[0] if names[1]=='locomotion_ground' else names[1];side=0 if bird.startswith('crow_left_') else 1 if bird.startswith('crow_right_') else -1
    if side>=0 and ('_foot' in bird or '_toe' in bird):foot[t][side]+=float(a[4])
 return state,foot
s,endpoint=read(A/'trace.log');ref,_=read(R/'trace.log');records={};errors=[]
for l in (A/'substeps.log').read_text().splitlines():
 a=l.split()
 if a and a[0]=='CROW_SUBSTEP_ERROR':errors.append(l)
 if a and a[0]=='CROW_TWO_TOE_SUBSTEP':records[tuple(map(int,a[1:4]))]={'required':int(a[4]),'capacity':int(a[5]),'v':list(map(float,a[6:]))}
missing=[(t,i) for t in range(1,121) for i in range(16) if (t,i,0) not in records];bad=[k for k,v in records.items() if v['required']>v['capacity'] or len(v['v'])!=37];diff=[t for t in range(120) if s.get(t)!=ref.get(t)];maxerr=0
for t in range(1,121):
 if (t,15,0) in records:
  v=records[t,15,0]['v'];maxerr=max(maxerr,*[abs(v[2+i]-endpoint[t][i]) for i in range(2)])
by=collections.defaultdict(lambda:[0.]*37)
for (t,sub,lane),r in records.items():
 for i,v in enumerate(r['v']):by[t][i]+=v
support=[t for t,v in by.items() if min(v[0],v[1])>.0002 and v[0]+v[1]>=.5*.45004*9.81*.02 and v[4]<=1e-7];other=[t for t,v in by.items() if v[4]>1e-7]
out={'observed_records':len(records),'missing':missing,'bad':bad,'errors':errors,'q_v_full_line_difference_steps':diff,'max_endpoint_normal_impulse_difference':maxerr,'bilateral_half_weight_support_intervals':len(support),'nonfoot_impulse_intervals':other,'note':'validation of observer on one initialized stationary perch case; no toe opposition, release, route or biological qualification'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
