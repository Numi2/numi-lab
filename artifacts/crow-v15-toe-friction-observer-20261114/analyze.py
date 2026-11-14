from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-toe-friction-observer-20261114');P=Path('artifacts/crow-v15-toe-effort-observer-20261114/trace.log.gz');kinds=('step','control','event','hipcommand','jointtarget','rearKnee','rearAnkle','rightHipCommand','balance','pitchcontrol','stage','pose','contact','stancetarget','pitchSupport')
def load(p):
 d={}
 with gzip.open(p,'rt') as f:
  for line in f:
   z=line.split()
   if z and z[0] in kinds:d.setdefault((int(z[1]),int(z[2]),z[0]),[]).append(z[3:])
 return d
cur=load(A/'trace.log.gz');old=load(P);prior=[t for t in range(450) if any(cur.get((0,t,k))!=old.get((0,t,k)) for k in kinds)];paired=[t for t in range(450) if any(cur.get((0,t,k))!=cur.get((1,t,k)) for k in kinds)]
contacts=[];totals=collections.defaultdict(lambda:[0.,0.]);normal={};errors=[]
with gzip.open(A/'substeps.log.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0] in ('CROW_SUBSTEP_ERROR','failure'):errors.append(line.strip())
  if z[0]=='CROW_TOE_FRICTION':
   t,sub,e,body=map(int,z[1:5]);jn,ju,jv,mu_s,mu_d=map(float,z[5:10]);side=0 if body==11 or 19<=body<=26 else 1;totals[e,t,sub][side]+=jn;contacts.append((e,t,sub,body,jn,ju,jv,mu_s,mu_d))
  elif z[0]=='CROW_TWO_TOE_SUBSTEP':
   t,sub,e=map(int,z[1:4]);normal[e,t,sub]=list(map(float,z[8:10]))
missing=[(e,t,k) for e in range(2) for t in range(1,451) for k in range(32) if (e,t,k) not in normal]
errs=[abs(totals[key][i]-n[i]) for key,n in normal.items() for i in (0,1)]
def group(b):return 'left_foot' if b==11 else 'right_foot' if b==15 else 'left_front' if 19<=b<=24 else 'left_rear' if 25<=b<=26 else 'right_front' if 27<=b<=32 else 'right_rear'
rows=[]
for e in (0,1):
 for lo,hi in ((270,290),(290,300),(300,305),(305,310),(310,330)):
  for g in ('right_foot','right_front','right_rear','left_foot','left_front','left_rear'):
   selected=[c for c in contacts if c[0]==e and lo<=c[1]<hi and group(c[3])==g and c[4]>=1e-6 and c[7]>0];v=sorted(math.hypot(c[5],c[6])/(c[7]*c[4]) for c in selected)
   rows.append({'lane':e,'trace_steps_inclusive_exclusive':[lo,hi],'group':g,'valid_contacts':len(v),'fraction_ratio_ge_0_95':sum(x>=.95 for x in v)/len(v) if v else None,'max_ratio':max(v,default=None),'median_ratio':v[len(v)//2] if v else None,'p95_ratio':v[min(len(v)-1,int(.95*len(v)))] if v else None,'static_coefficients':sorted(set(round(c[7],6) for c in selected))})
out={'prior_lane0_first_differing_zero_based':next(iter(prior),None),'between_lane_first_differing_zero_based':next(iter(paired),None),'substep_contact_summary_records':len(normal),'missing_summary_records':len(missing),'normal_sum_max_abs_error_Ns':max(errs,default=None),'logged_contact_count':len(contacts),'normal_threshold_Ns':1e-6,'errors':errors,'rows':rows,'limitations':'Ratios are solved tangent impulse magnitude divided by effective static coefficient times normal impulse; near-cone impulse alone is not proof of sliding. Relative contact tangential velocity and independent contact/state epoch alignment are not observed.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
