from pathlib import Path
import gzip,json,math,sys
A=Path('artifacts/crow-v15-boosted-launch-paired-long-20261114');arm=sys.argv[1];assert arm in ('push','hold');P=A/(arm+'-20261126-rebound-longpaired');d={e:{'wing':[],'pronation':[],'leg':[]} for e in (6,7)}
with gzip.open(P/'run.stderr.gz','rt') as f:
 for l in f:
  x=l.split()
  if not x or x[0] not in ('CROW_WING_EFFORT','CROW_PRONATION_EFFORT','CROW_LEG_EFFORT'):continue
  t,k,e=map(int,x[1:4])
  if e not in d:continue
  if x[0]=='CROW_WING_EFFORT' and 600<=t<=1200:d[e]['wing'].append((t,k,list(map(float,x[4:]))))
  if x[0]=='CROW_PRONATION_EFFORT' and 600<=t<=1200:d[e]['pronation'].append((t,k,list(map(float,x[4:]))))
  if x[0]=='CROW_LEG_EFFORT' and 369<=t<=390:d[e]['leg'].append((t,k,list(map(float,x[7:]))))
rows=[]
for e,r in d.items():
 out={'lane':e}
 for name,lo,hi in (('wing',600,1200),('pronation',600,1200),('leg',369,390)):
  v=r[name];expected={(t,k) for t in range(lo,hi+1) for k in range(16)};seen={(t,k) for t,k,_ in v};width=30 if name=='leg' else 14;out[name+'_unique_complete']=len(v)==len(expected) and seen==expected;out[name+'_finite_and_width']=all(len(a)==width and all(math.isfinite(x) for x in a) for _,_,a in v)
  if name!='leg':
   efforts=[abs(a[j]) for _,_,a in v if len(a)==width for j in (0,7)];out[name+'_max_effort_Nm']=max(efforts,default=None)
   if name=='wing':out['wing_saturated_fraction']=sum(x>=2.999 for x in efforts)/len(efforts) if efforts else None
   if name=='pronation':
    tracking=[abs(a[j]-a[j+1]) for _,_,a in v if len(a)==width for j in (1,8)];out['pronation_saturated_fraction']=sum(x>=.1499 for x in efforts)/len(efforts) if efforts else None;out['pronation_tracking_max_rad']=max(tracking,default=None)
  else:
   for j,limit in enumerate((.75,.75,.60,.75,.75,.60)):
    values=[abs(a[5*j]) for _,_,a in v if len(a)==width];out['leg_joint_%d_max_Nm'%j]=max(values,default=None);out['leg_joint_%d_saturated_fraction'%j]=sum(x>=limit-.0001 for x in values)/len(values) if values else None
 rows.append(out)
out={'arm':arm,'rows':rows,'scope':'modeled drive observations; flight windows 600–1200, leg window 369–390; no biological calibration or direct contact-time alignment'};(A/(arm+'-effort-audit.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
