from pathlib import Path
import gzip,json,collections,statistics
A=Path('artifacts/crow-v15-walk-contact-interval-sensitivity-20261112')
paths={'held':Path('artifacts/crow-v15-small-support-stroke-20261112/sham-20261127-paired'),'phase_lift':Path('artifacts/crow-v15-walk-hip-lift-screen-20261112/candidate-20261127-rebound')}
allout={}
for label,path in paths.items():
 data={}
 with gzip.open(path/'run.stderr.gz','rt') as f:
  for s in f:
   if not s.startswith('CROW_TWO_TOE_SUBSTEP '):continue
   z=s.split();t,k,e=map(int,z[1:4])
   if 201<=t<=300:data[e,t,k]=list(map(float,z[6:]))
 assert len(data)==12800
 result={}
 for gap in (0,1,2,4,8,16):
  counts=collections.Counter();durations=[];missingdur=[];bothdur=[];perlane=[]
  for e in range(8):
   lane=collections.Counter()
   for side in range(2):
    samples=[]
    for t in range(201,301):
     for k in range(16):
      z=data[e,t,k];v=z[5+side*8:13+side*8];samples.append((z[2+side]>1e-7,sum(v[:6])>1e-7,sum(v[6:])>1e-7))
    active=[i for i,v in enumerate(samples) if v[0]];groups=[]
    for i in active:
     if not groups or i-groups[-1][-1]>gap+1:groups.append([i])
     else:groups[-1].append(i)
    for grp in groups:
     start,end=grp[0],grp[-1]
     if start==0 or end==len(samples)-1:lane['partial']+=1;continue
     d=(end-start+1)*.00125;durations.append(d)
     front=[i for i in range(start,end+1) if samples[i][1]];rear=[i for i in range(start,end+1) if samples[i][2]]
     if not front or not rear:lane['missing_group']+=1;missingdur.append(d);continue
     bothdur.append(d);lane['both_groups']+=1
     delta=rear[-1]-front[-1];lane['rear_first' if delta<0 else 'front_first' if delta>0 else 'unresolved']+=1
   counts.update(lane);perlane.append({'lane':e,**dict(lane)})
  def stats(x):return {'n':len(x),'median_s':statistics.median(x) if x else None,'max_s':max(x) if x else None,'below_10ms':sum(v<.01 for v in x),'below_20ms':sum(v<.02 for v in x)}
  result[str(gap)]={'counts':dict(counts),'duration':stats(durations),'missing_duration':stats(missingdur),'both_duration':stats(bothdur),'lanes':perlane}
 allout[label]=result
allout['method']='Normal impulse >1e-7 per foot per 1.25ms substep, bridge 0–16 silent substeps; contacts on entirely inside the 4–6s window only, count partial runs separately. Group by first/last normal contact; missing means at least one toe group absent. Bridging can hide real loss of support or merge distinct stance cycles. This is sensitivity analysis, not proof of a stride or contact ordering, no biological qualification.'
(A/'analysis.json').write_text(json.dumps(allout,indent=2)+'\n')
for arm in paths:
 for gap in ('0','2','4','8','16'):
  x=allout[arm][gap];print(arm,gap,x['counts'],'missing duration',x['missing_duration'])
