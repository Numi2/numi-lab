from pathlib import Path
import gzip,json,statistics,math,sys,collections
P=Path('artifacts/crow-v15-mirrored-amplitude-screen-20261113')/sys.argv[1];a=collections.defaultdict(lambda:[[],[]]);errors=[];seen=set()
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if not line.startswith('CROW_PRONATION_EFFORT '):continue
  z=line.split();t,k,e=map(int,z[1:4]);key=(t,k,e)
  if e<2 or not 650<=t<=1150:continue
  if key in seen:errors.append(['duplicate',key])
  seen.add(key);v=list(map(float,z[4:]));
  if len(v)!=14 or not all(map(math.isfinite,v)):errors.append(['invalid',key]);continue
  for side in range(2):a[e][side].append(v[side*7:(side+1)*7])
rows=[]
for lane in range(2,8):
 joints=[]
 for samples in a[lane]:
  eff=[x[0] for x in samples];pos=[x[3] for x in samples]
  joints.append({'samples':len(samples),'effort_saturation_fraction':sum(abs(x)>=.15-1e-5 for x in eff)/len(eff) if eff else None,'over_effort_limit':sum(abs(x)>.15001 for x in eff),'position_range':[min(pos),max(pos)] if pos else None,'position_mean':statistics.mean(pos) if pos else None})
 rows.append({'lane':lane,'joints':joints})
out={'rows':rows,'errors':errors,'complete_650_1150':all(j['samples']==501*16 for r in rows for j in r['joints']),'scope':'Modeled actuator effort, not biological muscle force; completeness only expected for lanes surviving the observation interval.'};(P/'effort-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
