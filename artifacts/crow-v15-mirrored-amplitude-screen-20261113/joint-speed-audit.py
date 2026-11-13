from pathlib import Path
import collections,gzip,json,sys,math
P=Path('artifacts/crow-v15-mirrored-amplitude-screen-20261113')/sys.argv[1];model={};hits=collections.Counter();peak=collections.defaultdict(float);counts=collections.Counter();errors=[];indices=list(range(6,22))+[38,39]
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0]=='CROW_MODEL_DOF_LIMIT':model[int(z[2])]=(int(z[3]),float(z[4]),z[5])
  elif z[0]=='CROW_BODY_JOINT_V_SUBSTEP':
   t,k,e=map(int,z[1:4]);v=list(map(float,z[4:]));
   if len(v)!=len(indices) or not all(map(math.isfinite,v)):errors.append((t,k,e));continue
   if t<500:continue
   counts[e]+=1
   for j,x in zip(indices,v):
    peak[e,j]=max(peak[e,j],abs(x));a=model.get(j)
    if a and a[0]&8 and a[1]>0 and abs(x)>=a[1]-1e-4:hits[e,j]+=1
out={'counts':dict(counts),'hits':[{'lane':e,'joint':j,'name':model[j][2],'at_limit_substeps':n,'max_speed':peak[e,j],'limit':model[j][1]} for (e,j),n in sorted(hits.items())],'errors':errors,'scope':'Additional observed joint-speed diagnostic; thresholds are current model limits, not anatomical calibration.'};(P/'joint-speed-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
