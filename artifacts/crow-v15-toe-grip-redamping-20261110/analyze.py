from pathlib import Path
from collections import defaultdict
from itertools import groupby
import json,gzip
A=Path("artifacts/crow-v15-toe-grip-redamping-20261110");C=defaultdict(float);phase={};results={}
for line in gzip.open(A/"results.txt.gz","rt").read().splitlines():
 a=line.split()
 if a[0]=="contact":
  c,arm,t=map(int,a[1:4]); name=a[4]; nx=float(a[5]); imp=float(a[7]); side="left" if "_left_" in name else "right"; digit="rear" if "_rear_" in name else "front"; direction="positive" if nx>.5 else "negative" if nx<-.5 else "other"; C[c,arm,t,side,digit,direction]+=imp
 elif a[0]=="trace":phase[int(a[1]),int(a[2]),int(a[3])]=int(a[4])
 elif a[0]=="case_result":results[int(a[1]),int(a[2])]=line
rows=[]
for c in range(8):
 for arm in range(4):
  hits=[t for t in range(150) if phase.get((c,arm,t))==3 and all(C[c,arm,t,side,"front","negative"]>.00005 and C[c,arm,t,side,"rear","positive"]>.00005 for side in ("left","right"))]
  run=max((len(list(g)) for _,g in groupby(enumerate(hits),lambda q:q[1]-q[0])),default=0);release=next((t for t in range(150) if phase.get((c,arm,t))==4),None)
  loss={side:next((t for t in range(release if release is not None else 150,150) if C[c,arm,t,side,"front","negative"]<=.00005 or C[c,arm,t,side,"rear","positive"]<=.00005),None) for side in ("left","right")}
  result=results.get((c,arm),"missing"); rows.append(dict(case=c,arm=arm,result=result,max_consecutive_hold=run,release_step=release,post_release_loss=loss,passed=run>=3 and release is not None and all(v is not None for v in loss.values()) and "completed" in result))
out={"arms":{str(a):{"passes":sum(r["passed"] for r in rows if r["arm"]==a),"failures":sum(" failed " in r["result"] for r in rows if r["arm"]==a),"transitioned":sum(any(phase.get((r["case"],a,t)) in (2,3) for t in range(150)) for r in rows if r["arm"]==a)} for a in range(4)},"rows":rows,"scope":"already-supported developmental pinch; no landing or biological qualification"};(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out["arms"],indent=2))
