from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-return-support-feedback-20261114');B=Path('artifacts/crow-v15-roll32-return-placement-ablation-20261114')
def read(p):
 d={};events={}
 with gzip.open(p/'trace.log.gz','rt') as f:
  for line in f:
   x=line.split()
   if not x or x[0] not in ('step','control','hipcommand','jointtarget','rearKnee','rearAnkle','rightHipCommand','balance','placement','event'):continue
   e,t=map(int,x[1:3]); d.setdefault((e,t),[]).append((x[0],x[3:]))
   if x[0]=='event':events[e,t]=x
 return d,events
a,ev=read(A);b,be=read(B)
first=None
for t in range(1,450):
 x=be[0,t-1]; up,front,rear=map(float,(x[6],x[9],x[10]));finished=bool(int(x[11]))
 if finished and up>.001 and not(front>.000001 and rear<-.000001):first=t;break
assert first is not None
rows=[]
for e in (0,1):
 end=450 if e==0 else first
 bad=[t for t in range(end) if a.get((e,t))!=b.get((0,t))]
 rows.append({'lane':e,'comparison_intervals':end,'mismatch_count':len(bad),'first_mismatches':bad[:5]})
out={'first_changed_support_decision_zero_based':first,'rows':rows,'scope':'Exact logged state, command and event parity; contact/state epochs remain unverified.'};(A/'baseline-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
