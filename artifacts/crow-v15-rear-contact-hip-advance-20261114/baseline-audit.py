from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-rear-contact-hip-advance-20261114'); B=Path('artifacts/crow-v15-roll32-return-placement-ablation-20261114')
def read(p):
 d={}; hip={}
 with gzip.open(p/'trace.log.gz','rt') as f:
  for line in f:
   x=line.split()
   if not x or x[0] not in ('step','control','hipcommand','jointtarget','rearKnee','rearAnkle','rightHipCommand','balance','placement','event','hipRegrip'):continue
   e,t=map(int,x[1:3]);
   if x[0]=='hipRegrip':hip[e,t]=float(x[3])
   else:d.setdefault((e,t),[]).append((x[0],x[3:]))
 return d,hip
a,hip=read(A);b,_=read(B); first=next((t for t in range(450) if hip.get((1,t),0)>0),None); assert first is not None
rows=[]
for e in (0,1):
 end=450 if e==0 else first
 bad=[t for t in range(end) if a.get((e,t))!=b.get((0,t))]
 rows.append({'lane':e,'comparison_intervals':end,'mismatches':len(bad),'first_mismatches':bad[:5]})
out={'first_candidate_hip_intervention_zero_based':first,'peak_hip_offset_rad':max(hip.values()),'rows':rows,'scope':'Exact logged state, action and contact summary parity; contact/state epochs remain unverified.'};(A/'baseline-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
