from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-coordinated-return-isolated-20261114');B=Path('artifacts/crow-v15-rear-contact-front-curl-20261114')
def read(p):
 d={};coord={}
 with gzip.open(p/'trace.log.gz','rt') as f:
  for line in f:
   x=line.split()
   if not x or x[0] not in ('step','control','hipcommand','jointtarget','rearKnee','rearAnkle','rightHipCommand','balance','placement','event','frontCouple','coord'):continue
   e,t=map(int,x[1:3]);
   if x[0]=='coord':coord[e,t]=list(map(float,x[3:]))
   else:d.setdefault((e,t),[]).append((x[0],x[3:]))
 return d,coord
a,coord=read(A);b,_=read(B);first=next((t for t in range(450) if coord.get((1,t),[.75,0])[0]>.750001 or coord.get((1,t),[.75,0])[1]>0),None);assert first is not None
rows=[]
for e in (0,1):
 end=450 if e==0 else first;bad=[t for t in range(end) if (e,t) not in a or (1,t) not in b or a[e,t]!=b[1,t]];rows.append({'lane':e,'comparison_intervals':end,'mismatch_count':len(bad),'first_mismatches':bad[:5]})
out={'first_candidate_intervention_zero_based':first,'peak_rear_coupling_ratio':max(x[0] for x in coord.values()),'peak_hip_offset_rad':max(x[1] for x in coord.values()),'rows':rows,'scope':'Exact logged state, command and event parity; contact/state epochs remain unverified.'};(A/'baseline-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
