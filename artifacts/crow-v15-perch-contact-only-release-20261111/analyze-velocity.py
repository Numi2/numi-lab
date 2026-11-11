from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-perch-contact-only-release-20261111/offset-0.02')
endpoints={}; release={}
with gzip.open(A/'trace.log.gz','rt') as f:
 for l in f:
  a=l.split()
  if a and a[0]=='step':
   e,t=int(a[1]),int(a[2])+1;i=a.index('v');endpoints[e,t]=list(map(float,a[i+1:]))
  elif a and a[0]=='event' and int(a[3]) and int(a[1]) not in release:release[int(a[1])]=int(a[2])+1
counts=collections.Counter(); rows=[{'lane':e,'samples':0,'max_abs_toe_velocity_rad_s':0.,'at_or_above_11_99_count':0,'at_or_above_11_99_after_release_count':0,'max_last_substep_error':0.} for e in range(2)];bad=[]
with gzip.open(A/'substeps.log.gz','rt') as f:
 for l in f:
  a=l.split()
  if a and a[0]=='CROW_TOE_VELOCITY':
   t,sub,e=map(int,a[1:4]); v=list(map(float,a[4:]));counts[e,t,sub]+=1
   if len(v)!=len(endpoints.get((e,t),[])) or not all(map(math.isfinite,v)):
    bad.append([e,t,sub,len(v)]);continue
   row=rows[e];row['samples']+=1; toe=v[-18:-2]; mx=max(map(abs,toe));row['max_abs_toe_velocity_rad_s']=max(row['max_abs_toe_velocity_rad_s'],mx)
   if mx>=11.99:
    row['at_or_above_11_99_count']+=1
    if t>=release.get(e,10**9):row['at_or_above_11_99_after_release_count']+=1
   if sub==31:row['max_last_substep_error']=max(row['max_last_substep_error'],max(abs(x-y) for x,y in zip(v,endpoints[e,t])))
missing=sum(counts[e,t,s]!=1 for e in range(2) for t in range(1,551) for s in range(32))
out={'rows':rows,'missing_or_duplicate_samples':missing,'bad_records':bad[:30],'bad_record_count':len(bad),'interpretation':'Velocity samples are end-of-substep generalized velocities; they do not prove no instantaneous within-substep excursion. Check observer agreement and trajectory repeat before use.'};(A/'substep-velocity-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
