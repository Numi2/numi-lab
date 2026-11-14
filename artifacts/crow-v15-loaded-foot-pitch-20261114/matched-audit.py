from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-loaded-foot-pitch-20261114');records={}
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if x and x[0] in ('step','control','hipcommand','jointtarget','rearKnee','rearAnkle','rightHipCommand','balance','placement'):
   e,t=int(x[1]),int(x[2]);records.setdefault((e,t),[]).append((x[0],x[3:]))
def rows(e,t,kind):
 return [r for r in records.get((e,t),[]) if (r[0]=='step')==(kind=='state')]
out=[]
for kind in ('state','logged_targets'):
 diff=[t for t in range(450) if rows(0,t,kind)!=rows(1,t,kind)];out.append({'measure':kind,'first_difference_zero_based':next(iter(diff),None),'compared_intervals':450})
result={'rows':out,'scope':'between-lane exact logged comparison; state/command epochs not independently synchronized'};(A/'matched-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
