from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-clearance-coordination-20261114'); B=Path('artifacts/crow-v15-qualified-release-regrip-20261114')
kinds={'step','control','hipcommand','jointtarget','rearKnee','rearAnkle','rightHipCommand','balance','placement','event','frontCouple','geometry','pitchcontrol','contact'}
def read(p):
 d={}
 with gzip.open(p/'trace.log.gz','rt') as f:
  for s in f:
   z=s.split()
   if z and z[0] in kinds and z[1]=='1':d.setdefault(int(z[2]),[]).append((z[0],z[3:]))
 return d
a,b=read(A),read(B)
diff=[t for t in range(450) if a.get(t)!=b.get(t)]
r={'compared_intervals':450,'first_difference_zero_based':diff[0] if diff else None,'different_intervals':len(diff),'scope':'lane 1 logged comparison; contact epochs not independently synchronized'}
(A/'prior-candidate-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
