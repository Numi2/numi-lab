from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-staged-regrip-30-20261114');B=Path('artifacts/crow-v15-slow-return-regrip-20261114')
kinds={'step','control','hipcommand','jointtarget','rearKnee','rearAnkle','rightHipCommand','balance','placement','event','frontCouple','geometry','pitchcontrol'}
def read(p):
 d={}
 with gzip.open(p/'trace.log.gz','rt') as f:
  for s in f:
   z=s.split()
   if z and z[0] in kinds:
    e,t=map(int,z[1:3]);d.setdefault((e,t),[]).append((z[0],z[3:]))
 return d
a=read(A);b=read(B)
bad=[t for t in range(450) if a.get((0,t))!=b.get((1,t))]
first=next((t for t in range(450) if a.get((1,t))!=a.get((0,t))),None)
print(json.dumps({'control_intervals':450,'control_mismatch_count':len(bad),'control_first_mismatch':bad[:5],'candidate_first_logged_difference_zero_based':first,'scope':'logged observations; contact epochs not independently synchronized'},indent=2))
