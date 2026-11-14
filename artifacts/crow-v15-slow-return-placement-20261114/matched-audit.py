from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-slow-return-placement-20261114');B=Path('artifacts/crow-v15-centered-toe-regrip-20261114')
def read(path):
 out={}
 with gzip.open(path/'trace.log.gz','rt') as f:
  for line in f:
   a=line.split()
   if a and a[0] in ('step','control','hipcommand','jointtarget','rearKnee','rearAnkle','rightHipCommand','event','balance','placement'):
    lane,t=int(a[1]),int(a[2]);out.setdefault((lane,t),[]).append((a[0],a[3:]))
 return out
x,y=read(A),read(B)
def state(rows):return [r for r in rows if r[0]=='step']
def actions(rows):return [r for r in rows if r[0] in ('control','hipcommand','jointtarget','rearKnee','rearAnkle','rightHipCommand','balance','placement')]
rows=[]
for field,fn in [('state',state),('logged_targets',actions)]:
 d=[t for t in range(450) if fn(x.get((0,t),[]))!=fn(x.get((1,t),[]))]
 ref=[t for t in range(450) if fn(x.get((0,t),[]))!=fn(y.get((0,t),[]))]
 rows.append({'measure':field,'first_between_lane_divergence_zero_based':next(iter(d),None),'lane0_vs_previous_lane0_mismatch_count':len(ref),'first_reference_mismatch':ref[:5]})
o={'rows':rows,'scope':'exact logged state/target comparison; observation and command timing not independently synchronized'};(A/'matched-audit.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
