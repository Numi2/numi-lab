from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-bilateral-pitch-support-20261114');P=Path('artifacts/crow-v15-toe-friction-observer-20261114/trace.log.gz');kinds=('step','control','event','hipcommand','jointtarget','rearKnee','rearAnkle','rightHipCommand','balance','pitchcontrol','stage','pose','contact','stancetarget','pitchSupport')
def load(p):
 d={}
 with gzip.open(p,'rt') as f:
  for l in f:
   z=l.split()
   if z and z[0] in kinds:d.setdefault((int(z[1]),int(z[2]),z[0]),[]).append(z[3:])
 return d
cur=load(A/'trace.log.gz');old=load(P)
def first(fn):return next((t for t in range(450) if fn(t)),None)
gate=first(lambda t:cur.get((0,t,'pitchSupport'))!=cur.get((1,t,'pitchSupport')))
state=first(lambda t:any(cur.get((0,t,k))!=cur.get((1,t,k)) for k in kinds if k!='pitchSupport'))
prior=first(lambda t:any(cur.get((0,t,k))!=old.get((0,t,k)) for k in kinds))
out={'first_pitch_gate_difference_zero_based':gate,'first_other_logged_difference_zero_based':state,'first_lane0_prior_difference_zero_based':prior,'state_and_commands_match_before_gate':gate is not None and (state is None or state>=gate),'scope':'Exact logged comparisons, not independent synchronization of solver epochs.'};(A/'matched-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
