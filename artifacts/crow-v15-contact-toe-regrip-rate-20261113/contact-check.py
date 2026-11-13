from pathlib import Path
import gzip,json,collections,math
A=Path('artifacts/crow-v15-contact-toe-regrip-rate-20261113');host=collections.defaultdict(lambda:[0.,0.]);observed={};seen=set();errors=[]
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0]=='step':seen.add((int(z[1]),int(z[2])+1))
  if z[0]=='contact':
   e,t=int(z[1]),int(z[2])+1;an,bn=z[3:5];imp=float(z[5]);
   if imp>0 and (an=='locomotion_ground' or bn=='locomotion_ground'):
    bird=bn if an=='locomotion_ground' else an
    for side,name in enumerate(('crow_left_','crow_right_')):
     if bird.startswith(name) and ('foot' in bird or 'toe' in bird):host[e,t][side]+=imp
with gzip.open(A/'substeps.log.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0] in ('failure','CROW_SUBSTEP_ERROR'):errors.append(line.strip())
  if z[0]=='CROW_TWO_TOE_SUBSTEP' and int(z[2])==31:
   t,e=int(z[1]),int(z[3]);assert (e,t) not in observed;observed[e,t]=[float(z[8]),float(z[9])]
missing=seen-set(observed);diff=[];nonzero=0
for k in seen&set(observed):
 x=host[k];y=observed[k];nonzero+=sum(v>1e-7 for v in x);diff.append(max(abs(x[i]-y[i]) for i in (0,1)))
out={'accepted_state_records':len(seen),'last_substep_records':len(observed),'missing_last_substep':len(missing),'nonzero_foot_samples':nonzero,'max_normal_impulse_error_ns':max(diff,default=None),'observer_errors':errors,'passed':not missing and bool(diff) and nonzero>=10 and max(diff)<1e-6 and not errors,'scope':'Checks host end-of-interval foot normal impulse against the corrected last-substep observer. Does not by itself establish grip, flight, or model fidelity.'};(A/'observer-validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
