from pathlib import Path
from collections import defaultdict
import gzip,json,math
D=Path('artifacts/crow-v15-hip-roll-negative-shift-20261111');named=defaultdict(lambda:[0.,0.]);states=set();obs={}
with gzip.open(D/'trace.log.gz','rt') as f:
 for l in f:
  a=l.split()
  if not a:continue
  if a[0]=='step':states.add((int(a[1]),int(a[2])+1))
  elif a[0]=='contact':
   e,t=int(a[1]),int(a[2])+1;an,bn=a[3:5];imp=float(a[5]);bird=bn if an=='locomotion_ground' else an if bn=='locomotion_ground' else ''
   side=0 if bird.startswith('crow_left_') else 1 if bird.startswith('crow_right_') else -1
   if imp>0 and side>=0 and ('foot' in bird or 'toe' in bird):named[e,t][side]+=imp
with gzip.open(D/'substeps.log.gz','rt') as f:
 for l in f:
  a=l.split()
  if a and a[0]=='CROW_TWO_TOE_SUBSTEP' and int(a[2])==31:
   t,e=int(a[1]),int(a[3]);obs[e,t]=[float(a[8]),float(a[9])]
errs=[];diff=[]
for key in sorted(states&obs.keys()):
 for side in range(2):
  error=abs(named[key][side]-obs[key][side]);errs.append(error)
  if error>1e-6:diff.append((key,side,error,named[key][side],obs[key][side]))
r={'observed_endpoint_state_count':len(states),'compared_last_substeps':len(states&obs.keys()),'max_absolute_normal_impulse_difference_ns':max(errs) if errs else None,'differences_over_1e-6_ns':len(diff),'first_differences':diff[:8],'limitations':'Checks summed foot/toe normal impulses against native named endpoint contacts in the last substep, not friction signs, all-substep coverage, or state reconstruction.'};(D/'observer-validation.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
