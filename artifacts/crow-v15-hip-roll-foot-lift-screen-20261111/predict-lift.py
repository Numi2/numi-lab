from pathlib import Path
import gzip,subprocess,json
D=Path('artifacts/crow-v15-hip-roll-foot-lift-screen-20261111');p=Path('artifacts/crow-v15-hip-roll-strong-shift-20261111/trace.log.gz');q=v=None
with gzip.open(p,'rt') as f:
 for l in f:
  a=l.split()
  if a and a[0]=='step' and int(a[1])==0 and int(a[2])==180:
   i=a.index('q');j=a.index('v');q=list(map(float,a[i+1:j]));v=a[j+1:];break
if q is None:raise RuntimeError('reference state unavailable')
deltas=[-.30,-.15,0,.15,.30,.50];s=''
for i,delta in enumerate(deltas):
 qq=q.copy();qq[15]+=delta;qq[16]-=delta*.5;s+=f'{i} 180 '+' '.join(map(str,qq))+' '+' '.join(v)+'\n'
p=subprocess.run(['build/crow-v15-hip-roll-filtered-prototype-20261111/kinematics'],input=s,text=True,stdout=subprocess.PIPE,check=True);foot={}
for l in p.stdout.splitlines():
 a=l.split()
 if a[0]=='foot' and a[3]=='crow_left_foot':foot[int(a[1])]=list(map(float,a[4:7]))
base=foot[2];rows=[{'knee_delta_rad':delta,'ankle_delta_rad':-.5*delta,'foot_center_m':foot[i],'relative_to_baseline_m':[foot[i][k]-base[k] for k in range(3)]} for i,delta in enumerate(deltas)]
r={'reference':'crow-v15-hip-roll-strong-shift-20261111 lane 0 zero-based step 180','rows':rows,'limitations':'A fixed-root kinematic perturbation, not dynamic execution, toe-shape clearance, force, or biological evidence.'};(D/'lift-prediction.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
