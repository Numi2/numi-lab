from pathlib import Path
import gzip,json,math
D=Path('artifacts/crow-v15-toe-release-ramped-return-20261111');rows=[{'lane':i,'toe_max_abs_rad_s':0.,'toe_samples_near_speed_limit':0,'hip_max_abs_rad_s':0.,'hip_samples_near_speed_limit':0,'endpoint_count':0} for i in range(8)]
with gzip.open(D/'trace.log.gz','rt') as f:
 for line in f:
  a=line.split()
  if not a or a[0]!='step':continue
  e=int(a[1]);v=list(map(float,a[a.index('v')+1:]));r=rows[e];toe=v[-18:-2];hip=v[-2:];r['endpoint_count']+=1;r['toe_max_abs_rad_s']=max(r['toe_max_abs_rad_s'],*(abs(x) for x in toe));r['toe_samples_near_speed_limit']+=sum(abs(x)>=11.999 for x in toe);r['hip_max_abs_rad_s']=max(r['hip_max_abs_rad_s'],*(abs(x) for x in hip));r['hip_samples_near_speed_limit']+=sum(abs(x)>=7.999 for x in hip)
out={'rows':rows,'limitations':'Only control-interval endpoint velocities; intermediate solver substeps are not recorded. Absence of endpoint cap hits does not establish absence of substep cap hits. This constructed contact test is not a flight qualification.'};(D/'endpoint-speeds.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
