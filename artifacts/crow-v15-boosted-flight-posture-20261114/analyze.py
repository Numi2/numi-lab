from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-boosted-flight-posture-20261114');p=Path('artifacts/crow-v15-boosted-phase-turn-20261114/straight-reference-20261126.json.gz')
with gzip.open(p,'rt') as f:r=json.load(f)
rows=[]
for e in (6,7):
 a=[x for x in r if x['lane']==e and 450<=x['step']<=1800];tilt=[];angular=[];lat=[]
 for x in a:
  q=x['q'];v=x['v'];tilt.append((math.acos(max(-1,min(1,1-2*(q[3]**2+q[4]**2)))),x['step']));angular.append((math.sqrt(sum(z*z for z in v[3:6])),x['step']));lat.append((abs(v[1]),x['step']))
 rows.append({'lane':e,'state_count':len(a),'max_tilt_rad_and_step':max(tilt),'max_root_angular_speed_rad_s_and_step':max(angular),'max_abs_world_y_speed_m_s_and_step':max(lat)})
out={'source':str(p),'rows':rows,'scope':'Supplemental used-seed observed statistics, not a preregistered attitude or flight qualification gate; candidate observer timing remains inferred.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
