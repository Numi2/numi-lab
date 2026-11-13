from pathlib import Path
import gzip,json,re,statistics,math
A=Path('artifacts/crow-v15-aero-sideslip-observer-20261113');s=(A/'run.stderr').read_text();m=re.search(r'baseline_max_abs_error ([\deE+.-]+) baseline_rms_error ([\deE+.-]+) n (\d+)',s);rows={};duplicates=0
with gzip.open(A/'moments.tsv.gz','rt') as f:
 for line in f:
  z=line.split();c,e,t,k=map(int,z[:4]);v=float(z[4]);key=e,t,k
  if (c,key) in rows:duplicates+=1
  rows[c,key]=v
expected=4032;base=bool(m) and float(m[1])<=1e-5 and int(m[3])==expected*4*6 and len(rows)==expected*5 and not duplicates and all(math.isfinite(x) for x in rows.values());out={'baseline_max_abs_error':float(m[1]) if m else None,'baseline_rms_error':float(m[2]) if m else None,'baseline_observations':int(m[3]) if m else 0,'record_count':len(rows),'duplicates':duplicates,'baseline_passed':base,'lanes':[],'scope':'Fixed accepted-state instantaneous model sensitivity; not gust recovery, a dynamic identification, measured crow data or calibration.'}
if base:
 for e in (6,7):
  keys=sorted(key for c,key in rows if c==0 and key[0]==e);assert len(keys)==2016;wind=[(rows[1,k]-rows[2,k])/.2 for k in keys];yaw=[(rows[3,k]-rows[4,k])/.2 for k in keys];out['lanes'].append({'lane':e,'samples':len(keys),'yaw_moment_sensitivity_to_world_y_wind_mean_nms':statistics.mean(wind),'yaw_moment_sensitivity_to_world_y_wind_min':min(wind),'yaw_moment_sensitivity_to_world_y_wind_max':max(wind),'yaw_moment_sensitivity_to_world_z_yaw_rate_mean_nms':statistics.mean(yaw),'yaw_moment_sensitivity_to_world_z_yaw_rate_min':min(yaw),'yaw_moment_sensitivity_to_world_z_yaw_rate_max':max(yaw)})
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
