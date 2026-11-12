from pathlib import Path
import gzip,json,math,statistics
A=Path('artifacts/crow-v15-path-body-heading-corrected-20261112')
samples={i:[] for i in range(8)}
with gzip.open(A/'candidate-20261126-rebound-flexed/run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1])
   if 1001<=x['step']<=1200:samples[x['lane']].append(x)
rows=[]
for lane,s in samples.items():
 if not s:
  rows.append({'lane':lane,'samples':0});continue
 target=[0,0,.35,-.35,.35,-.35,.35,-.35][lane]
 errors=[abs(math.atan2(math.sin(target-math.atan2(x['v'][1],x['v'][0])),math.cos(target-math.atan2(x['v'][1],x['v'][0])))) for x in s]
 body_errors=[]
 for x in s:
  qx,qy,qz,qw=x['q'][3:7]
  yaw=math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
  body_errors.append(abs(math.atan2(math.sin(target-yaw),math.cos(target-yaw))))
 speeds=[math.hypot(x['v'][0],x['v'][1]) for x in s]
 r={'lane':lane,'target_rad':target,'samples':len(s),'mean_abs_body_heading_error_rad':statistics.mean(body_errors),'mean_abs_velocity_heading_error_rad':statistics.mean(errors),'max_abs_velocity_heading_error_rad':max(errors),'mean_planar_speed_m_s':statistics.mean(speeds),'max_planar_speed_m_s':max(speeds)}
 rows.append(r);print(r)
(A/'target-summary.json').write_text(json.dumps({'window_seconds':[20,24],'rows':rows},indent=2)+'\n')
