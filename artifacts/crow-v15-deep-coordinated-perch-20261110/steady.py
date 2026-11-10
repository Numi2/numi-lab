from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-deep-coordinated-perch-20261110');vals=[];last=None
with gzip.open(A/'development-j-deep-coordinated-perch.stderr.gz','rt') as f:
 for l in f:
  if not l.startswith('CROW_SUPPORT_SAMPLE '):continue
  x=json.loads(l.split(' ',1)[1]);t=x['step']
  if 4751<=t<=5001:
   q=x['q'];v=x['v'];pitch=math.asin(max(-1,min(1,2*(q[6]*q[4]-q[5]*q[3]))));speed=math.sqrt(sum(i*i for i in v[:3]));ang=math.sqrt(sum(i*i for i in v[3:6]));vals.append((t,q[0],q[1],q[2],pitch,speed,ang,q[14],q[16]));last=x
D={'sample_count':len(vals),'x_range_m':[min(v[1] for v in vals),max(v[1] for v in vals)],'y_range_m':[min(v[2] for v in vals),max(v[2] for v in vals)],'height_range_m':[min(v[3] for v in vals),max(v[3] for v in vals)],'pitch_range_rad':[min(v[4] for v in vals),max(v[4] for v in vals)],'max_sampled_linear_speed_m_s':max(v[5] for v in vals),'max_sampled_angular_speed_rad_s':max(v[6] for v in vals),'hip_range_rad':[min(v[7] for v in vals),max(v[7] for v in vals)],'ankle_range_rad':[min(v[8] for v in vals),max(v[8] for v in vals)],'last_step':vals[-1][0]}
(A/'steady.json').write_text(json.dumps(D,indent=2)+'\n');print(json.dumps(D,indent=2))
