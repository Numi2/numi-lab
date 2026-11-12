from pathlib import Path
import gzip,json,math,statistics
A=Path('artifacts/crow-v15-flight-sideslip-audit-20261112');B=Path('artifacts/crow-v15-wing-bank-feedback-screen-20261112');d={}
with gzip.open(B/'bank.stderr.gz','rt') as f:
 for l in f:
  if not l.startswith('CROW_WING_WRENCH '):continue
  z=l.split();t,k,e=map(int,z[1:4]);x=list(map(float,z[4:]));q=x[22:63];v=x[63:];a,b,c,w=q[3:7];yaw=math.atan2(2*(w*c+a*b),1-2*(b*b+c*c));vf=math.cos(yaw)*v[0]+math.sin(yaw)*v[1];vl=-math.sin(yaw)*v[0]+math.cos(yaw)*v[1];d[e,t,k]=(vf,vl,yaw)
assert len(d)==9600
rows=[]
for e in range(6):
 for lo,hi in [(451,500),(701,750)]:
  x=[d[e,t,k] for t in range(lo,hi+1) for k in range(16)];rows.append({'lane':e,'steps':[lo,hi],'forward_ground_speed_mean_m_s':statistics.mean(v[0] for v in x),'lateral_ground_speed_mean_m_s':statistics.mean(v[1] for v in x),'heading_mean_rad':statistics.mean(v[2] for v in x)})
flight=json.loads((B/'bank-analysis.json').read_text());
for e in range(6):
 vals=flight['rows'][e]['samples'];a=next(x for x in vals if x['step']==500);b=next(x for x in vals if x['step']==800);rows.append({'lane':e,'steps':[500,800],'world_displacement_x_y_m':[b['position'][i]-a['position'][i] for i in range(2)],'heading_start_end_rad':[a['yaw_rad'],b['yaw_rad']]})
out={'rows':rows,'scope':'Ground-relative horizontal velocity rotated by sampled yaw only, not full body-frame airspeed or biological sideslip. Source states precede kernel evaluation. Selected windows and endpoints are not continuous collision or flight qualification. Used development seed; known direct moments off only after 9s.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
