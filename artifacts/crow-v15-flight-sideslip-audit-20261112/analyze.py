from pathlib import Path
import gzip,hashlib,json,math
R=Path.cwd(); A=R/'artifacts/crow-v15-flight-sideslip-audit-20261112'
src=R/'artifacts/crow-v15-bank-pitch-feedback-screen-20261112'
result=json.loads((src/'feedback.stdout').read_text())['first_episode_navigation']
ends=result['steps_by_environment']; samples=[]
with gzip.open(src/'feedback.stderr.gz','rt') as f:
 for line in f:
  if not line.startswith('CROW_FLIGHT_SAMPLE '):continue
  x=json.loads(line.split(' ',1)[1]); e,t=x['lane'],x['step']
  if e>=6 or t<500 or t>ends[e]:continue
  q=x['q'];v=x['v'];qx,qy,qz,qw=q[3:7]
  yaw=math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
  speed=math.hypot(v[0],v[1])
  if speed<.05:continue
  direction=math.atan2(v[1],v[0]); slip=math.atan2(math.sin(direction-yaw),math.cos(direction-yaw))
  forward=math.cos(yaw)*v[0]+math.sin(yaw)*v[1]
  lateral=-math.sin(yaw)*v[0]+math.cos(yaw)*v[1]
  samples.append({'lane':e,'step':t,'height_m':q[2],'yaw_rad':yaw,'horizontal_speed_m_s':speed,'body_relative_forward_m_s':forward,'body_relative_lateral_m_s':lateral,'velocity_direction_minus_yaw_rad':slip})
rows=[]
for e in range(6):
 s=[x for x in samples if x['lane']==e]
 assert s
 rows.append({'lane':e,'count':len(s),'end_step':ends[e],'fraction_abs_slip_over_45deg':sum(abs(x['velocity_direction_minus_yaw_rad'])>math.pi/4 for x in s)/len(s),'last_sample':s[-1]})
out={'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [src/'feedback.stderr.gz',src/'feedback.stdout']},'rows':rows,'samples':samples,'scope':'Retained 100-control-interval endpoint snapshots, one previously used seed; uses projected body x-axis and simulated world linear velocity. The assumption is corroborated by the aerodynamic kernel, but not independently by filmed crow motion. This is descriptive sideslip of the simulation, not integrated displacement, aerodynamic cause, or qualified flight.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(rows,indent=2))
