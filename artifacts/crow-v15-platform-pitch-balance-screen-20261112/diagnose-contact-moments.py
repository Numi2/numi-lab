from pathlib import Path
import subprocess,gzip,json,collections,math
p=Path('artifacts/crow-v15-platform-pitch-balance-screen-20261112');d=p/'autonomous-20261126-posture';ends=json.loads((d/'run.stdout').read_text())['first_episode_navigation']['steps_by_environment'];s={};c=collections.defaultdict(list)
z=subprocess.Popen(['cat',*[str(x) for x in sorted(d.glob('run.stderr.gz.part[0-9][0-9][0-9]'))]],stdout=subprocess.PIPE)
with gzip.open(z.stdout,'rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step']
   if e in (5,6,7) and 3750<=t<=min(4700,ends[e]) and t%25==0:s[e,t]=x
  elif line.startswith('CROW_SCENE_CONTACT '):
   x=line.split();t,k,e,a,b=map(int,x[1:6])
   if e not in (5,6,7) or t<3750 or t>min(4700,ends[e]) or t%25 or 40 not in (a,b):continue
   body=b if a==40 else a
   if not (body in (11,15) or 19<=body<=34):continue
   normal,t1,t2=map(float,x[6:9]);point=list(map(float,x[9:12]));n=list(map(float,x[13:16]));u=list(map(float,x[16:19]));v=[n[1]*u[2]-n[2]*u[1],n[2]*u[0]-n[0]*u[2],n[0]*u[1]-n[1]*u[0]];sign=1 if a==40 else -1
   force=[sign*(n[i]*normal+u[i]*t1+v[i]*t2) for i in range(3)];c[e,t].append((point,force))
assert z.wait()==0
rows=[]
for (e,t),st in sorted(s.items()):
 q,v=st['q'],st['v'];qnorm=math.sqrt(sum(q[i]*q[i] for i in range(3,7)));assert abs(qnorm-1)<1e-4
 moment=0.;up=0.;cop=0.
 for point,force in c[e,t]:
  rx=point[0]-q[0];rz=point[2]-q[2];moment+=rz*force[0]-rx*force[2];up+=force[2];cop+=point[0]*force[2]
 rows.append({'lane':e,'step':t,'root_x':q[0],'root_z':q[2],'root_pitch_rad':math.asin(max(-1,min(1,2*(q[6]*q[4]-q[5]*q[3])))),'root_pitch_rate':v[4], 'hip_q':[q[14],q[17]],'knee_q':[q[15],q[18]],'ankle_q':[q[16],q[19]],'vertical_foot_impulse_ns':up,'vertical_impulse_weighted_contact_x_relative_root_m':cop/up-q[0] if abs(up)>1e-7 else None,'foot_contact_pitch_angular_impulse_about_root_nms':moment})
out={'scope':'Contact impulses sampled every 25 intervals and expressed about root origin, not full center of mass. Vertical weighted x is a diagnostic and can be misleading with negative vertical contact impulses; includes tangential impulse in contact moment. Applied motor and aerodynamic moments and the platform mass are not accounted. Development trajectory only.','rows':rows}
(p/'contact-moment-diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:
 if r['step']%100==0:print(r)
