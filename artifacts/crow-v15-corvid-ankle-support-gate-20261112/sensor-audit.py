from pathlib import Path
import collections,gzip,json,math,os,statistics,subprocess
A=Path('artifacts/crow-v15-corvid-ankle-support-gate-20261112');D=A/'candidate-20261127-rebound-fine';q={};v={};pts=collections.defaultdict(list);sensor={};ref={}
with gzip.open(D/'run.stderr.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0] in ('CROW_Q_SUBSTEP','CROW_V_SUBSTEP'):
   t,k,e=map(int,z[1:4]);
   if 201<=t<300 and k==15:(q if z[0]=='CROW_Q_SUBSTEP' else v)[e,t]=list(map(float,z[4:]))
  elif z[0]=='CROW_LOADED_POINT':
   t,k,e,b=map(int,z[1:5]);
   if 201<=t<300 and k==15:pts[e,t].append((b,list(map(float,z[5:]))))
  elif z[0]=='CROW_SENSOR_VALUE':
   t,e=map(int,z[1:3]);
   if 201<=t<300:sensor[e,t]=int(float(z[3]))
assert len(q)==len(v)==len(sensor)==8*99,(len(q),len(v),len(sensor))
keys=sorted(q);inp=''.join(f'{e} {t} '+' '.join(map(str,q[e,t]+v[e,t]))+'\n' for e,t in keys)
env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())}
p=subprocess.run(['build/crow-v15-gait-slip-observer-20261112/kinematics'],input=inp,text=True,capture_output=True,check=True,env=env)
bodies={};names={11:'crow_left_foot',15:'crow_right_foot'}
with gzip.open(D/'replay.json.gz','rt') as f:replay=json.load(f)['payload']
names={i:s for i,s in enumerate(replay['body_names'])}
for line in p.stdout.splitlines():
 z=line.split()
 if z and z[0]=='body':
  e,t=int(z[1]),int(z[2]);bodies[e,t,z[3]]=list(map(float,z[4:]))
for e,t in keys:
 rot=q[e,t][3:7];xx,yy,zz,ww=rot;h=[1-2*(yy*yy+zz*zz),2*(xx*yy+ww*zz)];norm=math.hypot(*h);h=[a/norm for a in h]
 sums=[0.,0.];weights=[0.,0.]
 for bi,a in pts[e,t]:
  side=0 if bi==11 or 19<=bi<=26 else 1 if bi==15 or 27<=bi<=34 else None
  if side is None:continue
  b=bodies[e,t,names[bi]];rx,ry,rz=[a[j]-b[j] for j in range(3)];wx,wy,wz=b[10:13];vx=b[7]+wy*rz-wz*ry;vy=b[8]+wz*rx-wx*rz;w=a[-1];sums[side]+=(vx*h[0]+vy*h[1])*w;weights[side]+=w
 ref[e,t]=[sums[i]/weights[i] if weights[i] else 0. for i in range(2)]
rows=[]
for e in range(2,8):
 decoded=[];expected=[]
 for t in range(201,300):
  packed=sensor[e,t];decoded.append([((packed//1024)-512)/511,((packed%1024)-512)/511]);expected.append(ref[e,t])
 dif=[abs(a-b) for x,y in zip(decoded,expected) for a,b in zip(x,y)]
 rows.append({'lane':e,'unique_packed_values':len(set(sensor[e,t] for t in range(201,300))),'most_common_packed':collections.Counter(sensor[e,t] for t in range(201,300)).most_common(3),'decoded_abs_mean_m_s':statistics.mean(abs(a) for x in decoded for a in x),'reconstructed_abs_mean_m_s':statistics.mean(abs(a) for x in expected for a in x),'mean_abs_difference_m_s':statistics.mean(dif),'max_abs_difference_m_s':max(dif)})
out={'interval_steps':[201,299],'rows':rows,'limits':'Independent reconstruction uses final-substep q/v, ground confirmed stationary in prior replay, and positive impulses >1e-9, matching the shader threshold in magnitude; negative impulses would remain a difference. Contact geometry and q/v have substep timing offsets; this is a measurement audit, not ground truth slip.'};(A/'sensor-audit.json').write_text(json.dumps(out,indent=2)+'\n');[print(x) for x in rows]
