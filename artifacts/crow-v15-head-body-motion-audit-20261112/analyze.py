from pathlib import Path
import collections,gzip,json,math,os,statistics,subprocess
A=Path('artifacts/crow-v15-head-body-motion-audit-20261112')
D=Path('artifacts/crow-v15-substep-slip-observer-20261112/candidate-20261127-rebound-low')
q={};v={};contact={}
with gzip.open(D/'run.stderr.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0] in ('CROW_Q_SUBSTEP','CROW_V_SUBSTEP'):
   t,k,e=map(int,z[1:4]);key=e,t,k;d=q if z[0]=='CROW_Q_SUBSTEP' else v;assert key not in d;d[key]=list(map(float,z[4:]))
  elif z[0]=='CROW_TWO_TOE_SUBSTEP':
   t,k,e=map(int,z[1:4])
   if 201<=t<=300:
    key=e,t,k;assert key not in contact;contact[key]=list(map(float,z[6:]));assert len(contact[key])==37
assert len(q)==len(v)==len(contact)==12800 and q.keys()==v.keys()==contact.keys()
inp=''.join(str(e)+' '+str((t-201)*16+k+1)+' '+' '.join(map(str,q[e,t,k]+v[e,t,k]))+'\n' for e,t,k in sorted(q))
env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())}
p=subprocess.run(['build/crow-v15-gait-slip-observer-20261112/kinematics'],input=inp,text=True,capture_output=True,check=True,env=env)
needed={f'crow_{side}_{name}' for side in ('left','right') for name in ('thigh','shank','foot','front_toe_2_proximal')}|{'crow_head'}
b={}
for line in p.stdout.splitlines():
 z=line.split()
 if z and z[0]=='body' and z[3] in needed:b[int(z[1]),int(z[2])-1,z[3]]=list(map(float,z[4:]))
assert len(b)==8*1600*9
with gzip.open(D/'replay.json.gz','rt') as f:r=json.load(f)['payload']
names=r['body_names'];err=0.
for fr in r['frames']:
 t=fr['step']+1
 if not 201<=t<=300:continue
 i=(t-201)*16+15
 for name in needed:
  idx=names.index(name)*13;raw=fr['body_states'][idx:idx+13];calc=b[5,i,name]
  err=max(err,math.dist(raw[:3],calc[:3]))
assert err<1e-5
def rot(q,v):
 x,y,z,w=q;u=(x,y,z);d=sum(u[i]*v[i] for i in range(3));c=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);s=w*w-sum(t*t for t in u);return [s*v[i]+2*d*u[i]+2*w*c[i] for i in range(3)]
def point(body,local):
 z=rot(body[3:7],local);return [z[i]+body[i] for i in range(3)]
def angle(h,k,a):
 x=[h[i]-k[i] for i in range(3)];y=[a[i]-k[i] for i in range(3)];d=sum(x[i]*y[i] for i in range(3))/(math.sqrt(sum(i*i for i in x))*math.sqrt(sum(i*i for i in y)));return math.degrees(math.acos(max(-1,min(1,d))))
def angles(q):
 x,y,z,w=q;return [math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),math.asin(max(-1,min(1,2*(w*y-z*x)))),math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))]
def corr(a,b):
 ma=statistics.mean(a);mb=statistics.mean(b);aa=[x-ma for x in a];bb=[x-mb for x in b];d=math.sqrt(sum(x*x for x in aa)*sum(x*x for x in bb));return sum(x*y for x,y in zip(aa,bb))/d if d else None
rows=[]
for e in range(8):
 root=[q[e,t,k] for t in range(201,301) for k in range(16)];ra=[angles(x[3:7]) for x in root];ha=[angles(b[e,i,'crow_head'][3:7]) for i in range(len(root))]
 axes=[]
 for j,name in enumerate(('roll','pitch','yaw')):
  rv=[x[j] for x in ra];hv=[x[j] for x in ha];rel=[math.atan2(math.sin(h-r),math.cos(h-r)) for h,r in zip(hv,rv)]
  axes.append({'axis':name,'torso_world_span_rad':max(rv)-min(rv),'head_world_span_rad':max(hv)-min(hv),'relative_angle_span_rad':max(rel)-min(rel),'head_torso_correlation':corr(rv,hv)})
 rows.append({'lane':e,'axes':axes})
out={'rows':rows,'scope':'Passive 800 Hz mechanical reconstruction, one used development seed over 4–6 seconds. Euler angle spans are small-angle diagnostics, not direct eye-gaze or crow head-stabilization measurement. No visual or vestibular behavior target is implied.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print(r)
