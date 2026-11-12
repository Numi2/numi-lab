from pathlib import Path
import collections,gzip,json,math,os,statistics,subprocess
A=Path('artifacts/crow-v15-corvid-ankle-fine-screen-20261112')
D=Path('artifacts/crow-v15-corvid-ankle-fine-screen-20261112/candidate-20261127-rebound-fine')
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
records=[];rejected=collections.Counter()
for e in range(8):
 root=[q[e,t,k] for t in range(201,301) for k in range(16)]
 for side,sd in enumerate(('left','right')):
  load=[]
  for t in range(201,301):
   for k in range(16):
    z=contact[e,t,k];load.append(z[2+side]>1e-7)
  active=[i for i,val in enumerate(load) if val];groups=[]
  for i in active:
   if not groups or i-groups[-1][-1]>3:groups.append([i])
   else:groups[-1].append(i)
  for group in groups:
   start,end=group[0],group[-1]
   if start==0 or end==len(load)-1:rejected['partial']+=1;continue
   if end-start+1<32:rejected['under_40ms']+=1;continue
   hip=[];knee=[];ankle=[]
   for i in range(start,end+1):
    thigh=b[e,i,'crow_'+sd+'_thigh'];sh=b[e,i,'crow_'+sd+'_shank'];foot=b[e,i,'crow_'+sd+'_foot'];hip_joint=point(thigh,[0,0,.031625]);kn=point(thigh,[0,0,-.031625]);an=point(sh,[0,0,-.031625]);foot_end=point(foot,[.02875,0,.0092]);r=root[i];femur=rot([-r[3],-r[4],-r[5],r[6]],[kn[j]-hip_joint[j] for j in range(3)])
    hip.append(math.degrees(math.atan2(femur[0],-femur[2])));knee.append(angle(hip_joint,kn,an));ankle.append(angle(kn,an,foot_end))
   def quintiles(x):
    return [statistics.mean(x[int(len(x)*a):max(int(len(x)*bb),int(len(x)*a)+1)]) for a,bb in ((0,.2),(.4,.6),(.8,1.))]
   hp,kp,ap=map(quintiles,(hip,knee,ankle));records.append({'lane':e,'side':sd,'start_index':start,'end_index':end,'seconds':(end-start+1)*.00125,'hip_forward_from_down_deg_early_mid_late':hp,'knee_interior_deg_early_mid_late':kp,'ankle_interior_deg_early_mid_late':ap,'knee_q_rad_early_mid_late':quintiles([root[i][15 if side==0 else 18] for i in range(start,end+1)]),'ankle_q_rad_early_mid_late':quintiles([root[i][16 if side==0 else 19] for i in range(start,end+1)]),'hip_change_deg':hp[2]-hp[0],'knee_change_deg':kp[2]-kp[0],'ankle_mid_minus_early_deg':ap[1]-ap[0],'ankle_late_minus_mid_deg':ap[2]-ap[1]})
result={'seed':20261127,'window_seconds':[4,6],'substep_seconds':.00125,'contact_threshold_ns':1e-7,'records':records,'rejections':dict(rejected),'scope':'Modeled v15 physical segment geometry and normal-load contact segmentation, eight closely related development starts. Hip forward angle decreasing is the chosen extension proxy, and increasing interior angles are extension proxies for knee/ankle. Segment and joint anatomy, 20 percent phase averaging, tiny contact thresholds and group merging differ from the magpie study. No direct biological validation.'}
(A/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print('stances',len(records),'rejected',dict(rejected))
for k in ('hip_change_deg','knee_change_deg','ankle_mid_minus_early_deg','ankle_late_minus_mid_deg'):
 vals=[r[k] for r in records];print(k,'median',statistics.median(vals),'range',min(vals),max(vals),'negative',sum(v<0 for v in vals),'positive',sum(v>0 for v in vals))
