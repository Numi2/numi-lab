from pathlib import Path
import gzip,json,math,collections
A=Path('artifacts/crow-v15-toe-slip-velocity-20261114');P=Path('artifacts/crow-v15-toe-friction-observer-20261114/trace.log.gz')
kinds=('step','control','event','hipcommand','jointtarget','rearKnee','rearAnkle','rightHipCommand','balance','pitchcontrol','stage','pose','contact','stancetarget','pitchSupport')
def load(p):
 d={}
 with gzip.open(p,'rt') as f:
  for line in f:
   z=line.split()
   if z and z[0] in kinds:d.setdefault((int(z[1]),int(z[2]),z[0]),[]).append(z[3:])
 return d
cur=load(A/'trace.log.gz');old=load(P)
prior=[t for t in range(450) if any(cur.get((0,t,k))!=old.get((0,t,k)) for k in kinds)]
paired=[t for t in range(450) if any(cur.get((0,t,k))!=cur.get((1,t,k)) for k in kinds)]
body={}
with (A/'kinematics-output.txt').open() as f:
 for line in f:
  z=line.split();assert z[0]=='body' and len(z)==14
  e,t,s,i=map(int,z[1:5]);key=e,t,s,i;assert key not in body;body[key]=list(map(float,z[5:]))
assert len(body)==2520
names={11:'crow_left_foot',15:'crow_right_foot',27:'crow_right_front_toe_1_proximal',28:'crow_right_front_toe_1_distal',29:'crow_right_front_toe_2_proximal',30:'crow_right_front_toe_2_distal',31:'crow_right_front_toe_3_proximal',32:'crow_right_front_toe_3_distal',33:'crow_right_rear_toe_proximal',34:'crow_right_rear_toe_distal'}
poseerrors=[]
for e in (0,1):
 for t in range(290,311):
  poses={r[0]:list(map(float,r[1:4])) for r in cur[e,t-1,'pose']}
  for i,name in names.items():poseerrors.append(math.dist(poses[name],body[e,t,31,i][:3]))
fd=[]
for e in (0,1):
 for t in range(290,311):
  for s in (0,15,30):
   for i in names:
    a=body[e,t,s,i];b=body[e,t,s+1,i];vel=[(b[j]-a[j])/(.02/32) for j in range(3)];fd.append(math.dist(vel,b[3:6]))
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def norm(a):return math.sqrt(sum(x*x for x in a))
def pct(v,p):
 v=sorted(v);return v[min(len(v)-1,int(p*(len(v)-1)))] if v else None
contacts=[];errors=[];skipped_unselected=0
with gzip.open(A/'substeps.log.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0] in ('CROW_SUBSTEP_ERROR','failure'):errors.append(line.strip())
  if z[0]!='CROW_SLIP_CONTACT':continue
  assert len(z)==16;t,s,e,i=map(int,z[1:5]);x=list(map(float,z[5:]));point=x[:3];n=x[3:6];jn,ju,jv,ms,md=x[6:]
  if i not in names:skipped_unselected+=1;continue
  assert (e,t,s,i) in body and jn>0 and abs(norm(n)-1)<1e-4
  a=body[e,t,s,i];r=[point[j]-a[j] for j in range(3)];omega_cross=cross(a[6:9],r);v=[a[3+j]+omega_cross[j] for j in range(3)];vn=sum(v[j]*n[j] for j in range(3));speed=norm([v[j]-vn*n[j] for j in range(3)]);ratio=math.hypot(ju,jv)/(md*jn) if md>0 else None
  contacts.append((e,t,s,i,jn,speed,ratio))
def group(i):return 'right_foot' if i==15 else 'right_front' if 27<=i<=32 else 'right_rear' if 33<=i<=34 else 'left_foot'
rows=[]
for e in (0,1):
 for lo,hi in ((290,300),(300,305),(305,311)):
  for g in ('right_foot','right_front','right_rear'):
   c=[x for x in contacts if x[0]==e and lo<=x[1]<hi and group(x[3])==g and x[4]>=1e-6 and x[6] is not None];v=[x[5] for x in c]
   rows.append({'lane':e,'intervals':[lo,hi],'group':g,'samples':len(c),'speed_m_s_p50':pct(v,.5),'speed_m_s_p95':pct(v,.95),'fraction_speed_ge_0_001':sum(x>=.001 for x in v)/len(v) if v else None,'fraction_speed_ge_0_01':sum(x>=.01 for x in v)/len(v) if v else None,'fraction_speed_ge_0_01_and_dynamic_ratio_ge_0_95':sum(x[5]>=.01 and x[6]>=.95 for x in c)/len(c) if c else None})
out={'trace_prior_first_difference':next(iter(prior),None),'trace_paired_first_difference':next(iter(paired),None),'qv_samples':252,'body_samples':len(body),'selected_contact_samples':len(contacts),'unselected_left_toe_samples':skipped_unselected,'pose_position_error_max_m':max(poseerrors),'finite_difference_velocity_error_m_s_p50':pct(fd,.5),'finite_difference_velocity_error_m_s_p95':pct(fd,.95),'finite_difference_velocity_error_m_s_max':max(fd),'errors':errors,'rows':rows,'limitations':'Post-solve generalized state is paired with solver contact geometry; exact pre-solve contact point timing is not established. Contact rows within lanes and steps are correlated. This estimates relative motion against stationary ground in the constructed model, not biological friction.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
