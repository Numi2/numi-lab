from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-toe-slip-velocity-20261114');body={}
with (A/'kinematics-orientation-output.txt').open() as f:
 for line in f:
  z=line.split();assert z[0]=='body' and len(z)==18
  e,t,s,i=map(int,z[1:5]);body[e,t,s,i]=list(map(float,z[5:]))
assert len(body)==2520
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def rot(q,v):
 a=cross(q[:3],v);b=cross(q[:3],a);return [v[j]+2*(q[3]*a[j]+b[j]) for j in range(3)]
def norm(x):return math.sqrt(sum(a*a for a in x))
def pct(x,p):
 x=sorted(x);return x[min(len(x)-1,int((len(x)-1)*p))] if x else None
contacts=[]
with gzip.open(A/'substeps.log.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z or z[0]!='CROW_SLIP_CONTACT':continue
  t,s,e,i=map(int,z[1:5])
  if s not in (0,15,30) or i not in (15,27,28,29,30,31,32,33,34):continue
  x=list(map(float,z[5:]));point=x[:3];n=x[3:6];jn,ju,jv,ms,md=x[6:]
  if jn<1e-6 or md<=0:continue
  a=body[e,t,s,i];b=body[e,t,s+1,i];qa=a[3:7];qb=b[3:7];r=[point[j]-a[j] for j in range(3)];local=rot([-qa[0],-qa[1],-qa[2],qa[3]],r);r2=rot(qb,local);p2=[b[j]+r2[j] for j in range(3)];v=[(p2[j]-point[j])/(.02/32) for j in range(3)];vn=sum(v[j]*n[j] for j in range(3));speed=norm([v[j]-vn*n[j] for j in range(3)]);ratio=math.hypot(ju,jv)/(md*jn);contacts.append((e,t,i,speed,ratio))
rows=[]
for e in (0,1):
 for lo,hi in ((290,300),(300,305),(305,311)):
  for group,ids in [('foot',(15,)),('foretoe',tuple(range(27,33))),('rear_toe',(33,34))]:
   c=[x for x in contacts if x[0]==e and lo<=x[1]<hi and x[2] in ids];v=[x[3] for x in c]
   rows.append({'lane':e,'intervals':[lo,hi],'group':group,'contact_points':len(c),'tangent_displacement_speed_m_s_p50':pct(v,.5),'tangent_displacement_speed_m_s_p95':pct(v,.95),'fraction_ge_0_01':sum(x>=.01 for x in v)/len(v) if v else None,'fraction_ge_0_01_and_dynamic_ratio_ge_0_95':sum(x[3]>=.01 and x[4]>=.95 for x in c)/len(c) if c else None})
out={'rows':rows,'scope':'Average material-point displacement from consecutive substep endpoint poses, using contact geometry at the first endpoint. Contacts need not persist through the next substep; CCD event timing and cone impulse association are not independently established. Samples and lanes are correlated.'}
(A/'material-point-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
