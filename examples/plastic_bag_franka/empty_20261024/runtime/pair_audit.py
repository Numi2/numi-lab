import json,sys,math,numpy as np
from pathlib import Path
R=.015305;h=.001075;theta=np.arange(128)*(2*np.pi/128);rings=[]
for y in [-h,-h/3,0,h/3,h]:rings.extend(np.stack((R*np.cos(theta),theta*0+y,R*np.sin(theta)),axis=1))
for y in [-h,h]:
 for radius in [.33*R,.66*R,.9*R]:rings.extend(np.stack((radius*np.cos(theta),theta*0+y,radius*np.sin(theta)),axis=1))
for y in [-h/2,0,h/2]:rings.extend(np.stack((.5*R*np.cos(theta),theta*0+y,.5*R*np.sin(theta)),axis=1))
rings.extend([[0,-h,0],[0,h,0],[0,0,0]]);samples=np.array(rings)
def rot(q,p):return p+2*q[3]*np.cross(q[:3],p)+2*np.cross(q[:3],np.cross(q[:3],p))
def sdf(p):
 radial=np.hypot(p[:,0],p[:,2])-R;axial=abs(p[:,1])-h
 return np.hypot(np.maximum(radial,0),np.maximum(axial,0))+np.minimum(np.maximum(radial,axial),0)
f=json.loads(Path(sys.argv[1]).read_text())['frames'];worst=0;event=None;queries=0
for frame in f:
 p=np.array(frame['coins']);q=np.array(frame['coin_orientations']);q/=np.linalg.norm(q,axis=1,keepdims=True);active=frame.get('coin_active',[True]*len(p));world=[rot(q[i],samples)+p[i] for i in range(10)]
 for i in range(10):
  if not active[i]:continue
  for j in range(i+1,10):
   if not active[j] or np.linalg.norm(p[i]-p[j])>2*math.hypot(R,h)+1e-5:continue
   qi=q[i].copy();qi[:3]*=-1;qj=q[j].copy();qj[:3]*=-1
   depth=max(0.,-float(sdf(rot(qj,world[i]-p[j])).min()),-float(sdf(rot(qi,world[j]-p[i])).min()));queries+=1
   if depth>worst:worst=depth;event={'time_s':frame['time_s'],'pair':[i,j]}
print(json.dumps({'max_boundary_penetration_mm':worst*1000,'event':event,'samples_per_coin':len(samples),'pair_queries':queries,'states':len(f),'method':'dense finite-cylinder surface and interior witnesses, both directions, normalized quaternions; positive sampled penetration proves overlap, no samples does not prove full nonintersection'},indent=2))
