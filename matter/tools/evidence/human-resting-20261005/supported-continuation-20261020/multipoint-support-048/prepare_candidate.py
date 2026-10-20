#!/usr/bin/env python3
"""Bounded048 placement from retained callback points; no per-step physics."""
import csv,itertools,json,pathlib,struct,math,sys
import numpy as np
E=pathlib.Path(__file__).resolve().parent
R=E.parents[5]
B=R/'build-integrated-validation-20261019/supported-skin-ligaments-20261020'
number=int(sys.argv[1])
assert 1<=number<=4
def qm(a,b):
 return np.r_[a[3]*b[:3]+b[3]*a[:3]+np.cross(a[:3],b[:3]),a[3]*b[3]-np.dot(a[:3],b[:3])]
def rot(q,v):return v+2*q[3]*np.cross(q[:3],v)+2*np.cross(q[:3],np.cross(q[:3],v))
def root(j):
 i=j['argv'].index('--root-pose')+1
 p=np.array(list(map(float,j['argv'][i:i+7])))
 p[3:]/=np.linalg.norm(p[3:])
 return p
baseline=json.loads((E/'observation-template.json').read_text());initial=root(baseline)
points=[];sources=[]
for k in range(number):
 d=B/('native-support-observation-048' if k==0 else 'native-support-candidate'+str(k)+'-048')
 j=json.loads((d/'invocation.json').read_text());p=root(j)
 rows=list(csv.DictReader((d/'output/resting-support-selected-skin-witness.csv').open()))
 # Initial noncommitting callback may duplicate pre-dynamics rows.
 assert len(rows)>=32 and all(rows[i]==rows[i%32] for i in range(len(rows)))
 pts=np.array([[float(x['selected_point_world_'+a+'_m']) for a in 'xyz'] for x in rows[:32]])
 assert all(int(x['pre_dynamics_step_index'])==0 and int(x['region_error'])==0 and int(x['global_error'])==0 for x in rows[:32])
 delta=qm(p[3:],np.r_[-initial[3:6],initial[6]])
 inverse=np.r_[-delta[:3],delta[3]]
 points.extend([rot(inverse,x-p[:3])+initial[:3] for x in pts])
 sources.append(str(d/'output/resting-support-selected-skin-witness.csv'))
points=np.array(points)
comrow=next(csv.DictReader((B/'native-support-observation-048/output/resting-com-momentum-diagnostic.csv').open()))
com=np.array([float(comrow['com_'+a+'_m']) for a in 'xyz'])
best=None
for ids in itertools.combinations(range(len(points)),3):
 p=points[list(ids)];A=np.c_[p[:,:2],np.ones(3)]
 if abs(np.linalg.det(A))<1e-10:continue
 a,b,c=np.linalg.solve(A,p[:,2])
 if np.min(points[:,2]-(a*points[:,0]+b*points[:,1]+c)) < -1e-8:continue
 angle=math.atan(math.hypot(a,b))
 if angle>math.radians(5):continue
 bary=np.linalg.solve(A.T,np.r_[com[:2],1.])
 if min(bary)<-1e-8:continue
 score=a*com[0]+b*com[1]+c
 if best is None or score>best['height_at_body_com_m']:
  best=dict(ids=ids,plane=[a,b,c],body_com_barycentric=bary.tolist(),angle_degrees=math.degrees(angle),height_at_body_com_m=score)
assert best is not None,'no sampled support plane within prescribed5degrees'
a,b,c=best['plane'];n=np.array([-a,-b,1.]);n/=np.linalg.norm(n)
dq=np.r_[np.cross(n,[0.,0.,1.]),1+n[2]];dq/=np.linalg.norm(dq)
relative=np.array([rot(dq,p-initial[:3]) for p in points])
newt=initial[:3].copy();newt[2]=-min(relative[:,2])
newq=qm(dq,initial[3:])
f32=lambda x:struct.unpack('<f',struct.pack('<f',float(x)))[0]
values=list(map(f32,np.r_[newt,newq]))
j=json.loads(json.dumps(baseline));ix=j['argv'].index('--root-pose')+1
j['argv'][ix:ix+7]=list(map(str,values))
template=E/('candidate-'+str(number)+'-template.json')
calculation=E/('candidate-'+str(number)+'-calculation.json')
assert not template.exists() and not calculation.exists()
template.write_text(json.dumps(j,indent=2)+'\n')
calculation.write_text(json.dumps(dict(sources=sources,points_in_baseline_world=points.tolist(),selected_lower_hull=best,new_float32_root=values,body_com_scope='articulated bodies only; excludes separately represented FEM; heuristic, not equilibrium',predicted_sample_min_gap_m=float(min(relative[:,2]+values[2])),scope='fresh full-skin callback and ordinary physical certificates decide acceptance'),indent=2)+'\n')
print('prepared',template,'root',values)
