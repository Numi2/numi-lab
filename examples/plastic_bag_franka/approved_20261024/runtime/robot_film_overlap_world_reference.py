"""Saved-state triangle intersections, robot presentation meshes against film."""
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
from mathutils import Vector,Quaternion,Matrix
from mathutils.bvhtree import BVHTree
import numpy as np
args=sys.argv[sys.argv.index('--')+1:];fs=json.loads(Path(args[0]).read_text())['frames'];a=Path('/Users/home/Documents/Codex/2026-10-21/i-wa/work/plastic_bag_franka/franka_assets');xml=ET.parse(a/'panda.xml').getroot();files={e.get('name',Path(e.get('file')).stem):e.get('file') for e in xml.findall('asset/mesh')};mapping={**{f'link{i}':i for i in range(8)},'hand':8,'left_finger':9,'right_finger':10}
meshes=[]
for body in xml.findall('.//worldbody//body'):
 idx=mapping[body.get('name')]
 for geom in body.findall('geom'):
  if geom.get('class')!='visual':continue
  path=a/'assets'/files[geom.get('mesh')]
  if idx in [9,10] and path.stem=='finger_1':continue
  vs=[];faces=[]
  for line in path.read_text().splitlines():
   if line.startswith('v '):vs.append(Vector(tuple(map(float,line.split()[1:4]))))
   elif line.startswith('f '):
    f=[int(x.split('/')[0]) for x in line.split()[1:]];f=[i-1 if i>0 else len(vs)+i for i in f]
    faces.extend((f[0],f[j],f[j+1]) for j in range(1,len(f)-1))
  pos=Vector(tuple(map(float,geom.get('pos','0 0 0').split())));q=Quaternion(tuple(map(float,geom.get('quat','1 0 0 0').split())))
  meshes.append((idx,[q@v+pos for v in vs],faces))
nu=25;nv=23;bagfaces=[]
for panel in range(2):
 off=panel*nu*nv
 for j in range(nv-1):
  for i in range(nu-1):
   v=off+j*nu+i;bagfaces.extend([(v,v+1,v+nu+1),(v,v+nu+1,v+nu)])
def crossing(tri,segs):
 # Three segment-triangle queries, including edge/vertex contacts, FP64.
 e1=tri[1]-tri[0];e2=tri[2]-tri[0];p0=segs;direction=np.roll(segs,-1,axis=0)-segs
 h=np.cross(direction,e2);det=h@e1;valid=abs(det)>1e-16;inv=np.zeros(3);inv[valid]=1/det[valid]
 s=p0-tri[0];u=np.sum(s*h,axis=-1)*inv;q=np.cross(s,e1);v=np.sum(direction*q,axis=-1)*inv;t=(q@e2)*inv
 return bool(np.any(valid&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(t>=-1e-8)&(t<=1+1e-8)))
events=[]
for f in fs:
 rv=[];rf=[];rl=[]
 for idx,vs,faces in meshes:
  pose=f['robot']['links'][idx];q=Quaternion((pose[6],*pose[3:6]));pos=Vector(pose[:3]);off=len(rv);rv.extend(q@v+pos for v in vs);rf.extend(tuple(off+i for i in face) for face in faces);rl.extend([idx]*len(faces))
 bv=[Vector(v) for v in f['front']+f['back']];bvh=BVHTree.FromPolygons(bv,bagfaces,all_triangles=True);r=BVHTree.FromPolygons(rv,rf,all_triangles=True);pairs=bvh.overlap(r)
 rva=np.array(rv);bva=np.array(bv);hits=[]
 for bi,ri in pairs:
  bt=bva[list(bagfaces[bi])];rt=rva[list(rf[ri])]
  if crossing(bt,rt) or crossing(rt,bt):hits.append((bi,ri,rl[ri]))
 if hits:events.append({'step':f['step'],'time_s':f['time_s'],'intersecting_triangle_pairs':len(hits),'links':sorted(set(h[2] for h in hits))})
print('ROBOT_FILM_AUDIT '+json.dumps({'states':len(fs),'through_s':fs[-1]['time_s'],'intersection_state_count':len(events),'events':events[:30],'scope':'Native saved film against authored robot visual meshes; excludes the compliant rubber contact patches and substep events.'}))
