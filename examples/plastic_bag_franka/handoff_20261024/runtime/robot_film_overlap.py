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
# Rigid backing is inside the original stock rubber volume. Its front
# overlaps the 3 mm pad and its back overlaps the white finger housing.
# Include it in the non-pad crossing audit; only the compliant faces are exempt.
for idx in [9,10]:
 vs=[Vector((x,y,z)) for x in [-.002,.002] for y in [.0025,.0105] for z in [.0375,.0515]]
 faces=[(0,4,6),(0,6,2),(1,3,7),(1,7,5),(0,1,5),(0,5,4),(2,6,7),(2,7,3),(0,2,3),(0,3,1),(4,5,7),(4,7,6)]
 meshes.append((idx,vs,faces))
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
def build_static(pad_z,carrier_recess=0):
 groups={}
 for mesh_number,(idx,vs,faces) in enumerate(meshes):
  vertices,triangles=groups.setdefault(idx,([],[]));off=len(vertices)
  if mesh_number>=len(meshes)-2:vs=[v+Vector((0,0,pad_z-.0445-(carrier_recess if v.z>.0445 else 0))) for v in vs]
  vertices.extend(vs);triangles.extend(tuple(off+i for i in face) for face in faces)
 result={}
 for idx,(vertices,triangles) in groups.items():
  va=np.asarray(vertices,dtype=float);lo=va.min(axis=0);hi=va.max(axis=0)
  corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
  result[idx]=(va,triangles,BVHTree.FromPolygons(vertices,triangles,all_triangles=True,epsilon=1e-7),corners)
 return result
static=build_static(.0445)
static2=build_static(fs[0].get('robot2',{}).get('pad_center_local_z_m',.0445),fs[0].get('robot2',{}).get('carrier_front_recess_m',0))
events=[];narrow_links=0
for f in fs:
 world=np.array(f['front']+f['back'],dtype=float);blo=world.min(axis=0);bhi=world.max(axis=0);hits=[]
 for arm in ['robot','robot2']:
  if arm not in f:continue
  for idx,(rva,rf,r,corners) in (static2 if arm=='robot2' else static).items():
   pose=f[arm]['links'][idx];q=Quaternion((pose[6],*pose[3:6]));rotation=np.asarray(q.to_matrix(),dtype=float);pos=np.array(pose[:3])
   bounds=corners@rotation.T+pos
   if np.any(bounds.max(axis=0)<blo-1e-6) or np.any(bounds.min(axis=0)>bhi+1e-6):continue
   # Query the film in a link's rigid local frame. Authored mesh BVHs are
   # immutable and reused, rather than rebuilding every large mesh per state.
   narrow_links+=1;bva=(world-pos)@rotation
   bvh=BVHTree.FromPolygons([Vector(v) for v in bva],bagfaces,all_triangles=True,epsilon=1e-7)
   for bi,ri in bvh.overlap(r):
    bt=bva[list(bagfaces[bi])];rt=rva[list(rf[ri])]
    if crossing(bt,rt) or crossing(rt,bt):hits.append((bi,ri,idx,arm))
 if hits:events.append({'step':f['step'],'time_s':f['time_s'],'intersecting_triangle_pairs':len(hits),'links':sorted(set((h[3],h[2]) for h in hits))})
print('ROBOT_FILM_AUDIT '+json.dumps({'states':len(fs),'through_s':fs[-1]['time_s'],'intersection_state_count':len(events),'events':events[:30],'narrow_phase_links':narrow_links,'scope':'Native saved film against authored robot visual meshes; excludes the compliant rubber contact patches and substep events.'}))
