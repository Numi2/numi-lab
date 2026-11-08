from pathlib import Path
import sys,json,xml.etree.ElementTree as ET,numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
root=Path('/Users/home/Documents/Codex/2026-10-21/i-wa');w=root/'work/plastic_bag_franka_stock_grip_20261024';a=root/'work/plastic_bag_franka/franka_assets';xml=ET.parse(a/'panda.xml').getroot();files={e.get('name',Path(e.get('file')).stem):e.get('file') for e in xml.findall('asset/mesh')};mapping={**{f'link{i}':i for i in range(8)},'hand':8,'left_finger':9,'right_finger':10};vertices={i:[] for i in range(11)}
for body in xml.findall('.//worldbody//body'):
 idx=mapping[body.get('name')]
 for geom in body.findall('geom'):
  if geom.get('class')!='visual':continue
  if idx in [9,10] and files[geom.get('mesh')]=='finger_1.obj':continue
  p=a/'assets'/files[geom.get('mesh')];v=np.array([list(map(float,l.split()[1:4])) for l in p.read_text().splitlines() if l.startswith('v ')])
  q=list(map(float,geom.get('quat','1 0 0 0').split()));v=Rotation.from_quat(q[1:]+q[:1]).apply(v)+np.array(list(map(float,geom.get('pos','0 0 0').split())));vertices[idx].append(v)
for i in [9,10]:vertices[i].append(np.array([[x,y,z] for x in [-.0085,.0085] for y in [0,.003] for z in [.0445-.0085,.0445+.0085]]))

for i in [9,10]:vertices[i].append(np.array([[x,y,z] for x in [-.002,.002] for y in [.0025,.0105] for z in [.0375,.0515]]))
for i,v in vertices.items():
 u=np.unique(np.vstack(v),axis=0);vertices[i]=u[ConvexHull(u).vertices]
fs=json.loads(Path(sys.argv[1]).read_text())['frames'];result=[];minimum=(1e9,None)
for i,v in vertices.items():
 poses=np.array([f['links'][i] for f in fs]);m=Rotation.from_quat(poses[:,3:]).as_matrix();height=np.einsum("fi,vi->fv",m[:,1,:],v)+poses[:,1,None]-.108;ix=np.unravel_index(np.argmin(height),height.shape);gap=float(height[ix]);result.append({'link':i,'minimum_vertex_table_gap_m':gap,'time_s':fs[ix[0]]['time_s']})
 if gap<minimum[0]:minimum=(gap,{'link':i,'time_s':fs[ix[0]]['time_s']})
r={'frames':len(fs),'all_visual_vertices_and_authored_rubber_pad_vertices':True,'rigid_central_pad_carriers_included':True,'convex_hull_reduction_exact_for_plane_support':True,'table_height_m':.108,'minimum_gap_m':minimum[0],'event':minimum[1],'per_link':result,'scope':'Plane clearance of authored kinematic robot visual meshes and pad boxes. Not robot self-collision or coupled motor dynamics qualification.'};print(json.dumps(r,indent=2))
