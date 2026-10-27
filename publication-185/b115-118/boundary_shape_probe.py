from pathlib import Path
import importlib.util,sys,json,hashlib
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
C=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz'
G=S/'per-bank-workers-20261027/b115-118/attempt-005/local-contour-geometry.json'
OUT=S/'per-bank-workers-20261027/b115-118/attempt-005/boundary-shape-probe.json'
AUDIT=Path('/Users/numi/Work/numi-human-connected-surface-069/audit_skin_bone_surfaces_159.py')
with np.load(C,allow_pickle=False) as z:
 P=z['points_world_m'].astype(np.float64);F=z['source_O_faces'].astype(np.int64);R=z['area_reference_037_points_world_m_f64'].astype(np.float64)
a=json.loads(G.read_text());nodes=np.asarray(a['local_patch']['fixed_outer_boundary_nodes'],np.int64);
spec=importlib.util.spec_from_file_location('a',AUDIT);m=importlib.util.module_from_spec(spec);sys.modules['a']=m;spec.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('h',m.BANKHELPER);h=importlib.util.module_from_spec(spec);sys.modules['h']=h;spec.loader.exec_module(h)
T,ids,gids,bodies,M,off,pos,keys,pins,u=h.read_bank_inputs(m.E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight',m.E)
tri=T[(ids[:,0]==115)&(ids[:,1]==118)]
# Count exact-coordinate edge incidence to assess whether this bank mesh supports a closed-solid winding test.
edge_count={}
for t in tri:
 v=[tuple(map(float,p)) for p in t]
 for i,j in ((0,1),(1,2),(2,0)):
  e=tuple(sorted((v[i],v[j])));edge_count[e]=edge_count.get(e,0)+1
boundary=sum(n==1 for n in edge_count.values()); nonmanifold=sum(n>2 for n in edge_count.values());
# Point-to-triangle solid-angle winding number, in the same bank-local coordinates.
local=((P-pos[115])@M[115].T-off[115])*1000.0
tril=tri*1000.0
wind=[]
for row in nodes:
 p=local[row]; aa=tril[:,0]-p;bb=tril[:,1]-p;cc=tril[:,2]-p
 la=np.linalg.norm(aa,axis=1);lb=np.linalg.norm(bb,axis=1);lc=np.linalg.norm(cc,axis=1)
 num=np.einsum('ij,ij->i',aa,np.cross(bb,cc))
 den=la*lb*lc+np.einsum('ij,ij->i',aa,bb)*lc+np.einsum('ij,ij->i',bb,cc)*la+np.einsum('ij,ij->i',cc,aa)*lb
 val=float(np.sum(2*np.arctan2(num,den))/(4*np.pi))
 wind.append({'node_row':int(row),'winding':val,'inside_by_abs_winding_gt_half':bool(abs(val)>0.5)})
# local area magnitude and orientation vs current geometry (all current triangles should orient as 1 by definition).
patchrows=sorted(set(a['crossing_face_rows'] for a in []))
# Pull the exact 57 face rows from topology report.
topo=json.loads((S/'per-bank-workers-20261027/b115-118/attempt-005/local-topology-report.json').read_text());cross=set(topo['bank']['crossing_skin_face_rows']);
# reconstruct one-vertex patch
nodefaces={}
for fi,t in enumerate(F):
 for v in t:nodefaces.setdefault(int(v),[]).append(fi)
patch=set(cross)
for fi in cross:
 for v in F[fi]:patch.update(nodefaces[int(v)])
pr=np.asarray(sorted(patch),np.int64)
cur=P[F[pr]];ref=R[F[pr]]
ca=np.cross(cur[:,1]-cur[:,0],cur[:,2]-cur[:,0]);ra=np.cross(ref[:,1]-ref[:,0],ref[:,2]-ref[:,0]);
mag=np.linalg.norm(ca,axis=1)/np.maximum(np.linalg.norm(ra,axis=1),1e-30)
current_area=0.5*np.linalg.norm(ca,axis=1)*1e6
report={'schema':'numi.human.bank115_118.boundary_shape_probe.v1','candidate_sha256':hashlib.sha256(C.read_bytes()).hexdigest(),'bank_115_118_triangle_count':int(len(tri)),'bank_exact_coordinate_edge_count':len(edge_count),'bank_boundary_edge_count':int(boundary),'bank_nonmanifold_edge_count':int(nonmanifold),'winding_test_available':bool(boundary==0 and nonmanifold==0),'fixed_outer_boundary_nodes':wind,'boundary_inside_count':sum(x['inside_by_abs_winding_gt_half'] for x in wind),'boundary_min_abs_winding':min(abs(x['winding']) for x in wind),'boundary_max_abs_winding':max(abs(x['winding']) for x in wind),'one_ring_shape':{'face_count':int(len(pr)),'min_current_area_mm2':float(current_area.min()),'max_current_area_mm2':float(current_area.max()),'minimum_fixed_reference_magnitude_ratio':float(mag.min()),'faces_below_0_1_magnitude_ratio':int(np.sum(mag<0.1)),'orientation_vs_current_baseline':'1.0 by construction; candidate itself is the fixed local orientation baseline'},'limitations':['Winding is a local geometric classification only if bank mesh is closed and consistently oriented. It does not certify physical clearance or native acceptance.']}
OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
