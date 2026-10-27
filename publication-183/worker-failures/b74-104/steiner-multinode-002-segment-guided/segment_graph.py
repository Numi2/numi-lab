import sys,json,hashlib,importlib.util
from pathlib import Path
from fractions import Fraction
import numpy as np
from collections import Counter
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
S=ROOT/'D/surface-clearance-546/component-sat-selector-638'
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
helper_path=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
exact_path=E/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py'
sys.path.insert(0,str(E/'incremental-exact-audit-owner-253/src'))
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
ex=module('numilab_human.cardiac_cavity_intersections',exact_path)
h=module('bankhelper',helper_path)
C=ROOT/'D/publication-182-stage-001/combined-653/candidate.npz'
Z=np.load(C,allow_pickle=False);X=np.asarray(Z['points_world_m'],np.float32);F=np.asarray(Z['source_O_faces'],np.int64)
W=json.load(open(S/'per-bank-workers-20261027/b74-104/baseline-witnesses.json'))
boundary=json.load(open(S/'per-bank-workers-20261027/b74-104/steiner-multinode-001/boundary-feasibility.json'))
old_rows=list(map(int,W['skin_child_faces']));loop=[int(x['node']) for x in boundary['fixed_boundary_vertices']]
EVID=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
T,ids,gids,bodies,M,off,pos,keys,pins,_=h.read_bank_inputs(EVID,E)
mask=(ids[:,0]==74)&(ids[:,1]==104);B=T[mask];bid=ids[mask,3];R=M[74];O=off[74];P=pos[74]
to_local=lambda v:(np.asarray(v,dtype=np.float64)-P)@R.T-O
base_local=to_local(X);skinF=F[old_rows]
def records(v,faces,face_ids):
 out=[]
 for fid,face in zip(face_ids,faces):
  tri=tuple(tuple(Fraction.from_float(float(x)) for x in v[int(k)]) for k in face)
  out.append((tri,tuple(min(q[a] for q in tri) for a in range(3)),tuple(max(q[a] for q in tri) for a in range(3)),int(fid),tuple(map(int,face))))
 return out
skinrec=records(base_local,skinF,old_rows)
bankrec=records(B.reshape(-1,3),np.arange(3*len(B),dtype=np.int64).reshape(-1,3),bid)
check=ex._audit_pair(skinrec,bankrec,same_surface=False)
by_bone={int(x):i for i,x in enumerate(bid)};point_ids={};points=[];edges=[];pairdata=[]
for skinrow,boneid in check['triangle_pairs']:
 si=old_rows.index(int(skinrow));s=base_local[skinF[si]];b=B[by_bone[int(boneid)]]
 raw=ex.triangle_intersection_points(tuple(tuple(Fraction.from_float(float(v)) for v in q) for q in s),tuple(tuple(Fraction.from_float(float(v)) for v in q) for q in b))
 uniq=[]
 for p in raw:
  if p not in uniq:uniq.append(p)
 if len(uniq)>=2:
  pids=[]
  for p in uniq[:2]:
   key=tuple(p)
   if key not in point_ids:point_ids[key]=len(points);points.append(np.asarray([float(v) for v in p],np.float64))
   pids.append(point_ids[key])
  if pids[0]!=pids[1]:edges.append((pids[0],pids[1]))
  pairdata.append({'skin_face_row':int(skinrow),'bone_triangle_row':int(boneid),'point_ids':pids,'segment_length_m':float(np.linalg.norm(points[pids[1]]-points[pids[0]]))})
uniqedges=sorted(set(tuple(sorted(e)) for e in edges));deg=Counter(k for e in uniqedges for k in e)
node_bary={}
for fr in old_rows:
 for n,b in zip(F[fr],Z['refinement_face_parent_barycentric'][fr]):node_bary[int(n)]=np.asarray(b,float)
corners=[next(n for n,b in node_bary.items() if np.array_equal(b,np.eye(3)[i])) for i in range(3)]
corner_xyz=base_local[corners];A=np.column_stack([corner_xyz[1]-corner_xyz[0],corner_xyz[2]-corner_xyz[0]])
proj=[];errs=[]
for p in points:
 uv=np.linalg.lstsq(A,p-corner_xyz[0],rcond=None)[0];q=corner_xyz[0]+A@uv
 proj.append([1-uv.sum(),uv[0],uv[1]]);errs.append(float(np.linalg.norm(p-q)))
mods={}
for name in ('triangle','shapely','meshpy.triangle'):
 try:mods[name]=bool(importlib.util.find_spec(name))
 except Exception:mods[name]=False
report={'schema':'numi.human.per_bank_worker_segment_graph.v1','candidate_sha256':hashlib.sha256(C.read_bytes()).hexdigest(),'body':74,'stable_bone_id':104,'raw_crossing_pairs':int(check['count']),'segment_count':len(uniqedges),'unique_exact_intersection_points':len(points),'segment_degrees':{str(k):int(v) for k,v in sorted(deg.items())},'degree_histogram':{str(k):list(deg.values()).count(k) for k in sorted(set(deg.values()))},'open_endpoints':[k for k,v in deg.items() if v==1],'branch_points':[k for k,v in deg.items() if v>2],'source_face_corners':corners,'max_projection_residual_m':max(errs,default=0.0),'intersection_points_barycentric':proj,'segments_point_indices':[list(e) for e in uniqedges],'segment_crossing_pairs':pairdata,'triangulation_packages':mods,'bank_input_pins':pins}
out=S/'per-bank-workers-20261027/b74-104/steiner-multinode-002-segment-guided'
(out/'segment-graph-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('intersection_points_barycentric','segment_crossing_pairs','segments_point_indices','bank_input_pins')},indent=2))

