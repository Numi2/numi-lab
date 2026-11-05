from pathlib import Path
import numpy as np,json,hashlib,itertools
w=Path('/Users/numi/Work/human-integrated-gaps-20261104')
b=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
source=b/'current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz'
cavity=b/'regional-compensation-1061-root/memory-performance-123/rejection-recovery-125/connected-conditioning-134/fixed-boundary-volume-proposal.npz'
outer=b/'all-quality-937-b5-backtracking-root/proposal/candidate-O-typed-current.npz'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
with np.load(source,allow_pickle=False) as z:
 p=z['points_reference_world_m_f64'].astype(np.float32)
 old=z['volume_tetrahedra']
 role=z['current_typed_role_source_row_to_joint_node_O']
with np.load(cavity,allow_pickle=False) as z:
 assert str(z['source_volume_sha256'].item())==sha(source)
 retained=z['retained_input_cell_rows'];replaced=z['replacement_tetrahedra']
 t=np.concatenate([old[retained],replaced])
with np.load(outer,allow_pickle=False) as z:
 raw=z['faces'];f=role[raw]
 f=f[np.all(f>=0,axis=1)]
def det(cells):
 q=p[cells].astype(np.float64)
 return np.einsum('ij,ij->i',np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,3]-q[:,0])
def link(cells,vertices):
 vertices=set(vertices);result=set()
 for cell in cells:
  if not vertices<=set(map(int,cell)):continue
  rest=sorted(set(map(int,cell))-vertices)
  for k in range(len(rest)+1):result.update(itertools.combinations(rest,k))
 return result
u,v=22677,22679
incident=np.flatnonzero(np.any((t==u)|(t==v),axis=1));cells=t[incident]
faces=f[np.any((f==u)|(f==v),axis=1)]
base=det(cells);out=[]
for removed,kept in [(u,v),(v,u)]:
 new=cells.copy();new[new==removed]=kept
 repeated=np.any(np.diff(np.sort(new,axis=1),axis=1)==0,axis=1)
 d=det(new[~repeated])
 newfaces=faces.copy();newfaces[newfaces==removed]=kept
 sfkeep=~np.any(np.diff(np.sort(newfaces,axis=1),axis=1)==0,axis=1)
 out.append(dict(removed_joint_node=removed,kept_joint_node=kept,
 edge_length_m=float(np.linalg.norm(p[removed].astype(float)-p[kept])),
 incident_tets=len(cells),baseline_nonpositive=int(np.count_nonzero(base<=0)),
 baseline_min_det_m3=float(base.min()),removed_tets=int(repeated.sum()),
 candidate_nonpositive=int(np.count_nonzero(d<=0)),
 candidate_min_det_m3=float(d.min()),removed_boundary_faces=int((~sfkeep).sum()),
 volume_link_condition=(link(cells,[u])&link(cells,[v]))==link(cells,[u,v]),
 surface_link_condition=(link(faces,[u])&link(faces,[v]))==link(faces,[u,v]),
 before_cavity_volume_m3=float(base.sum()/6),after_cavity_volume_m3=float(d.sum()/6),
 changed_volume_m3=float((d.sum()-base.sum())/6),
 nonpositive_retained_input_cell_rows=incident[~repeated][d<=0].tolist()))
report=dict(scope='Feasibility only; no source, runtime geometry, force, contact or mass mapping changed',
 source={'path':str(source),'sha256':sha(source)},cavity={'path':str(cavity),'sha256':sha(cavity)},
 mesh_tets=len(t),trials=out)
(w/'connected-fold-collapse-feasibility-62.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
