"""Bounded connected rest-geometry proposal. No runtime forces or acceptance authority."""
from pathlib import Path
import numpy as np,json,time,hashlib
from scipy.spatial import cKDTree
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def signed_det(x,t):
 a=x[t];return np.linalg.det(np.transpose(a[:,1:]-a[:,:1],(0,2,1)))
def face_normals(x,f):return np.cross(x[f[:,1]]-x[f[:,0]],x[f[:,2]]-x[f[:,0]])
def smooth_velocity(x,knots,residual,radius):
 tree=cKDTree(knots);neighbors=tree.query_ball_point(x,radius)
 counts=np.fromiter(map(len,neighbors),dtype=np.int64,count=len(x))
 rows=np.repeat(np.arange(len(x)),counts)
 cols=np.fromiter((j for r in neighbors for j in r),dtype=np.int64,count=int(counts.sum()))
 v=np.zeros_like(x)
 if len(rows):
  r=np.linalg.norm(x[rows]-knots[cols],axis=1)/radius
  w=(1-r)**4*(4*r+1)
  denom=np.full(len(x),.1);np.add.at(denom,rows,w)
  np.add.at(v,rows,w[:,None]*residual[cols]);v/=denom[:,None]
 return v
def main():
 begun=time.monotonic();out=D/'run-002';out.mkdir(exist_ok=False)
 mp=E/'E/connected-skin-mechanical-transfer-013/boundary-refinement-002/candidate.npz'
 rp=E/'connected-skin-ray-envelope-016/run-001/outer-envelope-proposal.npz'
 pp=E/'global-skin-crossing-patches-012/connected-physical-O-patches-012.npz'
 m=dict(np.load(mp));r=dict(np.load(rp));p=dict(np.load(pp))
 old=m['points_world_m'].astype(float);x=old.copy();F=m['physical_outer_faces'];T=m['tetrahedra'];n0=face_normals(old,F)
 # Detailed topology/source IDs remain untouched. These targets are geometry construction constraints.
 knots=np.flatnonzero(r['ray_lower_bounds_m']>0)
 delta=r['node_displacements_m'][knots];target=old[knots]+delta
 selected_parent=set(map(int,p['patch_face_ids']))
 active_faces=np.array([int(fi) in selected_parent for fi in m['physical_O_parent_face_rows']])
 active_nodes=np.unique(F[active_faces]);protected=np.setdiff1d(m['physical_outer_node_ids'],active_nodes)
 history=[];clock=0.;stopped=None
 for iteration in range(80):
  residual=target-x[knots];v=smooth_velocity(x,x[knots],residual,.012);v[protected]=0
  dt=min(.1,2.-clock)
  accepted=False
  while dt>=1e-5:
   proposal=(x+dt*v).astype(np.float32).astype(float);proposal[protected]=old[protected]
   det=signed_det(proposal,T);n=face_normals(proposal,F)
   orientation=np.einsum('ij,ij->i',n,n0)/(np.linalg.norm(n,axis=1)*np.linalg.norm(n0,axis=1))
   if np.all(orientation>.05):accepted=True;break
   dt*=.5
  if not accepted:stopped='no safe connected geometry step within bounded line search';break
  x=proposal;clock+=dt
  history.append({'iteration':iteration,'dt':dt,'construction_time':clock,'max_target_residual_mm':float(np.linalg.norm(target-x[knots],axis=1).max()*1000),
   'minimum_signed_tet_determinant_m3':float(det.min()),'minimum_face_normal_cosine':float(orientation.min())})
  if clock>=2.-1e-12 or np.max(np.linalg.norm(target-x[knots],axis=1))<2e-5:break
 new=x.astype(np.float32);changed=np.flatnonzero(np.any(new!=m['points_world_m'],axis=1));det=signed_det(x,T)
 edge=x[T][:,1:]-x[T][:,:1];den=np.linalg.norm(edge[:,0],axis=1)*np.linalg.norm(edge[:,1],axis=1)*np.linalg.norm(edge[:,2],axis=1)
 q=det/den
 dest=out/'connected-rest-proposal.npz';np.savez_compressed(dest,points_world_m=new,tetrahedra=T,physical_outer_faces=F,
  physical_O_parent_face_rows=m['physical_O_parent_face_rows'],node_displacements_m=x-old,changed_node_ids=changed,protected_node_ids=protected)
 report={'status':'SURFACE_RECONSTRUCTION_REQUIRES_CAVITY_REMESH','steps':len(history),'stopped_reason':stopped,'construction_time':clock,
  'targets':len(knots),'moved_nodes':len(changed),'max_displacement_mm':float(np.linalg.norm(x-old,axis=1).max()*1000),
  'nonpositive_tets':int(sum(det<=0)),'min_normalized_det':float(q.min()),'surface_normal_flips':int(sum(orientation<=0)),
  'protected_outer_nodes_byte_identical':bool(np.array_equal(new[protected],m['points_world_m'][protected])),
  'reference_volume_relative_delta':float((det.sum()-signed_det(old,T).sum())/signed_det(old,T).sum()),
  'history':history,'input_sha256':{str(p):sha(p) for p in (mp,rp,pp,Path(__file__))},'output_sha256':sha(dest),
  'physical_admission':False,'native_admission':False,'needs_exact_crossing_and_detailed_transfer_gates':True,'elapsed_s':time.monotonic()-begun}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
if __name__=='__main__':main()

