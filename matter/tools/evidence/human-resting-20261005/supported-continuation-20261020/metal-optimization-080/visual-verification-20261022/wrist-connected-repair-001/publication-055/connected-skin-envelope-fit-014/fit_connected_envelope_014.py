"""Constrained local outward envelope proposal. Offline geometry construction only."""
from pathlib import Path
import numpy as np,json,hashlib,time
from scipy import sparse
from scipy.optimize import minimize
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def vertex_directions(x,f):
 tri=x[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
 v=np.zeros_like(x)
 for j in range(3):np.add.at(v,f[:,j],n)
 lengths=np.linalg.norm(v,axis=1)
 mask=lengths>0;v[mask]/=lengths[mask,None]
 return v
def envelope_lower_bounds(f,face_normals,node_directions,pair_faces,heights,clearance):
 ids=f[np.asarray(pair_faces)]
 cos=np.einsum('ijk,ik->ij',node_directions[ids],face_normals[pair_faces])
 if np.any(cos<=0):raise ValueError('non-outward vertex direction needs a vector patch solve')
 need=(np.asarray(heights)[:,None]+clearance)/cos
 lower=np.zeros(len(node_directions))
 np.maximum.at(lower,ids.ravel(),need.ravel())
 return lower
def fit_connected_scalar_field(x,f,lower,rings=4,smooth=2.):
 edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
 active=lower>0
 for _ in range(rings):
  touching=np.any(active[edges],axis=1)
  active[edges[touching].ravel()]=True
 ids=np.flatnonzero(active);lookup=np.full(len(x),-1);lookup[ids]=np.arange(len(ids))
 selected=edges[np.any(active[edges],axis=1)]
 rr=[];cc=[];vv=[]
 for j in range(2):
  good=lookup[selected[:,j]]>=0
  rr.extend(np.flatnonzero(good));cc.extend(lookup[selected[good,j]]);vv.extend(np.full(np.sum(good),1. if j==0 else -1.))
 G=sparse.coo_matrix((vv,(rr,cc)),shape=(len(selected),len(ids))).tocsr()
 # millimetres condition the construction solve; no physical material coefficients are altered.
 lo=lower[ids]*1000
 H=sparse.eye(len(ids),format='csr')+smooth*(G.T@G)
 def fun(d):
  hd=H@d;return .5*float(d@hd),hd
 result=minimize(fun,lo,jac=True,method='L-BFGS-B',bounds=list(zip(lo,np.maximum(lo,40.))),
  options={'ftol':1e-12,'gtol':1e-9,'maxiter':600,'maxls':40})
 field=np.zeros(len(x));field[ids]=result.x/1000
 if np.min(field-lower)<-1e-10:raise ValueError('lower bound not respected')
 return field,ids,result
def fit_vector_envelope(x,f,pair_faces,heights,face_normals,rings=4,clearance=.00005):
 from scipy.sparse.linalg import splu
 from scipy.optimize import linprog
 # One maximum requirement per physical face, not repeated per crossing pair.
 need=np.zeros(len(f));np.maximum.at(need,pair_faces,np.asarray(heights)+clearance)
 core=need>0;active=np.zeros(len(x),bool);active[f[core].ravel()]=True
 edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
 for _ in range(rings):active[edges[np.any(active[edges],axis=1)].ravel()]=True
 ids=np.flatnonzero(active);lookup=np.full(len(x),-1);lookup[ids]=np.arange(len(ids))
 selected=edges[np.any(active[edges],axis=1)]
 er=[];ec=[];ev=[]
 for j in range(2):
  good=lookup[selected[:,j]]>=0
  er.extend(np.flatnonzero(good));ec.extend(lookup[selected[good,j]]);ev.extend(np.full(sum(good),1. if j==0 else -1.))
 G=sparse.coo_matrix((ev,(er,ec)),shape=(len(selected),len(ids))).tocsr()
 H=sparse.kron(sparse.eye(len(ids))+2.*(G.T@G),sparse.eye(3),format='csc')
 ar=[];ac=[];av=[];rhs=[]
 for fi in np.flatnonzero(core):
  for node in f[fi]:
   row=len(rhs)
   for j in range(3):ar.append(row);ac.append(3*lookup[node]+j);av.append(face_normals[fi,j])
   rhs.append(need[fi]*1000)
 A=sparse.coo_matrix((av,(ar,ac)),shape=(len(rhs),3*len(ids))).tocsr()
 rhs=np.asarray(rhs);cap=40.
 feasible=linprog(np.zeros(A.shape[1]),A_ub=-A,b_ub=-rhs,bounds=(-cap,cap),method='highs')
 if not feasible.success:raise ValueError('local normal cone constraints infeasible: '+feasible.message)
 C=sparse.vstack([A,sparse.eye(A.shape[1])],format='csc')
 low=np.r_[rhs,np.full(A.shape[1],-cap)];high=np.r_[np.full(len(rhs),np.inf),np.full(A.shape[1],cap)]
 rho=10.;solve=splu(H+rho*(C.T@C))
 q=feasible.x;z=np.clip(C@q,low,high);u=np.zeros(len(z));converged=False
 for iteration in range(5000):
  q=solve.solve(rho*(C.T@(z-u)))
  cq=C@q;old=z;z=np.clip(cq+u,low,high);u+=cq-z
  primal=float(np.max(np.abs(cq-z)));dual=float(np.max(np.abs(rho*C.T@(z-old))))
  if primal<1e-6 and dual<1e-5:converged=True;break
 # Polishing feasibility is mandatory; an optimizer flag is not a geometric gate.
 displacement=np.zeros_like(x);displacement[ids]=q.reshape(-1,3)/1000
 info={'optimizer_success':converged,'iterations':iteration+1,'primal_residual_mm':primal,'dual_residual':dual,
  'minimum_constraint_slack_mm':float(np.min(A@q-rhs)),'active_nodes':len(ids)}
 return displacement,ids,need,info
def main():
 begun=time.monotonic();out=D/'run-002';assert not out.exists();out.mkdir()
 mesh=E/'global-inner-p1-active-layer-refinement-008/candidate-with-transfer-partial-field.npz'
 pairs=E/'global-inner-p1-full-domain-clipping-009/validation-001/zero-support-physical-O-face-pairs-010.npz'
 aux=D/'baseline-outward-normals-and-pair-heights.npz'
 m=np.load(mesh);z=np.load(pairs);a=np.load(aux)
 x=m['points_world_m'].astype(float);f=m['physical_outer_faces'];n=a['physical_O_face_normals']
 delta,ids,need,info=fit_vector_envelope(x,f,z['physical_O_face_ids'],a['pair_plane_heights_m'],n)
 new=(x+delta).astype(np.float32)
 moved=np.flatnonzero(np.any(new!=m['points_world_m'],axis=1));of=z['physical_O_face_ids']
 plane=np.einsum('ijk,ik->ij',new[f[of]].astype(float)-x[f[of]],n[of])-a['pair_plane_heights_m'][:,None]
 tri0=x[f];tri1=new[f].astype(float)
 area0=np.linalg.norm(np.cross(tri0[:,1]-tri0[:,0],tri0[:,2]-tri0[:,0]),axis=1)
 normal1=np.cross(tri1[:,1]-tri1[:,0],tri1[:,2]-tri1[:,0]);area1=np.linalg.norm(normal1,axis=1)
 cos=np.einsum('ij,ij->i',normal1,n)/np.maximum(area1,1e-300)
 T=m['tetrahedra'];a0=x[T];a1=new[T].astype(float)
 det0=np.linalg.det(np.transpose(a0[:,1:]-a0[:,:1],(0,2,1)));det1=np.linalg.det(np.transpose(a1[:,1:]-a1[:,:1],(0,2,1)))
 np.savez_compressed(out/'outer-envelope-proposal.npz',points_world_m=new,physical_outer_faces=f,
  changed_node_ids=moved,node_displacements_m=new.astype(float)-x,active_node_ids=ids,face_plane_requirements_m=need,
  source_points_sha256=np.asarray(sha(mesh)))
 report={'status':'CONSTRUCTION_PROPOSAL_ONLY',**info,'old_crossing_pairs':len(of),'crossing_faces':len(np.unique(of)),
  'moved_nodes':len(moved),'maximum_displacement_mm':float(np.linalg.norm(new.astype(float)-x,axis=1).max()*1000),
  'minimum_old_normal_plane_clearance_mm':float(plane.min()*1000),'surface_normal_flips':int(sum(cos<=0)),
  'minimum_face_normal_cosine':float(cos.min()),'face_area_ratio_quantiles':np.quantile(area1/area0,[0,.01,.5,.99,1]).tolist(),
  'volume_transfer_required':True,'nonpositive_tets_if_only_outer_nodes_moved':int(sum(det0*det1<=0)),
  'minimum_tet_determinant_ratio_without_volume_transfer':float((det1/det0).min()),
  'input_sha256':{str(p):sha(p) for p in (mesh,pairs,aux,Path(__file__))},
  'physical_admission':False,'native_admission':False,'elapsed_s':time.monotonic()-begun}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
