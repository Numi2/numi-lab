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
def main():
 begun=time.monotonic();out=D/'run-001';assert not out.exists();out.mkdir()
 mesh=E/'global-inner-p1-active-layer-refinement-008/candidate-with-transfer-partial-field.npz'
 pairs=E/'global-inner-p1-full-domain-clipping-009/validation-001/zero-support-physical-O-face-pairs-010.npz'
 aux=D/'baseline-outward-normals-and-pair-heights.npz'
 m=np.load(mesh);z=np.load(pairs);a=np.load(aux)
 x=m['points_world_m'].astype(float);f=m['physical_outer_faces'];n=a['physical_O_face_normals']
 dirs=vertex_directions(x,f)
 # The tetrahedron-oriented normal must agree with the physical O winding.
 winding=np.cross(x[f[:,1]]-x[f[:,0]],x[f[:,2]]-x[f[:,0]])
 assert np.all(np.einsum('ij,ij->i',winding,n)>0)
 lo=envelope_lower_bounds(f,n,dirs,z['physical_O_face_ids'],a['pair_plane_heights_m'],.00005)
 d,ids,res=fit_connected_scalar_field(x,f,lo)
 new=(x+dirs*d[:,None]).astype(np.float32)
 # Construction plane constraints are necessary diagnostics, not final exact or physical admission.
 moved=np.flatnonzero(np.any(new!=m['points_world_m'],axis=1))
 h=a['pair_plane_heights_m'];of=z['physical_O_face_ids']
 plane=np.einsum('ijk,ik->ij',new[f[of]].astype(float)-x[f[of]],n[of])-h[:,None]
 tri0=x[f];tri1=new[f].astype(float)
 area0=np.linalg.norm(np.cross(tri0[:,1]-tri0[:,0],tri0[:,2]-tri0[:,0]),axis=1)
 normal1=np.cross(tri1[:,1]-tri1[:,0],tri1[:,2]-tri1[:,0])
 area1=np.linalg.norm(normal1,axis=1)
 cos=np.einsum('ij,ij->i',normal1,n)/np.maximum(area1,1e-300)
 T=m['tetrahedra'];a0=x[T];a1=new[T].astype(float)
 det0=np.linalg.det(np.transpose(a0[:,1:]-a0[:,:1],(0,2,1)))
 det1=np.linalg.det(np.transpose(a1[:,1:]-a1[:,:1],(0,2,1)))
 # Full volume deformation is intentionally not guessed by an outer-surface fitter.
 np.savez_compressed(out/'outer-envelope-proposal.npz',points_world_m=new,physical_outer_faces=f,
  changed_node_ids=moved,node_displacements_m=new.astype(float)-x,outward_scalar_displacements_m=d,
  constrained_lower_bounds_m=lo,active_node_ids=ids,source_points_sha256=np.asarray(sha(mesh)))
 report={'status':'CONSTRUCTION_PROPOSAL_ONLY','optimizer_success':bool(res.success),'optimizer_message':str(res.message),
  'iterations':res.nit,'old_crossing_pairs':len(of),'crossing_faces':len(np.unique(of)),'constrained_nodes':int(sum(lo>0)),
  'moved_nodes':len(moved),'maximum_displacement_mm':float(np.linalg.norm(new.astype(float)-x,axis=1).max()*1000),
  'minimum_old_normal_plane_clearance_mm':float(plane.min()*1000),'surface_normal_flips':int(sum(cos<=0)),
  'minimum_face_normal_cosine':float(cos.min()),'face_area_ratio_quantiles':np.quantile(area1/area0,[0,.01,.5,.99,1]).tolist(),
  'volume_transfer_required':True,'nonpositive_tets_if_only_outer_nodes_moved':int(sum(det0*det1<=0)),
  'minimum_tet_determinant_ratio_without_volume_transfer':float((det1/det0).min()),
  'input_sha256':{str(p):sha(p) for p in (mesh,pairs,aux,Path(__file__))},
  'physical_admission':False,'native_admission':False,'elapsed_s':time.monotonic()-begun}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()

