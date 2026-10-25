"""Independent static audit of one conforming, material-labelled tetrahedral volume."""
from pathlib import Path
from collections import Counter
import json,sys,time,hashlib,numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

def audit(P,T,M):
 P=np.asarray(P,np.float32).astype(float);T=np.asarray(T,np.int64);M=np.asarray(M,np.int64)
 if P.ndim!=2 or P.shape[1]!=3 or not np.isfinite(P).all():raise ValueError('invalid emitted points')
 if T.ndim!=2 or T.shape[1]!=4 or not len(T) or T.min()<0 or T.max()>=len(P):raise ValueError('invalid cells')
 if M.shape!=(len(T),) or not np.isin(M,[1,2]).all():raise ValueError('material IDs must be skin=1/fat=2')
 if np.any(np.diff(np.sort(T,axis=1),axis=1)==0):raise ValueError('repeated cell point')
 if len(np.unique(np.sort(T,axis=1),axis=0))!=len(T):raise ValueError('duplicate cells')
 X=P[T];D=np.einsum('ij,ij->i',X[:,1]-X[:,0],np.cross(X[:,2]-X[:,0],X[:,3]-X[:,0]))
 edges=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
 L=np.linalg.norm(X[:,edges[:,0]]-X[:,edges[:,1]],axis=2).max(1);Q=D/L**3
 local=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
 faces=T[:,local].reshape(-1,3);owner=np.repeat(np.arange(len(T)),4)
 ordered=np.sort(faces,axis=1);u,first,inv,count=np.unique(ordered,axis=0,return_index=True,return_inverse=True,return_counts=True)
 # A face's oriented parity relative to its sorted point IDs.
 inversions=(faces[:,0]>faces[:,1]).astype(int)+(faces[:,0]>faces[:,2])+(faces[:,1]>faces[:,2])
 parity=np.where(inversions%2,-1,1)
 sums=np.bincount(inv,weights=parity,minlength=len(u))
 internal=(count==2);bad_orientation=internal&(sums!=0)
 boundary_ids=np.flatnonzero(count==1);boundary=faces[first[boundary_ids]]
 E=np.sort(boundary[:,np.array([[0,1],[1,2],[2,0]])].reshape(-1,2),axis=1)
 ue,ec=np.unique(E,axis=0,return_counts=True)
 order=np.argsort(inv,kind='stable');beg=np.r_[0,np.cumsum(count)]
 pairs=np.stack((owner[order[beg[:-1][internal]]],owner[order[beg[:-1][internal]+1]]),axis=1)
 interface=pairs[M[pairs[:,0]]!=M[pairs[:,1]]]
 comps={}
 for mid,name in [(None,'union'),(1,'skin'),(2,'fat')]:
  mask=np.ones(len(T),bool) if mid is None else M==mid
  ids=np.flatnonzero(mask);rmap=np.full(len(T),-1);rmap[ids]=np.arange(len(ids))
  selected=pairs[mask[pairs].all(1)]
  if not len(ids): comps[name]={'count':0};continue
  a=rmap[selected[:,0]];b=rmap[selected[:,1]]
  graph=sparse.csr_matrix((np.ones(2*len(a)),(np.r_[a,b],np.r_[b,a])),shape=(len(ids),len(ids)))
  n,lab=connected_components(graph,directed=False)
  sizes=np.bincount(lab);volumes=np.bincount(lab,weights=D[ids]/6)
  comps[name]={'count':int(n),'largest_cell_counts':sorted(sizes.tolist(),reverse=True)[:12],'largest_volumes_m3':sorted(volumes.tolist(),reverse=True)[:12]}
 volumes={str(i):float(D[M==i].sum()/6)for i in (1,2)}
 rho={1:1000.000061,2:900.0};masses={str(i):volumes[str(i)]*rho[i]for i in (1,2)}
 r={'vertices':len(P),'cells':len(T),'nonpositive_or_invalid_cells':int((~np.isfinite(D)|(D<=0)).sum()),'minimum_determinant_m3':float(D.min()),'minimum_quality_det_over_longest_edge_cubed':float(Q.min()),'cells_below_q_1e_6':int((Q<1e-6).sum()),'nonmanifold_faces':int((count>2).sum()),'orientation_mismatched_internal_faces':int(bad_orientation.sum()),'closed_boundary_bad_edge_incidence':int((ec!=2).sum()),'boundary_triangles':len(boundary),'shared_skin_fat_interface_triangles':len(interface),'components':comps,'material_reference_volume_m3':volumes,'material_mass_at_declared_baseline_density_kg':masses,'density_policy':'No density renormalization; geometry-derived mass diagnostic only','native_admission':False,'accepted_physical_steps':0}
 r['static_topology_and_positivity_pass']=not(r['nonpositive_or_invalid_cells']or r['nonmanifold_faces']or r['orientation_mismatched_internal_faces']or r['closed_boundary_bad_edge_incidence'])
 return r,dict(boundary_faces=boundary,shared_interface_owner_cells=interface)
def main():
 C,O=map(Path,sys.argv[1:]);O.mkdir(exist_ok=False,parents=True)
 z=np.load(C);start=time.monotonic()
 r,a=audit(z['points_world_m'],z['tetrahedra_zero_based'],z['material_ids'])
 r.update(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256(C.read_bytes()).hexdigest(),wall_seconds=time.monotonic()-start)
 np.savez_compressed(O/'topology-witnesses.npz',**a)
 (O/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
