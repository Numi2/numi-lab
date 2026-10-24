from pathlib import Path
import json,hashlib,time,numpy as np
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019')
E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
D=E/'global-inner-p1-edge-support-403'
base=E/'global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/local-quality-cavities-398/candidate-with-transfer.npz'
candidate=D/'build-002/candidate.npz'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lex(a):return a[np.lexsort((a[:,2],a[:,1],a[:,0]))]
def edgekeys(T):
 e=np.asarray([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
 return np.unique(np.sort(T[:,e].reshape(-1,2),axis=1),axis=0)
def bndedge(F):return np.unique(np.sort(F[:,np.asarray([[0,1],[0,2],[1,2]])].reshape(-1,2),axis=1),axis=0)
def determinants(X,T):
 p=X[T].astype(float)
 return np.linalg.det(np.stack([p[:,1]-p[:,0],p[:,2]-p[:,0],p[:,3]-p[:,0]],axis=2))
def main():
 started=time.monotonic();assert sha(base)=='8abe15f941dbc01659aee693a07e3edf66f32617c784cf0334410ca11a357890'
 a=np.load(base,allow_pickle=False);b=np.load(candidate,allow_pickle=False)
 X=b['points_world_m'];T=b['tetrahedra'];F=b['physical_shell_faces'];A=b['region_attributes']
 assert X.dtype==np.float32 and np.isfinite(X).all() and T.dtype.kind in 'iu' and T.shape[1:]==(4,)
 assert T.min()>=0 and T.max()<len(X) and len(A)==len(T)
 assert X[:len(a['points_world_m'])].tobytes()==a['points_world_m'].tobytes()
 for k in ['physical_outer_faces','physical_shell_faces','construction_closure_faces','physical_outer_node_ids']:
  assert b[k].tobytes()==a[k].tobytes(),k
 det=determinants(X,T);old_det=determinants(a['points_world_m'],a['tetrahedra'])
 p=X[T].astype(float);L=np.zeros(len(T))
 for i,j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:L=np.maximum(L,np.linalg.norm(p[:,i]-p[:,j],axis=1))
 quality=det/L**3
 allfaces=np.concatenate([T[:,[1,2,3]],T[:,[0,3,2]],T[:,[0,1,3]],T[:,[0,2,1]]])
 keys=np.sort(allfaces,axis=1)
 uni,ix,inv,cnt=np.unique(keys,axis=0,return_index=True,return_inverse=True,return_counts=True)
 assert np.all((cnt==1)|(cnt==2))
 actual=allfaces[ix[cnt==1]]
 assert np.array_equal(lex(np.sort(actual,axis=1)),lex(np.sort(F,axis=1)))
 def canonical(f):
  k=np.argmin(f,axis=1)
  return np.stack([f[np.arange(len(f)),(k+i)%3] for i in range(3)],axis=1)
 ca=canonical(allfaces);sign=np.where(ca[:,1]<ca[:,2],1,-1)
 sums=np.bincount(inv,weights=sign,minlength=len(uni))
 internal_orientation_bad=int(np.count_nonzero(sums[cnt==2]))
 actualmap={tuple(sorted(f)):tuple(c) for f,c in zip(actual,canonical(actual))}
 boundary_orientation_bad=sum(actualmap[tuple(sorted(f))]!=tuple(c) for f,c in zip(F,canonical(F)))
 ids=b['parent_398_tet_row'];frac=b['parent_398_conservative_fraction']
 assert ids.shape==frac.shape==(len(T),) and ids.dtype.kind in 'iu'
 assert ids.min()>=0 and ids.max()<len(a['tetrahedra']) and np.isfinite(frac).all() and np.all(frac>0)
 assert np.array_equal(A,a['region_attributes'][ids])
 col=np.bincount(ids,weights=frac,minlength=len(a['tetrahedra']))
 ce=float(np.max(abs(col-1)))
 parent_child_v=np.bincount(ids,weights=det/6,minlength=len(old_det))
 per_parent_relative_volume=(parent_child_v-old_det/6)/(old_det/6)
 mass=1.0+(np.arange(len(old_det))%97)/97.0
 masserr=float(abs(np.dot(frac,mass[ids])-mass.sum())/mass.sum())
 ind=b['node_source_indptr'];idx=b['node_source_indices'];w=b['node_source_weights']
 assert ind.dtype.kind in 'iu' and idx.dtype.kind in 'iu' and ind.shape==(len(X)+1,) and ind[0]==0 and ind[-1]==len(idx)==len(w)
 assert np.all(np.diff(ind)>0) and np.isfinite(w).all() and np.all(w>0) and idx.min()>=0 and idx.max()<=a['node_source_indices'].max()
 oldn=len(a['points_world_m']);oldnnz=int(a['node_source_indptr'][-1])
 assert ind[:oldn+1].tobytes()==a['node_source_indptr'].tobytes()
 assert idx[:oldnnz].tobytes()==a['node_source_indices'].tobytes() and w[:oldnnz].tobytes()==a['node_source_weights'].tobytes()
 nodesum=np.add.reduceat(w,ind[:-1])
 O=np.zeros(len(X),bool);O[b['physical_outer_node_ids']]=True
 edges=edgekeys(T);bd=bndedge(F)
 enc=lambda e:e[:,0]*len(X)+e[:,1]
 badedge=edges[O[edges[:,0]]&O[edges[:,1]]&~np.isin(enc(edges),enc(bd))]
 internal=uni[cnt==2]
 badface=internal[np.all(O[internal],axis=1)]
 badtet=np.flatnonzero(np.all(O[T],axis=1))
 report={'candidate_sha256':sha(candidate),'baseline_398_sha256':sha(base),'nodes':len(X),'tets':len(T),'added_nodes':len(X)-oldn,'original_coordinates_bitwise_exact':True,'typed_boundary_arrays_bitwise_exact':True,'all_node_source_prefix_bytes_exact':True,'boundary_faces':len(F),'boundary_orientation_mismatches':boundary_orientation_bad,'internal_orientation_mismatches':internal_orientation_bad,'nonpositive_tets':int((det<=0).sum()),'minimum_normalized_determinant':float(quality.min()),'quality_below_1e5':int((quality<1e-5).sum()),'total_volume_m3':float(det.sum()/6),'original_volume_m3':float(old_det.sum()/6),'global_relative_volume_change':float((det.sum()-old_det.sum())/old_det.sum()),'max_parent_relative_volume_change':float(np.max(abs(per_parent_relative_volume))),'max_conservative_parent_column_error':ce,'synthetic_mass_relative_error':masserr,'node_source_row_sum_max_error':float(np.max(abs(nodesum-1))),'remaining_internal_all_O_edges':len(badedge),'remaining_internal_all_O_faces':len(badface),'remaining_all_O_tets':len(badtet),'unreferenced_nodes':len(X)-len(np.unique(T)),'native_admission':False,'elapsed_s':time.monotonic()-started,'limitations':['Topological parent-normalized transfer is conservative; not an exact clipped geometric-overlap map.','A positive support mesh is not a solved physical skin/fat layer; source boundary/solid-interior and aperture gates remain.']}
 report['topological_support_gate_pass']=not(len(badedge) or len(badface) or len(badtet))
 report['geometry_gate_pass']=bool(np.all(det>0) and quality.min()>=1e-5 and not boundary_orientation_bad and not internal_orientation_bad and ce<2e-12 and masserr<2e-12 and np.max(abs(nodesum-1))<2e-12)
 out=D/'root-verification-050';out.mkdir(exist_ok=False)
 (out/'validation-report.json').write_text(json.dumps(report,indent=2)+'\n')
 np.savez_compressed(out/'remaining-support-defects.npz',internal_all_O_edges=badedge,internal_all_O_faces=badface,all_O_tet_rows=badtet,bad_quality_tet_rows=np.flatnonzero(quality<1e-5))
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()

