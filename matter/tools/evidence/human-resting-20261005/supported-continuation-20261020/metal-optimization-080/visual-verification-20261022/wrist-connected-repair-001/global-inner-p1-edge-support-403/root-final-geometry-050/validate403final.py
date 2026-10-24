from pathlib import Path
import json,hashlib,time,itertools,numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=E/'global-inner-p1-edge-support-403';OUT=D/'root-final-geometry-050';OUT.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=E/'global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/local-quality-cavities-398/candidate-with-transfer.npz'
candidate=D/'quality-coordinate-005/candidate.npz';ref=D/'quality-bistellar-004/candidate-geometry-only.npz'
assert sha(candidate)=='5b54ffd34d74d717579fcb14677cbf2bfba01f4d4562652605b9207e65f8f698'
assert sha(base)=='8abe15f941dbc01659aee693a07e3edf66f32617c784cf0334410ca11a357890'
a=np.load(base);b=np.load(candidate);r=np.load(ref);X=b['points_world_m'];T=b['tetrahedra'];F=b['physical_shell_faces'];t0=time.monotonic()
assert X.dtype==np.float32 and np.isfinite(X).all() and T.min()>=0 and T.max()<len(X)
assert X[:len(a['points_world_m'])].tobytes()==a['points_world_m'].tobytes()
for k in ['physical_outer_faces','physical_shell_faces','construction_closure_faces','physical_outer_node_ids']:assert b[k].tobytes()==a[k].tobytes()
assert np.array_equal(T,r['tetrahedra']) and np.array_equal(b['region_attributes'],r['region_attributes'])
changed=np.flatnonzero(np.any(X!=r['points_world_m'],axis=1));assert changed.tolist()==[55529,55530]
p=X[T].astype(float);det=np.linalg.det(np.stack([p[:,1]-p[:,0],p[:,2]-p[:,0],p[:,3]-p[:,0]],axis=2))
pairs=np.asarray(list(itertools.combinations(range(4),2)));L=np.max(np.linalg.norm(p[:,pairs[:,0]]-p[:,pairs[:,1]],axis=2),axis=1);q=det/L**3
faces=np.concatenate([T[:,[1,2,3]],T[:,[0,3,2]],T[:,[0,1,3]],T[:,[0,2,1]]]);u,ix,inv,c=np.unique(np.sort(faces,axis=1),axis=0,return_index=True,return_inverse=True,return_counts=True)
def canon(f):
 k=np.argmin(f,axis=1);return np.stack([f[np.arange(len(f)),(k+i)%3] for i in range(3)],axis=1)
def order(f):return f[np.lexsort((f[:,2],f[:,1],f[:,0]))]
ca=canon(faces);sums=np.bincount(inv,weights=np.where(ca[:,1]<ca[:,2],1,-1),minlength=len(u))
boundaryexact=np.array_equal(order(canon(faces[ix[c==1]])),order(canon(F)))
orientationbad=int(np.count_nonzero(sums[c==2]));assert np.all(c<=2)
O=np.zeros(len(X),bool);O[b['physical_outer_node_ids']]=True
edges=np.unique(np.sort(T[:,pairs].reshape(-1,2),axis=1),axis=0);fp=np.asarray([[0,1],[0,2],[1,2]]);bedges=np.unique(np.sort(F[:,fp].reshape(-1,2),axis=1),axis=0)
enc=lambda e:e[:,0]*len(X)+e[:,1]
badedge=int(np.sum(np.all(O[edges],axis=1)&~np.isin(enc(edges),enc(bedges))));badface=int(np.sum((c==2)&np.all(O[u],axis=1)));badtet=int(np.sum(np.all(O[T],axis=1)))
ind=b['node_source_indptr'];ids=b['node_source_indices'];w=b['node_source_weights'];assert ind.shape==(len(X)+1,) and ind[0]==0 and ind[-1]==len(ids)==len(w) and np.all(np.diff(ind)>0)
assert np.isfinite(w).all() and np.all(w>0)
sumsw=np.add.reduceat(w,ind[:-1]);assert np.max(abs(sumsw-1))<2e-12
oldnnz=int(a['node_source_indptr'][-1]);assert np.array_equal(ind[:len(a['node_source_indptr'])],a['node_source_indptr']) and np.array_equal(ids[:oldnnz],a['node_source_indices']) and np.array_equal(w[:oldnnz],a['node_source_weights'])
reconstruct=[]
for node,parent,B in zip(b['coordinate_fit_node_rows'],b['coordinate_fit_parent_398_tet_rows'],b['coordinate_fit_parent_398_barycentric']):
 assert B.min()>=0 and abs(B.sum()-1)<1e-12
 px=a['points_world_m'][a['tetrahedra'][parent]].astype(float)
 err=float(np.linalg.norm(B@px-X[node].astype(float)));assert err<1e-12
 expected={}
 for corner,weight in zip(a['tetrahedra'][parent],B):
  for sid,sw in zip(a['node_source_indices'][a['node_source_indptr'][corner]:a['node_source_indptr'][corner+1]],a['node_source_weights'][a['node_source_indptr'][corner]:a['node_source_indptr'][corner+1]]):expected[int(sid)]=expected.get(int(sid),0)+weight*sw
 actual=dict(zip(map(int,ids[ind[node]:ind[node+1]]),w[ind[node]:ind[node+1]]));assert actual.keys()==expected.keys() and max(abs(actual[k]-expected[k]) for k in actual)<1e-12
 reconstruct.append({'node':int(node),'parent_398':int(parent),'coordinate_reconstruction_error_m':err})
aa=a['points_world_m'][a['tetrahedra']].astype(float);oldv=np.linalg.det(np.stack([aa[:,1]-aa[:,0],aa[:,2]-aa[:,0],aa[:,3]-aa[:,0]],axis=2)).sum()/6
v=det.sum()/6;delta=(v-oldv)/oldv
rep={'candidate_sha256':sha(candidate),'baseline398_sha256':sha(base),'nodes':len(X),'tets':len(T),'original398_coordinate_prefix_bitwise_exact':True,'typed_boundary_arrays_bitwise_exact':True,'boundary_oriented_exact':bool(boundaryexact),'internal_orientation_mismatches':orientationbad,'nonpositive_tets':int((det<=0).sum()),'minimum_normalized_determinant':float(q.min()),'quality_below_1e5':int((q<1e-5).sum()),'remaining_internal_all_O_edges':badedge,'remaining_internal_all_O_faces':badface,'remaining_all_O_tets':badtet,'total_volume_m3':float(v),'global_relative_volume_change':float(delta),'source_row_sum_error':float(np.max(abs(sumsw-1))),'changed_interior_nodes':changed.tolist(),'changed_node_displacements_m':np.linalg.norm(X[changed].astype(float)-r['points_world_m'][changed].astype(float),axis=1).tolist(),'new_source_rows_reconstruction':reconstruct,'unreferenced_nodes':len(X)-len(np.unique(T)),'elapsed_s':time.monotonic()-t0,'native_admission':False,'limitations':['No full material field, source anatomical clearance, physical skin/fat compilation, or native simulation qualification.','Construction closure facets remain separately typed, not physical skin.','Conservative material transfer requires separate validation.']}
rep['geometry_support_pass']=bool(boundaryexact and not orientationbad and np.all(det>0) and q.min()>=1e-5 and not badedge and not badface and not badtet and abs(delta)<1e-12)
(OUT/'validation-report.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep));assert rep['geometry_support_pass']

