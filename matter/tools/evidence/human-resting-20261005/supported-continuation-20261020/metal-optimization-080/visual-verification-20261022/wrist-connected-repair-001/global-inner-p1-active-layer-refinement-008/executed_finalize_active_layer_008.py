from pathlib import Path
import sys,importlib.util,json,hashlib,time
import numpy as np
from scipy.sparse import csr_matrix
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");OUT=E/'global-inner-p1-active-layer-refinement-008'
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
r=load(E/'global-inner-p1-edge-support-403/refine_internal_o_edges_403.py','r403')
a=load(E/'global-inner-p1-field-owner-integration-007/field_owner_integration_007.py','a007')
c=load(E/'robust-p1-construction-clip-owner-302/tools/human_skin_p1_obstacles.py','c302')
s=load(E/'global-inner-p1-field-402-absolute-variable-integration/absolute_field_402.py','s402')
z=np.load(OUT/'round-004.npz');b=np.load(E/'global-inner-p1-edge-support-403/quality-coordinate-005/candidate.npz')
X=z['points_world_m'];T=z['tetrahedra'];P=z['parent_005_tet_rows'];O=z['physical_outer_node_ids']
D,Q=r.tetra_metrics(X,T)
fractions,totals,closure=r.conservative_child_fractions(D/6,P,len(b['tetrahedra']))
assert np.max(np.abs(closure-1))<2e-12
_,oldQ=r.tetra_metrics(b['points_world_m'],b['tetrahedra'])
oldD,_=r.tetra_metrics(b['points_world_m'],b['tetrahedra'])
out={k:z[k] for k in z.files}
for k in ('physical_outer_faces','construction_closure_faces','coordinate_fit_node_rows','coordinate_fit_parent_398_tet_rows','coordinate_fit_parent_398_barycentric','coordinate_fit_displacements_m'):out[k]=b[k]
out['transfer_new_to_005_offsets']=np.arange(len(T)+1,dtype=np.int64)
out['transfer_new_to_005_parent_rows']=P
out['transfer_new_to_005_parent_fractions']=fractions
m=np.load(E/'global-inner-p1-edge-support-403/quality-transfer-006/local-overlap-005-to-002.npz')
B=csr_matrix((fractions,P,np.arange(len(T)+1)),shape=(len(T),len(b['tetrahedra'])))
A=csr_matrix((m['transfer_new_to_014_parent_fractions'],m['transfer_new_to_014_parent_rows'],m['transfer_new_to_014_offsets']),shape=(len(b['tetrahedra']),164176))
C=(B@A).tocsr();C.sort_indices()
out['transfer_new_to_014_offsets']=C.indptr.astype(np.int64);out['transfer_new_to_014_parent_rows']=C.indices.astype(np.int64);out['transfer_new_to_014_parent_fractions']=C.data
assert np.max(np.abs(np.asarray(C.sum(axis=0)).ravel()-1))<2e-12
bank=np.load(E/'registered-surrounding-tissue-native013-191/registered-tissue-banks.npz')
tri=bank['bank_0004_vertices'][bank['bank_0004_faces'][2562]]
cons,polys,meta=a.clip_typed_triangle_to_mesh(X,T,tri,c.triangle_tet_barycentric_polygon)
tids=np.asarray([ti for ti,bb in cons for _ in bb],np.int64);barys=np.concatenate([bb for ti,bb in cons])
caps=a.fixed_zero_capacity_report(T,cons,O,len(X));assert min(x['max_free_capacity'] for x in caps)>.341
# The owner requires strictly positive costs. Compact only referenced nodes
# for the LP, rather than fabricating a mass on inherited unused nodes.
used=np.unique(T);inv=np.full(len(X),-1,np.int64);inv[used]=np.arange(len(used))
assert np.all(inv[O]>=0)
lumped=np.bincount(T.ravel(),weights=np.repeat(D/24,4),minlength=len(X))
assert np.all(lumped[used]>0)
scale=float(np.mean(lumped[used]));cost=lumped[used]/scale
start=time.monotonic()
phi,report=s.solve_absolute_obstacle_field(np.zeros(len(used)),inv[T],tids,barys,inv[O],np.zeros(len(O)),cost,.341,time_limit_s=60)
assert phi is not None and report['pass'],report
full=np.zeros(len(X));full[used]=phi
values=np.einsum('ij,ij->i',barys,full[T[tids]])
assert np.array_equal(full[O],np.zeros(len(O))) and values.min()>=.341-1e-8
out['partial_single_face_phi']=full;out['clip_constraint_tet_ids']=tids;out['clip_constraint_tet_barycentric']=barys
out['lumped_volume_cost_m3']=lumped;out['solver_referenced_node_ids']=used
path=OUT/'candidate-with-transfer-partial-field.npz'
assert not path.exists();np.savez_compressed(path,**out)
g={'nodes':len(X),'tets':len(T),'min_quality':float(Q.min()),'bad_quality':int(np.sum(Q<1e-5)),'nonpositive':int(np.sum(D<=0)),'physical_outer_nodes':len(O),'max_raw_parent_volume_relative_error':float(np.max(np.abs(totals/(oldD/6)-1))),'child_005_column_sum_error':float(np.max(np.abs(closure-1))),'composed_014_column_sum_error':float(np.max(np.abs(np.asarray(C.sum(axis=0)).ravel()-1))),'global_volume_difference_m3':float(np.sum(D/6)-np.sum(oldD/6)),'all_old_coordinates_exact':bool(np.array_equal(X[:len(b['points_world_m'])],b['points_world_m']))}
rec={'geometry':g,'solver':report,'solver_elapsed_s':time.monotonic()-start,'minimum_constraint_value':float(values.min()),'constraints':len(values),'minimum_free_capacity':min(x['max_free_capacity'] for x in caps),'target':.341,'cost_policy':'existing311 volume-weighted minimum; positive referenced-node volume costs divided by one global constant; inherited unused nodes excluded from LP then assigned zero','lp_node_count':len(used),'inherited_unused_node_count':len(X)-len(used),'unscaled_volume_objective_m3':float(lumped@full),'candidate_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'physical_admission':False,'native_admission':False,'scope':'one typed bank0004 face2562 only; neither global field nor full tissue qualification'}
(OUT/'solver-and-transfer-report.json').write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps(rec),flush=True)

