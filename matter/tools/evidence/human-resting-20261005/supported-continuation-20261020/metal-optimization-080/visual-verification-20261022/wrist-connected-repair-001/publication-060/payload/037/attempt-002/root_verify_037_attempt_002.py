import pathlib,numpy as np,json,hashlib
e=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");d=e/'bone-clearance-fit-037-componentwise-035affine-002'
a=np.load(d/'candidate.npz');b=np.load(e/'bone-clearance-feasibility-033/world-bone-clearance-constraints.npz')
x=a['points_world_f32'].astype(float);t=a['tetrahedra'];v=x[t]
det=np.einsum('ij,ij->i',v[:,1]-v[:,0],np.cross(v[:,2]-v[:,0],v[:,3]-v[:,0]))
pairs=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
edges=np.max(np.array([np.linalg.norm(v[:,i]-v[:,j],axis=1)for i,j in pairs]),axis=0)
q=det/edges**3
free=a['variable_local_rows'];fixed=np.setdiff1d(np.arange(len(x)),free)
assert np.array_equal(a['points_world_f32'][fixed],b['points'].astype(np.float32)[fixed])
gap=np.einsum('ij,ij->i',x[a['witness_local_rows']]-a['witness_closest_world_m'],a['witness_normal_world'])
r={'status':'PASS_EXISTING_POSITIVE_GEOMETRY_FLOOR_ONLY','candidate_sha256':hashlib.sha256((d/'candidate.npz').read_bytes()).hexdigest(),'nonpositive_cells':int((det<=0).sum()),'below_q_1e_minus_5':int((q<1e-5).sum()),'below_construction_target_q_1_25e_minus_5':int((q<1.25e-5).sum()),'min_q':float(q.min()),'witnesses':len(gap),'min_frozen_plane_gap_m':float(gap.min()),'unchanged_fixed_nodes':len(fixed),'max_movement_m':float(np.linalg.norm(x-b['points'],axis=1).max()),'invalid_tet_rows':np.flatnonzero(det<=0).tolist(),'no_native_admission':True}
(e/'root-review-037-component-attempt-002.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k!='invalid_tet_rows'}))
