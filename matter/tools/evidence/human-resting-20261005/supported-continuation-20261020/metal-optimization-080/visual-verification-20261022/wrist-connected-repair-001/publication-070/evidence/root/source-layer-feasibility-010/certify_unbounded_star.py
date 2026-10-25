import json,numpy as np,sys
from pathlib import Path
from scipy.optimize import linprog
D=Path('/Users/numi/Work/numi-human-connected-surface-069');sys.path.insert(0,str(D/'canonical/matter/tools'));import human_skin_fat_connected_repair as r
z=np.load('/Users/n/Work/294-source-frame-audit-001/connected-interior-reference-fit-001/retessellation-001/volume-cavity-001/oid-cavity-replacement.npz');X=z['target_points_AP_m'];R=z['reference_points_AP_m'];T=z['candidate_tetrahedra'];J,G,_,_=r.cell_values_gradients(X,R,T)
rows=np.flatnonzero((np.sum(T<2411,axis=1)==3)&np.any(T==3656,axis=1));g=np.array([G[i,list(T[i]).index(3656)]for i in rows]);norm=np.linalg.norm(g,axis=1);g/=norm[:,None];h=J[rows]/norm
s=linprog([0,0,0,-1],A_ub=np.c_[-g,np.ones(len(g))],b_ub=h,bounds=[(None,None)]*3+[(None,.1)],method='highs')
assert s.success
# Dual weights certify no point can attain a positive altitude under these fixed faces.
dual=-s.ineqlin.marginals
out=dict(input_sha256='0feda312f87a58ac8d7f73b9d27d870911d86040396ac0c65c24feeac682ee2b',I_node=3656,O_node=1245,tet_rows=rows.tolist(),normals=g.tolist(),signed_altitudes_m=h.tolist(),unbounded_coordinate_box=True,maximum_positive_altitude_m=float(s.x[3]),displacement_m=s.x[:3].tolist(),dual_weights=dual.tolist(),dual_weight_sum=float(dual.sum()),dual_normal_residual=float(np.linalg.norm(dual@g)),dual_altitude_upper_bound_m=float(dual@h),geometry_admitted=False,accepted_physical_steps=0)
(D/'source-layer-feasibility-010/unbounded-star-certificate.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items()if k not in ('normals','signed_altitudes_m','tet_rows','dual_weights')},indent=2))
