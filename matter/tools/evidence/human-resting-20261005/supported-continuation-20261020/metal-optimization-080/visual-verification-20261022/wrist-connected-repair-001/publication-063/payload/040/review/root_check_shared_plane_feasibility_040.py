from pathlib import Path
import numpy as np,json,hashlib
from scipy.optimize import linprog
e=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");p=e/'bone-clearance-fit-040-shared-columns/attempt-004/fit-input.npz';z=np.load(p);X=z['points_baseline_f32'].astype(float);cols=z['constraint_endpoint_column_rows'];nodes=z['constraint_endpoint_node_rows'];N=z['constraint_normal_world'];H=z['constraint_plane_point_world_m'];m=z['constraint_clearance_m']
assert np.array_equal(nodes%2411,cols)
required=(m-np.einsum('ij,ij->i',N,X[nodes]-H))*1000
reports=[]
for col in np.unique(cols):
 sel=cols==col;ns=N[sel];rhs=required[sel]
 A=np.vstack([np.c_[-ns,np.zeros(len(ns))],np.c_[np.eye(3),-np.ones(3)],np.c_[-np.eye(3),-np.ones(3)]])
 b=np.r_[-rhs,np.zeros(6)]
 r=linprog([0,0,0,1],A_ub=A,b_ub=b,bounds=[(None,None)]*3+[(0,None)],method='highs')
 rec={'column':int(col),'constraints':int(sel.sum()),'feasible':bool(r.success),'status':int(r.status)}
 if r.success:rec.update(minimum_inf_translation_mm=float(r.x[3]),translation_mm=r.x[:3].tolist(),within_original_15mm_axis_bound=bool(r.x[3]<=15+1e-7))
 reports.append(rec)
out={'status':'LINEAR_SHARED_COLUMN_PLANE_FEASIBILITY_ONLY','columns':len(reports),'infeasible':sum(not x['feasible'] for x in reports),'over_15mm_axis_bound':sum(x.get('minimum_inf_translation_mm',0)>15+1e-7 for x in reports),'maximum_required_inf_translation_mm':max(x.get('minimum_inf_translation_mm',0) for x in reports),'witnesses':[x for x in reports if not x['feasible'] or x.get('minimum_inf_translation_mm',0)>15+1e-7],'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'limitation':'Necessary plane/bound feasibility only; ignores cell/material and solid geometry constraints.'}
(e/'root-shared-plane-linear-feasibility-040.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

