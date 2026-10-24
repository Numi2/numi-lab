from pathlib import Path
import csv,json,numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");d=E/"bone-clearance-feasibility-033";a=np.load(d/"world-bone-clearance-constraints.npz");p=a["points"];t=a["tetrahedra"];m=a["material_ids"];q=p.copy()
S=E/"native-candidate020-bone-witness-029/source-preflight-002"
skin={int(r["vertex"]):r for r in csv.DictReader((S/"registered-skin.csv").open())}
embed=list(csv.DictReader((E/"wholebody-attachment-source-rebind-037-candidate020-identity004-002/generated/embedding.csv").open()))
unresolved=[];dp=np.zeros((2411,3))
for r in embed:
 row=int(r["row"]);sv=int(r["source_vertex"])
 if sv not in skin:unresolved.append(row);continue
 v=skin[sv];dp[row]=np.array([float(v[k]) for k in ["wx","wy","wz"]])-p[row]
q=(p+np.tile(dp,(3,1))).astype(np.float32).astype(np.float64);v=q[t];e=v[:,1:]-v[:,:1];det=np.linalg.det(e);edges=np.stack([np.linalg.norm(v[:,i]-v[:,j],axis=1) for i,j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]],1);qual=det/edges.max(1)**3
out={"scope":"diagnostic current detailed skin O resampling with naive equal column displacement; not physical admission or valid layer fitting","unresolved_source_rows":unresolved,"maximum_outer_discrepancy_m":float(np.linalg.norm(dp,axis=1).max()),"outer_rows_over_1um":int(np.count_nonzero(np.linalg.norm(dp,axis=1)>1e-6)),"outer_rows_over_100um":int(np.count_nonzero(np.linalg.norm(dp,axis=1)>1e-4)),"nonpositive_cells":int(np.count_nonzero(det<=0)),"below_quality_1e5":int(np.count_nonzero(qual<1e-5)),"min_quality":float(qual.min()),"bad_cell_local_rows":np.flatnonzero(qual<1e-5).tolist(),"material_signed_volume_m3":{str(int(k)):float(det[m==k].sum()/6) for k in np.unique(m)}}
(d/"detailed-source-column-transfer-counterfactual.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out))

