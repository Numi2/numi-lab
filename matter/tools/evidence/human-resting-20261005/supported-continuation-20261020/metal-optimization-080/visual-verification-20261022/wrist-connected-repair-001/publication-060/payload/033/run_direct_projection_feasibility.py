from pathlib import Path
import json,csv,numpy as np,hashlib
from scipy.spatial.transform import Rotation
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"); D=E/"bone-clearance-feasibility-033";D.mkdir(exist_ok=True)
S=E/"native-candidate020-bone-witness-029/source-preflight-002"
N=E/"native-candidate020-conforming-prism-030/source-preflight-006/initial-skin-fem-nodes.csv"
P=E/"native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz"
a=np.load(P);t=a["tetrahedra"];m=a["material_ids"]
rows=list(csv.DictReader(N.open()));rows.sort(key=lambda x:int(x["global_node"]))
ids=np.array([int(r["global_node"]) for r in rows]);p=np.array([[float(r[k]) for k in ["x_m","y_m","z_m"]] for r in rows]);ind={int(v):i for i,v in enumerate(ids)}
poses={r["body_index"]:r["current"] for r in json.load((S/"registered-skin-poses.json").open())["all_bodies"]}
w=list(csv.DictReader((S/"initial-skin-bone-contact.csv").open()))
inside=[r for r in w if int(r["solid_audited"]) and int(r["solid_inside_rays"])==3 and not int(r["solid_ambiguous"])]
assert len(inside)==138 and len(rows)==7233
cs=[];margin=1e-4
for r in inside:
 i=ind[int(r["node"])];qv=np.array([float(r["actual_local_"+k]) for k in "xyz"]);c=np.array([float(r["closest_"+k]) for k in "xyz"])
 R=Rotation.from_quat(poses[int(r["body"])]["orientation_world_xyzw"])
 delta=R.apply(c-qv);d=np.linalg.norm(delta);normal=delta/d
 assert abs(d-float(r["distance_m"]))<1e-10
 cs.append((i,p[i]+delta,normal,d,int(r["row"]),int(r["layer"]),int(r["body"])))
np.savez_compressed(D/"world-bone-clearance-constraints.npz",points=p,tetrahedra=t,material_ids=m,global_nodes=ids,constrained_nodes=np.array([v[0] for v in cs]),closest_world=np.array([v[1] for v in cs]),outward_normals=np.array([v[2] for v in cs]),inside_depth_m=np.array([v[3] for v in cs]),columns=np.array([v[4] for v in cs]),layers=np.array([v[5] for v in cs]),bodies=np.array([v[6] for v in cs]),inferred_margin_m=margin)
def metrics(x):
 v=x[t];e=v[:,1:]-v[:,:1];det=np.linalg.det(e)
 edge=np.stack([np.linalg.norm(v[:,i]-v[:,j],axis=1) for i,j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]],axis=1)
 quality=det/np.maximum(edge.max(axis=1)**3,1e-300)
 return {"nonpositive":int(np.count_nonzero(det<=0)),"min_determinant_m3":float(det.min()),"min_det_over_max_edge_cubed":float(quality.min()),"below_1e_minus_5_quality":int(np.count_nonzero(quality<1e-5)),"material_signed_volume_m3":{str(int(k)):float(det[m==k].sum()/6) for k in np.unique(m)}}
out={"scope":"naive fixed nearest-point feasibility only; no anatomy/physics acceptance or fresh containment requery","inferred_clearance_margin_m":margin,"confirmed_inside_constraints":len(cs),"baseline_emitted_Float32":metrics(p),"trials":{}}
for mode in ["independent_nodes","whole_column_translation"]:
 x=p.copy();disp=np.zeros_like(p)
 if mode=="independent_nodes":
  for i,c,n,d,col,layer,b in cs:disp[i]=(d+margin)*n
 else:
  for i,c,n,d,col,layer,b in sorted(cs,key=lambda z:z[3]):
   for j in [col,col+2411,col+4822]:disp[j]=(d+margin)*n
 x=(x+disp).astype(np.float32).astype(np.float64)
 mm=metrics(x);mm["moved_nodes"]=int(np.count_nonzero(np.linalg.norm(x-p,axis=1)>0));mm["maximum_displacement_m"]=float(np.linalg.norm(x-p,axis=1).max())
 gaps=np.array([np.dot(x[i]-c,n) for i,c,n,*_ in cs])
 mm["minimum_original_witness_plane_gap_m"]=float(gaps.min());mm["original_planes_below_margin"]=int(np.count_nonzero(gaps<margin-1e-7))
 out["trials"][mode]=mm
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out["pins"]={str(v):sha(v) for v in [N,P,S/"registered-skin-poses.json",S/"initial-skin-bone-contact.csv",D/"world-bone-clearance-constraints.npz"]}
(D/"direct-projection-feasibility.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out))

