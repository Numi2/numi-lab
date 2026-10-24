from pathlib import Path
import csv,json,numpy as np,hashlib
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");S=E/"native-candidate020-bone-witness-029/source-preflight-002"
embed={int(r["row"]):r for r in csv.DictReader((E/"wholebody-attachment-source-rebind-037-candidate020-identity004-002/generated/embedding.csv").open())}
skin={int(r["vertex"]):r for r in csv.DictReader((S/"registered-skin.csv").open())}
fem={int(r["global_node"]):r for r in csv.DictReader((S/"initial-skin-fem-nodes.csv").open())}
contact=list(csv.DictReader((S/"initial-skin-bone-contact.csv").open()))
out=[]
for r in contact:
 if int(r["layer"])!=0 or int(r["solid_inside_rays"])!=3 or int(r["solid_ambiguous"]):continue
 col=int(r["row"]);sv=int(embed[col]["source_vertex"]);f=fem[int(r["node"])];v=skin[sv]
 p=np.array([float(f[k]) for k in ["x_m","y_m","z_m"]]);p2=np.array([float(v[k]) for k in ["wx","wy","wz"]])
 out.append(dict(global_fem_node=int(r["node"]),source_column=col,exact_source_vertex=sv,nearest_bone_body=int(r["body"]),inside_depth_m=float(r["distance_m"]),detailed_vs_mechanical_distance_m=float(np.linalg.norm(p-p2)),detailed_skin_world_m=p2.tolist(),mechanical_outer_world_m=p.tolist()))
d=E/"bone-clearance-feasibility-033";v={"scope":"12 confirmed outer-node penetrations mapped to original detailed skin source vertices; no new solid query","rows":out,"max_detailed_to_coarse_position_error_m":max(x["detailed_vs_mechanical_distance_m"] for x in out)}
(d/"outer-source-identity-diagnosis.json").write_text(json.dumps(v,indent=2)+"\n");print(json.dumps(v))

