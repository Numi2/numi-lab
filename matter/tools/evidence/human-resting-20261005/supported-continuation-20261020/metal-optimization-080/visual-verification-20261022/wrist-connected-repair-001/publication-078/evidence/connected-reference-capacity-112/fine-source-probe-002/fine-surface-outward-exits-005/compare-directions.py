import json,hashlib
from pathlib import Path
import numpy as np
D=Path("/Users/n/Work/numi-human-connected-surface-069")
R=D/"connected-reference-capacity-112/fine-source-probe-002/fine-surface-outward-exits-005/planes.json"
C=D/"connected-cell-witness-103/run-001/baseline/candidate.npz"
OLD=D/"connected-reference-capacity-112/fine-source-probe-002/first-outward-exits-004/planes.json"
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
r=json.loads(R.read_text());old=json.loads(OLD.read_text())
oldmap={(int(x["column_row"]),x["layer"],int(x["body"]),int(x["bone_stable_id"])):x["first_exit_distance_m"] for x in old["results"]}
P=np.load(C,allow_pickle=False)["points_world_f32"]
rows=[]
for x in r["results"]:
 c=int(x["column_row"]);n=np.asarray(x["fine_surface_outward_normal_world_unit"]);v=P[c+2411].astype(np.float64)-P[c].astype(np.float64)
 rows.append({"column_row":c,"layer":x["layer"],"query_row":int(x["query_row"]),"bank":[int(x["body"]),int(x["bone_stable_id"])],
  "fine_normal_dot_current_O_to_I_m":float(n@v),"fine_exit_distance_m":x["first_exit_distance_m"],
  "OD_exit_distance_m":oldmap.get((c,x["layer"],int(x["body"]),int(x["bone_stable_id"]))),
  "fine_plane_deficit_at_start_m":x["plane_deficit_at_start_m"],
  "fine_vs_OD_direction_cosine":x["fine_normal_dot_098_source_OD_direction"]})
out={"schema":"fine-surface-vs-OD-exit-comparison-005","input_report_sha256":sha(R),"old_OD_report_sha256":sha(OLD),
 "baseline_candidate_sha256":sha(C),"interpretation":"negative fine-normal dot O-to-I confirms the signed BFS/layer-order choice points opposite the current inward O-to-I displacement at each sampled column; this is a construction orientation consistency check, not anatomical proof.",
 "all_fine_normals_opposed_current_O_to_I":all(x["fine_normal_dot_current_O_to_I_m"]<0 for x in rows),"rows":rows}
p=R.parent/"direction-comparison.json";p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps({"path":str(p),"sha256":sha(p),"rows":len(rows),"all_opposed":out["all_fine_normals_opposed_current_O_to_I"],"min_OI_dot_mm":min(x["fine_normal_dot_current_O_to_I_m"] for x in rows)*1000},indent=2))
