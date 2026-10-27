from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
W=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027")
B=W/"combined-d071-b109-b61-001/candidate.npz"
L=W/"combined-d071-b109-b61-001/skin-fat-layers-541.npz"
C=W/"b109-114/attempt-029-target-ranked-local-fit-001/candidate-alpha-0p25.npz"
OWNER=W/"b65-99/attempt-024-topology-independent-layer-transfer-001/transfer_topology_independent_541.py"
OUT=W/"root-b109-alpha-0p25-regression-001/report.json"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(B)=="806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756"
assert sha(C)=="e09d08936361dbfab4d6f92ecb5fcde3bfbeaeb1ac11a26f2e01fa3eeee18771"
assert sha(L)=="eb41bf0d1c11f1ee0a94672058063c51350e8e0dca1483a6ae7568cb03a78b57"
sp=importlib.util.spec_from_file_location("layer541",OWNER);m=importlib.util.module_from_spec(sp);sys.modules["layer541"]=m;sp.loader.exec_module(m)
with np.load(B,allow_pickle=False) as b,np.load(C,allow_pickle=False) as c,np.load(L,allow_pickle=False) as l:
 assert np.array_equal(b["source_O_faces"],c["source_O_faces"])
 for k in ("source_prolongation_indptr","source_prolongation_node_rows","source_prolongation_weights","prior_input_006__source_prolongation_indptr","prior_input_006__source_prolongation_node_rows","prior_input_006__source_prolongation_weights","prior_run028__source_prolongation_indptr","prior_run028__source_prolongation_node_rows","prior_run028__source_prolongation_weights"):
  assert np.array_equal(b[k],c[k]),k
 o=np.asarray(l["outer_points_f64"]);i=np.asarray(l["inner_points_f64"]);d=np.asarray(l["deep_points_f64"])
 assert np.array_equal(np.asarray(b["points_world_m"],dtype=np.float32),np.asarray(o,dtype=np.float32))
 co=np.asarray(c["points_world_m"],dtype=np.float32).astype(np.float64)
 delta=co-o;ci=i+delta;cd=d+delta
 F=np.asarray(l["outer_faces"])
 result={}
 for name,a0,a1,b0,b1,t in (
  ("skin",o,i,co,ci,l["skin_thickness_transferred_m"]),("fat",i,d,ci,cd,l["fat_thickness_transferred_m"])):
  old=m.face_flags(a0,a1,F,t);new=m.face_flags(b0,b1,F,t);states={}
  for key in ("authored_bad","conditioned","nonpositive"):
   x=np.asarray(old[key],bool);y=np.asarray(new[key],bool)
   added=np.argwhere(~x&y).tolist();removed=np.argwhere(x&~y).tolist()
   states[key]={"before_count":int(x.sum()),"after_count":int(y.sum()),"added":added,"removed":removed}
  result[name]=states
payload={"status":"POSITION_ONLY_LAYER_FLAG_COMPARISON","baseline_sha256":sha(B),"candidate_sha256":sha(C),"layer_sha256":sha(L),"owner_sha256":sha(OWNER),
 "method":"Compare all Boolean prism flags on identical triangles after applying each outer-node displacement equally to its two inner positions; no new layer fitting. Same unchanged offset ancestry was asserted.",
 "physical_admission":False,"results":result}
OUT.write_text(json.dumps(payload,indent=2)+"\n")
for name,states in result.items():
 for key,s in states.items():print(name,key,"before",s["before_count"],"after",s["after_count"],"added",len(s["added"]),"removed",len(s["removed"]))
print("report sha",sha(OUT))
