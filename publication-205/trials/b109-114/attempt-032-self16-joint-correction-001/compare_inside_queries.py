from pathlib import Path
import csv,hashlib,json
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
W=ROOT/"D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027"
D=W/"b109-114/attempt-032-self16-joint-correction-001"
BASE=W/"combined-d071-b109-b61-001/candidate.npz"
CAND=D/"candidate-segment-partial-alpha-0p0625.npz"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with Path(p).open(newline="") as f:return list(csv.DictReader(f))
def signed(r):
 d=float(r["nearest_distance_m"])*1000.0
 return -d if r["classification"]=="inside" else d
def main():
 with np.load(BASE,allow_pickle=False) as z:p0=np.asarray(z["points_world_m"],np.float32)
 with np.load(CAND,allow_pickle=False) as z:p1=np.asarray(z["points_world_m"],np.float32)
 a=load(W/"combined-d071-b109-b61-001/audit-130/query/pairs.csv")
 b=load(D/"audit-130/query/pairs.csv")
 key=lambda r:(int(r["query_row"]),int(r["bank_body"]),int(r["bank_stable_id"]))
 da={key(r):r for r in a};db={key(r):r for r in b}
 inside_a={k for k,r in da.items() if r["classification"]=="inside"}
 inside_b={k for k,r in db.items() if r["classification"]=="inside"}
 depth=[]
 for k in sorted(inside_a):
  if k[1:]!=(109,114):continue
  if k not in db:raise ValueError("target inside key missing candidate query "+str(k))
  ra=da[k];rb=db[k]
  n=k[0];motion=float(np.linalg.norm((p1[n].astype(np.float64)-p0[n].astype(np.float64))*1000.))
  depth.append({"node_row":n,"baseline_signed_nearest_depth_mm":signed(ra),"candidate_signed_nearest_depth_mm":signed(rb),"depth_change_mm":signed(rb)-signed(ra),"point_motion_mm":motion,"abs_distance_change_minus_motion_mm":abs(float(rb["nearest_distance_m"])-float(ra["nearest_distance_m"]))*1000-motion,"baseline_classification":ra["classification"],"candidate_classification":rb["classification"]})
 shared=set(da)&set(db);res=[]
 for k in shared:
  ra,rb=da[k],db[k];n=k[0];motion=float(np.linalg.norm((p1[n].astype(np.float64)-p0[n].astype(np.float64))*1000.))
  res.append(abs(float(rb["nearest_distance_m"])-float(ra["nearest_distance_m"]))*1000-motion)
 out={"schema":"numi.human.b109-114.joint-self16-inside-query-comparison.v1","baseline_sha256":sha(BASE),"candidate_sha256":sha(CAND),"baseline_pairs_csv_sha256":sha(W/"combined-d071-b109-b61-001/audit-130/query/pairs.csv"),"candidate_pairs_csv_sha256":sha(D/"audit-130/query/pairs.csv"),"candidate_audit130_sha256":sha(D/"audit-130/summary.json"),"baseline_query_rows":len(a),"candidate_query_rows":len(b),"shared_query_keys":len(shared),"inside_identity_count_baseline":len(inside_a),"inside_identity_count_candidate":len(inside_b),"inside_identity_added":[list(k) for k in sorted(inside_b-inside_a)],"inside_identity_removed":[list(k) for k in sorted(inside_a-inside_b)],"target_sample_depth_count":len(depth),"target_sample_depths":depth,"target_depth_change_min_median_max_mm":[min(x["depth_change_mm"] for x in depth),float(np.median([x["depth_change_mm"] for x in depth])),max(x["depth_change_mm"] for x in depth)],"target_depth_improved_count":sum(x["depth_change_mm"]>1e-9 for x in depth),"target_depth_worsened_count":sum(x["depth_change_mm"] < -1e-9 for x in depth),"classification_changed_target_count":sum(x["baseline_classification"]!=x["candidate_classification"] for x in depth),"max_nearest_distance_motion_bound_residual_mm":max(res,default=0),"all_inside_identities_unchanged":inside_a==inside_b}
 p=D/"inside-query-comparison.json";p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"report":str(p),"sha256":sha(p),"inside_counts":[len(inside_a),len(inside_b)],"added":len(inside_b-inside_a),"removed":len(inside_a-inside_b),"target_depth_stats_mm":out["target_depth_change_min_median_max_mm"],"target_improved_worsened":[out["target_depth_improved_count"],out["target_depth_worsened_count"]],"max_motion_bound_residual_mm":out["max_nearest_distance_motion_bound_residual_mm"]},indent=2))
if __name__=="__main__":main()

