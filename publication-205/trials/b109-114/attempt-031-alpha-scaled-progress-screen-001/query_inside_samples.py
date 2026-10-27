from pathlib import Path
import csv, hashlib, json, subprocess, time
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
S=ROOT/"D/surface-clearance-546/component-sat-selector-638"
W=S/"per-bank-workers-20261027"
D=W/"b109-114/attempt-031-alpha-scaled-progress-screen-001"
BASE=W/"combined-d071-b109-b61-001/candidate.npz"
CAND=D/"candidate-segment-partial-alpha-0p0625.npz"
BASE_PAIRS=W/"combined-d071-b109-b61-001/audit-130/query/pairs.csv"
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
BANK=E/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
EXPORTER=E/"E/generic-static-query-042/prepare_generic_query_points.py"
BINARY=Path("/tmp/generic-query-overlay-build-042-work/matter/numi-matter-human-skin-fat-static-query")
PY="/Users/numi/numi-human-prep-venv-20261005/bin/python"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 t=time.monotonic()
 with np.load(BASE,allow_pickle=False) as z: p0=np.asarray(z["points_world_m"],np.float32)
 with np.load(CAND,allow_pickle=False) as z: p1=np.asarray(z["points_world_m"],np.float32)
 rows=[]
 with BASE_PAIRS.open(newline="") as f:
  for r in csv.DictReader(f):
   if int(r["bank_body"])==109 and int(r["bank_stable_id"])==114 and r["classification"]=="inside":
    rows.append(int(r["query_row"]))
 rows=sorted(set(rows))
 if len(rows)!=24: raise ValueError("expected 24 pinned inside 109/114 samples, got "+str(len(rows)))
 pts=np.concatenate([p0[rows],p1[rows]],axis=0).astype(np.float32)
 pin=D/"inside-sample-points.npz"
 with pin.open("xb") as f: np.savez_compressed(f,points_world_f32=pts)
 points_csv=D/"inside-sample-points.csv"
 cmd=[PY,str(EXPORTER),"--candidate-npz",str(pin),"--input-sha256",sha(pin),"--field","points_world_f32","--output",str(points_csv)]
 cp=subprocess.run(cmd,capture_output=True,text=True)
 (D/"inside-sample-export.log").write_text(cp.stdout+cp.stderr)
 if cp.returncode: raise RuntimeError("point export failed "+str(cp.returncode))
 pairs=D/"inside-sample-query-pairs.csv"; nodes=D/"inside-sample-query-nodes.csv"
 cmd=[str(BINARY),str(BANK/"exact-emitted.triangles.csv"),str(BANK/"exact-emitted.bodies.csv"),str(points_csv),str(pairs),str(nodes),"185"]
 cp=subprocess.run(cmd,capture_output=True,text=True)
 (D/"inside-sample-query.log").write_text(cp.stdout+cp.stderr)
 if cp.returncode: raise RuntimeError("static query failed "+str(cp.returncode))
 q={}
 with pairs.open(newline="") as f:
  for r in csv.DictReader(f):
   if int(r["bank_body"])==109 and int(r["bank_stable_id"])==114:
    q[int(r["query_row"])]=r
 if len(q)!=48: raise ValueError("expected 48 exact target-bank query rows, got "+str(len(q)))
 rec=[]
 for i,n in enumerate(rows):
  a=q[i];b=q[i+len(rows)]
  da=float(a["nearest_distance_m"])*1000.0; db=float(b["nearest_distance_m"])*1000.0
  sa=-da if a["classification"]=="inside" else da
  sb=-db if b["classification"]=="inside" else db
  motion=float(np.linalg.norm((p1[n].astype(np.float64)-p0[n].astype(np.float64))*1000.0))
  rec.append({"node_row":n,"baseline_classification":a["classification"],"candidate_classification":b["classification"],"baseline_nearest_depth_signed_mm":sa,"candidate_nearest_depth_signed_mm":sb,"signed_depth_change_mm":sb-sa,"nearest_distance_change_mm":db-da,"point_motion_mm":motion,"distance_change_bound_residual_mm":abs(db-da)-motion})
 out={"schema":"numi.human.b109-114.scaled-backtrack-inside-sample-query.v1","baseline_candidate_sha256":sha(BASE),"candidate_sha256":sha(CAND),"candidate_points_sha256":sha(pin),"exact_binary_sha256":sha(BINARY),"static_bank_triangle_sha256":sha(BANK/"exact-emitted.triangles.csv"),"static_bank_body_sha256":sha(BANK/"exact-emitted.bodies.csv"),"bank":[109,114],"sampled_node_count":len(rows),"query_row_count":len(q),"rows":rec,"classifications_changed":sum(x["baseline_classification"]!=x["candidate_classification"] for x in rec),"inside_depth_improved_count":sum(x["signed_depth_change_mm"]>1e-9 for x in rec),"inside_depth_worsened_count":sum(x["signed_depth_change_mm"] < -1e-9 for x in rec),"max_motion_bound_residual_mm":max((x["distance_change_bound_residual_mm"] for x in rec),default=0),"elapsed_seconds":time.monotonic()-t}
 path=D/"inside-sample-depth-report.json";path.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"report":str(path),"sha256":sha(path),"sampled_nodes":len(rows),"inside_improved":out["inside_depth_improved_count"],"worsened":out["inside_depth_worsened_count"],"classification_changes":out["classifications_changed"],"max_motion_bound_residual_mm":out["max_motion_bound_residual_mm"],"depth_change_min_median_max_mm":[min(x["signed_depth_change_mm"] for x in rec),float(np.median([x["signed_depth_change_mm"] for x in rec])),max(x["signed_depth_change_mm"] for x in rec)]},indent=2))
if __name__=="__main__": main()

