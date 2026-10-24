#!/usr/bin/env python3
"""Prepare and run a source-only exact-bank query for a pinned 040 candidate."""
from pathlib import Path
import argparse, csv, hashlib, json, subprocess, sys
import numpy as np

ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=ROOT/"bone-clearance-fit-038-exact-bank-export-039"
FROZEN=D/"run-002/source-preflight"
BASE=D/"run-002/base-launch-manifest.json"
BASE_NODES=ROOT/"native-candidate020-all-bank-containment-035/source-preflight-003/initial-skin-fem-nodes.csv"
CHECKED=ROOT/"publication-058/payload/028/run_native_contact_checked.py"
APP=D/"build-001/numi-human-native"
RUNTIME=ROOT/"native-app-main-header-integration-151/build-runtime-004/libmetalrobo.dylib"
METALLIB=Path("/tmp/numi-human-joint-owner-build-033/matter/shaders/NumiMatter.metallib")
APPS=D/"apps"
ROUTE_DIR=ROOT/"bone-clearance-fit-040-shared-columns/native-source-route-requery-001"
PIN={
 BASE:"617dee1b93d08960daea7d006602b7df02e6f00e693de3b419aba7d424b1b655",
 BASE_NODES:"53318beccf8c2021b3f3a21a781feb4cf8343513bd599d39fe35221de92aaa4b",
 CHECKED:"67f0a7d6d37164765a730d029b1278e5f5b845833fccef13fb6d24efe9eea56d",
 APP:"dc463fb9c341448e0a7d4accd52ba392c4bd757c8226541de0890462bd04b6ab",
 RUNTIME:"b7cba8777b071eb1df0dbdadc24f9a4963eb92b2e85bf1121a97d29328d5c26b",
 METALLIB:"ae3e0c4ce1bd720cb52174c72d70f60c5eab984cade9c1e76c0f4ac3c29da125",
 D/"run-002/source-preflight/launch-manifest.snapshot.json":"e45c6709dfcc66518dd97e7a678b4294e31ef530ee6ce26cdabec1a12a88f565",
 FROZEN/"exact-emitted.triangles.csv":"0d3418064af1d4f34d19a1ec91992afceacaaa70b6c982d4a2637ded1d829eac",
 FROZEN/"exact-emitted.bodies.csv":"dae52ab190e02586421b56381b03a4453aff1058edca7b6a42afcb9051b50b90",
 ROOT/"root_exact_segment_witness.py":"d6a2efe451e7743b659d74c274d1372a5ed2a765511a8671b2c1a2370b7d03a2",
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
 if not ok: raise RuntimeError(msg)
def validate_pins():
 for p,h in PIN.items():
  require(p.is_file() and not p.is_symlink(),f"missing/redirected frozen input: {p}")
  require(sha(p)==h,f"frozen input hash mismatch: {p}")
 require(sha(APPS/"NumiHumanD37ExactBankDump.hpp")=="c116306940faa97b7ea22b3dea138491c7ed9879a5b6abd8e4785cb412a52fe8","039 source header changed")
 require(sha(APPS/"numilab_human_myosim_visual_probe.mm")=="72b0726f3c6190f36e706dcb83f3ecb4b5c4f3370c8abacabf57e9b298b1c762","039 source app changed")
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--candidate-npz",required=True,type=Path)
 ap.add_argument("--candidate-sha256",required=True)
 ap.add_argument("--candidate-report",required=True,type=Path)
 ap.add_argument("--candidate-report-sha256",required=True)
 ap.add_argument("--output-dir",required=True,type=Path)
 ap.add_argument("--timeout",type=float,default=500.)
 a=ap.parse_args();validate_pins()
 cand=Path(a.candidate_npz);rep=Path(a.candidate_report);out=Path(a.output_dir)
 require(cand.is_file() and not cand.is_symlink() and sha(cand)==a.candidate_sha256,"candidate NPZ pin invalid")
 require(rep.is_file() and not rep.is_symlink() and sha(rep)==a.candidate_report_sha256,"candidate report pin invalid")
 fr=json.loads(rep.read_text());final=fr.get("final",{})
 require(fr.get("schema")=="numi.human.shared-column-bone-entry-fit.v1","wrong candidate report schema")
 require(fr.get("candidate_sha256")==a.candidate_sha256,"candidate report does not pin NPZ")
 require(fr.get("status") in ("candidate_basic_geometry_pass_not_admitted","bounded_fit_passed_basic_gates"),"candidate basic geometry status is not passing")
 require(final.get("basic_gate_pass") is True,"candidate basic geometry gate failed")
 for k in ("float32_nonpositive_tets","float32_q_below_1e-5","float32_J_below_.35","float32_J_above_2.5","constraint_violation_count","route_endpoint_violation_count","baseline_O12_violation_count"):
  require(int(final.get(k,-1))==0,f"candidate basic gate failed: {k}")
 require(float(final["min_q_f32"])>=1e-5 and .35<=float(final["min_J_f32"])<=float(final["max_J_f32"])<=2.5,"candidate Float32 q/J range failed")
 lp=final.get("layer_projection",{})
 require(int(lp.get("skin_floor_violations",-1))==0 and int(lp.get("fat_floor_violations",-1))==0,"candidate layer floors failed")
 with np.load(cand,allow_pickle=False) as z:
  require("points_world_f32" in z.files,"candidate lacks points_world_f32")
  idkey="global_node_ids" if "global_node_ids" in z.files else "global_nodes"
  require(idkey in z.files,"candidate lacks global node IDs")
  X=np.asarray(z["points_world_f32"]);ids=np.asarray(z[idkey])
 require(X.dtype==np.float32 and X.shape==(7233,3) and np.isfinite(X).all(),"candidate position array invalid")
 require(ids.dtype.kind in "iu" and ids.shape==(7233,),"candidate global node IDs invalid")
 with BASE_NODES.open(newline="") as f:
  rd=csv.DictReader(f)
  require(tuple(rd.fieldnames or ())==("global_node","layer","row","x_m","y_m","z_m","mass_kg"),"baseline node CSV schema mismatch")
  base=list(rd)
 gids=np.asarray([int(r["global_node"]) for r in base],np.int64)
 require(len(base)==7233 and np.array_equal(ids.astype(np.int64),gids),"candidate IDs/order differ from 035 baseline")
 require(not out.exists() and not out.is_symlink(),"output directory must be new")
 out.mkdir(parents=True)
 override=out/"candidate-node-override.csv"
 with override.open("x",newline="") as f:
  w=csv.writer(f,lineterminator="\n");w.writerow(("global_node","layer","row","x_m","y_m","z_m","mass_kg"))
  for r,x in zip(base,X):
   w.writerow((r["global_node"],r["layer"],r["row"],*(format(float(v),".9g") for v in x),r["mass_kg"]))
 with override.open(newline="") as f: rows=list(csv.DictReader(f))
 require(len(rows)==7233,"candidate override row count mismatch")
 for i,r in enumerate(rows):
  y=np.asarray([np.float32(r[k]) for k in ("x_m","y_m","z_m")],np.float32)
  require(np.array_equal(y.view(np.uint32),X[i].view(np.uint32)),f"candidate override Float32 mismatch at {i}")
 override_sha=sha(override)
 base=json.loads(BASE.read_text());env=dict(base["environment"])
 env.pop("NUMI_HUMAN_INITIAL_SKIN_BONE_WITNESS_CSV",None)
 env.update({"NUMI_HUMAN_D37_CANDIDATE_NODE_OVERRIDE_CSV":str(override),
  "NUMI_HUMAN_D37_CANDIDATE_NODE_OVERRIDE_SHA256":override_sha,
  "NUMI_HUMAN_INITIAL_SKIN_BONE_ALL_BANK_CSV":str(out/"source-preflight/all-bank-containment.csv"),
  "NUMI_HUMAN_D37_EXACT_EMITTED_BANK_DUMP_PREFIX":str(out/"source-preflight/exact-emitted")})
 base["environment"]=env;base["scope"]="040 candidate source-only exact D37 query; no simulation or physical admission"
 derived=out/"base-launch-manifest.json";derived.write_text(json.dumps(base,indent=2,sort_keys=True)+"\n")
 run=out/"source-preflight"
 command=[sys.executable,str(CHECKED),"--base-manifest",str(derived),"--app",str(APP),"--runtime",str(RUNTIME),
  "--metallib",str(METALLIB),"--run-dir",str(run),"--source-root",str(APPS),"--mode","source-preflight",
  "--static-bvh","--timeout",str(a.timeout)]
 proc=subprocess.run(command,text=True,capture_output=True)
 (out/"checked-runner.stdout.txt").write_text(proc.stdout);(out/"checked-runner.stderr.txt").write_text(proc.stderr)
 result={"schema":"numi.human.040-candidate-source-query.v1","candidate_npz":str(cand),"candidate_npz_sha256":sha(cand),
  "candidate_report":str(rep),"candidate_report_sha256":sha(rep),"candidate_fit_status":fr["status"],
  "candidate_global_id_field":idkey,"override_csv":str(override),"override_csv_sha256":override_sha,
  "base_manifest_sha256":sha(BASE),"checked_runner_sha256":sha(CHECKED),"native_app_sha256":sha(APP),
  "runtime_sha256":sha(RUNTIME),"metallib_sha256":sha(METALLIB),"exact_segment_witness_sha256":sha(ROOT/"root_exact_segment_witness.py"),"candidate_route_runner_sha256":sha(ROUTE_DIR/"rerun_material_routes_041_candidate.py"),"candidate_route_adapter_sha256":sha(ROUTE_DIR/"convert_route_planes_041_candidate.py"),"runner_returncode":proc.returncode,
  "source_only":True,"physical_qualification":False,"candidate_rows":7233}
 if proc.returncode!=0:
  result["status"]="SOURCE_QUERY_FAILED_OUTPUTS_RETAINED";result["stderr_tail"]=proc.stderr[-4000:]
  (out/"source-query-report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  print(json.dumps(result,indent=2,sort_keys=True));return 2
 receipt=run/"execution-receipt.json";require(receipt.is_file(),"checked runner receipt missing")
 rr=json.loads(receipt.read_text())
 require(rr.get("status")=="SOURCE_ONLY_INTENTIONAL_STOP" and rr.get("physical_qualification") is False,"run was not source-only")
 for name in ("all-bank-containment.csv","all-bank-containment.csv.banks.csv","all-bank-containment.csv.nodes.csv",
  "exact-emitted.triangles.csv","exact-emitted.nodes.csv","exact-emitted.bodies.csv","registered-skin-poses.json"):
  f=run/name;require(f.is_file() and f.stat().st_size>0,f"source-only output missing: {name}")
 tri=run/"exact-emitted.triangles.csv";bodies=run/"exact-emitted.bodies.csv"
 require(sha(tri)==PIN[FROZEN/"exact-emitted.triangles.csv"],"native exact triangle export changed from frozen 039")
 require(sha(bodies)==PIN[FROZEN/"exact-emitted.bodies.csv"],"native body transforms changed from frozen 039")
 result.update({"status":"SOURCE_QUERY_PASS_BANKS_IDENTICAL_TO_039","execution_receipt_sha256":sha(receipt),
  "nodes_sha256":sha(run/"exact-emitted.nodes.csv"),"all_bank_query_sha256":sha(run/"all-bank-containment.csv"),
  "bank_inventory_sha256":sha(run/"all-bank-containment.csv.banks.csv"),
  "bank_node_classification_sha256":sha(run/"all-bank-containment.csv.nodes.csv"),
  "triangles_sha256":sha(tri),"bodies_sha256":sha(bodies),"query_run_dir":str(run)})
 result["route_audit_command"]=[sys.executable,str(ROUTE_DIR/"rerun_material_routes_041_candidate.py"),
  "--root",str(ROOT),"--bank-export-dir",str(FROZEN),"--query-run-dir",str(run),
  "--candidate-npz",str(cand),"--candidate-sha256",sha(cand),"--candidate-report",str(rep),
  "--candidate-report-sha256",sha(rep),"--candidate-override-csv",str(override),"--output-dir",str(out/"routes-041")]
 (out/"source-query-report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=="__main__": raise SystemExit(main())

