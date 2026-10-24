#!/usr/bin/env python3
"""Run hash-pinned source-only all-per-bone containment on a qualified candidate node set."""
import argparse,csv,hashlib,json,os,subprocess,sys
from pathlib import Path
import prepare_candidate_nodes as exporter

ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
BASE=ROOT/"native-candidate020-all-bank-containment-035/source-preflight-003"
BASE_MANIFEST=BASE/"launch-manifest.snapshot.json"
BASE_NODES=BASE/"initial-skin-fem-nodes.csv"
BASE_SUMMARY=BASE/"all-bank-containment.csv.nodes.csv"
CHECKED_RUNNER=ROOT/"native-candidate020-bone-witness-028/run_native_contact_checked.py"
APP=ROOT/"bone-clearance-fit-038-candidate-query/build-001/numi-human-native"
RUNTIME=ROOT/"native-app-main-header-integration-151/build-runtime-004/libmetalrobo.dylib"
METALLIB=Path("/tmp/numi-human-joint-owner-build-033/matter/shaders/NumiMatter.metallib")
APPS=ROOT/"bone-clearance-fit-038-candidate-query/apps"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def check_node_map(path,baseline_nodes):
    with Path(path).open(newline="") as f: result=list(csv.DictReader(f))
    if len(result)!=7233: raise ValueError("candidate all-bank node summary row count mismatch")
    if [int(r["local_node"]) for r in result]!=list(range(7233)):
        raise ValueError("candidate all-bank local-node ordering mismatch")
    if len(baseline_nodes)!=7233: raise ValueError("pinned baseline node map row count mismatch")
    for i,(r,b) in enumerate(zip(result,baseline_nodes)):
        if (int(r["global_node"]),int(r["layer"]),int(r["row"])) != (
            int(b["global_node"]),int(b["layer"]),int(b["row"])):
            raise ValueError(f"candidate all-bank ID/layer/row mismatch at local row {i}")
        counts=sum(int(r[k]) for k in ("aabb_proven_outside_banks","parity_outside_banks",
            "inside_banks","ambiguous_banks","invalid_banks"))
        if counts!=int(r["total_bone_banks"]):
            raise ValueError(f"candidate all-bank coverage does not close at local row {i}")
    totals={key:sum(int(r[field]) for r in result) for key,field in {
      "node_bank_pairs":"total_bone_banks","aabb_proven_outside":"aabb_proven_outside_banks",
      "parity_outside":"parity_outside_banks","inside":"inside_banks",
      "ambiguous":"ambiguous_banks","invalid":"invalid_banks"}.items()}
    if totals["node_bank_pairs"]!=1338105:
        raise ValueError("candidate per-bone inventory/coverage differs from 035 7,233x185 domain")
    if totals["aabb_proven_outside"]+totals["parity_outside"]+totals["inside"]+totals["ambiguous"]+totals["invalid"]!=totals["node_bank_pairs"]:
        raise ValueError("candidate all-bank aggregate coverage does not close")
    return totals

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidate-npz",required=True,type=Path)
    ap.add_argument("--candidate-sha256",required=True)
    ap.add_argument("--candidate-report",required=True,type=Path)
    ap.add_argument("--candidate-report-sha256",required=True)
    ap.add_argument("--baseline-node-sha256",required=True)
    ap.add_argument("--base-manifest-sha256",required=True)
    ap.add_argument("--run-dir",required=True,type=Path)
    a=ap.parse_args()
    for p in (a.candidate_npz,a.candidate_report,BASE_NODES,BASE_MANIFEST,BASE_SUMMARY,CHECKED_RUNNER,APP,RUNTIME,METALLIB):
        if not p.is_file() or p.is_symlink(): raise ValueError(f"required input absent/redirected: {p}")
    if sha(BASE_MANIFEST)!=a.base_manifest_sha256: raise ValueError("035 base launch manifest SHA mismatch")
    if sha(BASE_NODES)!=a.baseline_node_sha256: raise ValueError("035 baseline node CSV SHA mismatch")
    if a.run_dir.exists() or a.run_dir.is_symlink(): raise ValueError("query run directory must be new")
    work=a.run_dir.parent
    if work.exists() or work.is_symlink(): raise ValueError("038 candidate work directory must be new")
    work.mkdir(parents=True)
    candidate_csv=work/"candidate-node-override.csv"
    override_manifest=exporter.prepare(a.candidate_npz,a.candidate_sha256,a.candidate_report,
        a.candidate_report_sha256,BASE_NODES,a.baseline_node_sha256,candidate_csv)
    base=json.loads(BASE_MANIFEST.read_text())
    env=dict(base["environment"])
    # The legacy nearest-only witness remains disabled; this run asks only the existing
    # per-(body,stable-bone) closed-solid/all-bank owner to query candidate positions.
    env.pop("NUMI_HUMAN_INITIAL_SKIN_BONE_WITNESS_CSV",None)
    env["NUMI_HUMAN_INITIAL_SKIN_BONE_ALL_BANK_CSV"]=str(a.run_dir/"all-bank-containment.csv")
    base["environment"]=env
    base["scope"]="candidate-position all-bank node containment only; source-only, no simulation"
    base_copy=work/"base-launch-manifest.json"
    base_copy.write_text(json.dumps(base,indent=2,sort_keys=True)+"\n")
    candidate_csv_sha=sha(candidate_csv)
    args=[
      sys.executable,str(CHECKED_RUNNER),"--base-manifest",str(base_copy),
      "--app",str(APP),"--runtime",str(RUNTIME),"--metallib",str(METALLIB),
      "--run-dir",str(a.run_dir),"--source-root",str(APPS),
      "--mode","source-preflight","--static-bvh",
      "--env",f"NUMI_HUMAN_D37_CANDIDATE_NODE_OVERRIDE_CSV={candidate_csv}",
      "--env",f"NUMI_HUMAN_D37_CANDIDATE_NODE_OVERRIDE_SHA256={candidate_csv_sha}"]
    proc=subprocess.run(args,text=True,capture_output=True)
    (work/"checked-runner.stdout.txt").write_text(proc.stdout)
    (work/"checked-runner.stderr.txt").write_text(proc.stderr)
    if proc.returncode!=0: raise RuntimeError(f"checked source preflight failed ({proc.returncode}); see preserved run receipt")
    receipt_path=a.run_dir/"execution-receipt.json"
    receipt=json.loads(receipt_path.read_text())
    if receipt.get("status")!="SOURCE_ONLY_INTENTIONAL_STOP" or receipt.get("physical_qualification") is not False:
        raise RuntimeError("native source-only runner did not report intentional nonphysical stop")
    stdout=(a.run_dir/"stdout.log").read_text()
    if "d37_candidate_node_override=hash_verified" not in stdout or "d37_initial_skin_bone_all_bank=written" not in stdout:
        raise RuntimeError("candidate override/all-bank owner markers missing")
    candidate_nodes=a.run_dir/"initial-skin-fem-nodes.csv"
    if not candidate_nodes.is_file() or sha(candidate_nodes)!=a.baseline_node_sha256:
        raise RuntimeError("source-only app node output changed; candidate override must affect audit copy only")
    pose_out=a.run_dir/"registered-skin-poses.json"
    baseline_pose=BASE/"registered-skin-poses.json"
    if not pose_out.is_file() or sha(pose_out)!=sha(baseline_pose):
        raise RuntimeError("source-only current pose snapshot differs from pinned 035")
    bank_csv=a.run_dir/"all-bank-containment.csv"
    summary_csv=a.run_dir/"all-bank-containment.csv.nodes.csv"
    inventory_csv=a.run_dir/"all-bank-containment.csv.banks.csv"
    if any(not p.is_file() or p.stat().st_size==0 for p in (bank_csv,summary_csv,inventory_csv)):
        raise RuntimeError("candidate all-bank output set is incomplete")
    with BASE_NODES.open(newline="") as f: baseline_rows=list(csv.DictReader(f))
    totals=check_node_map(summary_csv,baseline_rows)
    report={
      "schema":"numi.human.d37-candidate-all-bank-node-query.v1",
      "scope":"exact-hash-gated candidate node closed-solid containment against per-bone banks; source-only CPU audit",
      "candidate_npz_sha256":a.candidate_sha256,"candidate_report_sha256":a.candidate_report_sha256,
      "candidate_override_csv_sha256":candidate_csv_sha,
      "base_manifest_sha256":a.base_manifest_sha256,"baseline_node_csv_sha256":a.baseline_node_sha256,
      "pose_snapshot_sha256":sha(pose_out),"all_bank_csv_sha256":sha(bank_csv),
      "all_bank_nodes_csv_sha256":sha(summary_csv),"bank_inventory_csv_sha256":sha(inventory_csv),
      "coverage":totals,"physical_qualification":False,"native_simulation_steps":0,
      "limitation":"This is per-node closed-solid containment only; it does not certify triangle or tetrahedron intersection clearance."}
    (work/"candidate-query-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0
if __name__=="__main__":
    try: raise SystemExit(main())
    except Exception as e: print(f"candidate query refused/failed: {e}",file=sys.stderr); raise
