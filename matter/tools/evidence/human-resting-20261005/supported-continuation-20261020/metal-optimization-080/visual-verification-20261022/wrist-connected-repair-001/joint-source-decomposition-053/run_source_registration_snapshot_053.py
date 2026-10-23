#!/usr/bin/env python3
"""Matched current294 source-only pose observations; no physical horizon."""
from pathlib import Path
import argparse,csv,hashlib,json,os,subprocess,time
HERE=Path(__file__).resolve().parent;E=HERE.parent
CASES={"reference":{},"shoulder-elevation-050":{36:.5,74:.5},"elbow-flexion-080":{39:.8,77:.8},"knee-flexion-025":{106:.25,120:.25}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--case",choices=CASES,required=True);p.add_argument("--construction-only",action="store_true");a=p.parse_args()
 original=E/"current294-joint-source-check-052"/a.case/"invocation.json";b=json.loads(original.read_text())
 for k in ("binary","private_runtime_library","metallib"):
  if sha(b[k]["path"])!=b[k]["sha256"]:raise RuntimeError("changed "+k)
 d=HERE/(a.case+(".construction-only" if a.construction_only else ""));d.mkdir(exist_ok=False)
 argv=list(b["argv"]);argv[4]=str(d/"output")
 env=dict(b["actual_applied_simulation_environment"])
 env["NUMI_HUMAN_SKIN_REFERENCE_DUMP"]=str(d/"registered-skin.csv")
 env["NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP"]=str(d/"registered-skin-poses.json")
 env["NM_FEM_WRITE_NATIVE_TANGENT_SOURCE"]=str(d/"prepared-tangent.metal")
 env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(d/"common-field-failure.json")
 if a.construction_only:
  # Existing construction owner: source q with a fixed +0.35m root lift,
  # projected by loaded NHEQ. No source-support fitting or physical horizon.
  i=argv.index("--root-pose");del argv[i:i+8]
  env["NUMI_HUMAN_ROPE_ANCHOR"]="0,0,0.8,0.8"
  env["NUMI_HUMAN_ROPE_SITE"]="-0.0545973483,0.1411030829,-0.00279765767"
 if env.get("NM_FEM_WRITE_NATIVE_TANGENT_SOURCE_ONLY")!="1":raise RuntimeError("source-only return missing")
 inv={**b,"template":str(original),"template_sha256":sha(original),"argv":argv,"actual_applied_simulation_environment":env,"construction_only":a.construction_only,"boundary":"Exact current294 source initialization and existing all-binding pose serializer; no physical horizon, no support/contact/ROM qualification. Construction-only modes use the existing declared fixed root-lift owner solely for source geometry."}
 (d/"invocation.json").write_text(json.dumps(inv,indent=2,sort_keys=True)+"\n")
 t=time.monotonic()
 with (d/"stdout.log").open("wb") as so,(d/"stderr.log").open("wb") as se:r=subprocess.run(argv,env={**os.environ,**env},cwd="/Users/n/numi-human-aortic-contact-opt-20261019",stdout=so,stderr=se,timeout=100)
 text=(d/"stdout.log").read_text();err=(d/"stderr.log").read_text()
 if any(x in text for x in ("native_horizon_begin","authoritative_segment_begin","resting_FEM_terminal_audit=passed")):raise RuntimeError("unexpected physical horizon")
 csvpath=d/"registered-skin.csv";posepath=d/"registered-skin-poses.json";rows=0
 if csvpath.exists():
  with csvpath.open() as f:
   for row in csv.DictReader(f):
    if int(row["vertex"])!=rows:raise RuntimeError("unordered CSV")
    rows+=1
 snap=json.loads(posepath.read_text()) if posepath.exists() else {}
 result={"native_exit_code":r.returncode,"wall_seconds":time.monotonic()-t,"registered_rows":rows,"registered_csv_sha256":sha(csvpath) if csvpath.exists() else None,"paired_pose_snapshot_sha256":sha(posepath) if posepath.exists() else None,"paired_pose_bodies":len(snap.get("bodies",[])),"physical_horizon_executed":False,"last_error_lines":err.splitlines()[-3:],"scope":inv["boundary"]}
 (d/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result),flush=True)
 if rows!=61950 or len(snap.get("bodies",[]))!=86:raise RuntimeError("incomplete paired source snapshot")
 if not a.construction_only:
  prior=Path(original).parent/"registered-skin.csv"
  if sha(csvpath)!=sha(prior):raise RuntimeError("paired observation perturbed source geometry")
if __name__=="__main__":main()
