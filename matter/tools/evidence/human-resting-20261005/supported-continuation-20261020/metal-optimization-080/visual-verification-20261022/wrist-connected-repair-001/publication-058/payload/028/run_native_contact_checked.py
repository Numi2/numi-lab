#!/usr/bin/env python3
"""Run a matched native Human build using existing launch/receipt evidence."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,time
ap=argparse.ArgumentParser()
ap.add_argument("--base-manifest",required=True,type=Path)
ap.add_argument("--app",required=True,type=Path)
ap.add_argument("--runtime",required=True,type=Path)
ap.add_argument("--metallib",required=True,type=Path)
ap.add_argument("--run-dir",required=True,type=Path)
ap.add_argument("--source-root",action="append",type=Path,default=[])
ap.add_argument("--mode",choices=["source-preflight","native-1ms"],required=True)
ap.add_argument("--static-bvh",action="store_true")
ap.add_argument("--env",action="append",default=[])
ap.add_argument("--timeout",type=float,default=500.)
ns=ap.parse_args()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p in [ns.base_manifest,ns.app,ns.runtime,ns.metallib]:
 if not p.is_file():ap.error("missing required input: "+str(p))
base=json.loads(ns.base_manifest.read_text());N=ns.run_dir
argv=list(base["argv"]);argv[0]=str(ns.app)
# This existing reference command's output directory is its fourth argument.
if len(argv)<5:ap.error("invalid reference command")
argv[4]=str(N/"output")
env=dict(base["environment"])
env["DYLD_LIBRARY_PATH"]=str(ns.runtime.parent)
env["NUMI_MATTER_METALLIB_OVERRIDE"]=str(ns.metallib)
if ns.static_bvh:env["NUMI_HUMAN_SKIN_BONE_STATIC_BVH"]="1"
for row in ns.env:
 if "=" not in row:ap.error("--env requires KEY=VALUE")
 key,value=row.split("=",1);env[key]=value
# A copied launch must never overwrite the reference run's diagnostic outputs.
for key,name in {
 "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT":"common-field-failure.json",
 "NUMI_HUMAN_SKIN_REFERENCE_DUMP":"registered-skin.csv",
 "NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP":"registered-skin-poses.json",
 "NUMI_HUMAN_TISSUE_MASS_REFERENCE_DUMP":"tissue-mass-reference.txt",
 "NUMI_HUMAN_INITIAL_SKIN_BONE_WITNESS_CSV":"initial-skin-bone-contact.csv",
 "NUMI_HUMAN_INITIAL_SKIN_BONE_FULLBANK_COMPARISON_CSV":"initial-skin-bone-fullbank-comparison.csv",
 "NUMI_HUMAN_INITIAL_TET_WITNESS_JSON":"initial-tet.json",
 "NUMI_HUMAN_INITIAL_SKIN_FEM_NODES_CSV":"initial-skin-fem-nodes.csv",
}.items():
 if key in env or key=="NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT":env[key]=str(N/name)
if ns.mode=="source-preflight":
 env["NM_FEM_WRITE_NATIVE_TANGENT_SOURCE_ONLY"]="1"
 env["NM_FEM_WRITE_NATIVE_TANGENT_SOURCE"]=str(N/"prepared-tangent.metal")
else:
 env.pop("NM_FEM_WRITE_NATIVE_TANGENT_SOURCE_ONLY",None)
 env.pop("NM_FEM_WRITE_NATIVE_TANGENT_SOURCE",None)
inputs=[ns.app,ns.runtime,ns.metallib]
for value in [*argv[1:],*env.values()]:
 try:
  p=Path(value)
  if p.is_file():inputs.append(p)
 except OSError:pass
for root in ns.source_root:
 if root.is_file():inputs.append(root)
 elif root.is_dir():
  inputs += [p for p in root.rglob("*") if p.is_file() and p.suffix in [".h",".hpp",".inc",".mm",".metal",".metalinc",".cpp"]]
 else:ap.error("missing source root: "+str(root))
cwd=Path(base.get("cwd",os.getcwd()))
manifest=dict(argv=argv,cwd=str(cwd),environment=env,
 base_manifest_path=str(ns.base_manifest),base_manifest_sha256=sha(ns.base_manifest),
 source_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=cwd,text=True).strip(),
 created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 scope="Source preflight only" if ns.mode=="source-preflight" else "Short native integration regression; not anatomy or sustained physiology qualification",
 input_hashes={str(p):sha(p) for p in sorted(set(inputs))})
N.mkdir(exist_ok=False);(N/"output").mkdir()
snapshot=N/"launch-manifest.snapshot.json";snapshot.write_text(json.dumps(manifest,indent=2))
t=time.monotonic();rc=None;error=None
try:
 with (N/"stdout.log").open("x") as out,(N/"stderr.log").open("x") as err:
  rc=subprocess.run(argv,cwd=cwd,env=dict(os.environ,**env),stdout=out,stderr=err,timeout=ns.timeout).returncode
except Exception as ex:error=repr(ex)
out=(N/"stdout.log").read_text();err=(N/"stderr.log").read_text()
marker="native tangent source prepared; source-only initialization intentionally not admitted"
intentional=ns.mode=="source-preflight" and marker in out+err and error is None
# An intentional source-only stop is not evidence that an optional capture
# was implemented. Require each requested initialization diagnostic output.
required_outputs = {
 "NUMI_HUMAN_SKIN_REFERENCE_DUMP": "registered-skin.csv",
 "NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP": "registered-skin-poses.json",
 "NUMI_HUMAN_TISSUE_MASS_REFERENCE_DUMP": "tissue-mass-reference.txt",
 "NUMI_HUMAN_INITIAL_SKIN_BONE_WITNESS_CSV": "initial-skin-bone-contact.csv",
 "NUMI_HUMAN_INITIAL_SKIN_BONE_FULLBANK_COMPARISON_CSV": "initial-skin-bone-fullbank-comparison.csv",
 "NUMI_HUMAN_INITIAL_TET_WITNESS_JSON": "initial-tet.json",
 "NUMI_HUMAN_INITIAL_SKIN_FEM_NODES_CSV": "initial-skin-fem-nodes.csv",
}
missing_outputs = [dict(environment_key=key, path=str(N/name))
 for key,name in required_outputs.items()
 if env.get(key) and (not (N/name).is_file() or (N/name).stat().st_size == 0)]
capture_complete = not missing_outputs
intentional = intentional and capture_complete
receipt=dict(returncode=rc,error=error,wall_seconds=time.monotonic()-t,
 launch_manifest_sha256=sha(snapshot),stdout_sha256=sha(N/"stdout.log"),stderr_sha256=sha(N/"stderr.log"),
 status="SOURCE_ONLY_INTENTIONAL_STOP" if intentional else ("SHORT_RUN_COMPLETED" if rc==0 and error is None else "FAILED"),
 physical_qualification=False,source_only_marker_present=marker in out+err,
 requested_diagnostics_complete=capture_complete,missing_requested_outputs=missing_outputs)
(N/"execution-receipt.json").write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
raise SystemExit(0 if capture_complete and (intentional or (rc==0 and error is None)) else 1)
