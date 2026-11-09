#!/usr/bin/env python3
"""Run the preregistered near-band fixture and retain all outcomes."""
import json,os,pathlib,subprocess,time
root=pathlib.Path(__file__).resolve().parents[2]
p=root/"docs/evidence/midpoint-first-failure-20261109"
b=pathlib.Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109/evidence-stage-native-001")
paths={s:p/("stage-near-001"+s) for s in (".json",".log","-execution.json","-analysis.json")}
if any(f.exists() for f in paths.values()):raise SystemExit("near attempt already exists")
if not json.loads((p/"near-reference-analysis.json").read_text())["passes"]:raise SystemExit("near reference not qualified")
env=os.environ.copy();env["NM_MATTER_MIDPOINT_VELOCITY_COMPENSATION"]="1";env["NM_MATTER_MIDPOINT_STAGE_COMPENSATION"]="1"
cmd=[str(b/"numi_plane_contact_probe_scaled"),str(b/"NumiMatter.metallib"),"0.00000390625","2560","8","ipc_area_midpoint","0.000095","physical","contact","reference"]
start=time.monotonic()
with paths[".json"].open("w") as out,paths[".log"].open("w") as err:
 try:proc=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=2400);code=proc.returncode;timeout=False
 except subprocess.TimeoutExpired:code=None;timeout=True
meta={"command":cmd,"input_hashes":"stage-native-001-inputs.sha256","returncode":code,"timed_out":timeout,"wall_seconds_observational_only":time.monotonic()-start}
paths["-execution.json"].write_text(json.dumps(meta,indent=2)+"\n")
if code==0:
 check=subprocess.run(["python3",str(root/"tests/contact/native_trace_analysis.py"),str(paths[".json"]),str(p/"near-rk4-n8-h025.json")],capture_output=True,text=True)
 paths["-analysis.json"].write_text(check.stdout)
 meta["analysis_returncode"]=check.returncode;meta["analysis_stderr"]=check.stderr
 paths["-execution.json"].write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta),flush=True)
