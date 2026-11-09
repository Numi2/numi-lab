#!/usr/bin/env python3
"""Retain the BE precision full-contact trial; timing includes diagnostics."""
import hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path(__file__).resolve().parents[2]
p=root/"docs/evidence/midpoint-first-failure-20261109"
b=pathlib.Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109/evidence-be-precision-002")
ref=root/"docs/evidence/midpoint-spatial-cost-dynamic-20261108/rk4-n2-h025.json"
name="be-precision-002-full";paths={s:p/(name+s) for s in (".json",".log","-execution.json","-analysis.json")}
if any(q.exists() for q in paths.values()):raise SystemExit("attempt already exists")
exe=b/"numi_plane_contact_probe_scaled";shader=b/"NumiMatter.metallib"
hashes={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (exe,shader,ref)}
cmd=[str(exe),str(shader),"0.000001","10000","2","ipc_area","0.0002","physical","contact","reference"]
env=os.environ.copy();env["NM_MATTER_BE_VELOCITY_COMPENSATION"]="1"
for k in ("NM_MATTER_MIDPOINT_VELOCITY_COMPENSATION","NM_MATTER_MIDPOINT_STAGE_COMPENSATION"):env.pop(k,None)
start=time.monotonic()
with paths[".json"].open("w") as out,paths[".log"].open("w") as err:
 try:r=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=2400);code=r.returncode;timed_out=False
 except subprocess.TimeoutExpired:code=None;timed_out=True
meta={"command":cmd,"input_sha256":hashes,"flag":"NM_MATTER_BE_VELOCITY_COMPENSATION=1","returncode":code,"timed_out":timed_out,"wall_seconds_observational_only":time.monotonic()-start}
try:
 state=json.loads(paths[".json"].read_text())
 check=subprocess.run(["python3",str(root/"tests/contact/native_trace_analysis.py"),str(paths[".json"]),str(ref)],capture_output=True,text=True)
 paths["-analysis.json"].write_text(check.stdout)
 meta.update({"analysis_returncode":check.returncode,"analysis_stderr":check.stderr,"all_gates_pass":code==0 and check.returncode==0 and state["accepted_steps"]==10000 and state["rejected_steps"]==0 and state["independent_max_local_scaled_without_floor"]<=.005 and state["restore_replay"] and state["continuation_replay"] and state["forced_rollback_exact"] and state["identity_mismatch_rejected"]})
except (ValueError,KeyError) as e:meta.update({"analysis_error":repr(e),"all_gates_pass":False})
paths["-execution.json"].write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta),flush=True)
