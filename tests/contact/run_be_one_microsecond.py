#!/usr/bin/env python3
"""Retain one further backward-Euler accuracy trial, never a benchmark."""
import hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path(__file__).resolve().parents[2]
p=root/"docs/evidence/midpoint-first-failure-20261109"
b=pathlib.Path("/Users/numi/numi-matter-midpoint-reference-build-20261108/evidence-be-h001953125")
ref=root/"docs/evidence/midpoint-spatial-cost-dynamic-20261108/rk4-n2-h025.json"
prev=pathlib.Path("/Users/numi/numi-lab-matter-midpoint-reference-20261108/docs/evidence/midpoint-first-failure-20261109/be-n2-h001953125-execution.json")
previous=json.loads(prev.read_text())
print("prior metadata", json.dumps(previous),flush=True)
name="be-n2-h0001";paths={s:p/(name+s) for s in (".json",".log","-execution.json","-analysis.json")}
if any(q.exists() for q in paths.values()):raise SystemExit("attempt exists")
exe=b/"metalrobo_plane_contact_probe";shader=b/"NumiMatter.metallib"
hashes={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (exe,shader)}
for q in (exe,shader):
 expected=next((v for k,v in previous["input_sha256"].items() if pathlib.Path(k).name==q.name),None)
 if hashes[str(q)]!=expected:raise SystemExit("frozen input does not match previous trial: "+q.name)
cmd=[str(exe),str(shader),"0.000001","10000","2","ipc_area","0.0002","physical","contact","reference"]
env=os.environ.copy();env.pop("NM_MATTER_MIDPOINT_VELOCITY_COMPENSATION",None);env.pop("NM_MATTER_MIDPOINT_STAGE_COMPENSATION",None)
start=time.monotonic()
with paths[".json"].open("w") as out,paths[".log"].open("w") as err:
 try:r=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=1800);code=r.returncode;timeout=False
 except subprocess.TimeoutExpired:code=None;timeout=True
meta={"command":cmd,"hashes":hashes,"returncode":code,"timed_out":timeout,"wall_seconds_observational_only":time.monotonic()-start}
try:
 state=json.loads(paths[".json"].read_text())
 result=subprocess.run(["python3",str(root/"tests/contact/native_trace_analysis.py"),str(paths[".json"]),str(ref)],capture_output=True,text=True)
 paths["-analysis.json"].write_text(result.stdout)
 meta.update({"analysis_returncode":result.returncode,"analysis_stderr":result.stderr,"all_gates_pass":code==0 and result.returncode==0 and state["independent_max_local_scaled_without_floor"]<=.005 and state["continuation_replay"] and state["identity_mismatch_rejected"]})
except (ValueError,KeyError) as exc:meta.update({"analysis_error":repr(exc),"all_gates_pass":False})
paths["-execution.json"].write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta),flush=True)
