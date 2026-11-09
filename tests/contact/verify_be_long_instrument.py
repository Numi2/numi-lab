#!/usr/bin/env python3
"""Compare the bounded-lifetime instrument to the retained short control."""
import json,os,pathlib,subprocess,time
root=pathlib.Path(__file__).resolve().parents[2];p=root/"docs/evidence/midpoint-first-failure-20261109"
b=pathlib.Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109/evidence-be-precision-005")
name="be-precision-005-short";paths={s:p/(name+s) for s in (".json",".log","-execution.json")}
if any(q.exists() for q in paths.values()):raise SystemExit("attempt already exists")
cmd=[str(b/"numi_plane_contact_probe_scaled"),str(b/"NumiMatter.metallib"),"0.000001","64","2","ipc_area","0.0002","physical","contact","reference"]
env=os.environ.copy();env["NM_MATTER_BE_VELOCITY_COMPENSATION"]="1"
for k in ("NM_MATTER_MIDPOINT_VELOCITY_COMPENSATION","NM_MATTER_MIDPOINT_STAGE_COMPENSATION"):env.pop(k,None)
start=time.monotonic()
with paths[".json"].open("w") as out,paths[".log"].open("w") as err:r=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=180)
meta={"command":cmd,"input_hashes":"be-precision-005-inputs.sha256","returncode":r.returncode,"wall_seconds_observational_only":time.monotonic()-start}
try:
 a=json.loads((p/"be-precision-002-short.json").read_text());d=json.loads(paths[".json"].read_text())
 def clean(x):
  for k in ("wall_seconds","gpu_seconds","accepted_simulated_seconds_per_wall_second"):x.pop(k,None)
  for q in x["traces"]:q.pop("gpu_seconds",None)
 clean(a);clean(d);meta["physical_and_trace_parity"]=a==d
except (ValueError,KeyError) as e:meta.update({"physical_and_trace_parity":False,"error":repr(e)})
paths["-execution.json"].write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta),flush=True)
raise SystemExit(0 if r.returncode==0 and meta["physical_and_trace_parity"] else 1)
