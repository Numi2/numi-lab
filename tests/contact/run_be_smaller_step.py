#!/usr/bin/env python3
"""One preregistered small-step accuracy run; not a timing benchmark."""
import hashlib,json,pathlib,subprocess,time
root=pathlib.Path(__file__).resolve().parents[2]
evidence=root/"docs/evidence/midpoint-first-failure-20261109"
build=pathlib.Path("/Users/numi/numi-matter-midpoint-reference-build-20261108")
name="be-n2-h001953125"
paths={suffix:evidence/(name+suffix) for suffix in (".json",".log","-analysis.json","-execution.json")}
if any(p.exists() for p in paths.values()):raise SystemExit("attempt already exists")
cmd=[str(build/"metalrobo_plane_contact_probe"),str(build/"NumiMatter.metallib"),"0.000001953125","5120","2","ipc_area","0.0002","physical","contact","reference"]
hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [pathlib.Path(cmd[0]),pathlib.Path(cmd[1]),root/"apps/plane_contact_probe.mm",root/"tests/contact/native_trace_analysis.py"]}
start=time.monotonic()
with paths[".json"].open("w") as out, paths[".log"].open("w") as err:
 try:
  proc=subprocess.run(cmd,stdout=out,stderr=err,timeout=1800);code=proc.returncode;timeout=False
 except subprocess.TimeoutExpired:code=None;timeout=True
meta={"command":cmd,"input_sha256":hashes,"returncode":code,"timed_out":timeout,"wall_seconds_observational_only":time.monotonic()-start}
paths["-execution.json"].write_text(json.dumps(meta,indent=2)+"\n")
if code==0:
 check=subprocess.run(["python3",str(root/"tests/contact/native_trace_analysis.py"),str(paths[".json"]),str(root/"docs/evidence/midpoint-spatial-cost-dynamic-20261108/rk4-n2-h025.json")],capture_output=True,text=True)
 paths["-analysis.json"].write_text(check.stdout)
 meta["analysis_returncode"]=check.returncode;meta["analysis_stderr"]=check.stderr
 paths["-execution.json"].write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta),flush=True)
