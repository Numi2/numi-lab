#!/usr/bin/env python3
"""Resource-bounded preregistered fine BE control, including exact replay."""
import hashlib,json,os,pathlib,shutil,subprocess,time
root=pathlib.Path(__file__).resolve().parents[2]
p=root/"docs/evidence/midpoint-first-failure-20261109"
b=pathlib.Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109/evidence-be-precision-005")
ref=root/"docs/evidence/midpoint-spatial-cost-dynamic-20261108/rk4-n2-h025.json"
name="be-precision-005-n2-h00003125"; paths={s:p/(name+s) for s in (".json",".log","-execution.json","-analysis.json")}
if any(q.exists() for q in paths.values()):raise SystemExit("attempt already exists")
exe=b/"numi_plane_contact_probe_scaled";shader=b/"NumiMatter.metallib"
hashes={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (exe,shader,ref)}
cmd=[str(exe),str(shader),"0.00000003125","320000","2","ipc_area","0.0002","physical","contact","reference"]
env=os.environ.copy();env["NM_MATTER_BE_VELOCITY_COMPENSATION"]="1"
for k in ("NM_MATTER_MIDPOINT_VELOCITY_COMPENSATION","NM_MATTER_MIDPOINT_STAGE_COMPENSATION"):env.pop(k,None)
start=time.monotonic();reason=None;peak=0
with paths[".json"].open("w") as out,paths[".log"].open("w") as err:
 proc=subprocess.Popen(cmd,env=env,stdout=out,stderr=err)
 while proc.poll() is None:
  rss=subprocess.run(["ps","-o","rss=","-p",str(proc.pid)],capture_output=True,text=True)
  try:peak=max(peak,int(rss.stdout.strip()))
  except ValueError:pass
  if peak>4*1024*1024:reason="resident-memory limit"
  elif shutil.disk_usage(p).free<int(3.2*1024**3):reason="free-space limit"
  elif time.monotonic()-start>24*3600:reason="24-hour limit"
  if reason:
   proc.terminate()
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:proc.kill()
   break
  time.sleep(2)
 code=proc.wait()
meta={"command":cmd,"input_sha256":hashes,"flag":"NM_MATTER_BE_VELOCITY_COMPENSATION=1","returncode":code,"stopped_reason":reason,"wall_seconds_observational_only":time.monotonic()-start,"peak_rss_kib":peak,"free_bytes_after":shutil.disk_usage(p).free,"limits":{"rss_kib":4*1024*1024,"free_bytes":int(3.2*1024**3),"wall_seconds":24*3600}}
try:
 state=json.loads(paths[".json"].read_text())
 check=subprocess.run(["python3",str(root/"tests/contact/native_trace_analysis.py"),str(paths[".json"]),str(ref)],capture_output=True,text=True)
 paths["-analysis.json"].write_text(check.stdout)
 meta.update({"analysis_returncode":check.returncode,"analysis_stderr":check.stderr,"all_gates_pass":code==0 and reason is None and check.returncode==0 and state["accepted_steps"]==320000 and state["rejected_steps"]==0 and state["independent_max_local_scaled_without_floor"]<=.005 and state["restore_replay"] and state["continuation_replay"] and state["forced_rollback_exact"] and state["identity_mismatch_rejected"]})
except (ValueError,KeyError) as e:meta.update({"analysis_error":repr(e),"all_gates_pass":False})
paths["-execution.json"].write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta),flush=True)
