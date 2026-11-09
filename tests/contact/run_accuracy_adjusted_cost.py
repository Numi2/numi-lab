#!/usr/bin/env python3
"""Run one complete cost pair only after both accuracy gates pass."""
import gzip,hashlib,json,math,os,pathlib,shutil,signal,subprocess,sys,time
root=pathlib.Path(__file__).resolve().parents[2];p=root/"docs/evidence/midpoint-first-failure-20261109"
b=pathlib.Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109/evidence-be-precision-005")
manifest=p/"be-precision-005-inputs.sha256"
from cost_provenance import verify_accuracy_provenance
from cost_selection import select_largest_certified
if len(sys.argv)!=4:raise SystemExit("usage: cost.py MIDPOINT_PREFIX BE_PREFIX OUTPUT_TAG")
mid,be,tag=sys.argv[1:]
if not tag.replace("-","").isalnum() or any("/" in x for x in (mid,be)):raise SystemExit("invalid evidence name")
def load(name,suffix):
 path=p/(name+suffix); compressed=path.with_name(path.name+".gz")
 if path.exists() and compressed.exists():raise SystemExit("ambiguous evidence: "+str(path))
 if compressed.exists():
  with gzip.open(compressed,"rt") as f:return json.load(f)
 return json.loads(path.read_text())
def qualified(name,arm):
 d=load(name,".json");a=load(name,"-analysis.json");e=load(name,"-execution.json")
 verify_accuracy_provenance(e, manifest, b)
 if not (a.get("passes") and e.get("all_gates_pass") and d["arm"]==arm and d["rejected_steps"]==0 and d["restore_replay"] and d["continuation_replay"] and d["forced_rollback_exact"] and d["identity_mismatch_rejected"]):raise SystemExit("accuracy gate not satisfied: "+name)
 return d
verify=lambda e: verify_accuracy_provenance(e, manifest, b)
selected_mid, midpoint_ladder=select_largest_certified(p,"stage-ladder-n2-h*","ipc_area_midpoint",verify)
selected_be, be_ladder=select_largest_certified(p,"be-precision-006-n2-h*","ipc_area",verify)
if (mid,be)!=(selected_mid,selected_be):raise SystemExit("use largest certified tested timesteps: "+selected_mid+" "+selected_be)
m=qualified(mid,"ipc_area_midpoint");c=qualified(be,"ipc_area")
for key in ("divisions","nodes","initial_gap","contact_area_m2","contact_area_weights_m2","area_ipc_k_N_per_m5","residual_policy","coordinate_mode","freefall_control","mechanical_relative_tolerance"):
 if m[key]!=c[key]:raise SystemExit("mismatched physical input: "+key)
for key in ("mechanical_absolute_tolerance_Ns","mechanical_impulse_scale_Ns"):
 if not math.isclose(m[key]/m["h"],c[key]/c["h"],rel_tol=2e-6):raise SystemExit("inconsistent per-time impulse scale: "+key)
exe=b/"metalrobo_plane_contact_benchmark";shader=b/"NumiMatter.metallib"
hashes={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (exe,shader)}
def active_numi_workers():
 lines=subprocess.check_output(["ps","-Ao","pid=,comm="],text=True).splitlines()
 workers=[]
 for line in lines:
  parts=line.strip().split(None,1)
  if len(parts)==2 and (pathlib.Path(parts[1]).name.startswith("numi_") or
                         pathlib.Path(parts[1]).name.startswith("metalrobo_")):
   workers.append({"pid":int(parts[0]),"executable":parts[1]})
 return workers
results=[]
for name,d,arm in ((be,c,"ipc_area"),(mid,m,"ipc_area_midpoint")):
 workers=active_numi_workers()
 if workers:raise SystemExit("recognized Numi workload is active; wait for an isolated measurement: "+json.dumps(workers))
 if shutil.disk_usage("/System/Volumes/Data").free<3.1*1024**3:raise SystemExit("storage safety stop before benchmark")
 thermal=subprocess.run(["pmset","-g","therm"],capture_output=True,text=True).stdout.strip()
 path=p/(tag+"-"+arm+".json");log=p/(tag+"-"+arm+".log");meta=p/(tag+"-"+arm+"-execution.json")
 if any(q.exists() for q in (path,log,meta)):raise SystemExit("attempt exists: "+str(path))
 env=os.environ.copy()
 for k in ("NM_MATTER_BE_VELOCITY_COMPENSATION","NM_MATTER_MIDPOINT_VELOCITY_COMPENSATION","NM_MATTER_MIDPOINT_STAGE_COMPENSATION","NM_MATTER_MIDPOINT_PRECONTACT_POLISH","NM_MATTER_MIDPOINT_DIAGNOSTIC_ROOT"):env.pop(k,None)
 if arm=="ipc_area":env["NM_MATTER_BE_VELOCITY_COMPENSATION"]="1"
 else:env["NM_MATTER_MIDPOINT_VELOCITY_COMPENSATION"]="1";env["NM_MATTER_MIDPOINT_STAGE_COMPENSATION"]="1"
 cmd=[str(exe),str(shader),str(d["h"]),str(d["accepted_steps"]),str(d["divisions"]),arm];start=time.monotonic()
 with path.open("w") as out,log.open("w") as err:
  r=subprocess.Popen(cmd,env=env,stdout=out,stderr=err,start_new_session=True);reason="";conflicts=[];next_scan=start+10
  while r.poll() is None:
   if shutil.disk_usage("/System/Volumes/Data").free<3.1*1024**3:reason="storage safety stop"
   elif time.monotonic()-start>24*3600:reason="24 hour safety stop"
   elif time.monotonic()>=next_scan:
    conflicts=[w for w in active_numi_workers() if w["pid"]!=r.pid]
    if conflicts:reason="recognized competing Numi workload"
    next_scan=time.monotonic()+10
   if reason:
    os.killpg(r.pid,signal.SIGTERM)
    try:r.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(r.pid,signal.SIGKILL)
    break
   time.sleep(1)
  code=r.wait()
 record={"command":cmd,"input_sha256":hashes,"accuracy_manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),"accuracy_prefix":name,"tested_midpoint_ladder":midpoint_ladder,"tested_be_ladder":be_ladder,"returncode":code,
         "stopped_reason":reason,"thermal_snapshot_before":thermal,
         "recognized_numi_workers_before":workers,"recognized_conflicts_during":conflicts,"worker_check_scope":"process names beginning numi_ or metalrobo_; not a comprehensive GPU activity detector",
         "outer_wall_seconds":time.monotonic()-start}
 try:
  result=json.loads(path.read_text());record["accepted_and_matches_accuracy_state"]=code==0 and result["accepted_steps"]==d["accepted_steps"] and result["rejected_steps"]==0 and result["status"]==0 and result["per_step_snapshots"]==0 and result["final_physical_sha256"]==d["final_physical_sha256"] and result["device_fingerprint"]==d["device_fingerprint"]
 except (ValueError,KeyError) as e:record.update({"accepted_and_matches_accuracy_state":False,"parse_error":repr(e)})
 meta.write_text(json.dumps(record,indent=2)+"\n");print(json.dumps(record),flush=True);results.append(record)
 if not record["accepted_and_matches_accuracy_state"]:break
raise SystemExit(0 if len(results)==2 and all(x["accepted_and_matches_accuracy_state"] for x in results) else 1)
