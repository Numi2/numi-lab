#!/usr/bin/env python3
"""Run one complete cost pair only after both accuracy gates pass."""
import hashlib,json,math,os,pathlib,subprocess,sys,time
root=pathlib.Path(__file__).resolve().parents[2];p=root/"docs/evidence/midpoint-first-failure-20261109"
b=pathlib.Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109/evidence-be-precision-005")
if len(sys.argv)!=4:raise SystemExit("usage: cost.py MIDPOINT_PREFIX BE_PREFIX OUTPUT_TAG")
mid,be,tag=sys.argv[1:]
if not tag.replace("-","").isalnum() or any("/" in x for x in (mid,be)):raise SystemExit("invalid evidence name")
def load(name,suffix):return json.loads((p/(name+suffix)).read_text())
def qualified(name,arm):
 d=load(name,".json");a=load(name,"-analysis.json");e=load(name,"-execution.json")
 if not (a.get("passes") and e.get("all_gates_pass") and d["arm"]==arm and d["rejected_steps"]==0 and d["restore_replay"] and d["continuation_replay"] and d["forced_rollback_exact"] and d["identity_mismatch_rejected"]):raise SystemExit("accuracy gate not satisfied: "+name)
 return d
m=qualified(mid,"ipc_area_midpoint");c=qualified(be,"ipc_area")
for key in ("divisions","nodes","initial_gap","contact_area_m2","contact_area_weights_m2","area_ipc_k_N_per_m5","residual_policy","coordinate_mode","freefall_control","mechanical_relative_tolerance"):
 if m[key]!=c[key]:raise SystemExit("mismatched physical input: "+key)
for key in ("mechanical_absolute_tolerance_Ns","mechanical_impulse_scale_Ns"):
 if not math.isclose(m[key]/m["h"],c[key]/c["h"],rel_tol=2e-6):raise SystemExit("inconsistent per-time impulse scale: "+key)
exe=b/"metalrobo_plane_contact_benchmark";shader=b/"NumiMatter.metallib"
hashes={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (exe,shader)}
results=[]
for name,d,arm in ((be,c,"ipc_area"),(mid,m,"ipc_area_midpoint")):
 path=p/(tag+"-"+arm+".json");log=p/(tag+"-"+arm+".log");meta=p/(tag+"-"+arm+"-execution.json")
 if any(q.exists() for q in (path,log,meta)):raise SystemExit("attempt exists: "+str(path))
 env=os.environ.copy()
 for k in ("NM_MATTER_BE_VELOCITY_COMPENSATION","NM_MATTER_MIDPOINT_VELOCITY_COMPENSATION","NM_MATTER_MIDPOINT_STAGE_COMPENSATION"):env.pop(k,None)
 if arm=="ipc_area":env["NM_MATTER_BE_VELOCITY_COMPENSATION"]="1"
 else:env["NM_MATTER_MIDPOINT_VELOCITY_COMPENSATION"]="1";env["NM_MATTER_MIDPOINT_STAGE_COMPENSATION"]="1"
 cmd=[str(exe),str(shader),str(d["h"]),str(d["accepted_steps"]),str(d["divisions"]),arm];start=time.monotonic()
 with path.open("w") as out,log.open("w") as err:
  try:r=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=24*3600);code=r.returncode;timeout=False
  except subprocess.TimeoutExpired:code=None;timeout=True
 record={"command":cmd,"input_sha256":hashes,"accuracy_prefix":name,"returncode":code,"timed_out":timeout,"outer_wall_seconds":time.monotonic()-start}
 try:
  result=json.loads(path.read_text());record["accepted_and_matches_accuracy_state"]=code==0 and result["accepted_steps"]==d["accepted_steps"] and result["rejected_steps"]==0 and result["status"]==0 and result["per_step_snapshots"]==0 and result["final_physical_sha256"]==d["final_physical_sha256"] and result["device_fingerprint"]==d["device_fingerprint"]
 except (ValueError,KeyError) as e:record.update({"accepted_and_matches_accuracy_state":False,"parse_error":repr(e)})
 meta.write_text(json.dumps(record,indent=2)+"\n");print(json.dumps(record),flush=True);results.append(record)
 if not record["accepted_and_matches_accuracy_state"]:break
raise SystemExit(0 if len(results)==2 and all(x["accepted_and_matches_accuracy_state"] for x in results) else 1)
