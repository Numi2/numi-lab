#!/usr/bin/env python3
"""Run the registered n=2 midpoint precision-candidate ladder without timing claims."""
import json,os,pathlib,subprocess,time
root=pathlib.Path(__file__).resolve().parents[2]
p=root/"docs/evidence/midpoint-first-failure-20261109"
b=pathlib.Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109/evidence-stage-native-001")
ref=root/"docs/evidence/midpoint-spatial-cost-dynamic-20261108/rk4-n2-h025.json"
env=os.environ.copy();env["NM_MATTER_MIDPOINT_VELOCITY_COMPENSATION"]="1";env["NM_MATTER_MIDPOINT_STAGE_COMPENSATION"]="1"
for tag,h in [("125","0.000125"),("0625","0.0000625"),("03125","0.00003125"),("015625","0.000015625"),("0078125","0.0000078125")]:
 name="stage-ladder-n2-h"+tag
 paths={s:p/(name+s) for s in (".json",".log","-execution.json","-analysis.json")}
 if any(q.exists() for q in paths.values()):raise SystemExit("attempt exists: "+name)
 count=round(.01/float(h));cmd=[str(b/"numi_plane_contact_probe_scaled"),str(b/"NumiMatter.metallib"),h,str(count),"2","ipc_area_midpoint","0.0002","physical","contact","reference"]
 start=time.monotonic()
 with paths[".json"].open("w") as out,paths[".log"].open("w") as err:
  try:r=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=2400);code=r.returncode;timedout=False
  except subprocess.TimeoutExpired:code=None;timedout=True
 meta={"command":cmd,"input_hashes":"stage-native-001-inputs.sha256","returncode":code,"timed_out":timedout,"wall_seconds_observational_only":time.monotonic()-start}
 try:
  state=json.loads(paths[".json"].read_text())
  analysis=subprocess.run(["python3",str(root/"tests/contact/native_trace_analysis.py"),str(paths[".json"]),str(ref)],capture_output=True,text=True)
  paths["-analysis.json"].write_text(analysis.stdout)
  checks=code==0 and analysis.returncode==0 and state["accepted_steps"]==count and state["rejected_steps"]==0 and state["restore_replay"] and state["continuation_replay"] and state["forced_rollback_exact"] and state["identity_mismatch_rejected"] and state["independent_max_local_scaled_without_floor"]<=.005
  meta.update({"analysis_returncode":analysis.returncode,"analysis_stderr":analysis.stderr,"all_gates_pass":checks})
 except (ValueError,KeyError) as exc:
  meta.update({"analysis_error":repr(exc),"all_gates_pass":False})
 paths["-execution.json"].write_text(json.dumps(meta,indent=2)+"\n")
 print(name,json.dumps(meta),flush=True)
 if timedout or code is None:break
 if meta["all_gates_pass"]:break
