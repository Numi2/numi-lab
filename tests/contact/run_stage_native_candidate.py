#!/usr/bin/env python3
"""Qualify a scoped candidate; never treat a short pass as a full rollout."""
import json,os,pathlib,subprocess,time
root=pathlib.Path(__file__).resolve().parents[2]
evidence=root/"docs/evidence/midpoint-first-failure-20261109"
build=pathlib.Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109/evidence-stage-native-001")
env=os.environ.copy();env["NM_MATTER_MIDPOINT_VELOCITY_COMPENSATION"]="1";env["NM_MATTER_MIDPOINT_STAGE_COMPENSATION"]="1"
for suffix,steps in [("short",205),("full",2560)]:
 name="stage-native-001-"+suffix
 paths={s:evidence/(name+s) for s in (".json",".log","-execution.json")}
 if any(path.exists() for path in paths.values()):raise SystemExit("attempt already exists: "+name)
 command=[str(build/"numi_plane_contact_probe_scaled"),str(build/"NumiMatter.metallib"),"0.00000390625",str(steps),"8","ipc_area_midpoint","0.0002","physical","contact","reference"]
 start=time.monotonic()
 with paths[".json"].open("w") as out,paths[".log"].open("w") as err:
  try: result=subprocess.run(command,env=env,stdout=out,stderr=err,timeout=2400);code=result.returncode;timeout=False
  except subprocess.TimeoutExpired:code=None;timeout=True
 meta={"command":command,"returncode":code,"timed_out":timeout,"wall_seconds_observational_only":time.monotonic()-start,"input_hashes":"stage-native-001-inputs.sha256"}
 paths["-execution.json"].write_text(json.dumps(meta,indent=2)+"\n")
 try:
  trace=json.loads(paths[".json"].read_text())
  good=code==0 and trace["accepted_steps"]==steps and trace["rejected_steps"]==0 and trace["restore_replay"] and trace["continuation_replay"] and trace["forced_rollback_exact"] and trace["identity_mismatch_rejected"]
 except (ValueError,KeyError):good=False
 print(name,"accepted_and_replayed",good,flush=True)
 if not good:break
 if suffix=="full":
  output=evidence/(name+"-analysis.json")
  reference=root/"docs/evidence/midpoint-spatial-cost-dynamic-20261108/rk4-n8-h025.json"
  check=subprocess.run(["python3",str(root/"tests/contact/native_trace_analysis.py"),str(paths[".json"]),str(reference)],capture_output=True,text=True)
  output.write_text(check.stdout)
  meta["analysis_returncode"]=check.returncode;meta["analysis_stderr"]=check.stderr
  paths["-execution.json"].write_text(json.dumps(meta,indent=2)+"\n")
  print(name,"accuracy_analysis_returncode",check.returncode,flush=True)
