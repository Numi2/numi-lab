#!/usr/bin/env python3
"""Exercise BE precision dispatch, acceptance and transaction restoration."""
import json,os,subprocess,sys
if len(sys.argv)!=3:raise SystemExit("usage: be_precision_step_regression.py PROBE METALLIB")
env=os.environ.copy();env["NM_MATTER_BE_VELOCITY_COMPENSATION"]="1"
for k in ("NM_MATTER_MIDPOINT_VELOCITY_COMPENSATION","NM_MATTER_MIDPOINT_STAGE_COMPENSATION"):env.pop(k,None)
r=subprocess.run([sys.argv[1],sys.argv[2],"0.000001","2","2","ipc_area","0.0002","physical","contact","reference"],env=env,capture_output=True,text=True,timeout=120)
if r.returncode:raise SystemExit("probe failed: returncode="+str(r.returncode)+"\n"+r.stderr[-4000:])
d=json.loads(r.stdout)
checks={"accepted_steps":d["accepted_steps"]==2,"no_rejection":d["rejected_steps"]==0,"restore_replay":d["restore_replay"],"continuation_replay":d["continuation_replay"],"forced_rollback_exact":d["forced_rollback_exact"],"identity_mismatch_rejected":d["identity_mismatch_rejected"],"local_residual":d["independent_max_local_scaled_without_floor"]<=.005}
if not all(checks.values()):raise SystemExit("BE step checks failed: "+str(checks))
print("BE precision dispatch, accepted steps and restore verified")
