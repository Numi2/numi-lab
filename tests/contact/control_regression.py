#!/usr/bin/env python3
"""Residual/transaction regression. Force refinement is a separate gate."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("binary",type=Path);p.add_argument("shader",type=Path);p.add_argument("out",type=Path);a=p.parse_args();a.out.mkdir(exist_ok=True)
for name,h,count,mode in [("freefall",.0005,20,"freefall"),("contact",.0000625,160,"contact")]:
 cmd=[str(a.binary),str(a.shader),str(h),str(count),"2","ipc",".0002","physical",mode,"reference"]
 with (a.out/(name+".json")).open("w") as out,(a.out/(name+".stderr")).open("w") as err:
  subprocess.run(cmd,stdout=out,stderr=err,check=True)
 x=json.loads((a.out/(name+".json")).read_text())
 assert x["accepted_steps"]==count and x["rejected_steps"]==0
 assert x["restore_replay"] and x["forced_rollback_exact"] and x["identity_mismatch_rejected"]
 assert x["independent_max_local_scaled_residual"]<=.005
 assert x["momentum_max_absolute_error_Ns"]<=1e-7
 if mode=="freefall":
  assert all(t["normal_impulse"]==0 and t["contacts"]==0 for t in x["traces"])
  assert x["freefall_velocity_max_defect_mps"]<2e-6
 else:
  assert x["minimum_gap"]>1e-7
  assert sum(t["accepted_normal_impulse"] for t in x["traces"])>0
print("complete-duration local residual, noncontact update, geometry and transaction regressions passed; no force-refinement claim")
