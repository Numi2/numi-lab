#!/usr/bin/env python3
"""Summarize selected-root GPU trace without modifying physical state."""
import json,math,pathlib,struct,subprocess,sys
p=pathlib.Path(__file__).resolve().parents[2]/"docs/evidence/midpoint-first-failure-20261109"
a=json.loads((p/"be-n2-h0001.json").read_text());b=json.loads((p/"be-first-failure-001.json").read_text())
keys=("accepted_steps","rejected_steps","minimum_gap","final_physical_sha256","rejected_state_unchanged","continuation_replay","restore_replay","forced_rollback_exact","identity_mismatch_rejected")
reproduced=all(a[k]==b[k] for k in keys) and len(a["traces"])==len(b["traces"]) and all({k:v for k,v in x.items() if k!="gpu_seconds"}=={k:v for k,v in y.items() if k!="gpu_seconds"} for x,y in zip(a["traces"],b["traces"]))
items=[]
seen={}
repeated=0
for line in (p/"be-first-failure-001.log").read_text().splitlines():
 if line.startswith("fem_newton_iterate="):
  x=json.loads(line.split("=",1)[1])
  if x["stage"]=="final_line_search":
   iteration=x["iteration"]
   if iteration in seen:
    if x!=seen[iteration]:raise SystemExit("repeated trace differs")
    repeated+=1
    continue
   seen[iteration]=x
   n=x["fem_node_count"];nodes=bytes.fromhex(x["state_arenas"]["working_nodes"]);corr=bytes.fromhex(x["state_arenas"]["correction"]);axis=[]
   for j in range(3):
    nonzero=lost=0
    for i in range(n):
     v=struct.unpack_from("f",nodes,i*96+16+j*4)[0];q=struct.unpack_from("f",corr,i*16+j*4)[0]
     if q:
      nonzero+=1;lost+=struct.unpack("f",struct.pack("f",v+x["alpha"][0]*q))[0]==v
    axis.append({"nonzero":nonzero,"rounded_away":lost})
   items.append({"iteration":x["iteration"],"alpha":x["alpha"][0],"fgmres":x["fgmres"][0],"components":axis})
  if x["iteration"]==15 and x["stage"]=="fully_assembled_rhs":
   for name in ("accepted_nodes","working_nodes","rhs_workspace","contact_samples"):
    (p/("be-first-failure-001-"+name+".bin")).write_bytes(bytes.fromhex(x["state_arenas"][name]))
if len(items)!=16:raise SystemExit("missing or duplicate Newton trace")
result={"prefix_and_restored_state_reproduced":reproduced,"identical_repeated_iterations":repeated,"iterations":items}
print(json.dumps(result,indent=2));sys.exit(0 if reproduced else 1)
