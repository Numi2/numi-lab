#!/usr/bin/env python3
"""Retain every attempted native complete-step comparison, including rejection."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("build",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
exe=a.build/"metalrobo_plane_contact_probe";shader=a.build/"NumiMatter.metallib"
identity={"executable_sha256":hashlib.sha256(exe.read_bytes()).hexdigest(),"metallib_sha256":hashlib.sha256(shader.read_bytes()).hexdigest()}
(a.output/"identity.json").write_text(json.dumps(identity,indent=2)+"\n")
runs=[]
for n,h in [(2,.000125),(4,.0000625)]:runs.append(("ref",n,h,"ipc",.0002))
for h in [.00025,.0005,.001,.002]:
 for n in [1,2]:
  for k,arm in enumerate(["ipc","al","al","ipc"]):runs.append((f"pair{k}",n,h,arm,.0002))
for g in [.00005,.0000005]:
 for h in [.0005,.002]:
  for arm in ["ipc","al"]:runs.append((f"near{round(g*1e9)}",1,h,arm,g))
ledger=[]
for tag,n,h,arm,g in runs:
 name=f"{tag}-n{n}-h{round(h*1e9)}-{arm}"
 cmd=[str(exe),str(shader),str(h),str(round(.01/h)),str(n),arm,str(g)]
 start=time.monotonic()
 with (a.output/(name+".json")).open("w") as out,(a.output/(name+".stderr")).open("w") as err:
  result=subprocess.run(cmd,stdout=out,stderr=err,timeout=600)
 row={"name":name,"command":cmd,"exit":result.returncode,"process_seconds":time.monotonic()-start};ledger.append(row)
 (a.output/"attempt-ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
 print(json.dumps(row),flush=True)
