import json,subprocess
from pathlib import Path
libs={"baseline":"/Users/numi/numi-reconciliation-build-20261108/shaders/MetalRobo.metallib","candidate":"/Users/numi/rod-axis-probe-20261108/candidate.metallib"}
results={}
for name,lib in libs.items():
 for axis in ("x","y"):
  command=["./probe",lib,axis];p=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);key=name+"-"+axis
  results[key]=dict(command=command,exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
  Path(key+".stdout").write_text(p.stdout);Path(key+".stderr").write_text(p.stderr)
  Path("results.json").write_text(json.dumps(results,indent=2)+"\n");print(key,p.returncode,flush=True)
