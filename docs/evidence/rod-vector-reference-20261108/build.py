import json, subprocess
from pathlib import Path
out=Path("/Users/numi/rod-vector-20261108")
old=Path("/Users/numi/rod-axis-probe-20261108")
steps=json.loads((old/"build.json").read_text())
commands=[]
for name in ("prior","vector"):
 for i in (0,1):
  command=steps[i]["command"][:]
  for j,v in enumerate(command):
   v=v.replace(str(old/"candidate.metal"),str(old/"candidate.metal") if name=="prior" else str(out/"candidate.metal"))
   v=v.replace(str(old/"candidate.air"),str(out/(name+".air")))
   v=v.replace(str(old/"candidate.metallib"),str(out/(name+".metallib")))
   command[j]=v
  commands.append(command)
results=[]
for c in commands:
 p=subprocess.run(c,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 results.append({"command":c,"exit":p.returncode,"output":p.stdout})
 (out/"build.json").write_text(json.dumps(results,indent=2)+"\n")
 print(Path(c[-1]).name,p.returncode,p.stdout,flush=True)
 if p.returncode: raise SystemExit(p.returncode)
