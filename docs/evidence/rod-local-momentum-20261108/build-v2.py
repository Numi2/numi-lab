import json,subprocess
from pathlib import Path
p=Path("/Users/numi/rod-microbend-20261108");base=Path("/Users/numi/rod-vector-20261108")
b=json.loads((base/"build-v2.json").read_text())
commands=[]
for c in b[:2]:
 args=[x.replace(str(base/"guarded.metal"),str(p/"residual.metal")).replace(str(base/"guarded.air"),str(p/"residual.air")).replace(str(base/"guarded.metallib"),str(p/"residual.metallib")) for x in c["command"]]
 commands.append(args)
r=[]
for c in commands:
 q=subprocess.run(c,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);r.append({"command":c,"exit":q.returncode,"output":q.stdout});(p/"build-v2.json").write_text(json.dumps(r,indent=2)+"\n");print(q.returncode,q.stdout,flush=True)
 if q.returncode:raise SystemExit(q.returncode)
