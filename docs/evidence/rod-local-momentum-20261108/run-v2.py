import json,subprocess
from pathlib import Path
p=Path("/Users/numi/rod-microbend-20261108");lib=p/"residual.metallib"
deltas=[0]+[s*x*1e-9 for x in (25,50,100,150,200,300,1000) for s in (-1,1)]
with (p/"raw-v2.jsonl").open("w") as f:
 for axis in ("x","oblique"):
  for bend in deltas:
   for repeat in range(2):
    c=[str(p/"probe"),str(lib),axis,format(bend,".17g")];r=subprocess.run(c,capture_output=True,text=True)
    f.write(json.dumps({"arm":"residual","axis":axis,"bend":bend,"repeat":repeat,"command":c,"exit":r.returncode,"stdout":r.stdout,"stderr":r.stderr})+"\n");f.flush()
    if r.returncode or r.stderr:print("ERROR",c,repeat,r.returncode,r.stderr,flush=True)
print("completed 60",flush=True)
