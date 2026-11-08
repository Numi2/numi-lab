import json,subprocess
from pathlib import Path
p=Path("/Users/numi/rod-microbend-20261108")
libs={"production":Path("/Users/numi/numi-reconciliation-build-20261108/shaders/MetalRobo.metallib"),"guarded":Path("/Users/numi/rod-vector-20261108/guarded.metallib")}
deltas=[0]+[s*x*1e-9 for x in (25,50,100,150,200,300,1000) for s in (-1,1)]
with (p/"raw.jsonl").open("w") as f:
 for arm,lib in libs.items():
  for axis in ("x","oblique"):
   for bend in deltas:
    for repeat in range(2):
     c=[str(p/"probe"),str(lib),axis,format(bend,".17g")];r=subprocess.run(c,capture_output=True,text=True)
     item={"arm":arm,"axis":axis,"bend":bend,"repeat":repeat,"command":c,"exit":r.returncode,"stdout":r.stdout,"stderr":r.stderr}
     f.write(json.dumps(item)+"\n");f.flush()
     if r.returncode or r.stderr:print("ERROR",arm,axis,bend,repeat,r.returncode,r.stderr,flush=True)
print("completed",len(libs)*2*len(deltas)*2,flush=True)
