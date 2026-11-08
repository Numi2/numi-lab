import json,subprocess
from pathlib import Path
p=Path("/Users/numi/rod-vector-20261108")
libs={"production":Path("/Users/numi/numi-reconciliation-build-20261108/shaders/MetalRobo.metallib"),"vector":p/"vector.metallib","guarded":p/"guarded.metallib"}
count=0
with (p/"results-v3.jsonl").open("w") as f:
 for arm,lib in libs.items():
  for n in (5,8):
   for axis in ("x","oblique3"):
    for offset in ("0","0.25"):
     for kind in ("fixed","moving","internal"):
      for repeat in range(2):
       c=[str(p/"multichain"),str(lib),str(n),axis,offset,kind]
       r=subprocess.run(c,capture_output=True,text=True);f.write(json.dumps({"arm":arm,"repeat":repeat,"command":c,"exit":r.returncode,"stdout":r.stdout,"stderr":r.stderr})+"\n");f.flush();count+=1
       if r.returncode or r.stderr:print("ERROR",c,repeat,r.returncode,r.stderr,flush=True)
print("completed",count,flush=True)
