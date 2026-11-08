import json,subprocess
from pathlib import Path
p=Path("/Users/numi/rod-vector-20261108")
libs={"vector":p/"vector.metallib","guarded":p/"guarded.metallib"}
axes=("x","y","z","minusx","oblique1","oblique2","oblique3","oblique4","oblique5")
cases=[(a,o,"axial") for o in ("0","0.5","100") for a in axes]+[("x","0",m) for m in ("bent","transverse")]
count=0
with (p/"results-v2.jsonl").open("w") as f:
 for arm,lib in libs.items():
  for axis,offset,kind in cases:
   for repeat in range(2):
    command=[str(p/"probe-v2"),str(lib),axis,offset,kind]
    r=subprocess.run(command,capture_output=True,text=True)
    f.write(json.dumps({"arm":arm,"axis":axis,"offset":offset,"kind":kind,"repeat":repeat,"command":command,"exit":r.returncode,"stdout":r.stdout,"stderr":r.stderr})+"\n");f.flush();count+=1
    if r.returncode or r.stderr: print("ERROR",arm,axis,offset,kind,repeat,r.returncode,r.stderr,flush=True)
print("completed",count,flush=True)
