import json, subprocess, hashlib
from pathlib import Path
p=Path("/Users/numi/rod-vector-20261108")
base=Path("/Users/numi/numi-reconciliation-build-20261108")
libs={"production":base/"shaders/MetalRobo.metallib","prior":p/"prior.metallib","vector":p/"vector.metallib"}
axes=("x","y","z","minusx","oblique1","oblique2")
cases=[(a,o,"axial") for o in ("0","0.5","100") for a in axes]+[("x","0",m) for m in ("bent","transverse")]
results=[]
with (p/"results.jsonl").open("w") as stream:
 for mode,lib in libs.items():
  for axis,offset,kind in cases:
   for repeat in range(2):
    cmd=[str(p/"probe"),str(lib),axis,offset,kind]
    r=subprocess.run(cmd,capture_output=True,text=True)
    item={"arm":mode,"axis":axis,"offset":offset,"kind":kind,"repeat":repeat,"command":cmd,"exit":r.returncode,"stdout":r.stdout,"stderr":r.stderr}
    stream.write(json.dumps(item)+"\n");stream.flush();results.append(item)
    if r.returncode or r.stderr:
     print("ERROR",mode,axis,offset,kind,repeat,r.returncode,r.stderr,flush=True)
 print("completed",len(results),"executions",flush=True)
