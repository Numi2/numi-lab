import json,subprocess
from pathlib import Path
p=Path("/Users/numi/rod-microbend-20261108");lib=p/"residual.metallib";probe=Path("/Users/numi/rod-vector-20261108/multichain")
with (p/"straight-v2.jsonl").open("w") as f:
 for n in (5,8):
  for axis in ("x","oblique3"):
   for offset in ("0","0.25"):
    for kind in ("fixed","moving","internal"):
     for repeat in range(2):
      c=[str(probe),str(lib),str(n),axis,offset,kind];r=subprocess.run(c,capture_output=True,text=True)
      f.write(json.dumps({"arm":"residual","repeat":repeat,"command":c,"exit":r.returncode,"stdout":r.stdout,"stderr":r.stderr})+"\n");f.flush()
      if r.returncode or r.stderr:print("ERROR",c,repeat,r.returncode,r.stderr,flush=True)
print("completed 48",flush=True)
