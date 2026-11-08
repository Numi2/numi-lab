import json,subprocess
from pathlib import Path
p=Path("/Users/numi/rod-vector-20261108")
b=json.loads((p/"build.json").read_text())
metal=[x.replace(str(p/"candidate.metal"),str(p/"guarded.metal")).replace(str(p/"vector.air"),str(p/"guarded.air")) for x in b[2]["command"]]
link=[x.replace(str(p/"vector.air"),str(p/"guarded.air")).replace(str(p/"candidate.metallib"),str(p/"guarded.metallib")) for x in b[3]["command"]]
clang=["xcrun","clang++","-std=c++20","-O2","-fobjc-arc","-I","/Users/numi/numi-lab-reconciliation-20261108/include",str(p/"probe-v2.mm"),"-L","/Users/numi/numi-reconciliation-build-20261108/lib","-lmetalrobo","-Wl,-rpath,/Users/numi/numi-reconciliation-build-20261108/lib","-framework","Foundation","-framework","Metal","-o",str(p/"probe-v2")]
r=[]
for c in (metal,link,clang):
 a=subprocess.run(c,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);r.append({"command":c,"exit":a.returncode,"output":a.stdout});(p/"build-v2.json").write_text(json.dumps(r,indent=2)+"\n");print(a.returncode,a.stdout,flush=True)
 if a.returncode:raise SystemExit(a.returncode)
