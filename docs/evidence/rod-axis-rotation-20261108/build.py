import json,subprocess
from pathlib import Path
base="/Users/numi/numi-reconciliation-build-20261108"
root="/Users/numi/rod-axis-probe-20261108"
source="/Users/numi/numi-lab-reconciliation-20261108"
a=json.loads(Path("/Users/numi/mixed-sustained-20261108/cap-build.json").read_text())
metal=["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-I",source+"/include","-c",root+"/candidate.metal","-o",root+"/candidate.air"]
link=[x.replace("/Users/numi/mixed-sustained-20261108/candidate-2e-4-cap128.air",base+"/shaders/MetalRoboWorldContact.air").replace("/Users/numi/mixed-sustained-20261108/candidate-2e-4-cap128.metallib",root+"/candidate.metallib").replace(base+"/shaders/MetalRoboDiscreteElasticRod.air",root+"/candidate.air") for x in a[1]["command"]]
clang=["xcrun","clang++","-std=c++20","-O2","-fobjc-arc","-I",source+"/include",root+"/probe.mm","-L",base+"/lib","-lmetalrobo","-Wl,-rpath,"+base+"/lib","-framework","Foundation","-framework","Metal","-o",root+"/probe"]
out=[]
for args in (metal,link,clang):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);out.append(dict(command=args,exit=p.returncode,output=p.stdout));Path("build.json").write_text(json.dumps(out,indent=2)+"\n");print(p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
