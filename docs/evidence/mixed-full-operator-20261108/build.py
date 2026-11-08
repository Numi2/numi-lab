import json, pathlib, subprocess, hashlib
root=pathlib.Path(__file__).resolve().parent
checkout=root.parents[2]
build=pathlib.Path("/Users/numi/numi-reconciliation-build-20261108")
records=[]
def run(cmd):
 p=subprocess.run([str(x) for x in cmd],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 records.append(dict(command=list(map(str,cmd)),returncode=p.returncode,output=p.stdout))
 (root/"build.json").write_text(json.dumps(records,indent=2)+"\n")
 if p.returncode: raise SystemExit(p.returncode)
run(["xcrun","clang++","-std=c++20","-O2","-fobjc-arc","-I",checkout/"include",root/"probe.mm","-L",build/"lib","-lmetalrobo","-Wl,-rpath,"+str(build/"lib"),"-framework","Foundation","-framework","Metal","-o",root/"probe"])
link=json.load(open(root.parent/"mixed-failing-row-witness-20261108/build.json"))[2]["command"]
inputs=[pathlib.Path(x) for x in link if x.endswith(".air") and not x.endswith("/baseline.air")]
(root/"air-inputs.sha256").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in inputs))
for arm in ("baseline","axis0","axis1","axis2"):
 run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-I",checkout/"include","-c",root/(arm+".metal"),"-o",root/(arm+".air")])
 cmd=[str(root/(arm+".air")) if x.endswith("/baseline.air") else str(root/(arm+".metallib")) if x.endswith("/baseline.metallib") else x for x in link]
 run(cmd)
