import json,pathlib,subprocess,hashlib
root=pathlib.Path(__file__).resolve().parent
checkout=root.parents[2]
link=json.load(open(root.parent/"mixed-failing-row-witness-20261108/build.json"))[2]["command"]
records=[]
def run(cmd):
 p=subprocess.run([str(x) for x in cmd],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 records.append(dict(command=list(map(str,cmd)),returncode=p.returncode,output=p.stdout))
 (root/"factor-build.json").write_text(json.dumps(records,indent=2)+"\n")
 if p.returncode:raise SystemExit(p.returncode)
for arm in ("rigid0","rigid1","rigid2","factor0","factor1","factor2"):
 run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-I",checkout/"include","-c",root/(arm+".metal"),"-o",root/(arm+".air")])
 cmd=[str(root/(arm+".air")) if x.endswith("/baseline.air") else str(root/(arm+".metallib")) if x.endswith("/baseline.metallib") else x for x in link]
 run(cmd)
(root/"factor-inputs.sha256").write_text("".join(hashlib.sha256((root/(arm+ext)).read_bytes()).hexdigest()+"  "+arm+ext+"\n" for arm in ("rigid0","rigid1","rigid2","factor0","factor1","factor2") for ext in (".metal",".metallib")))
