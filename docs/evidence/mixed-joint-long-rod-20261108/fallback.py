from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parent;repo=root.parents[2]
work=Path("/Users/numi/Work/mixed-joint-long-rod-artifacts-20261108");out=root/"fallback";out.mkdir(exist_ok=True)
base=work/"long.metal"
assert hashlib.sha256(base.read_bytes()).hexdigest()==(root/"artifacts.sha256").read_text().splitlines()[0].split()[0]
s=base.read_text();old="""            if (valid) valid=mrResearchRodWitnessesFitSelectedBand(
                witnessNodeA[0],witnessNodeB[0],witnessEdge[0],
                witnessNodeA[1],witnessNodeB[1],witnessEdge[1]);"""
assert s.count(old)==1
s=s.replace(old,"            if (valid) valid=false; // diagnostic forced admission denial",1)
shader=work/"forced-fallback.metal";air=work/"forced-fallback.air";library=work/"forced-fallback.metallib";shader.write_text(s)
link=json.loads((root.parent/"mixed-failing-row-witness-20261108/build.json").read_text())[2]["command"]
cmds=[["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-I",str(repo/"include"),"-c",str(shader),"-o",str(air)],
 [str(air) if x.endswith("/baseline.air") else str(library) if x.endswith("/baseline.metallib") else x for x in link]]
records=[]
def run(label,cmd):
 print(label,flush=True);p=subprocess.run([str(x) for x in cmd],cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 (out/(label+".stdout")).write_text(p.stdout);(out/(label+".stderr")).write_text(p.stderr)
 records.append({"label":label,"command":[str(x) for x in cmd],"returncode":p.returncode});(out/"commands.json").write_text(json.dumps(records,indent=2)+"\n")
 if p.returncode:raise RuntimeError(label+" exit "+str(p.returncode))
for i,cmd in enumerate(cmds):run("build-"+str(i),cmd)
for n in (5,9):run("rod%d"%n,[work/("rod%d"%n),library,"2"])
(out/"artifacts.sha256").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in (base,shader,air,library)))
