import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
repo=root.parents[2]
work=Path("/Users/numi/Work/mixed-joint-contact-controls-artifacts-20261108")
work.mkdir(parents=True,exist_ok=True)
out=root/"controls";out.mkdir(exist_ok=True)
source=root.parent/"mixed-full-operator-20261108/probe.mm"
original=source.read_text()
ref=Path("/Users/numi/frictionless-mixed-rod-20261108/split.metallib")
candidate=Path("/Users/numi/Work/mixed-joint-contact-v2-artifacts-20261108/joint.metallib")
trace=Path("/Users/numi/Work/mixed-joint-contact-v2-artifacts-20261108/trace.metallib")
lib=Path("/Users/numi/numi-reconciliation-build-20261108/lib/libmetalrobo.dylib")
assert hashlib.sha256(ref.read_bytes()).hexdigest()=="eb50d360aa00d89390efb52ec8f59637e8b21398b968aab689b2443db1511ccf"
assert hashlib.sha256(lib.read_bytes()).hexdigest()=="c5bb0e7432a2df5bda2b89c5fecdd66896acf907d4274e61245a8dca97d901f1"
assert hashlib.sha256(candidate.read_bytes()).hexdigest()==(root/"v2-artifacts.sha256").read_text().splitlines()[2].split()[0]
assert hashlib.sha256(trace.read_bytes()).hexdigest()==(root/"v2-trace-artifacts.sha256").read_text().splitlines()[2].split()[0]
def exposed(x):
 if isinstance(x,dict): return {k:exposed(v) for k,v in x.items() if k not in ("scratch","row_trace")}
 if isinstance(x,list): return [exposed(v) for v in x]
 return x
for fresh,prior in (("fresh-v2.json","v2-joint.json"),("fresh-fixed128.json","fixed128-correct.json")):
 a=json.loads((root/fresh).read_text()); b=json.loads((root/prior).read_text())
 assert exposed(a)==exposed(b) and all(a[k] for k in ("capture_noninvasive","repeat","exact_trace"))
scenarios={
 "reverse": [("m.defaultV[0]=.05f;","m.defaultV[0]=-.035f;")],
 "raised_rod": [("p[1]=.0228;","p[1]=.02315;")],
 "frictionless": [("x.friction={.5f,.5f,0,0}","x.friction={0.f,0.f,0,0}")],
 "separating": [("m.defaultQ[1]=.0099f;","m.defaultQ[1]=.0115f;"),("m.defaultV[1]=0.f;","m.defaultV[1]=.04f;")],
}
records=[]
def run(label,cmd):
 print(label,flush=True)
 p=subprocess.run([str(x) for x in cmd],cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 (out/(label+".stdout")).write_text(p.stdout);(out/(label+".stderr")).write_text(p.stderr)
 records.append({"label":label,"command":[str(x) for x in cmd],"returncode":p.returncode})
 (out/"commands.json").write_text(json.dumps(records,indent=2)+"\n")
 return p.returncode
(out/"inputs.sha256").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in (source,ref,candidate,trace,lib)))
for name,edits in scenarios.items():
 s=original
 for old,new in edits:
  assert s.count(old)==1,(name,old,s.count(old))
  s=s.replace(old,new,1)
 cpp=work/(name+".mm");probe=work/name;cpp.write_text(s)
 cmd=["xcrun","clang++","-std=c++20","-O2","-fobjc-arc","-I",repo/"include",cpp,"-L",lib.parent,"-lmetalrobo","-Wl,-rpath,"+str(lib.parent),"-framework","Foundation","-framework","Metal","-o",probe]
 if run(name+"-build",cmd):continue
 with (out/"artifacts.sha256").open("a") as f:
  for p in (cpp,probe):f.write(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n")
 for arm,shader,iters in (("reference",ref,"128"),("candidate",candidate,"2"),("trace",trace,"2")):
  run(name+"-"+arm,[probe,shader,iters])
