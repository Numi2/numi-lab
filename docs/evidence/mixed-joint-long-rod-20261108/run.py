import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent;repo=root.parents[2]
work=Path("/Users/numi/Work/mixed-joint-long-rod-artifacts-20261108");work.mkdir(parents=True,exist_ok=True)
out=root/"runs";out.mkdir(exist_ok=True)
source=root.parent/"mixed-full-operator-20261108/probe.mm"
original=source.read_text()
ref=Path("/Users/numi/frictionless-mixed-rod-20261108/split.metallib")
old=Path("/Users/numi/Work/mixed-joint-contact-v2-artifacts-20261108/joint.metallib")
long=work/"long.metallib";trace=work/"trace.metallib"
lib=Path("/Users/numi/numi-reconciliation-build-20261108/lib/libmetalrobo.dylib")
assert hashlib.sha256(ref.read_bytes()).hexdigest()=="eb50d360aa00d89390efb52ec8f59637e8b21398b968aab689b2443db1511ccf"
assert hashlib.sha256(lib.read_bytes()).hexdigest()=="c5bb0e7432a2df5bda2b89c5fecdd66896acf907d4274e61245a8dca97d901f1"
assert hashlib.sha256(long.read_bytes()).hexdigest()==(root/"artifacts.sha256").read_text().splitlines()[2].split()[0]
assert hashlib.sha256(trace.read_bytes()).hexdigest()==(root/"trace-artifacts.sha256").read_text().splitlines()[2].split()[0]
assert hashlib.sha256(old.read_bytes()).hexdigest()==(root.parent/"mixed-joint-contact-candidate-20261108/v2-artifacts.sha256").read_text().splitlines()[2].split()[0]
records=[]
def run(label,cmd):
 print(label,flush=True)
 p=subprocess.run([str(x) for x in cmd],cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 (out/(label+".stdout")).write_text(p.stdout);(out/(label+".stderr")).write_text(p.stderr)
 records.append({"label":label,"command":[str(x) for x in cmd],"returncode":p.returncode})
 (out/"commands.json").write_text(json.dumps(records,indent=2)+"\n")
 return p.returncode
(out/"inputs.sha256").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in (source,ref,old,long,trace,lib)))
for n,length,center in ((5,".08",".04"),(9,".16",".08"),(17,".32",".16")):
 name="rod%d"%n;s=original
 edits=[("makeStraightSutureRod(3,.04,mat)","makeStraightSutureRod(%d,%s,mat)"%(n,length)),("p[0]-=.02;","p[0]-="+center+";")]
 for a,b in edits:
  assert s.count(a)==1,(name,a,s.count(a));s=s.replace(a,b,1)
 cpp=work/(name+".mm");exe=work/name;cpp.write_text(s)
 cmd=["xcrun","clang++","-std=c++20","-O2","-fobjc-arc","-I",repo/"include",cpp,"-L",lib.parent,"-lmetalrobo","-Wl,-rpath,"+str(lib.parent),"-framework","Foundation","-framework","Metal","-o",exe]
 if run(name+"-build",cmd):continue
 with (out/"artifacts.sha256").open("a") as f:
  for p in (cpp,exe):f.write(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n")
 for arm,shader,niter in (("reference",ref,"128"),("old",old,"2"),("long",long,"2"),("trace",trace,"2")):
  run(name+"-"+arm,[exe,shader,niter])
