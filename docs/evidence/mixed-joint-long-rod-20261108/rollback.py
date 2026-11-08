import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
repo=root.parents[2]
work=Path("/Users/numi/Work/mixed-joint-long-rod-rollback-artifacts-20261108");work.mkdir(parents=True,exist_ok=True)
out=root/"rollback";out.mkdir(exist_ok=True)
source=Path("/Users/numi/Work/mixed-joint-long-rod-artifacts-20261108/trace.metal")
trace_lib=Path("/Users/numi/Work/mixed-joint-long-rod-artifacts-20261108/trace.metallib")
probe_source=root.parent/"mixed-full-operator-20261108/probe.mm"
lib=Path("/Users/numi/numi-reconciliation-build-20261108/lib/libmetalrobo.dylib")
assert hashlib.sha256(source.read_bytes()).hexdigest()==(root/"trace-artifacts.sha256").read_text().splitlines()[0].split()[0]
assert hashlib.sha256(trace_lib.read_bytes()).hexdigest()==(root/"trace-artifacts.sha256").read_text().splitlines()[2].split()[0]
assert hashlib.sha256(lib.read_bytes()).hexdigest()=="c5bb0e7432a2df5bda2b89c5fecdd66896acf907d4274e61245a8dca97d901f1"
s=source.read_text();start=s.index("inline void mrWorldWave32SolvePacket(")
needle="        result.environment = environment;\n";pos=s.index(needle,start)
injection="        if (result.code == MR_STEP_SUCCESS && rodWork && pass.controlStep == 10u) result.code = MR_STEP_DID_NOT_CONVERGE;\n"
s=s[:pos]+injection+s[pos:]
shader=work/"reject.metal";air=work/"reject.air";reject_lib=work/"reject.metallib";shader.write_text(s)
link=json.loads((root.parent/"mixed-failing-row-witness-20261108/build.json").read_text())[2]["command"]
cmds=[["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-I",str(repo/"include"),"-c",str(shader),"-o",str(air)],
 [str(air) if x.endswith("/baseline.air") else str(reject_lib) if x.endswith("/baseline.metallib") else x for x in link]]
records=[]
def run(label,cmd):
 print(label,flush=True)
 p=subprocess.run([str(x) for x in cmd],cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 (out/(label+".stdout")).write_text(p.stdout);(out/(label+".stderr")).write_text(p.stderr)
 records.append({"label":label,"command":[str(x) for x in cmd],"returncode":p.returncode})
 (out/"commands.json").write_text(json.dumps(records,indent=2)+"\n")
 if p.returncode:raise RuntimeError(label+" exit "+str(p.returncode))
for i,cmd in enumerate(cmds):run("shader-"+str(i),cmd)
base=probe_source.read_text()
for a,b in (("makeStraightSutureRod(3,.04,mat)","makeStraightSutureRod(9,.16,mat)"),("p[0]-=.02;","p[0]-=.08;")):
 assert base.count(a)==1
 base=base.replace(a,b,1)
helper="""template<class T> std::string bytehex(const std::vector<T>& v){
 static const char d[]="0123456789abcdef";std::string s;const auto*p=reinterpret_cast<const unsigned char*>(v.data());
 for(size_t i=0;i<v.size()*sizeof(T);++i){s.push_back(d[p[i]>>4]);s.push_back(d[p[i]&15]);}return s;
}
"""
needle="struct Capture {"
assert base.count(needle)==1;base=base.replace(needle,helper+needle,1)
old="std::cout<<std::setprecision(10)<<\"{\\\"plain_published\\\":\""
new="""std::cout<<std::setprecision(10)<<"{\\\"final_bytes\\\":{\\\"q\\\":\\\""<<bytehex(recorded.finalQ)<<"\\\",\\\"v\\\":\\\""<<bytehex(recorded.finalV)<<"\\\",\\\"scene\\\":\\\""<<bytehex(recorded.finalSceneBodies)<<"\\\",\\\"rod_nodes\\\":\\\""<<bytehex(recorded.finalRodNodes)<<"\\\",\\\"rod_edges\\\":\\\""<<bytehex(recorded.finalRodEdges)<<"\\\"},\\\"plain_published\\\":""" + chr(34)
assert base.count(old)==1;base=base.replace(old,new,1)
for n in (10,11):
 text=base.replace("constexpr unsigned N=20;","constexpr unsigned N="+str(n)+";")
 assert text!=base
 cpp=work/("probe%d.mm"%n);exe=work/("probe%d"%n);cpp.write_text(text)
 run("probe%d-build"%n,["xcrun","clang++","-std=c++20","-O2","-fobjc-arc","-I",repo/"include",cpp,"-L",lib.parent,"-lmetalrobo","-Wl,-rpath,"+str(lib.parent),"-framework","Foundation","-framework","Metal","-o",exe])
 for label,shader_path in (("clean",trace_lib),("injected",reject_lib)):
  if n==10 and label=="injected":continue
  run("%s%d"%(label,n),[exe,shader_path,"2"])
paths=[source,trace_lib,probe_source,lib,shader,air,reject_lib,work/"probe10.mm",work/"probe10",work/"probe11.mm",work/"probe11"]
(out/"inputs-and-artifacts.sha256").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in paths))
