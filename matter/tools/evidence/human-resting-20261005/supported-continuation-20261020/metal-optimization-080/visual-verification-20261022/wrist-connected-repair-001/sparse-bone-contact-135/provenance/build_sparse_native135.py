from pathlib import Path
import json,subprocess,shlex,time,hashlib
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019");E=R/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001";D=E/"sparse-bone-contact-135";O=D/"source-overlay";B=D/"build-native-002";B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(); stages=[]
def run(name,argv,cwd=R):
 t=time.monotonic();p=subprocess.run(argv,cwd=cwd,capture_output=True,text=True);(B/(name+".log")).write_text(p.stdout+p.stderr);stages.append(dict(name=name,argv=argv,cwd=str(cwd),returncode=p.returncode,wall_seconds=time.monotonic()-t));(B/"build-stages.json").write_text(json.dumps(stages,indent=2));assert p.returncode==0,name
incs=["-I",str(R/"matter/include"),"-I",str(R/"include")]
flags=["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math"]
run("baseline-metal",flags+incs+["-c",str(R/"matter/src/matter_kernels.metal"),"-o",str(B/"baseline.air")])
run("baseline-metallib",["xcrun","-sdk","macosx","metallib",str(B/"baseline.air"),"-o",str(B/"baseline.metallib")])
run("candidate-metal",flags+["-I",str(O/"matter/src"),"-I",str(R/"matter/src")]+incs+["-c",str(O/"matter/src/matter_kernels.metal"),"-o",str(B/"candidate.air")])
run("candidate-metallib",["xcrun","-sdk","macosx","metallib",str(B/"candidate.air"),"-o",str(B/"candidate.metallib")])
b=Path("/tmp/numi-human-joint-owner-build-033");cmd=next(x for x in json.loads((b/"compile_commands.json").read_text()) if x["file"].endswith("matter/src/numi_human.mm"));a=shlex.split(cmd["command"]);a[1:1]=["-I",str(R/"matter/src")];a[a.index("-o")+1]=str(B/"numi_human.mm.o");a[-1]=str(O/"matter/src/numi_human.mm");run("host",a,Path(cmd["directory"]))
link=(E/"selector6-scalar-inverse-fix-001/build/link-001/link-command.sh").read_text();line=next(x for x in link.splitlines() if x.startswith(": && "));a=shlex.split(line[5:-5]);a[a.index("-o")+1]=str(B/"libmetalrobo.dylib");ix=a.index("matter/CMakeFiles/numi_matter_runtime_objects.dir/src/numi_human.mm.o");a[ix]=str(B/"numi_human.mm.o");run("link",a,b)
old=b/"matter/shaders/NumiMatter.metallib"
r={"stages":stages,"baseline033_metallib_sha256":sha(old),"rebuilt_baseline_sha256":sha(B/"baseline.metallib"),"baseline_binary_parity":sha(old)==sha(B/"baseline.metallib"),"candidate_metallib_sha256":sha(B/"candidate.metallib"),"candidate_runtime_sha256":sha(B/"libmetalrobo.dylib"),"source_hashes":{str(O/p):sha(O/p) for p in ["matter/src/numi_human.mm","matter/src/metal/numi_human.metalinc","matter/src/matter_kernels.metal"]},"status":"BUILD_PASS_not_native_qualified"}
(B/"build-manifest.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
