from pathlib import Path
import difflib,hashlib,json,shlex,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-walk-traction-screen-20261112";B=R/"build/crow-v15-walk-traction-screen-20261112";W=R/"build/crow-v15-wide-contact-isolation-20261111";H=R/"build/crow-v15-substep-target-interpolation-screen-20261112"
old=(R/"src/core/TaskProgram.cpp").read_text();s=old
for n in ("actionHash","observationHash"):
 needle=n+".scalar(world.fingerprint());";assert s.count(needle)==1,(n,s.count(needle));s=s.replace(needle,"if (std::getenv(\"CROW_LAYOUT_AUDIT_IGNORE_WORLD\") == nullptr) "+needle)
s="#include <cstdlib>\n"+s;(B/"TaskProgram-layout-audit.cpp").write_text(s);(A/"layout-audit.patch").write_text("".join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile="a/TaskProgram.cpp",tofile="b/layout-audit-only.cpp")))
lines=subprocess.check_output(["/opt/homebrew/bin/ninja","-C",str(R/"build"),"-t","commands"],text=True).splitlines();line=next(x for x in lines if " -c "+str(R/"src/core/TaskProgram.cpp") in x);args=shlex.split(line);obj=B/"TaskProgram-layout-audit.cpp.o"
for k,v in [("-o",str(obj)),("-MF",str(obj)+".d"),("-MT",str(obj))]:args[args.index(k)+1]=v
args[-1]=str(B/"TaskProgram-layout-audit.cpp");subprocess.run(args,cwd=R/"build",check=True)
line=next(x for x in lines if x.startswith(": && /usr/bin/c++") and "-dynamiclib" in x and " -o lib/libmetalrobo.dylib " in x);args=shlex.split(line[len(": && "):-len(" && :")]);args[args.index("-o")+1]=str(B/"lib-layout-audit.dylib");args[args.index("CMakeFiles/metalrobo.dir/src/c_api.cpp.o")]=str(B/"c_api.cpp.o");args[args.index("CMakeFiles/metalrobo.dir/src/core/TaskProgram.cpp.o")]=str(obj)
files=["src/core/CompiledMetalMultiArticulatedProgram.cpp","src/core/CompiledMetalMultiArticulatedContactProgram.cpp","src/metal/MetalArticulatedInverseMass.mm","src/metal/MetalArticulatedABA.mm","src/metal/MetalWorld.mm","src/metal/MetalMultiArticulatedContact.mm","src/metal/MetalMultiArticulatedABA.mm","src/metal/MetalMultiArticulatedInverseMass.mm"]
for i,n in enumerate(files):
 key="CMakeFiles/metalrobo.dir/"+n+".o";assert key in args;args[args.index(key)]=str(H/"MetalWorld.mm.o" if i==4 else W/(str(i)+".o"))
subprocess.run(args,cwd=R/"build",check=True);h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/"layout-audit-host-sha256.txt").write_text(h(B/"lib-layout-audit.dylib")+"\n")

d=B/"layout-audit";d.mkdir(exist_ok=True);link=d/"libmetalrobo.dylib"
if not link.exists():link.symlink_to("../lib-layout-audit.dylib")
