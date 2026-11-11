from pathlib import Path
import difflib,hashlib,shlex,subprocess
R=Path.cwd(); A=R/"artifacts/crow-v15-neck-consistent-aba-screen-20261111"; B=R/"build/crow-v15-neck-consistent-aba-screen-20261111"
(B/"include/metalrobo").mkdir(parents=True,exist_ok=True)
original=(R/"include/metalrobo/engine_types.h").read_text(); modified=original
for name,old,new in [("BODIES",32,40),("DOFS",40,48),("Q",41,49)]:
 before=f"#define MR_ARTICULATED_ABA_MAX_{name} {old}u"; after=f"#define MR_ARTICULATED_ABA_MAX_{name} {new}u"; assert modified.count(before)==1; modified=modified.replace(before,after)
assert modified.count("sizeof(MRBoundedDriveFailureGPU) == 14288")==1; modified=modified.replace("sizeof(MRBoundedDriveFailureGPU) == 14288","sizeof(MRBoundedDriveFailureGPU) == 20208")
(B/"include/metalrobo/engine_types.h").write_text(modified)
(A/"engine-types.patch").write_text("".join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile="a/engine_types.h",tofile="b/engine_types.h")))
lines=subprocess.check_output(["/opt/homebrew/bin/ninja","-C",str(R/"build"),"-t","commands"],text=True).splitlines()
sources=["src/core/CompiledMetalMultiArticulatedProgram.cpp","src/core/CompiledMetalMultiArticulatedContactProgram.cpp","src/metal/MetalArticulatedInverseMass.mm","src/metal/MetalArticulatedABA.mm","src/metal/MetalWorld.mm","src/metal/MetalMultiArticulatedContact.mm","src/metal/MetalMultiArticulatedABA.mm","src/metal/MetalMultiArticulatedInverseMass.mm"]
objects={}
for i,name in enumerate(sources):
 line=next(x for x in lines if " -c "+str(R/name) in x);args=shlex.split(line);args.insert(1,"-I"+str(B/"include"));obj=B/(str(i)+".o");objects[args[args.index("-o")+1]]=str(obj)
 for k,v in [("-o",str(obj)),("-MF",str(obj)+".d"),("-MT",str(obj))]:args[args.index(k)+1]=v
 subprocess.run(args,cwd=R/"build",check=True);print("compiled",name,flush=True)
line=next(x for x in lines if x.startswith(": && /usr/bin/c++") and "-dynamiclib" in x and " -o lib/libmetalrobo.dylib " in x);args=shlex.split(line[len(": && "):-len(" && :")]);lib=B/"libmetalrobo.dylib";args[args.index("-o")+1]=str(lib)
for old,new in objects.items():assert old in args;args[args.index(old)]=new
subprocess.run(args,cwd=R/"build",check=True)
air=B/"aba.air";ml=B/"aba.metallib";shader=R/"build/crow-v15-wide-contact-isolation-20261111/source/ArticulatedABA.metal";subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(B/"include"),"-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(shader),"-o",str(air)],check=True);subprocess.run(["xcrun","-sdk","macosx","metallib",str(air),"-o",str(ml)],check=True)
src=(R/"artifacts/crow-v15-articulated-neck-prototype-20261111/probe.cpp").read_text();old="/Users/n/Developer/numi-crow-rl-20261107/build/crow-v15-wide-contact-isolation-20261111/MetalRobo.metallib";assert src.count(old)==1;src=src.replace(old,str(ml)); (A/"probe.cpp").write_text(src)
exe=B/"probe";subprocess.run(["xcrun","clang++","-std=c++20","-O2","-I"+str(B/"include"),"-Iinclude","-Isrc",str(A/"probe.cpp"),"-L"+str(B),"-lmetalrobo","-Wl,-rpath,"+str(B),"-o",str(exe)],check=True)
for p in (shader,lib,ml,exe):print(str(p),hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
