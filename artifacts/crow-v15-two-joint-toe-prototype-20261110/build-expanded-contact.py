from pathlib import Path
import subprocess,shlex,difflib,hashlib,json
R=Path.cwd();A=R/"artifacts/crow-v15-two-joint-toe-prototype-20261110";B=R/"build/crow-v15-expanded-contact-20261110";B.mkdir(exist_ok=True);inc=B/"include/metalrobo";inc.mkdir(parents=True,exist_ok=True)
orig=(R/"include/metalrobo/engine_types.h").read_text();new=orig.replace("#define MR_ARTICULATED_ABA_MAX_BODIES 32u","#define MR_ARTICULATED_ABA_MAX_BODIES 40u");assert new!=orig;(inc/"engine_types.h").write_text(new);(A/"expanded-contact-header.patch").write_text("".join(difflib.unified_diff(orig.splitlines(True),new.splitlines(True),fromfile="a/engine_types.h",tofile="b/engine_types.h")))
shaderNames=["ArticulatedABA","ArticulatedInverseMass","FunctionBasedDenseDynamics"];airs=[]
for n in shaderNames:
 out=B/(n+".air");subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(B/"include"),"-I",str(R/"include"),"-c",str(R/("src/metal/"+n+".metal")),"-o",str(out)],check=True);airs.append(out)
other=sorted(x for x in (R/"build/shaders").glob("MetalRobo*.air") if x.name not in {"MetalRobo"+n+".air" for n in shaderNames});lib=B/"expanded-contact.metallib";subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(x) for x in other+airs],"-o",str(lib)],check=True)
lines=subprocess.check_output(["/opt/homebrew/bin/ninja","-C",str(R/"build"),"-t","commands"],text=True).splitlines();names=["MetalArticulatedABA","MetalArticulatedInverseMass","MetalMultiArticulatedABA","MetalMultiArticulatedInverseMass","MetalMultiArticulatedContact","MetalWorld"];objs={}
for n in names:
 src=R/("src/metal/"+n+".mm"); line=next(x for x in lines if " -c "+str(src) in x);args=shlex.split(line);out=B/(n+".o");args[1:1]=["-I"+str(B/"include")];
 for k,v in [("-o",str(out)),("-MF",str(out)+".d"),("-MT",str(out))]:args[args.index(k)+1]=v
 subprocess.run(args,cwd=R/"build",check=True);objs["CMakeFiles/metalrobo.dir/src/metal/"+n+".mm.o"]=str(out)
line=next(x for x in lines if x.startswith(": && /usr/bin/c++") and " -dynamiclib " in x and " -o lib/libmetalrobo.dylib " in x);args=shlex.split(line[len(": && "):-len(" && :")]);out=B/"libmetalrobo.dylib";args[args.index("-o")+1]=str(out);args=[objs.get(x,x) for x in args];subprocess.run(args,cwd=R/"build",check=True)
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(A/"expanded-contact-bindings.json").write_text(json.dumps({"body_cap":40,"metallib_sha256":h(lib),"library_sha256":h(out),"recompiled_hosts":names,"recompiled_shaders":shaderNames,"warning":"isolated screening only"},indent=2)+"\n")
