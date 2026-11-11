from pathlib import Path
import difflib,hashlib,json,shlex,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-slow-fold-full-observer-20261111";B=R/"build/crow-v15-slow-fold-full-observer-20261111";W=R/"build/crow-v15-wide-contact-isolation-20261111";H=R/"build/crow-v15-hip-roll-obstacle-free-control-20261111";P=R/"build/crow-v15-hip-roll-journey-integration-20261111"
base=(P/"MetalWorld.mm").read_text()
needle="                            for(NSUInteger i=0;i<n;++i){const auto& c=contacts[lane*crowTraceCapacity+i];"
extra='''\n                                if (crowTraceStep>=380u && crowTraceStep<=1600u && c.bodyA<35u && c.bodyB<35u && c.impulses.x>1.0e-7f) std::fprintf(stderr,"CROW_SELF_CONTACT %llu %lu %lu %u %u %.9g %.9g\\n",crowTraceStep,(unsigned long)sub,(unsigned long)lane,c.bodyA,c.bodyB,c.impulses.x,c.pointAndSeparation.w);'''
assert base.count(needle)==1
s=base.replace(needle,needle+extra);(B/"MetalWorld.mm").write_text(s)
(A/"observer.patch").write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="a/observer.mm",tofile="b/self-contact-observer.mm")))
lines=subprocess.check_output(["/opt/homebrew/bin/ninja","-C",str(R/"build"),"-t","commands"],text=True).splitlines()
compile=next(x for x in lines if " -c "+str(R/"src/metal/MetalWorld.mm") in x)
link=next(x for x in lines if x.startswith(": && /usr/bin/c++") and "-dynamiclib" in x and " -o lib/libmetalrobo.dylib " in x)
args=shlex.split(compile);obj=B/"MetalWorld.mm.o"
for k,v in [("-o",str(obj)),("-MF",str(obj)+".d"),("-MT",str(obj))]:args[args.index(k)+1]=v
args[-1]=str(B/"MetalWorld.mm");args[1:1]=["-I"+str(W/"include"),"-I"+str(R/"src/metal")];subprocess.run(args,cwd=R/"build",check=True)
args=shlex.split(link[len(": && "):-len(" && :")]);lib=B/"libmetalrobo.dylib";args[args.index("-o")+1]=str(lib);args[args.index("CMakeFiles/metalrobo.dir/src/c_api.cpp.o")]=str(H/"c_api.cpp.o")
files=["src/core/CompiledMetalMultiArticulatedProgram.cpp","src/core/CompiledMetalMultiArticulatedContactProgram.cpp","src/metal/MetalArticulatedInverseMass.mm","src/metal/MetalArticulatedABA.mm","src/metal/MetalWorld.mm","src/metal/MetalMultiArticulatedContact.mm","src/metal/MetalMultiArticulatedABA.mm","src/metal/MetalMultiArticulatedInverseMass.mm"]
for i,n in enumerate(files):
 old="CMakeFiles/metalrobo.dir/"+n+".o";assert old in args;args[args.index(old)]=str(obj if i==4 else W/(str(i)+".o"))
subprocess.run(args,cwd=R/"build",check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(A/"hashes.json").write_text(json.dumps({"source_sha256":h(B/"MetalWorld.mm"),"host_sha256":h(lib),"source_host_sha256":h(H/"libmetalrobo.dylib")},indent=2)+"\n")
