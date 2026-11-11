from pathlib import Path
import difflib,hashlib,json,shlex,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-landing-target-observer-20261111";B=R/"build/crow-v15-landing-target-observer-20261111";W=R/"build/crow-v15-wide-contact-isolation-20261111";H=R/"build/crow-v15-hip-roll-journey-integration-20261111";P=R/"build/crow-v15-landing-phase-observer-20261111";base=(P/"MetalWorld.mm").read_text()
a="""            id<MTLBuffer> crowTaskStateCopy = nil;
            if (crowTraceWindow) {""";b="""            id<MTLBuffer> crowTargetCopy = nil;
            if (crowTraceWindow && crowTraceStep>=3200u && crowTraceStep<=4000u) {
                id<MTLBuffer> sourceTargets=selectedState->buffers[kEffortTrajectory];
                crowTargetCopy=[selectedState->device newBufferWithLength:sourceTargets.length options:MTLResourceStorageModeShared];
                if (crowTargetCopy == nil) throw std::bad_alloc();
                id<MTLBlitCommandEncoder> targetBlit=[commandBuffer blitCommandEncoder];
                [targetBlit copyFromBuffer:sourceTargets sourceOffset:0 toBuffer:crowTargetCopy destinationOffset:0 size:sourceTargets.length];
                [targetBlit endEncoding];
            }
            id<MTLBuffer> crowTaskStateCopy = nil;
            if (crowTraceWindow) {""";assert base.count(a)==1;s=base.replace(a,b)
a="""                    const auto* crowStates=static_cast<const MRTaskStateGPU*>(crowTaskStateCopy.contents);""";b="""                    if (crowTargetCopy != nil && crowTargetCopy.length>=crowTraceLanes*40u*sizeof(float)) {
                        const float* targets=static_cast<const float*>(crowTargetCopy.contents);
                        for (NSUInteger lane=0;lane<crowTraceLanes;++lane) {
                            std::fprintf(stderr,"CROW_DRIVE_TARGET %llu %lu",crowTraceStep,(unsigned long)lane);
                            for (NSUInteger i=0;i<40u;++i) std::fprintf(stderr," %.9g",targets[lane*40u+i]);
                            std::fprintf(stderr,"\\n");
                        }
                    }
                    const auto* crowStates=static_cast<const MRTaskStateGPU*>(crowTaskStateCopy.contents);""";assert s.count(a)==1;s=s.replace(a,b);(B/"MetalWorld.mm").write_text(s);(A/"observer.patch").write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="a/reference.mm",tofile="b/target-observer.mm")))
lines=subprocess.check_output(["/opt/homebrew/bin/ninja","-C",str(R/"build"),"-t","commands"],text=True).splitlines();compile=next(x for x in lines if " -c "+str(R/"src/metal/MetalWorld.mm") in x);link=next(x for x in lines if x.startswith(": && /usr/bin/c++") and "-dynamiclib" in x and " -o lib/libmetalrobo.dylib " in x);args=shlex.split(compile);obj=B/"MetalWorld.mm.o"
for k,v in [("-o",str(obj)),("-MF",str(obj)+".d"),("-MT",str(obj))]:args[args.index(k)+1]=v
args[-1]=str(B/"MetalWorld.mm");args[1:1]=["-I"+str(W/"include"),"-I"+str(R/"src/metal")];subprocess.run(args,cwd=R/"build",check=True)
args=shlex.split(link[len(": && "):-len(" && :")]);lib=B/"libmetalrobo.dylib";args[args.index("-o")+1]=str(lib);args[args.index("CMakeFiles/metalrobo.dir/src/c_api.cpp.o")]=str(H/"c_api.cpp.o");files=["src/core/CompiledMetalMultiArticulatedProgram.cpp","src/core/CompiledMetalMultiArticulatedContactProgram.cpp","src/metal/MetalArticulatedInverseMass.mm","src/metal/MetalArticulatedABA.mm","src/metal/MetalWorld.mm","src/metal/MetalMultiArticulatedContact.mm","src/metal/MetalMultiArticulatedABA.mm","src/metal/MetalMultiArticulatedInverseMass.mm"]
for i,n in enumerate(files):
 old="CMakeFiles/metalrobo.dir/"+n+".o";assert old in args;args[args.index(old)]=str(obj if i==4 else W/(str(i)+".o"))
subprocess.run(args,cwd=R/"build",check=True);h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/"hashes.json").write_text(json.dumps({"source_sha256":h(B/"MetalWorld.mm"),"host_sha256":h(lib),"reference_host_sha256":h(P/"libmetalrobo.dylib")},indent=2)+"\n")
