from pathlib import Path
import difflib,hashlib,json,shlex,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-takeoff-substep-wing-observer-20261112";B=R/"build/crow-v15-takeoff-substep-wing-observer-20261112";J=R/"build/crow-v15-hip-roll-journey-integration-20261111";W=R/"build/crow-v15-wide-contact-isolation-20261111"
L=R/"build/crow-v15-landing-phase-observer-20261111"
old=(L/"MetalWorld.mm").read_text();s=old
edits=[
 ("const NSUInteger crowTraceVStride = diagnostics.layout.dispatch.vStride;","const NSUInteger crowTraceVStride = diagnostics.layout.dispatch.vStride;\n            const NSUInteger crowTraceQStride = diagnostics.layout.dispatch.qStride;"),
 ("id<MTLBuffer> sourceVelocities = selectedState->buffers[destinationV];","id<MTLBuffer> sourceVelocities = selectedState->buffers[destinationV];\n                        id<MTLBuffer> sourceCoordinates = selectedState->buffers[destinationQ];"),
 ("id<MTLBuffer> velocityCopy = [selectedState->device newBufferWithLength:sourceVelocities.length options:MTLResourceStorageModeShared];","id<MTLBuffer> velocityCopy = [selectedState->device newBufferWithLength:sourceVelocities.length options:MTLResourceStorageModeShared];\n                        id<MTLBuffer> coordinateCopy = [selectedState->device newBufferWithLength:sourceCoordinates.length options:MTLResourceStorageModeShared];"),
 ("contactsCopy == nil || statusCopy == nil || velocityCopy == nil || crowTraceVStride < 40","contactsCopy == nil || statusCopy == nil || velocityCopy == nil || coordinateCopy == nil || crowTraceVStride < 40 || crowTraceQStride < 41"),
 ("[blit copyFromBuffer:sourceVelocities sourceOffset:0 toBuffer:velocityCopy destinationOffset:0 size:sourceVelocities.length];","[blit copyFromBuffer:sourceVelocities sourceOffset:0 toBuffer:velocityCopy destinationOffset:0 size:sourceVelocities.length];\n                        [blit copyFromBuffer:sourceCoordinates sourceOffset:0 toBuffer:coordinateCopy destinationOffset:0 size:sourceCoordinates.length];"),
 ("[crowTraceBuffers addObject:velocityCopy];","[crowTraceBuffers addObject:velocityCopy];\n                        [crowTraceBuffers addObject:coordinateCopy];"),
 ("sub<crowTraceBuffers.count/3; ++sub","sub<crowTraceBuffers.count/4; ++sub"),
 ("id<MTLBuffer> cb=crowTraceBuffers[3*sub], sb=crowTraceBuffers[3*sub+1], vb=crowTraceBuffers[3*sub+2];","id<MTLBuffer> cb=crowTraceBuffers[4*sub], sb=crowTraceBuffers[4*sub+1], vb=crowTraceBuffers[4*sub+2], qb=crowTraceBuffers[4*sub+3];\n                        const auto* coordinates=static_cast<const float*>(qb.contents);"),
 ("const auto n=std::min<NSUInteger>(statuses[lane].requiredConstraints,crowTraceCapacity);","const auto n=std::min<NSUInteger>(statuses[lane].requiredConstraints,crowTraceCapacity);\n                            if (crowTraceStep >= 335 && crowTraceStep <= 365) { std::fprintf(stderr,\"CROW_Q_SUBSTEP %llu %lu %lu\",crowTraceStep,(unsigned long)sub,(unsigned long)lane); for (NSUInteger j=0;j<crowTraceQStride;++j) std::fprintf(stderr,\" %.9g\",coordinates[lane*crowTraceQStride+j]); std::fprintf(stderr,\"\\n\"); }"),
]
for x,y in edits:
 assert s.count(x)==1,(x,s.count(x));s=s.replace(x,y)
src=B/"MetalWorld.mm";src.write_text(s);(A/"observer.patch").write_text("".join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile="a/MetalWorld.mm",tofile="b/MetalWorld.mm")))
lines=subprocess.check_output(["/opt/homebrew/bin/ninja","-C",str(R/"build"),"-t","commands"],text=True).splitlines();line=next(x for x in lines if " -c "+str(R/"src/metal/MetalWorld.mm") in x);args=shlex.split(line);obj=B/"MetalWorld.mm.o"
for k,v in [("-o",str(obj)),("-MF",str(obj)+".d"),("-MT",str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args[1:1]=["-I"+str(W/"include"),"-I"+str(R/"src/metal"),"-I"+str(R/"artifacts/crow-clearance-observer-20261109")];subprocess.run(args,cwd=R/"build",check=True)
line=next(x for x in lines if x.startswith(": && /usr/bin/c++") and "-dynamiclib" in x and " -o lib/libmetalrobo.dylib " in x);args=shlex.split(line[len(": && "):-len(" && :")]);args[args.index("-o")+1]=str(B/"libmetalrobo.dylib");args[args.index("CMakeFiles/metalrobo.dir/src/c_api.cpp.o")]=str(J/"c_api.cpp.o")
files=["src/core/CompiledMetalMultiArticulatedProgram.cpp","src/core/CompiledMetalMultiArticulatedContactProgram.cpp","src/metal/MetalArticulatedInverseMass.mm","src/metal/MetalArticulatedABA.mm","src/metal/MetalWorld.mm","src/metal/MetalMultiArticulatedContact.mm","src/metal/MetalMultiArticulatedABA.mm","src/metal/MetalMultiArticulatedInverseMass.mm"]
for i,n in enumerate(files):
 key="CMakeFiles/metalrobo.dir/"+n+".o";assert key in args;args[args.index(key)]=str(obj if i==4 else W/(str(i)+".o"))
subprocess.run(args,cwd=R/"build",check=True);h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/"hashes.json").write_text(json.dumps({"observer_source":h(src),"observer_host":h(B/"libmetalrobo.dylib"),"reference_host":h(L/"libmetalrobo.dylib")},indent=2)+"\n")
