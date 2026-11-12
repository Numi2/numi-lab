from pathlib import Path
import subprocess,shlex,difflib,hashlib,json
R=Path.cwd();A=R/'artifacts/crow-v15-tail-load-observer-20261112';B=R/'build/crow-v15-tail-load-observer-20261112';W=R/'build/crow-v15-wide-contact-isolation-20261111';base=R/'build/crow-v15-hip-roll-journey-integration-20261111/MetalWorld.mm';s=base.read_text()
needle='            const NSUInteger crowTraceCapacity = diagnostics.layout.contactDispatch.constraintCapacity;\n';assert s.count(needle)==1
s=s.replace(needle,'''            const bool crowAeroWindow = std::getenv("CROW_AERO_TRACE") != nullptr && ((crowTraceStep >= 451 && crowTraceStep <= 500) || (crowTraceStep >= 701 && crowTraceStep <= 750));
            NSMutableArray<id<MTLBuffer>>* crowAeroBuffers = crowAeroWindow ? [NSMutableArray array] : nil;
            const NSUInteger crowAeroBodyStride = world.model().bodies.size();
            const NSUInteger crowAeroQStride = diagnostics.layout.dispatch.qStride;
            const NSUInteger crowAeroVStride = diagnostics.layout.dispatch.vStride;
'''+needle)
needle='                    const bool encodedABA =\n';assert s.count(needle)==1
s=s.replace(needle,'''                    auto crowCaptureFlapping = [&]() -> bool {
                        if (!config.flappingWingProgram.valid()) return true;
                        if (!crowAeroWindow || controlStep != 0u) return encodeFlappingWingActuation(*selectedState,commandBuffer,sourceQ,sourceV,batch.environmentCount);
                        id<MTLBuffer> wrench=selectedState->buffers[kBodyWrenchPlaceholder], q=selectedState->buffers[sourceQ], v=selectedState->buffers[sourceV];
                        id<MTLBuffer> before=[selectedState->device newBufferWithLength:wrench.length options:MTLResourceStorageModeShared];
                        id<MTLBuffer> after=[selectedState->device newBufferWithLength:wrench.length options:MTLResourceStorageModeShared];
                        id<MTLBuffer> qc=[selectedState->device newBufferWithLength:q.length options:MTLResourceStorageModeShared];
                        id<MTLBuffer> vc=[selectedState->device newBufferWithLength:v.length options:MTLResourceStorageModeShared];
                        if (before==nil || after==nil || qc==nil || vc==nil) throw std::bad_alloc();
                        id<MTLBlitCommandEncoder> pre=[commandBuffer blitCommandEncoder];
                        [pre copyFromBuffer:wrench sourceOffset:0 toBuffer:before destinationOffset:0 size:wrench.length];
                        [pre copyFromBuffer:q sourceOffset:0 toBuffer:qc destinationOffset:0 size:q.length];
                        [pre copyFromBuffer:v sourceOffset:0 toBuffer:vc destinationOffset:0 size:v.length];
                        [pre endEncoding];
                        const bool ok=encodeFlappingWingActuation(*selectedState,commandBuffer,sourceQ,sourceV,batch.environmentCount);
                        id<MTLBlitCommandEncoder> post=[commandBuffer blitCommandEncoder];
                        [post copyFromBuffer:wrench sourceOffset:0 toBuffer:after destinationOffset:0 size:wrench.length];
                        [post endEncoding];
                        [crowAeroBuffers addObject:before];[crowAeroBuffers addObject:after];[crowAeroBuffers addObject:qc];[crowAeroBuffers addObject:vc];
                        return ok;
                    };
'''+needle)
old='''!config.flappingWingProgram.valid() ||
                            encodeFlappingWingActuation(
                                *selectedState,
                                commandBuffer,
                                sourceQ,
                                sourceV,
                                batch.environmentCount
                            )''';assert s.count(old)==1;s=s.replace(old,'crowCaptureFlapping()')
needle='            if (crowTraceWindow) {\n                [commandBuffer addCompletedHandler:';assert s.count(needle)==1
addition='''            if (crowAeroWindow) {
                [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                    if (completed.status != MTLCommandBufferStatusCompleted) { std::fprintf(stderr,"CROW_AERO_ERROR %llu\\n",crowTraceStep); return; }
                    for(NSUInteger sub=0;sub<crowAeroBuffers.count/4;++sub) {
                        const auto* before=static_cast<const MRABABodyWrenchGPU*>(crowAeroBuffers[4*sub].contents);
                        const auto* after=static_cast<const MRABABodyWrenchGPU*>(crowAeroBuffers[4*sub+1].contents);
                        const auto* q=static_cast<const float*>(crowAeroBuffers[4*sub+2].contents);
                        const auto* v=static_cast<const float*>(crowAeroBuffers[4*sub+3].contents);
                        for(NSUInteger lane=0;lane<crowTraceLanes;++lane) {
                            std::fprintf(stderr,"CROW_WING_WRENCH %llu %lu %lu %.9g %.9g %.9g %.9g",crowTraceStep,(unsigned long)sub,(unsigned long)lane,q[lane*crowAeroQStride+9],q[lane*crowAeroQStride+10],v[lane*crowAeroVStride+8],v[lane*crowAeroVStride+9]);
                            const int ids[4]={5,6,0,7};
                            for(int j=0;j<4;++j) { const auto& a=after[lane*crowAeroBodyStride+ids[j]]; const auto& b=before[lane*crowAeroBodyStride+ids[j]];
                                std::fprintf(stderr," %.9g %.9g %.9g %.9g %.9g %.9g",a.force.x-b.force.x,a.force.y-b.force.y,a.force.z-b.force.z,a.torque.x-b.torque.x,a.torque.y-b.torque.y,a.torque.z-b.torque.z);
                            }
                            for(NSUInteger j=0;j<crowAeroQStride;++j) std::fprintf(stderr," %.9g",q[lane*crowAeroQStride+j]);
                            for(NSUInteger j=0;j<crowAeroVStride;++j) std::fprintf(stderr," %.9g",v[lane*crowAeroVStride+j]);
                            std::fprintf(stderr,"\\n");
                        }
                    }
                }];
            }
'''
s=s.replace(needle,addition+needle);source=B/'MetalWorld.mm';source.write_text(s);(A/'observer.patch').write_text(''.join(difflib.unified_diff(base.read_text().splitlines(True),s.splitlines(True),fromfile='a/MetalWorld.mm',tofile='b/MetalWorld.mm')))
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();comp=next(x for x in lines if ' -c '+str(R/'src/metal/MetalWorld.mm') in x);link=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(comp);obj=B/'MetalWorld.mm.o'
for k,vv in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=vv
args[-1]=str(source);args[1:1]=['-I'+str(W/'include'),'-I'+str(R/'src/metal')];subprocess.run(args,cwd=R/'build',check=True)
args=shlex.split(link[len(': && '):-len(' && :')]);lib=B/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(R/'build/crow-v15-hip-roll-obstacle-free-control-20261111/c_api.cpp.o')
files=['src/core/CompiledMetalMultiArticulatedProgram.cpp','src/core/CompiledMetalMultiArticulatedContactProgram.cpp','src/metal/MetalArticulatedInverseMass.mm','src/metal/MetalArticulatedABA.mm','src/metal/MetalWorld.mm','src/metal/MetalMultiArticulatedContact.mm','src/metal/MetalMultiArticulatedABA.mm','src/metal/MetalMultiArticulatedInverseMass.mm']
for i,n in enumerate(files):
 old='CMakeFiles/metalrobo.dir/'+n+'.o';assert old in args;args[args.index(old)]=str(obj if i==4 else W/(str(i)+'.o'))
subprocess.run(args,cwd=R/'build',check=True);h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'build-hashes.json').write_text(json.dumps({'baseline_source_sha256':h(base),'observer_patch_sha256':h(A/'observer.patch'),'host_sha256':h(lib)},indent=2)+'\n')
