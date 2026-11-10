from pathlib import Path
import subprocess,shlex,difflib,hashlib
R=Path.cwd();A=R/"artifacts/crow-v15-substep-contact-observer-20261110";B=R/"build/crow-v15-substep-contact-observer-20261110";old=(R/"build/crow-v15-landing-phase-observer-20261110/MetalWorld.mm").read_text();s=old
anchor="""            for (std::uint32_t controlStep = 0u;
                 controlStep <
                     diagnostics.layout.dispatch.controlStepCount;
                 ++controlStep) {
                MRMetalWorldPassGPU pass{};"""
insert="""            static unsigned long long crowTraceSubmission = 0;
            const bool crowTraceEnabled = std::getenv("CROW_SUBSTEP_TRACE") != nullptr;
            const unsigned long long crowTraceStep = crowTraceEnabled ? ++crowTraceSubmission : 0;
            const bool crowTraceWindow = crowTraceEnabled && crowTraceStep >= 4375 && crowTraceStep <= 4410;
            NSMutableArray<id<MTLBuffer>>* crowTraceBuffers = crowTraceWindow ? [NSMutableArray array] : nil;
            const NSUInteger crowTraceCapacity = diagnostics.layout.contactDispatch.constraintCapacity;
            const NSUInteger crowTraceLanes = batch.environmentCount;
            for (std::uint32_t controlStep = 0u;
                 controlStep <
                     diagnostics.layout.dispatch.controlStepCount;
                 ++controlStep) {
                MRMetalWorldPassGPU pass{};"""
assert s.count(anchor)==1;s=s.replace(anchor,insert)
anchor="""                    std::swap(sourceQ, destinationQ);
                    std::swap(sourceV, destinationV);"""
insert="""                    if (crowTraceWindow && controlStep == 0u) {
                        id<MTLBuffer> sourceContacts = selectedState->buffers[kContacts];
                        id<MTLBuffer> sourceStatuses = selectedState->buffers[kContactStatuses];
                        id<MTLBuffer> contactsCopy = [selectedState->device newBufferWithLength:sourceContacts.length options:MTLResourceStorageModeShared];
                        id<MTLBuffer> statusCopy = [selectedState->device newBufferWithLength:sourceStatuses.length options:MTLResourceStorageModeShared];
                        if (contactsCopy == nil || statusCopy == nil) throw std::bad_alloc();
                        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
                        [blit copyFromBuffer:sourceContacts sourceOffset:0 toBuffer:contactsCopy destinationOffset:0 size:sourceContacts.length];
                        [blit copyFromBuffer:sourceStatuses sourceOffset:0 toBuffer:statusCopy destinationOffset:0 size:sourceStatuses.length];
                        [blit endEncoding];
                        [crowTraceBuffers addObject:contactsCopy];
                        [crowTraceBuffers addObject:statusCopy];
                    }
                    std::swap(sourceQ, destinationQ);
                    std::swap(sourceV, destinationV);"""
assert s.count(anchor)==1;s=s.replace(anchor,insert)
anchor="""            ++selectedState->stats.submissionCount;
            [commandBuffer commit];"""
insert="""            if (crowTraceWindow) {
                [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                    if (completed.status != MTLCommandBufferStatusCompleted) { std::fprintf(stderr,"CROW_SUBSTEP_ERROR %llu\\n",crowTraceStep); return; }
                    for (NSUInteger sub=0; sub<crowTraceBuffers.count/2; ++sub) {
                        id<MTLBuffer> cb=crowTraceBuffers[2*sub], sb=crowTraceBuffers[2*sub+1];
                        const auto* contacts=static_cast<const MRContactConstraintGPU*>(cb.contents);
                        const auto* statuses=static_cast<const MRMetalWorldContactStatusGPU*>(sb.contents);
                        for (NSUInteger lane=0;lane<crowTraceLanes;++lane) {
                            const auto n=std::min<NSUInteger>(statuses[lane].requiredConstraints,crowTraceCapacity);
                            std::fprintf(stderr,"CROW_SUBSTEP_STATUS %llu %lu %lu %u %u\\n",crowTraceStep,(unsigned long)sub,(unsigned long)lane,statuses[lane].requiredConstraints,statuses[lane].physicsSubstep);
                            for (NSUInteger i=0;i<n;++i) {
                                const auto& c=contacts[lane*crowTraceCapacity+i];
                                const bool toeA=c.bodyA>=17u && c.bodyA<=24u, toeB=c.bodyB>=17u && c.bodyB<=24u;
                                if ((toeA && c.bodyB==30u)||(toeB && c.bodyA==30u))
                                    std::fprintf(stderr,"CROW_SUBSTEP_TOE %llu %lu %lu %u %u %.9g %.9g %.9g\\n",crowTraceStep,(unsigned long)sub,(unsigned long)lane,c.bodyA,c.bodyB,c.impulses.x,c.pointAndSeparation.w,c.normal.z);
                            }
                        }
                    }
                }];
            }
            ++selectedState->stats.submissionCount;
            [commandBuffer commit];"""
assert s.count(anchor)==1;s=s.replace(anchor,insert);src=B/"MetalWorld.mm";src.write_text(s);(A/"observer.patch").write_text("".join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile="a/MetalWorld.mm",tofile="b/MetalWorld.mm")))
lines=subprocess.check_output(["/opt/homebrew/bin/ninja","-C",str(R/"build"),"-t","commands"],text=True).splitlines();line=next(x for x in lines if " -c "+str(R/"src/metal/MetalWorld.mm") in x);args=shlex.split(line);obj=B/"MetalWorld.mm.o"
for k,v in [("-o",str(obj)),("-MF",str(obj)+".d"),("-MT",str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,"-I"+str(R/"src/metal"));args.insert(1,"-I"+str(R/"artifacts/crow-clearance-observer-20261109"));subprocess.run(args,cwd=R/"build",check=True)
line=next(x for x in lines if x.startswith(": && /usr/bin/c++") and "-dynamiclib" in x and " -o lib/libmetalrobo.dylib " in x);args=shlex.split(line[len(": && "):-len(" && :")]);args[args.index("-o")+1]=str(B/"libmetalrobo.dylib");args[args.index("CMakeFiles/metalrobo.dir/src/c_api.cpp.o")]=str(R/"build/crow-v15-heavy-wing-route-20261110/c_api.cpp.o");args[args.index("CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o")]=str(obj);subprocess.run(args,cwd=R/"build",check=True);print(hashlib.sha256((B/"libmetalrobo.dylib").read_bytes()).hexdigest())
