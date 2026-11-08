from pathlib import Path
s=Path('/Users/numi/tracker-reset-cadence-20261108/probe.cpp').read_text()
s=s.replace('std::getenv("NUMI_TEST_SLOW_RATE") ? 25.0f : 50.0f','std::getenv("NUMI_TEST_LONG_RATE") ? 10.0f : (std::getenv("NUMI_TEST_SLOW_RATE") ? 25.0f : 50.0f)')
a=s.index('        if (std::getenv("NUMI_TEST_RESET_CADENCE") != nullptr) {'); b=s.index('        const std::vector<float> firstDepth(',a)
block=r'''        if (std::getenv("NUMI_TEST_RESET_CADENCE") != nullptr) {
            auto* bodies = static_cast<MRBodyStateGPU*>(bodyBuffer.contents);
            auto* tasks = static_cast<MRTaskStateGPU*>(taskStateBuffer.contents);
            auto* masks = static_cast<std::uint32_t*>(resetBuffer.contents);
            const std::size_t last=(kActorHistoryLength-1u)*kActorFrameSize;
            std::cout.precision(9);
            auto report=[&](int step) {
                metalrobo::HybridObservationBatch capture; require(renderer.readback(capture), "readback");
                const auto* output=static_cast<const float*>(observationBuffer.contents);
                const auto* first=output+kActorFrameSize*kActorHistoryLength;
                const std::size_t env0Far=std::count(first,first+visualPixels,1.0f);
                const auto* features=first+2u*visualPixels;
                float featureAbs=0; for(unsigned i=0;i<MR_TASK_MASKED_DEPTH_FEATURE_COUNT;i++) featureAbs+=std::abs(features[i]);
                std::cout << "CHECK step=" << step << " frame=" << capture.metadata.identity.z
                          << " env0_visible=" << output[last] << " env0_x=" << output[last+1u]
                          << " env0_far=" << env0Far << " pixels=" << visualPixels << " features_abs=" << featureAbs
                          << " env1_visible=" << output[kTrackWidth+last] << " env1_x=" << output[kTrackWidth+last+1u] << '\n';
                require(output[kTrackWidth+last]>0.5f && std::isfinite(output[kTrackWidth+last+1u]),"env1 invalid");
                bool stale=(std::getenv("NUMI_TEST_LONG_RATE") && step>0 && step<5) ||
                    (std::getenv("NUMI_TEST_SLOW_RATE") && !std::getenv("NUMI_TEST_LONG_RATE") && step==1);
                if (stale) require(output[last]==0.0f && output[last+1u]==0.0f && env0Far==visualPixels && featureAbs==0.0f,"stale reset frame leaked");
                else require(output[last]>0.5f && std::isfinite(output[last+1u]),"fresh env0 track invalid");
            };
            report(0);
            std::fill_n(masks,kEnvironmentCount,0u);
            bodies[trackedBody].position.x += 0.20f;
            id<MTLBuffer> statuses=[device newBufferWithLength:kEnvironmentCount*sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];
            require(statuses!=nil,"statuses"); std::memset(statuses.contents,0,statuses.length);
            for (std::uint32_t step=1; step<=5; ++step) {
                masks[0]=step==1 ? 1u : 0u; tasks[0].episode.x=step-1u; tasks[0].episode.y=1u; tasks[1].episode.x=step;
                id<MTLCommandBuffer> cb=[queue commandBuffer]; trackerPass.commandBuffer=(__bridge void*)cb;
                require(trackerProgram.begin(trackerProgram.context,trackerPass.commandBuffer),"begin");
                require(trackerProgram.encode(trackerProgram.context,trackerPass),"encode");
                const metalrobo::MetalWorldDeviceObservationStepResult outcome{
                    .commandBuffer=trackerPass.commandBuffer,.worldStatuses=(__bridge void*)statuses,
                    .environmentCount=kEnvironmentCount,.actorFrameSize=kActorFrameSize,
                    .actorHistoryLength=kActorHistoryLength,.controlStep=0u,.contactMode=false};
                require(trackerProgram.afterStep(trackerProgram.context,outcome),"afterStep");
                [cb commit];trackerProgram.finish(trackerProgram.context,trackerPass.commandBuffer,true);
                [cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,"completion");
                report(step);
            }
            return 0;
        }
'''
s=s[:a]+block+s[b:]
Path('/Users/numi/tracker-reset-cadence-20261108/validation.cpp').write_text(s)
