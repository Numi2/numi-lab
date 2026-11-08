from pathlib import Path
p=Path('/Users/numi/numi-lab-tracker-step-rollback-20261108/apps/visual_platform_probe.cpp')
s=p.read_text().replace('trackerConfig.timestepSeconds = 0.02f;', 'trackerConfig.timestepSeconds = 0.02f;\n        trackerConfig.nominalSensorRateHz = std::getenv("NUMI_TEST_SLOW_RATE") ? 25.0f : 50.0f;',1)
start=s.index('        if (std::getenv("NUMI_TEST_TRACKER_STEP_ROLLBACK") != nullptr) {')
end=s.index('        const std::vector<float> firstDepth(',start)
block=r'''        if (std::getenv("NUMI_TEST_RESET_CADENCE") != nullptr) {
            auto* bodies = static_cast<MRBodyStateGPU*>(bodyBuffer.contents);
            auto* tasks = static_cast<MRTaskStateGPU*>(taskStateBuffer.contents);
            auto* masks = static_cast<std::uint32_t*>(resetBuffer.contents);
            const std::size_t last=(kActorHistoryLength-1u)*kActorFrameSize;
            std::cout.precision(9);
            auto report=[&](const char* label) {
                metalrobo::HybridObservationBatch capture; require(renderer.readback(capture), "readback");
                const auto* output=static_cast<const float*>(observationBuffer.contents);
                std::cout << "RESET_OBSERVATION " << label << " frame=" << capture.metadata.identity.z
                          << " env0_visible=" << output[last] << " env0_x=" << output[last+1u]
                          << " env1_visible=" << output[kTrackWidth+last] << " env1_x=" << output[kTrackWidth+last+1u] << '\n';
                require(output[last]>0.5f && output[kTrackWidth+last]>0.5f &&
                        std::isfinite(output[last+1u]) && std::isfinite(output[kTrackWidth+last+1u]), "invalid tracks");
            };
            report("initial");
            std::fill_n(masks,kEnvironmentCount,0u); masks[0]=1u;
            bodies[trackedBody].position.x += 0.20f;
            tasks[0].episode.x=0u; tasks[0].episode.y=1u; tasks[1].episode.x=1u;
            id<MTLBuffer> statuses=[device newBufferWithLength:kEnvironmentCount*sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];
            require(statuses!=nil,"statuses"); std::memset(statuses.contents,0,statuses.length);
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
            report("reset"); return 0;
        }
'''
s=s[:start]+block+s[end:]
Path('/Users/numi/tracker-reset-cadence-20261108/probe.cpp').write_text(s)
