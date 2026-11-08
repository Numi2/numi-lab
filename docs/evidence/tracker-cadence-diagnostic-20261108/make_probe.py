from pathlib import Path
p=Path('/Users/numi/numi-lab-tracker-step-rollback-20261108/apps/visual_platform_probe.cpp')
s=p.read_text().replace('trackerConfig.timestepSeconds = 0.02f;', 'trackerConfig.timestepSeconds = 0.02f;\n        trackerConfig.nominalSensorRateHz = 25.0f;',1)
start=s.index('        if (std::getenv("NUMI_TEST_TRACKER_STEP_ROLLBACK") != nullptr) {')
end=s.index('        const std::vector<float> firstDepth(',start)
block=r'''        if (std::getenv("NUMI_TEST_CADENCE") != nullptr) {
            auto* bodyValues = static_cast<MRBodyStateGPU*>(bodyBuffer.contents);
            auto* taskValues = static_cast<MRTaskStateGPU*>(taskStateBuffer.contents);
            auto* masks = static_cast<std::uint32_t*>(resetBuffer.contents);
            std::fill_n(masks, kEnvironmentCount, 0u);
            id<MTLBuffer> statuses = [device newBufferWithLength:kEnvironmentCount * sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];
            require(statuses != nil, "status allocation");
            auto* statusValues = static_cast<MRMetalWorldStatusGPU*>(statuses.contents);
            const bool inject = std::getenv("NUMI_TEST_CADENCE_FAIL") != nullptr;
            for (std::uint32_t step = 1; step <= 4; ++step) {
                std::memset(statusValues, 0, statuses.length);
                if (inject && step == 1) statusValues[0].code = MR_STEP_FACTORIZATION_FAILED;
                for (std::uint32_t e=0; e<kEnvironmentCount; ++e) {
                    bodyValues[static_cast<std::size_t>(e)*worldTemplate.engineModel.bodies.size()+trackedBody].position.x += 0.02f;
                    taskValues[e].episode.x = step;
                }
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                trackerPass.commandBuffer = (__bridge void*)cb;
                require(trackerProgram.begin(trackerProgram.context, trackerPass.commandBuffer), "begin");
                require(trackerProgram.encode(trackerProgram.context, trackerPass), "encode");
                const metalrobo::MetalWorldDeviceObservationStepResult outcome{
                    .commandBuffer=trackerPass.commandBuffer, .worldStatuses=(__bridge void*)statuses,
                    .environmentCount=kEnvironmentCount, .actorFrameSize=kActorFrameSize,
                    .actorHistoryLength=kActorHistoryLength, .controlStep=0u, .contactMode=false};
                require(trackerProgram.afterStep(trackerProgram.context, outcome), "afterStep");
                [cb commit]; trackerProgram.finish(trackerProgram.context, trackerPass.commandBuffer, true);
                [cb waitUntilCompleted]; require(cb.status==MTLCommandBufferStatusCompleted, "completion");
                metalrobo::HybridObservationBatch captured; require(renderer.readback(captured), "renderer readback");
                const float* values = static_cast<const float*>(observationBuffer.contents);
                std::cout << "cadence step=" << step << " failure0=" << (inject && step==1)
                    << " frame_sequence=" << captured.metadata.identity.z
                    << " env0_last_x=" << values[(kActorHistoryLength-1u)*kActorFrameSize+1u]
                    << " env1_last_x=" << values[kTrackWidth+(kActorHistoryLength-1u)*kActorFrameSize+1u] << '\n';
            }
            return 0;
        }
'''
s=s[:start]+block+s[end:]
Path('/Users/numi/tracker-cadence-diagnostic-20261108/probe.cpp').write_text(s)
