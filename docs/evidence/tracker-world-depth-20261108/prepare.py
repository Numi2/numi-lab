from pathlib import Path
p=Path(__file__).resolve().parent
s=(p.parent/'tracker-observation-output-rollback-20261108'/'world_probe.cpp').read_text()
def replace(old,new):
    global s
    assert s.count(old)==1,(old[:100],s.count(old))
    s=s.replace(old,new)
replace('''            trackerConfig.maskedDepthInstanceIds.clear();
            trackerConfig.maskedDepthFrameOffsets.clear();
            trackerConfig.maximumActorHistoryLength = 32u;''','''            trackerConfig.maskedDepthActorFrameOffset = 102u;
            trackerConfig.maximumActorHistoryLength = 32u;''')
replace('''            manifest.task = makeFrankaPickPlaceTaskPack(manifest.sensors.observation, manifest.reality.reset);
            manifest.reality.id''','''            manifest.task = makeFrankaPickPlaceTaskPack(manifest.sensors.observation, manifest.reality.reset);
            auto& observation = manifest.sensors.observation;
            observation.visual.width = rendererLayout.width;
            observation.visual.height = rendererLayout.height;
            observation.visual.frameOffsets = {0u, 1u};
            observation.visual.nearDepthMeters = 0.1f;
            observation.visual.farDepthMeters = 5.0f;
            observation.visual.pixelDropoutProbability = 0.1f;
            observation.visual.depthJitterMeters = 0.15f;
            observation.visual.depthNoiseSigmaMeters = 0.03f;
            observation.visual.edgeFlickerProbability = 0.15f;
            observation.visual.includeDerivedFeatures = true;
            for (std::uint32_t component=0; component<2u*visualPixels+MR_TASK_MASKED_DEPTH_FEATURE_COUNT; ++component) {
                observation.actorFrame.push_back({.source=TaskObservationSource::maskedDepth, .component=component});
            }
            manifest.reality.id''')
replace('''            require(layout.actorFrameSize >= 7u && layout.actorHistoryLength <= 32u, "incompatible tracker layout");''','''            require(layout.actorFrameSize >= 7u && layout.actorHistoryLength <= 32u, "incompatible tracker layout");
            require(layout.actorFrameSize * layout.actorHistoryLength == 102u, "unexpected depth prefix");
            require(layout.actorObservationSize == 102u + 2u * visualPixels + MR_TASK_MASKED_DEPTH_FEATURE_COUNT, "unexpected depth suffix");
            std::cout << "depth_width=" << rendererLayout.width << " depth_height=" << rendererLayout.height << " depth_features=" << MR_TASK_MASKED_DEPTH_FEATURE_COUNT << "\\n";''')
replace('''                    auto copy=[&](NSUInteger phase) -> bool {
                        id<MTLBlitCommandEncoder> blit=[cmd blitCommandEncoder];''','''                    auto copy=[&](NSUInteger phase) -> bool {
                        if (std::getenv("NUMI_TEST_NO_CAPTURE") != nullptr) return true;
                        id<MTLBlitCommandEncoder> blit=[cmd blitCommandEncoder];''')
replace('''                    if (!c->tracker.afterStep(c->tracker.context,pass)) return false;
                    id<MTLCommandBuffer> cmd''','''                    if (!c->tracker.afterStep(c->tracker.context,pass)) return false;
                    if (std::getenv("NUMI_TEST_NO_CAPTURE") != nullptr) return true;
                    id<MTLCommandBuffer> cmd''')
(p/'world_depth_probe.mm').write_text(s)
