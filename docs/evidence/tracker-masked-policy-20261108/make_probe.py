from pathlib import Path
s=Path('/Users/numi/tracker-reset-cadence-20261108/world_probe.cpp').read_text()
def rep(old,new):
 global s
 assert s.count(old)==1,(old[:100],s.count(old))
 s=s.replace(old,new)
rep('''            trackerConfig.maskedDepthInstanceIds.clear();
            trackerConfig.maskedDepthFrameOffsets.clear();
            trackerConfig.maximumActorHistoryLength = 32u;''','''            trackerConfig.bindings[0].actorFrameOffset = 27u;
            trackerConfig.maskedDepthActorFrameOffset = 34u * 3u;
            trackerConfig.maximumActorHistoryLength = 32u;''')
rep('''            manifest.reality.id = "nominal_reality";''','''            auto& obs = manifest.sensors.observation;
            obs.visual = {.width=rendererLayout.width, .height=rendererLayout.height,
                .frameOffsets={0u,1u}, .nearDepthMeters=0.1f, .farDepthMeters=5.0f,
                .pixelDropoutProbability=0.1f, .depthJitterMeters=0.15f,
                .depthNoiseSigmaMeters=0.03f, .edgeFlickerProbability=0.15f,
                .includeDerivedFeatures=true};
            for (std::uint32_t i=0; i<2u*visualPixels+MR_TASK_MASKED_DEPTH_FEATURE_COUNT; ++i)
                obs.actorFrame.push_back({.source=TaskObservationSource::maskedDepth, .component=i});
            manifest.reality.id = "nominal_reality";''')
rep('''            struct ActionOwner {
                int rejectStep;''','''            const char* shaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct Params { uint envs; uint actions; uint stride; uint pixels; uint step; };
kernel void policy(device const float* obs [[buffer(0)]], device float* actions [[buffer(1)]],
                   device float* trace [[buffer(2)]], constant Params& p [[buffer(3)]],
                   uint e [[thread_position_in_grid]]) {
    if (e >= p.envs) return;
    float n=0.0f;
    for (uint i=0; i<p.pixels; ++i) n += 1.0f-clamp(obs[e*p.stride+102u+i],0.0f,1.0f);
    n /= float(p.pixels);
    for (uint a=0; a<p.actions; ++a) actions[e*p.actions+a] = a==0 ? 0.01f*n : 0.0f;
    trace[(p.step*p.envs+e)*2u]=n;
    trace[(p.step*p.envs+e)*2u+1u]=0.01f*n;
}
)METAL";
            id<MTLDevice> policyDevice=MTLCreateSystemDefaultDevice();
            NSError* shaderError=nil;
            id<MTLLibrary> policyLibrary=[policyDevice newLibraryWithSource:[NSString stringWithUTF8String:shaderSource] options:nil error:&shaderError];
            require(policyLibrary != nil, std::string{"policy shader: "} + (shaderError ? shaderError.localizedDescription.UTF8String : "unknown"));
            id<MTLComputePipelineState> pipeline=[policyDevice newComputePipelineStateWithFunction:[policyLibrary newFunctionWithName:@"policy"] error:&shaderError];
            require(pipeline != nil, "policy pipeline");
            id<MTLBuffer> policyTrace=[policyDevice newBufferWithLength:30u*kEnvironmentCount*2u*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> policyInputs=[policyDevice newBufferWithLength:30u*kEnvironmentCount*layout.actorObservationSize*sizeof(float) options:MTLResourceStorageModeShared];
            require(policyTrace != nil && policyInputs != nil, "policy trace allocation");
            struct ActionOwner {
                __strong id<MTLComputePipelineState> pipeline;
                __strong id<MTLBuffer> trace;
                __strong id<MTLBuffer> inputs;
                std::uint32_t pixels;
                int rejectStep;''')
rep('''            } action{rejection == nullptr ? -1 : (std::strcmp(rejection, "second") == 0 ? 1 : 0),
                     &renderer, &tracker};''','''            } action{pipeline,policyTrace,policyInputs,visualPixels,
                     rejection == nullptr ? -1 : (std::strcmp(rejection, "second") == 0 ? 1 : 0),
                     &renderer, &tracker};''')
rep('''                    id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
                    if (!blit || !actions) return false;
                    [blit fillBuffer:actions range:NSMakeRange(0, actions.length) value:0];
                    [blit endEncoding]; return true;''','''                    id<MTLBuffer> input=(__bridge id<MTLBuffer>)pass.actorObservations;
                    id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
                    NSUInteger bytes=pass.environmentCount*pass.actorObservationSize*sizeof(float);
                    if (!blit || !actions || !input || (pass.controlStep+1u)*bytes>owner->inputs.length) return false;
                    [blit copyFromBuffer:input sourceOffset:pass.actorObservationOffsetElements*sizeof(float)
                        toBuffer:owner->inputs destinationOffset:pass.controlStep*bytes size:bytes];
                    [blit endEncoding];
                    struct Params { std::uint32_t envs, actions, stride, pixels, step; } p{
                        pass.environmentCount,pass.actionCount,pass.actorObservationSize,owner->pixels,pass.controlStep};
                    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                    if (!enc) return false;
                    [enc setComputePipelineState:owner->pipeline];
                    [enc setBuffer:input offset:pass.actorObservationOffsetElements*sizeof(float) atIndex:0];
                    [enc setBuffer:actions offset:0 atIndex:1]; [enc setBuffer:owner->trace offset:0 atIndex:2];
                    [enc setBytes:&p length:sizeof(p) atIndex:3];
                    [enc dispatchThreads:MTLSizeMake(pass.environmentCount,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                    [enc endEncoding]; return true;''')
rep('''                std::ofstream resetFile(std::string(prefix)+".reset",std::ios::binary|std::ios::trunc);''','''                auto saveBuffer=[prefix](const char* suffix, id<MTLBuffer> b) {
                    std::ofstream f(std::string(prefix)+suffix,std::ios::binary|std::ios::trunc);
                    f.write(static_cast<const char*>(b.contents),static_cast<std::streamsize>(b.length));
                    f.close(); require(f.good(), "policy trace write");
                };
                saveBuffer(".policy",policyTrace); saveBuffer(".inputs",policyInputs);
                std::cout << "actor_size=" << layout.actorObservationSize << " visual_pixels=" << visualPixels << " actions=" << layout.actionCount << '\\n';
                std::ofstream resetFile(std::string(prefix)+".reset",std::ios::binary|std::ios::trunc);''')
Path('/Users/numi/tracker-masked-policy-20261108/probe.cpp').write_text(s)
