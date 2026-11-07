#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/RetinalObserver.hpp"
#include "metalrobo/FiniteResource.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
namespace metalrobo {
namespace {
void require(bool yes,const char* reason) {if (!yes) throw std::invalid_argument(reason);}
bool equal(mr_float4 a,mr_float4 b) {return a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w;}
}
namespace detail {
struct RetinalObserverRuntime : std::enable_shared_from_this<RetinalObserverRuntime> {
    MetalWorldFamilyContext worlds;
    std::array<std::unique_ptr<MetalHybridRenderer>,2> renderers;
    std::unique_ptr<RetinalStreamContext> stream;
    FiniteResourceDeviceView cueResources;
    __strong id<MTLBuffer> cueBindings=nil,cueVisibility=nil;
    __strong id<MTLComputePipelineState> cuePipeline=nil;
    mr_uint4 cueCounts{};
    RetinalStreamConfig timing;
    unsigned environments=0,bodies=0,features=0,offset=0;
    std::uint64_t fingerprint=0;
    float dt=0;
    mutable std::mutex mutex;
    std::uintptr_t activeToken=0;
    bool poisoned=false;
    static bool encode(void* opaque,const MetalWorldDeviceObservationPass& pass) {
        auto s=static_cast<RetinalObserverRuntime*>(opaque)->shared_from_this();
        const std::lock_guard lock(s->mutex);
        if (s->poisoned || !pass.commandBuffer || !pass.currentBodies || !pass.taskStates || !pass.actorObservations ||
            pass.environmentCount!=s->environments || pass.bodyCount!=s->bodies || pass.actorObservationSize!=s->features) return false;
        const auto token=reinterpret_cast<std::uintptr_t>(pass.commandBuffer);
        if (s->activeToken && s->activeToken!=token) return false;
        id<MTLCommandBuffer> command=(__bridge id<MTLCommandBuffer>)pass.commandBuffer;
        id<MTLBuffer> bodies=(__bridge id<MTLBuffer>)pass.currentBodies;
        if (command.status>MTLCommandBufferStatusEnqueued || bodies.device.registryID!=command.device.registryID ||
            bodies.length<std::size_t(s->environments)*s->bodies*sizeof(MRBodyStateGPU)) return false;
        if (!s->activeToken) {
            s->activeToken=token;
            [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                const std::lock_guard done(s->mutex);
                if (completed.status!=MTLCommandBufferStatusCompleted) s->poisoned=true;
                if (s->activeToken==token) s->activeToken=0;
            }];
        }
        // Noiseless instantaneous optics use one fixed lighting/sample phase.
        // Encoding counts change with bootstrap calls and are not sensor time.
        // The GPU TaskState clock alone owns retinal cadence, age and delay.
        HybridDeviceStateBatch live{.currentBodyStates=pass.currentBodies,.previousBodyStates=pass.currentBodies,
            .environmentCount=s->environments,.bodyCount=s->bodies,.frameIndex=0,
            .sensorSequence=0,.source=MR_VISUAL_SOURCE_SIMULATION,.captureTimestampSeconds=0};
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];if (encoder==nil) {s->poisoned=true;return false;}
        if (s->cuePipeline) {
            [encoder setComputePipelineState:s->cuePipeline];
            [encoder setBuffer:(__bridge id<MTLBuffer>)s->cueResources.states offset:0 atIndex:0];
            [encoder setBuffer:s->cueBindings offset:0 atIndex:1];
            [encoder setBuffer:s->cueVisibility offset:0 atIndex:2];
            [encoder setBytes:&s->cueCounts length:sizeof(s->cueCounts) atIndex:3];
            [encoder dispatchThreads:MTLSizeMake(std::size_t(s->cueCounts.x)*s->cueCounts.y,1,1)
                threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            live.currentMeshVisibility=(__bridge void*)s->cueVisibility;
        }
        for (unsigned eye=0;eye<2;++eye) {
            const auto status=s->renderers[eye]->encode(s->worlds,live,eye,(__bridge void*)encoder);
            if (!status.succeeded()) {[encoder endEncoding];s->poisoned=true;return false;}
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        [encoder endEncoding];
        const std::array<void*,2> rgb{s->renderers[0]->nativeBuffer(MetalHybridRendererBuffer::rgb),
            s->renderers[1]->nativeBuffer(MetalHybridRendererBuffer::rgb)};
        if (!s->stream->encode(pass.commandBuffer,pass.taskStates,rgb,pass.actorObservations,s->features,s->offset,
            pass.actorObservationOffsetElements)) {s->poisoned=true;return false;}
        return true;
    }
};
}
RetinalObserverContext::RetinalObserverContext(const CompiledRun& run,const WorldFamily& family,
    const VisualSceneManifestV3& scene,unsigned environments,std::uint64_t seed)
    :RetinalObserverContext(run,family,scene,environments,seed,nullptr) {}
RetinalObserverContext::RetinalObserverContext(const CompiledRun& run,const WorldFamily& family,
    const VisualSceneManifestV3& scene,unsigned environments,std::uint64_t seed,
    const FiniteResourceContext* cueResources) {
    const auto* p=run.visualSensorProgram();
    require(run.valid() && p && p->retinal && environments>0 && run.task().header().articulation.z==48,
        "Retinal observer requires a compiled native retinal task");
    const auto& r=*p->retinal;const auto& sensors=family.worldTemplate.sensors;
    FiniteResourceDeviceView cueView;
    if (!r.resourceCues.empty()) {
        require(cueResources,"Rendered resource cues require their actual native resource owner");
        cueView=cueResources->deviceView();const auto& c=cueView.config;
        require(c.worldFingerprint==run.world().fingerprint() && c.environmentCount==environments &&
            c.bodyCount==run.model().bodies.size() && c.nq==run.model().defaultQ.size() && c.publishTaskState &&
            c.allocation.version==1 && c.allocation.concealProximity==1 &&
            cueView.programFingerprint==run.task().header().deviceStateProgramFingerprint,
            "Rendered cue resources differ from the compiled native world/task or expose proximity labels");
        for (const auto& cue:r.resourceCues) require(cue.binding.y<c.patches.size() &&
            cue.binding.z<=c.allocation.availableAfterTicks,"Rendered cue target or lifetime window is invalid");
    } else require(!cueResources,"Native resource owner supplied without a bound visual cue program");
    require(family.program.variations.empty() && family.program.categoricalValues.empty() &&
        family.worldTemplate.artifacts.empty(),"Retinal development observer requires a fixed physical fixture");
    // Recompile authoring against the actual owning physics model. Never
    // trust a caller-mutated template, cached scene fingerprint or binding.
    EpisodeTwin episode;episode.id=family.worldTemplate.id;episode.assets=family.worldTemplate.assets;
    episode.sensors=sensors;episode.task=family.worldTemplate.task;
    WorldTemplate canonical;
    require(compileEpisodeTwin(episode,run.model(),canonical).succeeded() &&
        canonical.fingerprint==family.worldTemplate.fingerprint,"Retinal template does not reproduce the actual physics model");
    WorldProgram fixed;fixed.id=family.program.id;fixed.instanceFlags=family.program.instanceFlags;
    WorldFamily canonicalFamily;
    require(compileWorldFamily(canonical,fixed,canonicalFamily).succeeded() && canonicalFamily.fingerprint==family.fingerprint,
        "Retinal fixed world program differs from its compiled identity");
    std::vector<bool> covered(run.model().bodies.size(),false);
    for (const auto& asset:canonical.assets) {
        require(asset.uniformScale==1 && asset.massScale==1 && asset.frictionScale==1 && asset.restitutionScale==1 &&
            asset.dampingScale==1 && asset.controllerGainScale==1 && asset.controllerDampingScale==1 &&
            asset.controllerLatencySeconds==0 && asset.payloadScale==1 && asset.renderAlternative==0 && asset.collisionAlternative==0,
            "Retinal fixture transforms or physical variations are unsupported");
        for (auto body:asset.bodyIndices) {
            require(body<covered.size() && !covered[body],"Retinal asset physical ownership is missing or duplicated");covered[body]=true;
            const auto scenes=run.world().sceneBodyIndices();const auto it=std::find(scenes.begin(),scenes.end(),body);
            if (it!=scenes.end()) {
                const auto& state=run.defaultSceneBodies()[it-scenes.begin()];
                require((asset.initialPose.position.x==state.position.x && asset.initialPose.position.y==state.position.y && asset.initialPose.position.z==state.position.z) && equal(asset.initialPose.orientation,state.orientation),
                    "Retinal static asset pose differs from physical default state");
            }
        }
    }
    require(std::all_of(covered.begin(),covered.end(),[](bool v){return v;}),"Retinal fixture does not cover every physical body");
    require(p->fingerprint==visualSensorProgramFingerprint(*p) && p->rendererProfile=="sensor_fast" && p->environmentPath.empty(),
        "Retinal observer supports exact sensor_fast neutral-studio content only");
    require(sensors.size()==2 && family.worldTemplate.engineModel.bodies.size()==run.model().bodies.size(),"Retinal camera/world layout mismatch");
    const auto at=std::find(run.model().bodyNames.begin(),run.model().bodyNames.end(),p->cameraParentBody);
    require(at!=run.model().bodyNames.end(),"Retinal camera parent is absent");
    const auto head=unsigned(at-run.model().bodyNames.begin());
    const float focal=.5f*p->height/std::tan(p->verticalFieldOfViewDegrees*.5f*3.14159265358979323846f/180.0f);
    for (unsigned eye=0;eye<2;++eye) {
        const auto& camera=sensors[eye];
        require(camera.kind==MR_WORLD_SENSOR_RGBD && camera.parentKind==MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK &&
            camera.parentBodyIndex==head && camera.width==p->width && camera.height==p->height &&
            equal(camera.localPose.position,eye ? r.rightCameraPosition : p->cameraPosition) &&
            equal(camera.localPose.orientation,eye ? r.rightCameraOrientation : p->cameraOrientation) &&
            equal(camera.intrinsics,{focal,focal,.5f*p->width,.5f*p->height}) &&
            camera.minimumDepthMeters==r.minimumDepthMeters && camera.maximumDepthMeters==r.maximumDepthMeters &&
            camera.nominalRateHz==p->nominalRateHz && equal(camera.distortion,{0,0,0,0}) && camera.focalScale==1 &&
            camera.colorNoiseSigma==0 && camera.depthNoiseSigma==0 && camera.depthDropout==0 && camera.latencySeconds==0 &&
            camera.exposureSeconds==1.0f/240 && camera.shutterReadoutSeconds==0 && camera.shutterModel==MR_VISUAL_SHUTTER_GLOBAL &&
            camera.shutterDirection==MR_VISUAL_SHUTTER_TOP_TO_BOTTOM && camera.frameJitterSeconds==0 &&
            camera.depthQuantumMeters==.0001f && camera.motionBlurScale==0,"Retinal physical camera differs from bound sensor program");
    }
    require(scene.worldFingerprint==family.worldTemplate.fingerprint && scene.renderScene.bodyCount==run.model().bodies.size() &&
        scene.renderScene.visualPacks.size()==p->assets.size(),"Retinal scene provenance mismatch");
    std::string reason;require(scene.valid(&reason),"Invalid retinal visual scene");
    const auto neutral=makeNeutralStudioEnvironmentV2();const auto lighting=makeIndoorAreaLightRigV1();
    require(scene.environmentMapHash==neutral.contentHash && scene.lightRigHash==lighting.contentHash,
        "Retinal lighting/environment differs from fixed sensor semantics");
    std::vector<VisualAssetReferenceV3> references;
    std::uint64_t meshInstances=0;
    for (unsigned i=0;i<p->assets.size();++i) {
        const auto& asset=p->assets[i];const auto& pack=scene.renderScene.visualPacks[i];
        VisualAssetPackV2 actual;
        require(readVisualAssetPack(asset.path,actual,&reason) && actual.contentHash==asset.contentHash &&
            actual.sourceContentHash=="world:"+std::to_string(run.world().fingerprint()),"Retinal content or source-world hash mismatch");
        meshInstances+=actual.instances.size();
        references.push_back({asset.path,asset.contentHash,canonical.assetIndex(asset.assetId),asset.semanticId,asset.instanceId});
        require(pack.contentHash==asset.contentHash && pack.semanticId==asset.semanticId && pack.instanceId==asset.instanceId &&
            pack.assetIndex==family.worldTemplate.assetIndex(asset.assetId),"Retinal rendered asset differs from bound sensor program");
    }
    VisualSceneManifestV3 canonicalScene;
    for (const auto& cue:r.resourceCues) require(cue.binding.x<meshInstances,"Rendered cue mesh instance is absent");
    require(compileVisualSceneManifestV3(canonical,references,neutral,lighting,canonicalScene,&reason) &&
        canonicalScene.fingerprint==scene.fingerprint,"Retinal visual scene does not reproduce bound content and fixed lighting");
    auto s=std::make_shared<detail::RetinalObserverRuntime>();
    s->environments=environments;s->bodies=run.model().bodies.size();s->features=run.task().layout().actorObservationSize;
    s->fingerprint=run.task().observationFingerprint();s->dt=run.profile().controlTimestepSeconds;
    const unsigned count=2*r.width*r.height*3+2;
    require(s->features>=count && run.task().layout().actorHistoryLength==1,"Retinal actor layout mismatch");s->offset=s->features-count;
    auto status=s->worlds.compile(canonicalFamily,environments);require(status.succeeded(),"Retinal world compilation failed");
    status=s->worlds.sample(environments,seed);require(status.succeeded(),"Retinal world sampling failed");
    MetalHybridRendererConfig c;c.width=p->width;c.height=p->height;
    c.maximumRetainedBytes=p->maximumRetainedBytes ? p->maximumRetainedBytes : 512ull*1024*1024;
    for (auto& renderer:s->renderers) {
        renderer=std::make_unique<MetalHybridRenderer>(c);auto copy=canonicalScene.renderScene;
        require(renderer->compile(std::move(copy),VisualRendererProfileV1::sensorFast(),environments).succeeded(),"Retinal renderer compilation failed");
    }
    if (!r.resourceCues.empty()) {
        s->cueResources=std::move(cueView);
        const auto instances=s->renderers[0]->layout().meshInstanceCount;
        require(instances && instances==s->renderers[1]->layout().meshInstanceCount &&
            std::uint64_t(instances)*environments<=UINT32_MAX,"Rendered cue instance layout overflows");
        id<MTLBuffer> state=(__bridge id<MTLBuffer>)s->cueResources.states;auto device=state.device;
        id<MTLLibrary> library=(__bridge id<MTLLibrary>)s->cueResources.library;
        id<MTLFunction> function=[library newFunctionWithName:@"mr_finite_resource_visual_cues_v1"];
        require(function!=nil,"Resource visual cue kernel v1 unavailable");NSError* error=nil;
        s->cuePipeline=[device newComputePipelineStateWithFunction:function error:&error];
        require(s->cuePipeline!=nil,"Resource visual cue pipeline creation failed");
        s->cueBindings=[device newBufferWithBytes:r.resourceCues.data() length:r.resourceCues.size()*sizeof(MRFiniteResourceCueGPU) options:MTLResourceStorageModeShared];
        s->cueVisibility=[device newBufferWithLength:std::size_t(environments)*instances*sizeof(std::uint32_t) options:MTLResourceStorageModePrivate];
        require(s->cueBindings!=nil && s->cueVisibility!=nil,"Resource visual cue buffer allocation failed");
        s->cueCounts={environments,instances,std::uint32_t(r.resourceCues.size()),1};
    }
    RetinalStreamConfig timing;timing.environments=environments;timing.sourceWidth=p->width;timing.sourceHeight=p->height;
    timing.width=r.width;timing.height=r.height;timing.sampleNumerator=r.sampleNumerator;timing.sampleDenominator=r.sampleDenominator;
    timing.latencyTicks=r.latencyTicks;timing.tickSeconds=s->dt;s->timing=timing;s->stream=std::make_unique<RetinalStreamContext>(timing);runtime_=std::move(s);
}
RetinalObserverContext::~RetinalObserverContext()=default;
MetalWorldDeviceObservationProgram RetinalObserverContext::deviceProgram() noexcept {
    return {runtime_.get(),detail::RetinalObserverRuntime::encode,runtime_->fingerprint};
}
void RetinalObserverContext::reset(std::uint64_t seed) {
    const auto s=runtime_;const std::lock_guard lock(s->mutex);
    require(!s->activeToken,"Retinal reset requires a quiescent command boundary");
    auto fresh=std::make_unique<RetinalStreamContext>(s->timing);
    require(s->worlds.sample(s->environments,seed).succeeded(),"Retinal fixed-world reset failed");
    s->stream=std::move(fresh);s->poisoned=false;
}
RetinalStreamSnapshot RetinalObserverContext::snapshot() const {
    const auto s=runtime_;const std::lock_guard lock(s->mutex);
    require(!s->poisoned && !s->activeToken,"Retinal observer publication requires quiescent successful work");return s->stream->snapshot();
}
std::array<HybridObservationBatch,2> RetinalObserverContext::inspectImages() const {
    const auto s=runtime_;const std::lock_guard lock(s->mutex);
    require(!s->poisoned && !s->activeToken,"Retinal inspection requires quiescent successful work");
    std::array<HybridObservationBatch,2> images;
    for (unsigned eye=0;eye<2;++eye) require(s->renderers[eye]->readback(images[eye]).succeeded(),"Retinal image publication failed");
    return images;
}


std::vector<std::byte> RetinalObserverContext::checkpointState() const {
    const auto s=runtime_;const std::lock_guard lock(s->mutex);
    require(!s->poisoned && !s->activeToken,"Retinal observer checkpoint requires quiescent successful work");return s->stream->checkpointState();
}
void RetinalObserverContext::restoreState(std::span<const std::byte> state) {
    const auto s=runtime_;const std::lock_guard lock(s->mutex);
    require(!s->poisoned && !s->activeToken,"Retinal observer restore requires quiescent successful work");s->stream->restoreState(state);
}
}
