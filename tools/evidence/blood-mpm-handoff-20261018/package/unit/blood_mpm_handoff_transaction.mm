#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "numi/matter/human_physiology.hpp"
#include "numi/matter/human_respiration.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_MATTER_MATERIAL
#error NUMI_MATTER_MATERIAL is required
#endif
#ifndef NUMI_MATTER_METALLIB
#error NUMI_MATTER_METALLIB is required
#endif
#ifndef NUMI_BLOOD_MPM_HANDOFF_METAL
#error NUMI_BLOOD_MPM_HANDOFF_METAL is required
#endif

namespace {
using namespace numi::matter;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string readFile(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    require(stream.good(), std::string("cannot read source: ") + path);
    return std::string(std::istreambuf_iterator<char>(stream),
                       std::istreambuf_iterator<char>());
}

id<MTLComputePipelineState> pipeline(
    id<MTLDevice> device, id<MTLLibrary> library, const char* name) {
    NSError* error = nil;
    id<MTLFunction> function =
        [library newFunctionWithName:[NSString stringWithUTF8String:name]];
    require(function != nil, std::string("missing Metal function ") + name);
    id<MTLComputePipelineState> value =
        [device newComputePipelineStateWithFunction:function error:&error];
    require(value != nil, std::string("pipeline failed ") + name + ": " +
            (error.localizedDescription.UTF8String ?: "unknown"));
    return value;
}

id<MTLBuffer> makeBuffer(id<MTLDevice> device, const void* bytes,
                         std::size_t length, const char* label) {
    id<MTLBuffer> result = [device newBufferWithBytes:bytes
        length:length options:MTLResourceStorageModeShared];
    require(result != nil, std::string("buffer allocation failed: ") + label);
    result.label = [NSString stringWithUTF8String:label];
    return result;
}

struct Extension {
    id<MTLDevice> device = nil;
    id<MTLComputePipelineState> handoff = nil;
    id<MTLComputePipelineState> resolve = nil;
    id<MTLComputePipelineState> reject = nil;
    id<MTLComputePipelineState> prepareWorldStatus = nil;
    id<MTLBuffer> respirationAccepted = nil;
    id<MTLBuffer> respirationCandidate = nil;
    NMHumanRespirationDispatch dispatch{};
    bool rejectNext = false;
    std::uint32_t frameBeginCalls = 0;
    std::uint32_t microstepCalls = 0;
    std::uint32_t frameCompleteCalls = 0;
    std::uint32_t handoffDispatches = 0;
    std::uint32_t rejectionDispatches = 0;
};

bool encodeExtension(
    void* opaque,
    AcceptedStepExtensionPhase phase,
    const AcceptedStepExtensionView& view
) {
    auto& state = *static_cast<Extension*>(opaque);
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)view.commandBuffer;
    if (commandBuffer == nil || view.environmentCount != 1u ||
        view.particleCount == 0u || view.particleCount != state.dispatch.particleCount ||
        view.materialStateStride != state.dispatch.materialStateStride ||
        view.particleAccepted == nullptr || view.particleCandidate == nullptr ||
        view.particleCheckpoint == nullptr ||
        view.matterStatuses == nullptr ||
        (view.materialStateStride != 0u &&
            (view.particleMaterialStateDefaults == nullptr ||
             view.particleMaterialStateAccepted == nullptr ||
             view.particleMaterialStateCandidate == nullptr))) {
        return false;
    }

    if (phase == AcceptedStepExtensionPhase::frameBegin) {
        ++state.frameBeginCalls;
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        if (blit == nil) return false;
        [blit copyFromBuffer:state.respirationAccepted sourceOffset:0u
                    toBuffer:state.respirationCandidate destinationOffset:0u
                        size:sizeof(NMHumanRespirationState)];
        [blit endEncoding];

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) return false;
        [encoder setComputePipelineState:state.handoff];
        [encoder setBytes:&state.dispatch length:sizeof(state.dispatch) atIndex:0u];
        [encoder setBuffer:state.respirationCandidate offset:0u atIndex:1u];
        [encoder setBuffer:(__bridge id<MTLBuffer>)view.matterStatuses offset:0u atIndex:2u];
        [encoder setBuffer:(__bridge id<MTLBuffer>)view.particleAccepted offset:0u atIndex:3u];
        [encoder setBuffer:(__bridge id<MTLBuffer>)view.particleCandidate offset:0u atIndex:4u];
        [encoder setBuffer:(__bridge id<MTLBuffer>)view.particleCheckpoint offset:0u atIndex:5u];
        [encoder setBuffer:(__bridge id<MTLBuffer>)view.particleMaterialStateDefaults offset:0u atIndex:6u];
        [encoder setBuffer:(__bridge id<MTLBuffer>)view.particleMaterialStateAccepted offset:0u atIndex:7u];
        [encoder setBuffer:(__bridge id<MTLBuffer>)view.particleMaterialStateCandidate offset:0u atIndex:8u];
        [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
        [encoder endEncoding];
        ++state.handoffDispatches;
        return true;
    }

    if (phase == AcceptedStepExtensionPhase::microstepComplete) {
        ++state.microstepCalls;
        if (state.rejectNext) {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            if (encoder == nil) return false;
            [encoder setComputePipelineState:state.reject];
            [encoder setBuffer:(__bridge id<MTLBuffer>)view.matterStatuses
                        offset:0u atIndex:0u];
            [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
            [encoder endEncoding];
            state.rejectNext = false;
            ++state.rejectionDispatches;
        }
        return true;
    }

    if (phase == AcceptedStepExtensionPhase::frameComplete) {
        ++state.frameCompleteCalls;
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) return false;
        [encoder setComputePipelineState:state.resolve];
        [encoder setBytes:&state.dispatch length:sizeof(state.dispatch) atIndex:0u];
        [encoder setBuffer:state.respirationCandidate offset:0u atIndex:1u];
        [encoder setBuffer:state.respirationAccepted offset:0u atIndex:2u];
        [encoder setBuffer:(__bridge id<MTLBuffer>)view.matterStatuses
                    offset:0u atIndex:3u];
        [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
        [encoder endEncoding];
        return true;
    }

    return true;
}

template <typename T>
bool bytesEqual(const std::vector<T>& a, const std::vector<T>& b) {
    static_assert(std::is_trivially_copyable_v<T>);
    return a.size() == b.size() &&
        (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

CompiledWorld makeWorld(float& density, float& referenceVolume) {
    const auto parsed = parseMatterFile(NUMI_MATTER_MATERIAL);
    require(parsed.succeeded(), "blood proxy material did not parse");
    auto parameter = [&](const char* name) {
        const auto found = std::find_if(parsed.material.parameters.begin(),
            parsed.material.parameters.end(), [name](const Parameter& p) {
                return p.name == name;
            });
        require(found != parsed.material.parameters.end(),
            std::string("missing material parameter ") + name);
        return static_cast<float>(found->defaultValue);
    };
    density = parameter("density");
    require(std::isfinite(density) && density > 0.0f,
        "blood proxy density is invalid");
    referenceVolume = 1.0e-9f;

    VascularNetworkSource vascular;
    std::string networkError;
    const std::filesystem::path networkPath =
        "/Users/numi/numi-human-resting-evidence-20261005/reference-circulation-001/resting_reference_lv15.native.v3.json";
    require(readHumanPhysiologyNetwork(networkPath.string(), vascular, &networkError),
        "reference CVSim network failed to load: " + networkError);

    WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = 1.0 / 8000.0;
    source.gravity = {0.0, 0.0, 0.0};
    source.vascular = std::move(vascular);
    source.materials.push_back(parsed.material);

    ObjectSource object;
    object.name = "blood_mpm_receiver_transaction_fixture";
    object.materialIndex = 0u;
    object.representation = Representation::mpm;
    object.characteristicLength = 0.005;
    object.adaptive = false;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.mpmGridMinimum = {-0.015, -0.015, -0.015};
    object.mpmGridMaximum = { 0.015,  0.015,  0.015};
    ParticleSource reserve;
    reserve.position = {0.0, 0.0, 0.0};
    reserve.velocity = {0.0, 0.0, 0.0};
    reserve.referenceVolume = referenceVolume;
    reserve.mass = density * referenceVolume;
    reserve.active = false;
    object.particles.push_back(reserve);
    source.objects.push_back(std::move(object));

    CompileOptions options;
    options.maximumRateExponent = NM_MAX_RATE_EXPONENT;
    const auto compiled = compileWorld(source, options);
    for (const auto& diagnostic : compiled.diagnostics) {
        std::cerr << "compile_diagnostic=" << diagnostic.message << "\n";
    }
    require(compiled.succeeded(), "blood MPM receiver world did not compile");
    require(compiled.world.mpm.particles.size() == 1u &&
            compiled.world.mpm.grids.size() == 1u,
        "receiver fixture must own exactly one dormant particle and one MPM grid");
    require((compiled.world.mpm.particles[0].identity.w & NM_OBJECT_ACTIVE) == 0u,
        "receiver particle did not compile dormant");
    return compiled.world;
}

void encodeStep(Runtime& runtime, id<MTLCommandQueue> queue,
                id<MTLBuffer> worldStatus, std::uint32_t step,
                Extension& extension) {
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    require(commandBuffer != nil, "command buffer allocation failed");
    EncodeRequest request{};
    request.commandBuffer = (__bridge void*)commandBuffer;
    request.environmentStatuses = (__bridge void*)worldStatus;
    request.controlStep = step;
    request.physicsSubstep = 0u;
    request.physicsSubsteps = 1u;
    request.timestepSeconds = runtime.timestepSeconds();
    request.runAdaptiveTransfer = false;
    request.acceptedStepExtensionContext = nullptr;
    request.encodeAcceptedStepExtension = nullptr;

    // The context remains borrowed through postCommit/frameComplete in this
    // command buffer; the fixture owns each extension through completion.
    request.acceptedStepExtensionContext = &extension;
    request.encodeAcceptedStepExtension = &encodeExtension;

    id<MTLComputeCommandEncoder> prepareEncoder =
        [commandBuffer computeCommandEncoder];
    require(prepareEncoder != nil, "world-status preparation encoder unavailable");
    [prepareEncoder setComputePipelineState:extension.prepareWorldStatus];
    [prepareEncoder setBuffer:worldStatus offset:0u atIndex:0u];
    [prepareEncoder setBytes:&step length:sizeof(step) atIndex:1u];
    [prepareEncoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
    [prepareEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [prepareEncoder endEncoding];

    request.phase = EncodePhase::preDynamics;
    const auto pre = runtime.encode(request);
    require(pre.encoded, "preDynamics encode failed: " + pre.message);
    request.phase = EncodePhase::postCommit;
    const auto post = runtime.encode(request);
    require(post.encoded, "postCommit encode failed: " + post.message);
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    require(commandBuffer.status == MTLCommandBufferStatusCompleted,
        "transaction command buffer failed");
}


} // namespace

int main() {
    @autoreleasepool {
        try {
            float density = 0.0f;
            float volume = 0.0f;
            const auto world = makeWorld(density, volume);
            std::cout << "{\"event\":\"compiled\",\"particles\":"
                      << world.dispatch.particleCount
                      << ",\"rate_exponent\":" << world.dispatch.maximumRateExponent
                      << ",\"substeps\":"
                      << (1u << world.dispatch.maximumRateExponent)
                      << ",\"density_kg_m3\":" << density
                      << ",\"capacity_volume_m3\":" << volume << "}\n";

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "Metal device unavailable");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue != nil, "Metal queue unavailable");

            std::string shader = readFile(NUMI_BLOOD_MPM_HANDOFF_METAL);
            shader += R"(
kernel void nm_test_reject_after_mpm(
    device NMMatterStatusGPU* status [[buffer(0)]],
    uint i [[thread_position_in_grid]]) {
    if (i == 0u) {
        status[i].code = NM_STATUS_MULTIPHYSICS_FAILURE;
        status[i].environment = i;
        status[i].objectIndex = 0u;
        status[i].failingIndex = 0u;
        status[i].diagnostics = float4(-88.0f, 0.0f, 0.0f, 0.0f);
    }
}
kernel void nm_test_prepare_world_status(
    device MRMetalWorldStatusGPU* worldStatus [[buffer(0)]],
    constant uint& controlStep [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i == 0u) {
        MRMetalWorldStatusGPU s{};
        s.code = MR_STEP_SUCCESS;
        s.environment = 0u;
        s.controlStep = controlStep;
        worldStatus[0] = s;
    }
}
)";
            MTLCompileOptions* compileOptions = [MTLCompileOptions new];
            compileOptions.languageVersion = MTLLanguageVersion4_0;
            compileOptions.mathMode = MTLMathModeSafe;
            compileOptions.mathFloatingPointFunctions =
                MTLMathFloatingPointFunctionsPrecise;
            NSError* compileError = nil;
            id<MTLLibrary> humanLibrary = [device newLibraryWithSource:
                [NSString stringWithUTF8String:shader.c_str()]
                options:compileOptions error:&compileError];
            require(humanLibrary != nil, std::string("Human MSL compile failed: ") +
                (compileError.localizedDescription.UTF8String ?: "unknown"));

            Extension extension;
            extension.device = device;
            extension.handoff = pipeline(device, humanLibrary, "nm_human_blood_mpm_handoff");
            extension.resolve = pipeline(device, humanLibrary, "nm_human_respiration_resolve");
            extension.reject = pipeline(device, humanLibrary, "nm_test_reject_after_mpm");
            extension.prepareWorldStatus = pipeline(device, humanLibrary,
                "nm_test_prepare_world_status");

            constexpr float oxygen = 1.0e-10f;
            constexpr float carbonDioxide = 2.0e-10f;
            const float mass = density * volume;
            NMHumanRespirationState seed{};
            seed.externalBlood.total = {volume, mass, oxygen, carbonDioxide};
            seed.externalBlood.pendingParcel = seed.externalBlood.total;
            seed.externalBlood.pendingPositionMass = {0.0f, 0.0f, 0.0f, mass};
            extension.respirationAccepted = makeBuffer(device, &seed,
                sizeof(seed), "blood MPM external inventory accepted");
            extension.respirationCandidate = makeBuffer(device, &seed,
                sizeof(seed), "blood MPM external inventory candidate");

            const auto& sourceParticle = world.mpm.particles.front();
            const auto& grid = world.mpm.grids.front();
            extension.dispatch.environmentCount = 1u;
            extension.dispatch.bloodMpmFirstParticle = 0u;
            extension.dispatch.bloodMpmParticleCount = 1u;
            extension.dispatch.bloodMpmObjectIndex = sourceParticle.identity.x;
            extension.dispatch.bloodMpmMaterialIndex = sourceParticle.identity.y;
            extension.dispatch.bloodMpmTopologyGeneration = sourceParticle.identity.z;
            extension.dispatch.bloodMpmEnabled = 1u;
            extension.dispatch.bloodMpmMaterialStateCount = world.dispatch.materialStateStride;
            extension.dispatch.bloodMpmDensity = density;
            extension.dispatch.bloodMpmGridCellWidth = grid.metrics.x;
            extension.dispatch.bloodMpmGridNodeMinimum = {
                grid.nodeMinimumAndObject.x, grid.nodeMinimumAndObject.y,
                grid.nodeMinimumAndObject.z, 0
            };
            extension.dispatch.bloodMpmGridNodeDimensions = {
                grid.nodeDimensionsAndOffset.x, grid.nodeDimensionsAndOffset.y,
                grid.nodeDimensionsAndOffset.z, 0u
            };
            extension.dispatch.particleCount = world.dispatch.particleCount;
            extension.dispatch.materialStateStride = world.dispatch.materialStateStride;
            Extension controlExtension;
            controlExtension.device = device;
            controlExtension.handoff = extension.handoff;
            controlExtension.resolve = extension.resolve;
            controlExtension.reject = extension.reject;
            controlExtension.prepareWorldStatus = extension.prepareWorldStatus;
            controlExtension.dispatch = extension.dispatch;
            controlExtension.respirationAccepted = makeBuffer(device, &seed,
                sizeof(seed), "control respiration inventory accepted");
            controlExtension.respirationCandidate = makeBuffer(device, &seed,
                sizeof(seed), "control respiration inventory candidate");

            id<MTLBuffer> worldStatus = [device
                newBufferWithLength:sizeof(MRMetalWorldStatusGPU)
                options:MTLResourceStorageModeShared];
            require(worldStatus != nil, "world status allocation failed");
            auto* ws = static_cast<MRMetalWorldStatusGPU*>(worldStatus.contents);
            *ws = {};
            ws->code = MR_STEP_SUCCESS;
            ws->environment = 0u;
            id<MTLBuffer> controlWorldStatus = [device
                newBufferWithLength:sizeof(MRMetalWorldStatusGPU)
                options:MTLResourceStorageModeShared];
            require(controlWorldStatus != nil, "control world status allocation failed");
            auto* controlWS = static_cast<MRMetalWorldStatusGPU*>(controlWorldStatus.contents);
            *controlWS = {};
            controlWS->code = MR_STEP_SUCCESS;
            controlWS->environment = 0u;

            Runtime runtime;
            const RuntimeConfiguration configuration{
                .metallib = NUMI_MATTER_METALLIB,
                .environmentCount = 1u,
                .captureEvents = false,
                .captureDiagnostics = false,
                .automaticIdentification = false,
                .adaptiveTransfer = false,
            };
            const auto initialized = runtime.initialize(world, configuration);
            require(initialized.encoded && runtime.valid(),
                "Runtime initialization failed: " + initialized.message);
            const auto baseline = runtime.snapshot();
            require(baseline.available, "baseline snapshot unavailable: " + baseline.message);
            require(baseline.particles.size() == 1u &&
                (baseline.particles.front().identity.w & NM_OBJECT_ACTIVE) == 0u,
                "baseline receiver slot is not dormant");

            Runtime controlRuntime;
            const auto controlInitialized = controlRuntime.initialize(world, configuration);
            require(controlInitialized.encoded && controlRuntime.valid(),
                "control Runtime initialization failed: " + controlInitialized.message);
            const auto controlBaseline = controlRuntime.snapshot();
            require(controlBaseline.available &&
                bytesEqual(controlBaseline.particles, baseline.particles) &&
                bytesEqual(controlBaseline.particleMaterialState,
                           baseline.particleMaterialState) &&
                bytesEqual(controlBaseline.vascularState, baseline.vascularState) &&
                bytesEqual(controlBaseline.vascularClock, baseline.vascularClock) &&
                bytesEqual(controlBaseline.schedulers, baseline.schedulers) &&
                controlBaseline.controlStep == baseline.controlStep &&
                controlBaseline.physicsSubstep == baseline.physicsSubstep,
                "control and rollback runtimes did not start from identical accepted state");

            // Uninterrupted accepted control, using the same source world and
            // the same production handoff/resolve kernels, but no injected reject.
            encodeStep(controlRuntime, queue, controlWorldStatus, 0u, controlExtension);
            const auto controlAccepted = controlRuntime.snapshot();
            require(controlAccepted.available &&
                controlAccepted.statuses.size() == 1u &&
                controlAccepted.statuses.front().code == NM_STATUS_SUCCESS,
                "uninterrupted control transaction did not accept");

            extension.rejectNext = true;
            encodeStep(runtime, queue, worldStatus, 0u, extension);
            const auto rejected = runtime.snapshot();
            require(rejected.available, "rejected snapshot unavailable: " + rejected.message);
            require(rejected.statuses.size() == 1u &&
                rejected.statuses.front().code == NM_STATUS_MULTIPHYSICS_FAILURE,
                "GPU status injection did not reject the post-MPM transaction");
            require(bytesEqual(rejected.particles, baseline.particles) &&
                bytesEqual(rejected.particleMaterialState, baseline.particleMaterialState),
                "rejected MPM handoff changed accepted particle/material bytes");
            const auto* acceptedState =
                static_cast<const NMHumanRespirationState*>(
                    extension.respirationAccepted.contents);
            require(acceptedState != nullptr &&
                std::memcmp(acceptedState, &seed, sizeof(seed)) == 0,
                "rejected handoff changed accepted external inventory bytes");
            require(extension.handoffDispatches == 1u &&
                extension.rejectionDispatches == 1u,
                "first attempt did not execute one activation and one GPU rejection");

            extension.rejectNext = false;
            encodeStep(runtime, queue, worldStatus, 0u, extension);
            const auto accepted = runtime.snapshot();
            require(accepted.available, "accepted snapshot unavailable: " + accepted.message);
            if (accepted.statuses.size() != 1u ||
                accepted.statuses.front().code != NM_STATUS_SUCCESS) {
                std::string detail = "same-context retry failed to accept; statuses=" +
                    std::to_string(accepted.statuses.size());
                if (!accepted.statuses.empty()) {
                    const auto& status = accepted.statuses.front();
                    detail += " code=" + std::to_string(status.code) +
                        " object=" + std::to_string(status.objectIndex) +
                        " failing=" + std::to_string(status.failingIndex) +
                        " diag0=" + std::to_string(status.diagnostics.x) +
                        " diag1=" + std::to_string(status.diagnostics.y) +
                        " diag2=" + std::to_string(status.diagnostics.z) +
                        " diag3=" + std::to_string(status.diagnostics.w);
                }
                const auto* failedExternal =
                    static_cast<const NMHumanRespirationState*>(
                        extension.respirationAccepted.contents);
                if (failedExternal != nullptr) {
                    detail += " externalStatus=" +
                        std::to_string(failedExternal->status.w) +
                        " externalMass=" +
                        std::to_string(failedExternal->externalBlood.total.y) +
                        " pendingMass=" +
                        std::to_string(failedExternal->externalBlood.pendingParcel.y);
                }
                throw std::runtime_error(detail);
            }
            std::uint32_t activeCount = 0u;
            double activeMass = 0.0;
            for (const auto& particle : accepted.particles) {
                if ((particle.identity.w & NM_OBJECT_ACTIVE) != 0u) {
                    ++activeCount;
                    activeMass += particle.positionAndMass.w;
                }
            }
            require(activeCount == 1u &&
                std::abs(activeMass - mass) <= std::max(1.0e-12, mass * 4.0e-6),
                "retry did not activate exactly the transferred MPM mass");
            acceptedState = static_cast<const NMHumanRespirationState*>(
                extension.respirationAccepted.contents);
            const auto& inventory = acceptedState->externalBlood;
            require(inventory.pendingParcel.x == 0.0f &&
                inventory.pendingParcel.y == 0.0f &&
                inventory.total.x == volume && inventory.total.y == mass &&
                inventory.total.z == oxygen && inventory.total.w == carbonDioxide &&
                inventory.mpmRepresented.x == volume &&
                inventory.mpmRepresented.y == mass &&
                inventory.mpmRepresented.z == oxygen &&
                inventory.mpmRepresented.w == carbonDioxide,
                "accepted retry did not conserve the seeded external/gas inventory exactly");
            const NMHumanRespirationState externalAfterAccept = *acceptedState;
            const auto* controlAcceptedExternal =
                static_cast<const NMHumanRespirationState*>(
                    controlExtension.respirationAccepted.contents);
            require(controlAcceptedExternal != nullptr &&
                std::memcmp(controlAcceptedExternal, &externalAfterAccept,
                            sizeof(externalAfterAccept)) == 0,
                "same-context retry inventory differs from uninterrupted accepted control");
            require(bytesEqual(accepted.particles, controlAccepted.particles) &&
                bytesEqual(accepted.particleMaterialState,
                           controlAccepted.particleMaterialState) &&
                bytesEqual(accepted.vascularState, controlAccepted.vascularState) &&
                bytesEqual(accepted.vascularClock, controlAccepted.vascularClock) &&
                bytesEqual(accepted.schedulers, controlAccepted.schedulers) &&
                accepted.controlStep == controlAccepted.controlStep &&
                accepted.physicsSubstep == controlAccepted.physicsSubstep,
                "rejected run plus retry differs from uninterrupted accepted state");

            const RuntimeStateSnapshot beforeRestore = accepted;
            const auto restored = runtime.restore(beforeRestore);
            require(restored.encoded, "snapshot restore failed: " + restored.message);
            const auto afterRestore = runtime.snapshot();
            require(afterRestore.available &&
                bytesEqual(afterRestore.particles, beforeRestore.particles) &&
                bytesEqual(afterRestore.particleMaterialState,
                           beforeRestore.particleMaterialState) &&
                bytesEqual(afterRestore.vascularState,
                           beforeRestore.vascularState) &&
                bytesEqual(afterRestore.vascularClock,
                           beforeRestore.vascularClock) &&
                bytesEqual(afterRestore.schedulers, beforeRestore.schedulers) &&
                afterRestore.controlStep == beforeRestore.controlStep &&
                afterRestore.physicsSubstep == beforeRestore.physicsSubstep,
                "accepted particle/material/vascular/scheduler/cursor bytes changed across snapshot restore");
            acceptedState = static_cast<const NMHumanRespirationState*>(
                extension.respirationAccepted.contents);
            require(std::memcmp(acceptedState, &externalAfterAccept,
                                sizeof(externalAfterAccept)) == 0,
                "external ledger changed during Runtime snapshot restore");

            std::cout << "{\"schema\":\"numi.human.blood_mpm_handoff_transaction.v1\""
                      << ",\"scope\":\"source-seeded external inventory handoff unit; not circulation-stock integration or free-surface blood\""
                      << ",\"active_receiver_slots\":1,\"accepted_mass_kg\":"
                      << activeMass << ",\"expected_mass_kg\":" << mass
                      << ",\"accepted_volume_m3\":" << volume
                      << ",\"external_mass_kg\":" << inventory.total.y
                      << ",\"mpm_represented_mass_kg\":" << inventory.mpmRepresented.y
                      << ",\"accepted_o2_stpd_m3\":" << inventory.total.z
                      << ",\"represented_o2_stpd_m3\":" << inventory.mpmRepresented.z
                      << ",\"accepted_co2_stpd_m3\":" << inventory.total.w
                      << ",\"represented_co2_stpd_m3\":" << inventory.mpmRepresented.w
                      << ",\"attempted_transactions\":2"
                      << ",\"accepted_transactions\":1"
                      << ",\"rejected_transactions\":1"
                      << ",\"control_accepted_transactions\":1"
                      << ",\"world_status_reset_gpu_per_attempt\":true"
                      << ",\"retry_matches_uninterrupted_particle_material_vascular_scheduler_cursor\":true"
                      << ",\"frame_begin_callbacks\":" << extension.frameBeginCalls
                      << ",\"microstep_callback_invocations\":" << extension.microstepCalls
                      << ",\"frame_complete_callbacks\":" << extension.frameCompleteCalls
                      << ",\"handoff_dispatches\":" << extension.handoffDispatches
                      << ",\"gpu_rejection_dispatches\":" << extension.rejectionDispatches
                      << ",\"rejected_physical_snapshot_restored\":true"
                      << ",\"same_context_retry_accepted\":true"
                      << ",\"snapshot_restore_exact\":true}\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "blood_mpm_handoff_failure=" << error.what() << "\n";
            return 1;
        }
    }
}

