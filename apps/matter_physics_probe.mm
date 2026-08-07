#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "numi/matter/metal_world.hpp"
#include "metalrobo/EngineModel.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/engine_types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#ifndef NUMI_MATTER_MATERIAL
#define NUMI_MATTER_MATERIAL ""
#endif

#ifndef NUMI_MATTER_METALLIB
#define NUMI_MATTER_METALLIB ""
#endif

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

numi::matter::CompiledWorld compileCase(
    const numi::matter::Representation representation,
    const bool includePlane,
    const bool singleMPMParticle = false,
    const bool fullRate = false,
    const bool nearPlane = false,
    const bool gentleMPMContact = false,
    const bool bodyBackedPlane = false,
    const double frameTimestep = 1.0 / 240.0,
    const std::uint32_t environmentCount = 1u,
    const std::uint32_t identificationCandidates = 0u
) {
    const auto parsed = numi::matter::parseMatterFile(NUMI_MATTER_MATERIAL);
    require(parsed.succeeded(), "reference silicone material did not parse");

    numi::matter::WorldSource source;
    source.environmentCount = environmentCount;
    source.frameTimestep = frameTimestep;
    source.identificationCandidates = identificationCandidates;
    source.gravity = {0.0, 0.0, -9.81};
    source.femPCGIterations = 8u;
    source.materials.push_back(parsed.material);

    if (includePlane) {
        numi::matter::RigidProxySource plane;
        plane.shape = NM_RIGID_PLANE;
        plane.localCenter = {0.0, 0.0, 1.0};
        plane.radiusOrOffset = 0.0;
        if (bodyBackedPlane) {
            // `makeFreeSphereEngineModel` owns its free articulated body at
            // global body index 1.  A plane attached there gives Matter a
            // real ABA wrench destination rather than a standalone proxy.
            plane.bodyIndex = 1u;
            plane.articulated = true;
        }
        source.rigidProxies.push_back(plane);
    }

    numi::matter::ObjectSource object;
    object.name = representation == numi::matter::Representation::mpm
        ? "mpm_drop"
        : "fem_drop";
    object.materialIndex = 0u;
    object.representation = representation;
    object.characteristicLength = 0.01;
    // This is deliberately the smallest fixed background domain that covers
    // the probe trajectory plus the compiler's quadratic-kernel halo.  The
    // production package still cooks its authored domain; a qualification
    // probe should exercise coupled particles without turning an O(P×G)
    // reference scatter into a minute-long benchmark.
    object.mpmGridMinimum = {-0.015, -0.015, -0.01};
    object.mpmGridMaximum = {0.015, 0.015, 0.06};
    if (representation == numi::matter::Representation::mpm) {
        constexpr double spacing = 0.005;
        constexpr double volume = spacing * spacing * spacing;
        constexpr int kParticlesPerAxis = 2;
        for (int z = 0; z < (singleMPMParticle ? 1 : kParticlesPerAxis); ++z) {
            for (int y = 0; y < (singleMPMParticle ? 1 : kParticlesPerAxis); ++y) {
                for (int x = 0; x < (singleMPMParticle ? 1 : kParticlesPerAxis); ++x) {
                    numi::matter::ParticleSource particle;
                    particle.position = {
                        -0.005 + spacing * x,
                        -0.005 + spacing * y,
                        // The MetalWorld bridge probe begins inside the
                        // quadratic-grid contact halo, so its first accepted
                        // microstep necessarily exercises the body-wrench
                        // handoff. Standalone drops retain their free-flight
                        // approach trajectories below.
                        (bodyBackedPlane
                             ? 0.001
                             : (gentleMPMContact ? 0.015 : 0.03)) +
                            spacing * z,
                    };
                    particle.velocity = {0.0, 0.0, gentleMPMContact ? 0.0 : -1.0};
                    particle.mass = 1100.0 * volume;
                    particle.referenceVolume = volume;
                    object.particles.push_back(particle);
                }
            }
        }
    } else {
        const double baseHeight = nearPlane ? 0.002 : 0.02;
        if (fullRate && !nearPlane) {
            object.femInitialVelocity = {0.0, 0.0, -1.0};
        }
        object.femNodes = {
            {-0.01, -0.01, baseHeight},
            { 0.01, -0.01, baseHeight},
            {-0.01,  0.01, baseHeight},
            {-0.01, -0.01, baseHeight + 0.02},
        };
        object.tetrahedra.push_back({{0u, 1u, 2u, 3u}});
    }
    source.objects.push_back(std::move(object));

    numi::matter::CompileOptions options;
    // FEM is qualified first at one implicit step per frame. MPM must retain
    // its material-selected CFL subdivision; forcing it to the FEM baseline
    // would knowingly test an unstable explicit material update.
    options.maximumRateExponent =
        representation == numi::matter::Representation::mpm || fullRate
        ? NM_MAX_RATE_EXPONENT
        : 0u;
    auto compiled = numi::matter::compileWorld(source, options);
    require(compiled.succeeded(), "Matter world compilation failed");
    require(
        !includePlane || compiled.world.dispatch.contactPairCount > 0u,
        "drop case has no continuum-to-plane contact pairs"
    );
    return std::move(compiled.world);
}

struct Outcome {
    std::uint32_t contactSamples = 0u;
    std::uint32_t completedMicrosteps = 0u;
    std::uint32_t pcgIterations = 0u;
    bool sawContactOnset = false;
    bool sawContactEvent = false;
    float minimumDeterminant = std::numeric_limits<float>::infinity();
    float minimumHeight = std::numeric_limits<float>::infinity();
    float maximumHeight = -std::numeric_limits<float>::infinity();
    float minimumVerticalVelocity = std::numeric_limits<float>::infinity();
    float maximumVerticalVelocity = -std::numeric_limits<float>::infinity();
};

MRBodyStateGPU staticSceneBody() {
    MRBodyStateGPU state{};
    state.position.w = 1.0f;
    state.orientation.w = 1.0f;
    state.flagsAndIndices[0] = MR_MOTION_STATIC;
    state.flagsAndIndices[1] = MR_INVALID_INDEX;
    state.flagsAndIndices[2] = 0u;
    return state;
}

void runIdentification() {
    @autoreleasepool {
        const auto world = compileCase(
            numi::matter::Representation::mpm,
            false,
            false,
            false,
            false,
            false,
            false,
            1.0 / 480.0,
            2u,
            2u
        );
        require(world.dispatch.identificationCandidateCount == 2u &&
                    !world.identification.empty(),
            "identification probe did not compile paired candidate state");

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "no Metal device is available");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        require(queue != nil, "failed to create Matter identification queue");
        id<MTLBuffer> statuses = [device
            newBufferWithLength:2u * sizeof(MRMetalWorldStatusGPU)
            options:MTLResourceStorageModeShared];
        require(statuses != nil, "failed to allocate identification world statuses");
        auto* worldStatuses = static_cast<MRMetalWorldStatusGPU*>(statuses.contents);
        require(worldStatuses != nullptr, "identification world statuses are unavailable");
        for (std::uint32_t environment = 0u; environment < 2u; ++environment) {
            worldStatuses[environment] = {};
            worldStatuses[environment].code = MR_STEP_SUCCESS;
            worldStatuses[environment].environment = environment;
        }

        numi::matter::Runtime runtime;
        const auto initialized = runtime.initialize(
            world,
            {
                .metallib = NUMI_MATTER_METALLIB,
                .environmentCount = 2u,
                .captureEvents = true,
                .captureDiagnostics = true,
                .automaticIdentification = true,
                .adaptiveTransfer = false,
            }
        );
        require(initialized.encoded && runtime.valid(),
            "identification runtime could not initialize: " + initialized.message);
        const auto matterStatuses = (__bridge id<MTLBuffer>)runtime.statusBuffer();
        const auto* matterStatusData = static_cast<const NMMatterStatusGPU*>(
            matterStatuses.contents
        );
        require(matterStatusData != nullptr,
            "identification Matter statuses are unavailable");

        const auto runStep = [&](const std::uint32_t controlStep) {
            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
            require(commandBuffer != nil, "failed to allocate identification command buffer");
            numi::matter::EncodeRequest request{};
            request.commandBuffer = (__bridge void*)commandBuffer;
            request.environmentStatuses = (__bridge void*)statuses;
            request.phase = numi::matter::EncodePhase::preDynamics;
            request.controlStep = controlStep;
            request.physicsSubstep = 0u;
            request.physicsSubsteps = 1u;
            request.timestepSeconds = runtime.timestepSeconds();
            request.runIdentification = true;
            request.runAdaptiveTransfer = false;
            auto encoded = runtime.encode(request);
            require(encoded.encoded,
                "identification pre-dynamics encoding failed: " + encoded.message);
            request.phase = numi::matter::EncodePhase::postCommit;
            request.runIdentification = false;
            encoded = runtime.encode(request);
            require(encoded.encoded,
                "identification post-commit encoding failed: " + encoded.message);
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            require(commandBuffer.status == MTLCommandBufferStatusCompleted,
                "identification command buffer did not complete");
            for (std::uint32_t environment = 0u; environment < 2u; ++environment) {
                require(matterStatusData[environment].code == NM_STATUS_SUCCESS,
                    "identification continuum step failed in environment " +
                        std::to_string(environment));
            }
        };

        runStep(0u);
        const auto sampled = runtime.snapshot();
        require(sampled.available && !sampled.identification.empty() &&
                    sampled.environmentParameters.size() ==
                        2u * world.dispatch.parameterCount,
            "identification sampling did not publish diagnostic state");
        const float priorMean = sampled.identification[0].momentsAndBounds.x;
        bool antitheticOverlay = false;
        for (std::uint32_t parameter = 0u;
             parameter < world.dispatch.parameterCount;
             ++parameter) {
            antitheticOverlay = antitheticOverlay ||
                std::abs(sampled.environmentParameters[parameter] -
                         sampled.environmentParameters[
                             world.dispatch.parameterCount + parameter]) >
                    1.0e-6f;
        }
        require(antitheticOverlay,
            "identification candidates did not produce paired environment overlays");

        const auto losses = (__bridge id<MTLBuffer>)runtime.identificationLossBuffer();
        require(losses != nil && losses.contents != nullptr,
            "identification loss boundary is unavailable");
        auto* lossData = static_cast<float*>(losses.contents);
        lossData[0] = 0.0f;
        lossData[1] = 100.0f;
        runStep(1u);
        const auto updated = runtime.snapshot();
        require(updated.available && updated.identification.size() ==
                    sampled.identification.size(),
            "identification update did not publish posterior state");
        require(std::abs(updated.identification[0].momentsAndBounds.x -
                         priorMean) > 1.0e-5f,
            "identification posterior did not respond to asymmetric losses");
        std::cout
            << "{\"schema\":\"numi.matter.physics-probe.v1\""
            << ",\"representation\":\"inverse_identification\""
            << ",\"prior_mean\":" << priorMean
            << ",\"posterior_mean\":"
            << updated.identification[0].momentsAndBounds.x
            << "}\n";
    }
}

numi::matter::CompiledWorld compileAdaptiveCase() {
    const auto parsed = numi::matter::parseMatterFile(NUMI_MATTER_MATERIAL);
    require(parsed.succeeded(), "reference silicone material did not parse");
    numi::matter::WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = 1.0 / 480.0;
    source.gravity = {0.0, 0.0, 0.0};
    source.femPCGIterations = 8u;
    source.materials.push_back(parsed.material);

    numi::matter::RigidProxySource fallback;
    fallback.shape = NM_RIGID_SPHERE;
    fallback.bodyIndex = 0u;
    fallback.sceneBodyIndex = 0u;
    fallback.radiusOrOffset = 0.01;
    fallback.dynamic = true;
    source.rigidProxies.push_back(fallback);

    numi::matter::ObjectSource object;
    object.name = "adaptive_mpm";
    object.materialIndex = 0u;
    object.representation = numi::matter::Representation::mpm;
    object.adaptive = true;
    object.rigidBinding = 0u;
    object.characteristicLength = 0.01;
    object.mpmGridMinimum = {-0.02, -0.02, -0.02};
    object.mpmGridMaximum = {0.02, 0.02, 0.02};
    constexpr double spacing = 0.005;
    constexpr double volume = spacing * spacing * spacing;
    for (int z = 0; z < 2; ++z) {
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 2; ++x) {
                object.particles.push_back({
                    .position = {
                        -0.005 + spacing * x,
                        -0.005 + spacing * y,
                        -0.005 + spacing * z,
                    },
                    .velocity = {0.0, 0.0, 0.0},
                    .mass = 1100.0 * volume,
                    .referenceVolume = volume,
                });
            }
        }
    }
    source.objects.push_back(std::move(object));
    numi::matter::CompileOptions options;
    options.maximumRateExponent = 0u;
    const auto compiled = numi::matter::compileWorld(source, options);
    require(compiled.succeeded(), "adaptive Matter world did not compile");
    return compiled.world;
}

MRBodyStateGPU adaptiveBodyState() {
    MRBodyStateGPU state{};
    state.position = {10.0f, 0.0f, 0.0f, 1.0f};
    state.orientation.w = 1.0f;
    state.linearVelocityAndInverseMass.w = 1.0f;
    state.inverseInertiaWorldRow0.x = 1.0f;
    state.inverseInertiaWorldRow1.y = 1.0f;
    state.inverseInertiaWorldRow2.z = 1.0f;
    state.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
    state.flagsAndIndices[1] = MR_INVALID_INDEX;
    state.flagsAndIndices[2] = 0u;
    return state;
}

void runAdaptiveDemotion() {
    @autoreleasepool {
        const auto world = compileAdaptiveCase();
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "no Metal device is available");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        require(queue != nil, "failed to create adaptive Matter queue");
        id<MTLBuffer> statuses = [device
            newBufferWithLength:sizeof(MRMetalWorldStatusGPU)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> currentBodies = [device
            newBufferWithLength:sizeof(MRBodyStateGPU)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> sceneBodies = [device
            newBufferWithLength:sizeof(MRBodyStateGPU)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> bodyWrenches = [device
            newBufferWithLength:sizeof(MRABABodyWrenchGPU)
            options:MTLResourceStorageModeShared];
        require(statuses != nil && currentBodies != nil && sceneBodies != nil &&
                    bodyWrenches != nil,
            "failed to allocate adaptive bridge arenas");
        auto* worldStatus = static_cast<MRMetalWorldStatusGPU*>(statuses.contents);
        auto* current = static_cast<MRBodyStateGPU*>(currentBodies.contents);
        auto* scene = static_cast<MRBodyStateGPU*>(sceneBodies.contents);
        require(worldStatus != nullptr && current != nullptr && scene != nullptr,
            "adaptive bridge arenas are unavailable");
        *worldStatus = {};
        worldStatus->code = MR_STEP_SUCCESS;
        *current = adaptiveBodyState();
        *scene = adaptiveBodyState();

        numi::matter::Runtime runtime;
        const auto initialized = runtime.initialize(
            world,
            {
                .metallib = NUMI_MATTER_METALLIB,
                .environmentCount = 1u,
                .captureEvents = true,
                .captureDiagnostics = true,
                .automaticIdentification = false,
                .adaptiveTransfer = true,
            }
        );
        require(initialized.encoded && runtime.valid(),
            "adaptive runtime could not initialize: " + initialized.message);
        const auto matterStatuses = (__bridge id<MTLBuffer>)runtime.statusBuffer();
        const auto* matterStatus = static_cast<const NMMatterStatusGPU*>(
            matterStatuses.contents
        );
        require(matterStatus != nullptr, "adaptive Matter status is unavailable");

        for (std::uint32_t step = 0u; step < 30u; ++step) {
            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
            require(commandBuffer != nil, "failed to allocate adaptive command buffer");
            numi::matter::EncodeRequest request{};
            request.commandBuffer = (__bridge void*)commandBuffer;
            request.rigid.currentBodies = (__bridge void*)currentBodies;
            request.rigid.bodyWrenches = (__bridge void*)bodyWrenches;
            request.rigid.sceneBodies = (__bridge void*)sceneBodies;
            request.rigid.currentBodyCount = 1u;
            request.rigid.currentBodyStride = 1u;
            request.rigid.bodyWrenchCount = 1u;
            request.rigid.sceneBodyCount = 1u;
            request.rigid.bodyWrenchStride = 1u;
            request.rigid.sceneStride = 1u;
            request.environmentStatuses = (__bridge void*)statuses;
            request.controlStep = step;
            request.physicsSubstep = 0u;
            request.physicsSubsteps = 1u;
            request.timestepSeconds = runtime.timestepSeconds();
            request.phase = numi::matter::EncodePhase::preDynamics;
            request.runAdaptiveTransfer = false;
            auto encoded = runtime.encode(request);
            require(encoded.encoded,
                "adaptive pre-dynamics encoding failed: " + encoded.message);
            request.phase = numi::matter::EncodePhase::postCommit;
            request.runAdaptiveTransfer = true;
            encoded = runtime.encode(request);
            require(encoded.encoded,
                "adaptive post-commit encoding failed: " + encoded.message);
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            require(commandBuffer.status == MTLCommandBufferStatusCompleted,
                "adaptive command buffer did not complete");
            require(matterStatus->code == NM_STATUS_SUCCESS,
                "adaptive continuum step failed at frame " + std::to_string(step));
        }

        const auto snapshot = runtime.snapshot();
        require(snapshot.available && snapshot.adaptive.size() == 1u,
            "adaptive transfer did not publish diagnostic state");
        const NMAdaptiveStateGPU& adaptive = snapshot.adaptive[0];
        require(adaptive.activeRepresentation == NM_REPRESENTATION_RIGID &&
                    adaptive.requestedRepresentation == NM_REPRESENTATION_RIGID,
            "low-strain adaptive object did not demote to rigid ownership");
        require(std::abs(scene->position.x - adaptive.centerAndRadius.x) < 1.0e-5f &&
                    (scene->flagsAndIndices[3] &
                     MR_BODY_STATE_COLLISION_DISABLED) == 0u,
            "adaptive demotion did not publish its rigid scene authority");
        std::cout
            << "{\"schema\":\"numi.matter.physics-probe.v1\""
            << ",\"representation\":\"adaptive_mpm_to_rigid\""
            << ",\"stable_frames\":" << adaptive.stableFrames
            << ",\"scene_x\":" << scene->position.x
            << "}\n";
    }
}

void runMetalWorldCoupling() {
    @autoreleasepool {
        constexpr std::uint32_t controlSteps = 1u;
        const auto matterWorld = compileCase(
            numi::matter::Representation::mpm,
            true,
            false,
            false,
            false,
            false,
            true,
            1.0 / 480.0
        );
        numi::matter::Runtime matter;
        const auto initialized = matter.initialize(
            matterWorld,
            {
                .captureEvents = true,
                .captureDiagnostics = true,
                .automaticIdentification = false,
                .adaptiveTransfer = false,
            }
        );
        require(initialized.encoded && matter.valid(),
            "MetalWorld coupling could not initialize Matter: " + initialized.message);

        const metalrobo::EngineModel model =
            metalrobo::makeFreeSphereEngineModel();
        metalrobo::CompiledWorld rigidWorld;
        const auto compiled = metalrobo::compileMetalWorld(model, 0u, rigidWorld);
        require(compiled.succeeded(),
            "MetalWorld coupling could not compile free body: " + compiled.message);
        std::vector<float> efforts(
            static_cast<std::size_t>(controlSteps) * rigidWorld.nv(),
            0.0f
        );
        const std::array<MRBodyStateGPU, 1u> scene{staticSceneBody()};
        const metalrobo::MetalWorldBatch batch{
            .environmentCount = 1u,
            .controlStepCount = controlSteps,
            .initialQ = model.defaultQ,
            .initialV = model.defaultV,
            .efforts = efforts,
            .initialSceneBodies = scene,
        };
        metalrobo::MetalWorldStepConfig config{};
        config.timestepSeconds = 1.0f / 480.0f;
        config.physicsSubsteps = 1u;
        config.solverMode = metalrobo::MetalWorldSolverMode::temporalCone;
        config.velocityIterations = 2u;
        config.finalVelocityIterations = 1u;
        config.matrixFreeArticulatedContact = false;
        config.streamedArticulatedContactResponses = false;
        config.captureContactEvidence = true;
        config.devicePhysicsProgram =
            numi::matter::makeMetalWorldDevicePhysicsProgram(matter);
        require(config.devicePhysicsProgram.valid(),
            "Matter did not produce a valid MetalWorld adapter");

        metalrobo::MetalWorldContext context;
        metalrobo::MetalWorldResult result;
        const auto ran = context.run(rigidWorld, batch, config, result);
        const id<MTLBuffer> matterStatusBuffer =
            (__bridge id<MTLBuffer>)matter.statusBuffer();
        const auto* matterStatus = static_cast<const NMMatterStatusGPU*>(
            matterStatusBuffer.contents
        );
        std::string matterFailure;
        if (matterStatus != nullptr && matterStatus->code != NM_STATUS_SUCCESS) {
            matterFailure = " matter_status=" +
                std::to_string(matterStatus->code) +
                " object=" + std::to_string(matterStatus->objectIndex) +
                " index=" + std::to_string(matterStatus->failingIndex) +
                " diagnostics=(" + std::to_string(matterStatus->diagnostics.x) +
                "," + std::to_string(matterStatus->diagnostics.y) +
                "," + std::to_string(matterStatus->diagnostics.z) +
                "," + std::to_string(matterStatus->diagnostics.w) + ")";
        }
        const std::string layoutFailure =
            " dispatch=(abi=" +
            std::to_string(ran.layout.dispatch.abiVersion) +
            ",flags=" + std::to_string(ran.layout.dispatch.flags) +
            ",steps=" + std::to_string(ran.layout.dispatch.controlStepCount) +
            ",substeps=" + std::to_string(ran.layout.dispatch.physicsSubsteps) +
            ",nq=" + std::to_string(ran.layout.dispatch.nq) +
            ",nv=" + std::to_string(ran.layout.dispatch.nv) + ")";
        const std::string operatorFailure =
            ran.layout.kinematicsDispatches.empty()
            ? " operator=(missing)"
            : " operator=(articulation=" + std::to_string(
                ran.layout.kinematicsDispatches[0].articulationIndex
            ) + ",env=" + std::to_string(
                ran.layout.kinematicsDispatches[0].environmentCount
            ) + ",flags=" + std::to_string(
                ran.layout.kinematicsDispatches[0].flags
            ) + ",point_count=" + std::to_string(
                ran.layout.kinematicsDispatches[0].pointCount
            ) + ",q_stride=" + std::to_string(
                ran.layout.kinematicsDispatches[0].qStride
            ) + ",body_stride=" + std::to_string(
                ran.layout.kinematicsDispatches[0].bodyPoseStride
            ) + ")";
        const std::string contactFailure = result.contactStatuses.empty()
            ? " contact=(missing)"
            : " contact=(code=" + std::to_string(
                result.contactStatuses[0].code
            ) + ",constraint=" + std::to_string(
                result.contactStatuses[0].firstFailingConstraint
            ) + ",stable_low=" + std::to_string(
                result.contactStatuses[0].firstFailingStableKeyLow
            ) + ",stable_high=" + std::to_string(
                result.contactStatuses[0].firstFailingStableKeyHigh
            ) + ",flags=" + std::to_string(
                result.contactStatuses[0].flags
            ) + ",diagnostics=(" + std::to_string(
                result.contactStatuses[0].diagnostics.x
            ) + "," + std::to_string(
                result.contactStatuses[0].diagnostics.y
            ) + "," + std::to_string(
                result.contactStatuses[0].diagnostics.z
            ) + "," + std::to_string(
                result.contactStatuses[0].diagnostics.w
            ) + ")";
        require(ran.succeeded(),
            "MetalWorld/Matter coupling failed: " + ran.message +
                matterFailure + layoutFailure + operatorFailure + contactFailure);
        require(
            result.environmentStatuses.size() == 1u &&
                result.environmentStatuses[0].code == MR_STEP_SUCCESS &&
                result.finalV.size() == rigidWorld.nv(),
            "MetalWorld/Matter coupling did not publish an accepted state"
        );
        const auto snapshot = matter.snapshot();
        require(snapshot.available && !snapshot.reactions.empty(),
            "MetalWorld/Matter coupling did not publish Matter reactions");
        const float reactionZ = snapshot.reactions[0].impulseAndCount.z;
        // Matter sees a downward particle impact, so the equal-and-opposite
        // impulse applied to the body-backed plane is downward as well.
        require(reactionZ < -1.0e-6f,
            "continuum impact did not accumulate a rigid reaction: reaction_z=" +
                std::to_string(reactionZ) +
                " final_body_velocity_z=" +
                std::to_string(result.finalV[2]));
        require(
            result.finalV[2] < -1.0e-6f &&
                std::abs(result.finalV[2] - reactionZ) < 1.0e-5f,
            "MetalWorld ABA did not consume the continuum rigid reaction: reaction_z=" +
                std::to_string(reactionZ) +
                " final_body_velocity_z=" +
                std::to_string(result.finalV[2])
        );
        std::cout
            << "{\"schema\":\"numi.matter.physics-probe.v1\""
            << ",\"representation\":\"mpm_articulated_coupling\""
            << ",\"rigid_reaction_z\":" << reactionZ
            << ",\"accepted_body_velocity_z\":" << result.finalV[2]
            << ",\"gpu_milliseconds\":" << ran.gpuElapsedMilliseconds
            << "}\n";
    }
}

Outcome runCase(
    const numi::matter::CompiledWorld& world,
    const char* label,
    const bool requireContact,
    const bool requireDescent,
    const std::uint32_t controlSteps,
    const bool forceRollback = false
) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "no Metal device is available");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        require(queue != nil, "failed to create Matter probe command queue");
        id<MTLBuffer> worldStatuses = [device
            newBufferWithLength:sizeof(MRMetalWorldStatusGPU)
            options:MTLResourceStorageModeShared];
        require(worldStatuses != nil, "failed to allocate world status buffer");
        auto* worldStatus = static_cast<MRMetalWorldStatusGPU*>(
            worldStatuses.contents
        );
        *worldStatus = {};
        worldStatus->code = MR_STEP_SUCCESS;
        worldStatus->environment = 0u;

        numi::matter::Runtime runtime;
        const auto initialized = runtime.initialize(
            world,
            {
                .metallib = NUMI_MATTER_METALLIB,
                .environmentCount = 1u,
                .captureEvents = true,
                .captureDiagnostics = true,
                .automaticIdentification = false,
                .adaptiveTransfer = false,
            }
        );
        require(initialized.encoded && runtime.valid(), initialized.message);

        const auto events = (__bridge id<MTLBuffer>)runtime.eventBuffer();
        const auto statuses = (__bridge id<MTLBuffer>)runtime.statusBuffer();
        require(events != nil && statuses != nil, "Matter probe diagnostics are unavailable");
        auto* eventData = static_cast<NMEventTokenGPU*>(events.contents);
        auto* statusData = static_cast<NMMatterStatusGPU*>(statuses.contents);
        require(eventData != nullptr && statusData != nullptr, "Matter diagnostics are not CPU-visible");
        const auto baseline = forceRollback
            ? runtime.snapshot()
            : numi::matter::RuntimeStateSnapshot{};
        if (forceRollback) {
            require(baseline.available, label + std::string(" baseline: ") + baseline.message);
        }

        Outcome outcome;
        auto recordSnapshot = [&](const numi::matter::RuntimeStateSnapshot& snapshot) {
            require(snapshot.available, label + std::string(" snapshot: ") + snapshot.message);
            if (!snapshot.particles.empty()) {
                for (const NMParticleStateGPU& particle : snapshot.particles) {
                    outcome.minimumHeight = std::min(
                        outcome.minimumHeight, particle.positionAndMass.z
                    );
                    outcome.maximumHeight = std::max(
                        outcome.maximumHeight, particle.positionAndMass.z
                    );
                    outcome.minimumVerticalVelocity = std::min(
                        outcome.minimumVerticalVelocity, particle.velocityAndReferenceVolume.z
                    );
                    outcome.maximumVerticalVelocity = std::max(
                        outcome.maximumVerticalVelocity, particle.velocityAndReferenceVolume.z
                    );
                }
            }
            if (!snapshot.femNodes.empty()) {
                for (const NMFEMNodeStateGPU& node : snapshot.femNodes) {
                    outcome.minimumHeight = std::min(
                        outcome.minimumHeight, node.positionAndMass.z
                    );
                    outcome.maximumHeight = std::max(
                        outcome.maximumHeight, node.positionAndMass.z
                    );
                    outcome.minimumVerticalVelocity = std::min(
                        outcome.minimumVerticalVelocity, node.velocityAndInverseMass.z
                    );
                    outcome.maximumVerticalVelocity = std::max(
                        outcome.maximumVerticalVelocity, node.velocityAndInverseMass.z
                    );
                }
            }
        };
        for (std::uint32_t step = 0u; step < controlSteps; ++step) {
            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
            require(commandBuffer != nil, "failed to allocate Matter command buffer");
            numi::matter::EncodeRequest request{};
            request.commandBuffer = (__bridge void*)commandBuffer;
            request.environmentStatuses = (__bridge void*)worldStatuses;
            request.phase = numi::matter::EncodePhase::preDynamics;
            request.controlStep = step;
            request.physicsSubstep = 0u;
            request.physicsSubsteps = 1u;
            request.timestepSeconds = runtime.timestepSeconds();
            request.runAdaptiveTransfer = false;
            auto encoded = runtime.encode(request);
            require(encoded.encoded, label + std::string(" pre-dynamics: ") + encoded.message);
            if (forceRollback) {
                // Simulate MetalWorld rejecting this enclosing transaction
                // after Matter has encoded its tentative continuum update.
                worldStatus->code = MR_STEP_DID_NOT_CONVERGE;
            }
            request.phase = numi::matter::EncodePhase::postCommit;
            encoded = runtime.encode(request);
            require(encoded.encoded, label + std::string(" post-commit: ") + encoded.message);
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            require(
                commandBuffer.status == MTLCommandBufferStatusCompleted,
                label + std::string(" command buffer did not complete")
            );

            const NMMatterStatusGPU status = statusData[0];
            require(
                status.code == (forceRollback
                    ? NM_STATUS_RIGID_WORLD_FAILURE
                    : NM_STATUS_SUCCESS),
                label + std::string(" reported Matter status ") +
                    std::to_string(status.code) +
                    " step=" + std::to_string(step) +
                    " object=" + std::to_string(status.objectIndex) +
                    " index=" + std::to_string(status.failingIndex) +
                    " diagnostics=(" +
                    std::to_string(status.diagnostics.x) + "," +
                    std::to_string(status.diagnostics.y) + "," +
                    std::to_string(status.diagnostics.z) + "," +
                    std::to_string(status.diagnostics.w) + ")"
            );
            if (!forceRollback) {
                require(
                    !std::isnan(status.diagnostics.x) &&
                        status.diagnostics.x > 0.0f,
                    label + std::string(" lost a valid deformation determinant")
                );
            }
            outcome.contactSamples = std::max(outcome.contactSamples, status.contactCount);
            outcome.completedMicrosteps = std::max(
                outcome.completedMicrosteps,
                status.completedMicrosteps
            );
            outcome.pcgIterations = std::max(outcome.pcgIterations, status.pcgIterations);
            outcome.minimumDeterminant = std::min(
                outcome.minimumDeterminant,
                status.diagnostics.x
            );
            for (std::uint32_t event = 0u;
                 event < world.dispatch.eventStride;
                 ++event) {
                const NMEventTokenGPU& token = eventData[event];
                outcome.sawContactEvent = outcome.sawContactEvent || (
                    token.eventClass == NM_EVENT_CONTACT_ONSET &&
                    token.payload.z > 0.0f
                );
            }
            recordSnapshot(runtime.snapshot());
        }
        if (forceRollback) {
            const auto restored = runtime.snapshot();
            require(restored.available, label + std::string(" restored: ") + restored.message);
            const auto equalBytes = [](const auto& left, const auto& right) {
                return left.size() == right.size() &&
                    (left.empty() || std::memcmp(
                        left.data(), right.data(), left.size() * sizeof(left.front())
                    ) == 0);
            };
            require(
                equalBytes(restored.particles, baseline.particles) &&
                    equalBytes(restored.femNodes, baseline.femNodes) &&
                    equalBytes(restored.schedulers, baseline.schedulers),
                label + std::string(" did not restore continuum state after rejection")
            );
        }
        if (!forceRollback) {
            require(outcome.completedMicrosteps > 0u, label + std::string(" executed no microsteps"));
        }
        if (requireContact) {
            require(
                outcome.contactSamples > 0u || outcome.sawContactEvent,
                label + std::string(" never produced continuum contact")
            );
        }
        outcome.sawContactOnset = outcome.contactSamples > 0u;
        if (requireDescent) {
            require(
                outcome.minimumHeight < 0.019f,
                label + std::string(" did not advance under gravity")
            );
        }
        if (!forceRollback) {
            require(
                !std::isnan(outcome.minimumDeterminant) &&
                    outcome.minimumDeterminant > 0.0f,
                label + std::string(" produced an invalid deformation determinant")
            );
        }
        return outcome;
    }
}

} // namespace

int main(int argc, const char* argv[]) {
    try {
        const bool femOnly = argc == 2 && std::string_view(argv[1]) == "--fem";
        const bool mpmOnly = argc == 2 && std::string_view(argv[1]) == "--mpm";
        const bool mpmFree = argc == 2 && std::string_view(argv[1]) == "--mpm-free";
        const bool mpmSingle = argc == 2 && std::string_view(argv[1]) == "--mpm-single";
        const bool mpmSingleContact = argc == 2 && std::string_view(argv[1]) == "--mpm-single-contact";
        const bool mpmGentle = argc == 2 && std::string_view(argv[1]) == "--mpm-gentle-contact";
        const bool mpmRollback = argc == 2 && std::string_view(argv[1]) == "--mpm-rollback";
        const bool metalWorldCoupling = argc == 2 && std::string_view(argv[1]) == "--metal-world-coupling";
        const bool identification = argc == 2 && std::string_view(argv[1]) == "--identification";
        const bool adaptiveDemotion = argc == 2 && std::string_view(argv[1]) == "--adaptive-demotion";
        const bool femFree = argc == 2 && std::string_view(argv[1]) == "--fem-free";
        const bool femHighRate = argc == 2 && std::string_view(argv[1]) == "--fem-high-rate";
        const bool femHighDrop = argc == 2 && std::string_view(argv[1]) == "--fem-high-drop";
        require(
            argc == 1 || femOnly || mpmOnly || mpmFree || mpmSingle || mpmSingleContact || mpmGentle || mpmRollback || metalWorldCoupling || identification || adaptiveDemotion || femFree || femHighRate || femHighDrop,
            "usage: metalrobo_matter_physics_probe [--mpm|--mpm-free|--mpm-single|--mpm-single-contact|--mpm-gentle-contact|--mpm-rollback|--metal-world-coupling|--identification|--adaptive-demotion|--fem|--fem-free|--fem-high-rate|--fem-high-drop]"
        );
        if (identification) {
            runIdentification();
        }
        if (adaptiveDemotion) {
            runAdaptiveDemotion();
        }
        if (metalWorldCoupling) {
            runMetalWorldCoupling();
        }
        if (!identification && !adaptiveDemotion && !metalWorldCoupling &&
            !femOnly && !femFree && !femHighRate && !femHighDrop) {
            const bool withPlane = !mpmFree && !mpmSingle;
            const auto mpm = runCase(
                compileCase(
                    numi::matter::Representation::mpm,
                    withPlane || mpmSingleContact || mpmGentle,
                    mpmSingle || mpmSingleContact,
                    false,
                    false,
                    mpmGentle
                ),
                withPlane ? "MPM" : "MPM freefall",
                !mpmRollback && (withPlane || mpmSingleContact || mpmGentle),
                !mpmRollback,
                mpmRollback ? 1u : (mpmFree || mpmSingle ? 8u : 4u),
                mpmRollback
            );
            std::cout
                << "{\"schema\":\"numi.matter.physics-probe.v1\""
                << ",\"representation\":\"mpm\""
                << ",\"contact_samples\":" << mpm.contactSamples
                << ",\"minimum_J\":";
            if (mpmRollback) {
                std::cout << "null";
            } else {
                std::cout << mpm.minimumDeterminant;
            }
            std::cout
                << ",\"minimum_height\":" << mpm.minimumHeight
                << ",\"maximum_height\":" << mpm.maximumHeight
                << ",\"vertical_velocity_range\":[" << mpm.minimumVerticalVelocity
                << ',' << mpm.maximumVerticalVelocity << ']'
                << ",\"contact_event\":" << (mpm.sawContactEvent ? "true" : "false")
                << ",\"transaction_rollback\":" << (mpmRollback ? "true" : "false")
                << "}\n";
        }
        if (!mpmOnly && !mpmFree && !mpmSingle && !mpmSingleContact &&
            !mpmGentle && !mpmRollback && !metalWorldCoupling &&
            !identification && !adaptiveDemotion) {
            const bool withPlane = !femFree;
            const auto fem = runCase(
                compileCase(
                    numi::matter::Representation::fem,
                    withPlane,
                    false,
                    femHighRate || femHighDrop,
                    femHighRate
                ),
                femHighRate || femHighDrop
                    ? "FEM high-rate"
                    : (withPlane ? "FEM" : "FEM freefall"),
                withPlane,
                true,
                femHighRate || femHighDrop ? 8u : 20u
            );
            require(fem.pcgIterations > 0u, "FEM did not execute its implicit PCG solve");
            std::cout
                << "{\"schema\":\"numi.matter.physics-probe.v1\""
                << ",\"representation\":\"fem\""
                << ",\"contact_samples\":" << fem.contactSamples
                << ",\"pcg_iterations\":" << fem.pcgIterations
                << ",\"minimum_J\":" << fem.minimumDeterminant
                << ",\"minimum_height\":" << fem.minimumHeight
                << ",\"maximum_height\":" << fem.maximumHeight
                << ",\"vertical_velocity_range\":[" << fem.minimumVerticalVelocity
                << ',' << fem.maximumVerticalVelocity << ']'
                << ",\"contact_event\":" << (fem.sawContactEvent ? "true" : "false")
                << "}\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "matter_physics_probe: " << error.what() << '\n';
        return 1;
    }
}
