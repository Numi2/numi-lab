#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "numi/matter/metal_world.hpp"
#include "metalrobo/EngineModel.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/engine_types.h"

#include <algorithm>
#include <array>
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

#ifndef NUMI_MATTER_STATEFUL_MATERIAL
#define NUMI_MATTER_STATEFUL_MATERIAL ""
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

metalrobo::EngineModel makeTwoFreeSphereEngineModel() {
    metalrobo::EngineModel model =
        metalrobo::makeFreeSphereEngineModel();
    model.name = "two_free_sphere_articulations";
    const auto firstQ = model.defaultQ;

    MRArticulationGPU second = model.articulations.front();
    second.rootBody = 2u;
    second.firstBody = 2u;
    second.qOffset = 7u;
    second.vOffset = 6u;
    model.articulations.push_back(second);

    MRBodyPropertiesGPU secondBody = model.bodies[1];
    secondBody.articulationIndex = 1u;
    model.bodies.push_back(secondBody);
    for (std::uint32_t local = 0u; local < 6u; ++local) {
        MRDofPropertiesGPU dof = model.dofs[local];
        dof.articulationIndex = 1u;
        dof.qIndex = local < 3u ? 7u + local : MR_INVALID_INDEX;
        dof.vIndex = 6u + local;
        model.dofs.push_back(dof);
    }
    MRShapeGPU secondShape = model.shapes[1];
    secondShape.bodyIndex = 2u;
    secondShape.slotGeneration = 2u;
    model.shapes.push_back(secondShape);
    model.defaultQ.insert(
        model.defaultQ.end(),
        firstQ.begin(),
        firstQ.end()
    );
    model.defaultV.resize(12u, 0.0f);
    model.world.bodyCount = static_cast<mr_u32>(model.bodies.size());
    model.world.articulationCount =
        static_cast<mr_u32>(model.articulations.size());
    model.world.shapeCount = static_cast<mr_u32>(model.shapes.size());
    model.world.nq = static_cast<mr_u32>(model.defaultQ.size());
    model.world.nv = static_cast<mr_u32>(model.defaultV.size());
    model.world.pairCapacity = std::max(model.world.pairCapacity, 3u);
    model.world.contactCapacity =
        std::max(model.world.contactCapacity, 8u);
    model.world.constraintCapacity =
        std::max(model.world.constraintCapacity, 24u);
    std::string reason;
    require(model.valid(&reason),
        "two-free-sphere model is invalid: " + reason);
    return model;
}

void encodeRigidWorldFailure(
    id<MTLDevice> device,
    id<MTLCommandBuffer> commandBuffer,
    id<MTLBuffer> destination,
    const std::uint32_t successfulSubsteps = 0u
) {
    require(
        device != nil && commandBuffer != nil && destination != nil &&
            destination.length >= sizeof(MRMetalWorldStatusGPU),
        "cannot encode a rigid-world failure into an invalid status arena"
    );
    MRMetalWorldStatusGPU failure{};
    failure.code = MR_STEP_DID_NOT_CONVERGE;
    failure.environment = 0u;
    failure.successfulSubsteps = successfulSubsteps;
    id<MTLBuffer> source = [device
        newBufferWithBytes:&failure
        length:sizeof(failure)
        options:MTLResourceStorageModeShared];
    require(source != nil, "failed to allocate rigid-world failure record");
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    require(blit != nil, "failed to encode rigid-world failure handoff");
    blit.label = @"Matter probe rigid-world failure injection";
    [blit
        copyFromBuffer:source
        sourceOffset:0u
        toBuffer:destination
        destinationOffset:0u
        size:sizeof(failure)];
    [blit endEncoding];
    // Retain the source record until the GPU has consumed the ordered blit.
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        (void)completed;
        (void)source;
    }];
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
    const std::uint32_t identificationCandidates = 0u,
    const std::uint32_t maximumRateExponentOverride = NM_INVALID_INDEX,
    const bool secondBodyBackedPlane = false,
    const bool enableMultiphysics = false,
    const bool enableMutation = false,
    const bool enableLearned = false
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
    if (enableLearned) {
        numi::matter::LearnedMaterialSource learned;
        learned.invariantCount = 4u;
        learned.softplusBeta = 2.0f;
        learned.determinantFloor = 0.05f;
        learned.growthCoefficient = 0.01f;
        numi::matter::LearnedLayerSource layer;
        layer.inputWidth = 4u;
        layer.outputWidth = 1u;
        layer.inputWeights = {0.04f, 0.03f, 0.02f, 0.01f};
        layer.biases = {-0.1f};
        learned.layers.push_back(std::move(layer));
        source.materials[0].hint =
            numi::matter::ConstitutiveHint::polyconvexICNN;
        source.materials[0].learned = std::move(learned);
    }

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
        if (bodyBackedPlane && secondBodyBackedPlane) {
            plane.bodyIndex = 2u;
            source.rigidProxies.push_back(plane);
        }
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
        if (enableMultiphysics) {
            object.multiphysics.enabled = true;
            object.multiphysics.initialTemperature = 300.0;
            object.multiphysics.initialPorePressure = 2.0;
            object.multiphysics.initialElectricPotential = 0.25;
            object.multiphysics.initialActivation = 0.0;
            numi::matter::FieldBoundarySource hot;
            hot.node = 0u;
            hot.flags = NM_FIELD_DIRICHLET_TEMPERATURE |
                NM_FIELD_DIRICHLET_ELECTRIC_POTENTIAL;
            hot.value = {350.0, 0.0, 1.0, 0.0};
            object.fieldBoundaries.push_back(hot);
            numi::matter::FieldBoundarySource ground;
            ground.node = 1u;
            ground.flags = NM_FIELD_DIRICHLET_TEMPERATURE |
                NM_FIELD_DIRICHLET_ELECTRIC_POTENTIAL;
            ground.value = {300.0, 0.0, 0.0, 0.0};
            object.fieldBoundaries.push_back(ground);
            source.materials[0].mixed.heatCapacity = 1.0;
            source.materials[0].mixed.thermalConductivity = 0.1;
            source.materials[0].mixed.poreStorage = 1.0;
            source.materials[0].mixed.poreMobility = 0.1;
            source.materials[0].mixed.electricalConductivity = 1.0;
            source.materials[0].mixed.activationDiffusivity = 0.1;
            source.materials[0].mixed.activationOnRate = 8.0;
            source.materials[0].mixed.activationOffRate = 1.0;
            source.materials[0].mixed.activationThreshold = 0.2;
            source.materials[0].mixed.activationSlope = 12.0;
            source.materials[0].mixed.maximumActiveTension = 50.0;
        }
        if (enableMutation) {
            object.mutationPolicy.enabled = true;
            object.femCapacity.tetrahedra = 1u;
            object.femCapacity.mutationCommands = 1u;
            numi::matter::MutationCommandSource command;
            command.kind = NM_MUTATION_DEACTIVATE_TETRAHEDRON;
            command.controlStep = 0u;
            command.target = 0u;
            command.priority = 1u;
            command.stableIdentifier = 17u;
            object.mutationCommands.push_back(command);
        }
    }
    source.objects.push_back(std::move(object));

    numi::matter::CompileOptions options;
    // FEM is qualified first at one implicit step per frame. MPM must retain
    // its material-selected CFL subdivision; forcing it to the FEM baseline
    // would knowingly test an unstable explicit material update.
    options.maximumRateExponent =
        maximumRateExponentOverride != NM_INVALID_INDEX
        ? maximumRateExponentOverride
        : (representation == numi::matter::Representation::mpm || fullRate
            ? NM_MAX_RATE_EXPONENT
            : 0u);
    auto compiled = numi::matter::compileWorld(source, options);
    require(compiled.succeeded(), "Matter world compilation failed");
    require(
        !includePlane || compiled.world.dispatch.contactPairCount > 0u,
        "drop case has no continuum-to-plane contact pairs"
    );
    return std::move(compiled.world);
}

numi::matter::CompiledWorld compileMixedCase() {
    auto parsed = numi::matter::parseMatterFile(NUMI_MATTER_MATERIAL);
    require(parsed.succeeded(), "reference silicone material did not parse");
    for (auto& parameter : parsed.material.parameters) {
        if (parameter.name == "mu") {
            parameter.defaultValue = 1.0e3;
            parameter.lower = 5.0e2;
            parameter.upper = 2.0e3;
        } else if (parameter.name == "lambda") {
            parameter.defaultValue = 4.0e3;
            parameter.lower = 1.0e3;
            parameter.upper = 8.0e3;
        }
    }

    numi::matter::WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = 1.0 / 240.0;
    source.gravity = {0.0, 0.0, -9.81};
    source.femPCGIterations = 6u;
    source.materials.push_back(parsed.material);

    numi::matter::ObjectSource mpm;
    mpm.name = "mixed_mpm";
    mpm.materialIndex = 0u;
    mpm.representation = numi::matter::Representation::mpm;
    mpm.characteristicLength = 0.01;
    mpm.mpmGridMinimum = {-0.04, -0.02, -0.01};
    mpm.mpmGridMaximum = {-0.01, 0.02, 0.06};
    constexpr double particleSpacing = 0.005;
    constexpr double particleVolume =
        particleSpacing * particleSpacing * particleSpacing;
    for (int index = 0; index < 2; ++index) {
        numi::matter::ParticleSource particle;
        particle.position = {
            -0.03 + particleSpacing * index,
            0.0,
            0.03,
        };
        particle.velocity = {0.0, 0.0, -0.2};
        particle.mass = 1100.0 * particleVolume;
        particle.referenceVolume = particleVolume;
        mpm.particles.push_back(particle);
    }
    source.objects.push_back(std::move(mpm));

    numi::matter::ObjectSource fem;
    fem.name = "mixed_fem";
    fem.materialIndex = 0u;
    fem.representation = numi::matter::Representation::fem;
    fem.characteristicLength = 0.02;
    fem.femInitialVelocity = {0.0, 0.0, -0.1};
    fem.femNodes = {
        {0.015, -0.01, 0.025},
        {0.035, -0.01, 0.025},
        {0.015,  0.01, 0.025},
        {0.015, -0.01, 0.045},
    };
    fem.tetrahedra.push_back({{0u, 1u, 2u, 3u}});
    source.objects.push_back(std::move(fem));

    numi::matter::CompileOptions options;
    options.maximumRateExponent = 2u;
    auto compiled = numi::matter::compileWorld(source, options);
    require(compiled.succeeded(), "mixed Matter world compilation failed");
    require(
        compiled.world.dispatch.particleCount != 0u &&
            compiled.world.dispatch.femNodeCount != 0u &&
            compiled.world.dispatch.tetrahedronCount != 0u,
        "mixed Matter world did not retain both backends"
    );
    return std::move(compiled.world);
}


numi::matter::CompiledWorld compileStatefulCase(
    const numi::matter::Representation representation
) {
    auto parsed = numi::matter::parseMatterFile(
        NUMI_MATTER_STATEFUL_MATERIAL
    );
    require(parsed.succeeded(), "stateful silicone material did not parse");
    for (auto& parameter : parsed.material.parameters) {
        if (parameter.name == "mu") {
            parameter.defaultValue = 5.0e3;
            parameter.lower = 5.0e2;
            parameter.upper = 2.0e4;
        } else if (parameter.name == "lambda") {
            parameter.defaultValue = 2.0e4;
            parameter.lower = 2.0e3;
            parameter.upper = 8.0e4;
        } else if (parameter.name == "eta") {
            parameter.defaultValue = 100.0;
            parameter.lower = 10.0;
            parameter.upper = 1.0e3;
        } else if (parameter.name == "damage_rate") {
            parameter.defaultValue = 50.0;
            parameter.lower = 0.0;
            parameter.upper = 200.0;
        } else if (parameter.name == "damage_threshold") {
            parameter.defaultValue = 1.0e-4;
            parameter.lower = 0.0;
            parameter.upper = 0.1;
        }
    }

    numi::matter::WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = 1.0 / 960.0;
    source.gravity = {0.0, 0.0, 0.0};
    source.femPCGIterations = 8u;
    source.materials.push_back(parsed.material);

    numi::matter::ObjectSource object;
    object.name = representation == numi::matter::Representation::mpm
        ? "stateful_mpm"
        : "stateful_fem";
    object.materialIndex = 0u;
    object.representation = representation;
    object.characteristicLength = 0.01;
    constexpr double strainRate = 20.0;
    if (representation == numi::matter::Representation::mpm) {
        object.mpmGridMinimum = {-0.02, -0.02, -0.02};
        object.mpmGridMaximum = {0.02, 0.02, 0.02};
        constexpr double spacing = 0.005;
        constexpr double volume = spacing * spacing * spacing;
        for (int z = 0; z < 2; ++z) {
            for (int y = 0; y < 2; ++y) {
                for (int x = 0; x < 2; ++x) {
                    numi::matter::ParticleSource particle;
                    particle.position = {
                        -0.0025 + spacing * x,
                        -0.0025 + spacing * y,
                        -0.0025 + spacing * z,
                    };
                    particle.velocity = {
                        strainRate * particle.position[0],
                        -0.5 * strainRate * particle.position[1],
                        -0.5 * strainRate * particle.position[2],
                    };
                    particle.mass = 1100.0 * volume;
                    particle.referenceVolume = volume;
                    object.particles.push_back(particle);
                }
            }
        }
    } else {
        object.femNodes = {
            {-0.005, -0.005, -0.005},
            { 0.005, -0.005, -0.005},
            {-0.005,  0.005, -0.005},
            {-0.005, -0.005,  0.005},
        };
        object.tetrahedra.push_back({{0u, 1u, 2u, 3u}});
    }
    source.objects.push_back(std::move(object));

    numi::matter::CompileOptions options;
    options.maximumRateExponent = 4u;
    auto compiled = numi::matter::compileWorld(source, options);
    require(compiled.succeeded(), "stateful Matter world compilation failed");
    require(
        compiled.world.dispatch.materialStateStride == 2u,
        "stateful Matter world has the wrong material-state stride"
    );
    if (representation == numi::matter::Representation::fem) {
        require(
            compiled.world.fem.nodes.size() == 4u,
            "stateful FEM world did not retain its tetrahedron"
        );
        for (NMFEMNodeStateGPU& node : compiled.world.fem.nodes) {
            node.velocityAndInverseMass.x =
                static_cast<float>(strainRate) * node.positionAndMass.x;
            node.velocityAndInverseMass.y =
                static_cast<float>(-0.5 * strainRate) *
                node.positionAndMass.y;
            node.velocityAndInverseMass.z =
                static_cast<float>(-0.5 * strainRate) *
                node.positionAndMass.z;
        }
        compiled.world.fingerprint =
            numi::matter::compiledWorldFingerprint(compiled.world);
    }
    return std::move(compiled.world);
}

void runStatefulMaterial(
    const numi::matter::Representation representation
) {
    @autoreleasepool {
        const auto world = compileStatefulCase(representation);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "no Metal device is available");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        require(queue != nil, "failed to create stateful Matter queue");
        id<MTLBuffer> worldStatuses = [device
            newBufferWithLength:sizeof(MRMetalWorldStatusGPU)
            options:MTLResourceStorageModeShared];
        require(
            worldStatuses != nil,
            "failed to allocate stateful world-status buffer"
        );
        auto* worldStatus = static_cast<MRMetalWorldStatusGPU*>(
            worldStatuses.contents
        );

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

        const auto runStep = [&](
            const std::uint32_t controlStep,
            const bool reject
        ) {
            *worldStatus = {};
            worldStatus->code = MR_STEP_SUCCESS;
            worldStatus->environment = 0u;
            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
            require(
                commandBuffer != nil,
                "failed to allocate stateful Matter command buffer"
            );
            numi::matter::EncodeRequest request{};
            request.commandBuffer = (__bridge void*)commandBuffer;
            request.environmentStatuses = (__bridge void*)worldStatuses;
            request.phase = numi::matter::EncodePhase::preDynamics;
            request.controlStep = controlStep;
            request.physicsSubstep = 0u;
            request.physicsSubsteps = 1u;
            request.timestepSeconds = runtime.timestepSeconds();
            request.runAdaptiveTransfer = false;
            auto encoded = runtime.encode(request);
            require(
                encoded.encoded,
                std::string("stateful pre-dynamics: ") + encoded.message
            );
            if (reject) {
                encodeRigidWorldFailure(
                    device,
                    commandBuffer,
                    worldStatuses
                );
            }
            request.phase = numi::matter::EncodePhase::postCommit;
            encoded = runtime.encode(request);
            require(
                encoded.encoded,
                std::string("stateful post-commit: ") + encoded.message
            );
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            require(
                commandBuffer.status == MTLCommandBufferStatusCompleted,
                "stateful Matter command buffer did not complete"
            );
        };

        const auto initial = runtime.snapshot();
        require(initial.available, initial.message);
        require(
            initial.materialStateStride == 2u,
            "stateful runtime snapshot has the wrong state stride"
        );
        const std::vector<float>& initialState =
            representation == numi::matter::Representation::mpm
            ? initial.particleMaterialState
            : initial.femMaterialState;
        require(
            !initialState.empty() &&
                std::ranges::all_of(initialState, [](const float value) {
                    return value == 0.0f;
                }),
            "stateful runtime did not initialize material state exactly"
        );

        for (std::uint32_t step = 0u; step < 4u; ++step) {
            runStep(step, false);
        }
        const auto evolved = runtime.snapshot();
        require(evolved.available, evolved.message);
        const std::vector<float>& evolvedState =
            representation == numi::matter::Representation::mpm
            ? evolved.particleMaterialState
            : evolved.femMaterialState;
        require(
            evolvedState.size() == initialState.size(),
            "stateful runtime changed material-state capacity"
        );
        bool accumulated = false;
        float maximumDamage = 0.0f;
        float maximumAccumulatedStrain = 0.0f;
        for (std::size_t base = 0u;
             base + 1u < evolvedState.size();
             base += evolved.materialStateStride) {
            const float damage = evolvedState[base];
            const float accumulatedStrain = evolvedState[base + 1u];
            require(
                std::isfinite(damage) &&
                    std::isfinite(accumulatedStrain) &&
                    damage >= 0.0f && damage <= 0.9501f &&
                    accumulatedStrain >= 0.0f,
                "stateful runtime produced invalid material state"
            );
            maximumDamage = std::max(maximumDamage, damage);
            maximumAccumulatedStrain = std::max(
                maximumAccumulatedStrain,
                accumulatedStrain
            );
            accumulated = accumulated || accumulatedStrain > 1.0e-6f;
        }
        require(
            accumulated,
            "rate-dependent material state did not evolve"
        );

        runStep(4u, true);
        const auto restored = runtime.snapshot();
        require(restored.available, restored.message);
        const std::vector<float>& restoredState =
            representation == numi::matter::Representation::mpm
            ? restored.particleMaterialState
            : restored.femMaterialState;
        require(
            restoredState.size() == evolvedState.size() &&
                std::memcmp(
                    restoredState.data(),
                    evolvedState.data(),
                    evolvedState.size() * sizeof(float)
                ) == 0,
            "rejected transaction did not restore material state exactly"
        );

        std::cout
            << "{\"schema\":\"numi.matter.stateful-runtime.v1\""
            << ",\"representation\":\""
            << (representation == numi::matter::Representation::mpm
                    ? "mpm"
                    : "fem")
            << "\",\"state_stride\":"
            << evolved.materialStateStride
            << ",\"maximum_damage\":" << maximumDamage
            << ",\"maximum_accumulated_strain\":"
            << maximumAccumulatedStrain
            << ",\"rollback_exact\":true}\n";
    }
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
    float minimumTemperature = std::numeric_limits<float>::infinity();
    float maximumTemperature = -std::numeric_limits<float>::infinity();
    float maximumActivation = 0.0f;
    float maximumElectricPotential = -std::numeric_limits<float>::infinity();
    std::uint32_t activeTetrahedra = 0u;
    std::uint32_t learnedRevision = 0u;
};

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

void runAdaptiveTransfer(
    const bool requirePromotion,
    const bool rejectPromotion = false
) {
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
        require(adaptive.inverseInertiaRow0.x > 0.0f &&
                    adaptive.inverseInertiaRow1.y > 0.0f &&
                    adaptive.inverseInertiaRow2.z > 0.0f,
            "adaptive demotion produced no valid rigid inverse inertia");
        if (requirePromotion) {
            // The direct probe uses the same typed arena MetalWorld exposes
            // after a solved contact substep.  The fallback body's real
            // pre-solve normal speed crosses the authored promotion threshold.
            *current = *scene;
            id<MTLBuffer> contactStatuses = [device
                newBufferWithLength:sizeof(MRMetalWorldContactStatusGPU)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> contacts = [device
                newBufferWithLength:sizeof(MRContactConstraintGPU)
                options:MTLResourceStorageModeShared];
            require(contactStatuses != nil && contacts != nil,
                "failed to allocate adaptive rigid contact evidence");
            auto* contactStatus = static_cast<MRMetalWorldContactStatusGPU*>(
                contactStatuses.contents
            );
            auto* contact = static_cast<MRContactConstraintGPU*>(contacts.contents);
            require(contactStatus != nullptr && contact != nullptr,
                "adaptive rigid contact evidence is unavailable");
            *contactStatus = {};
            contactStatus->code = MR_STEP_SUCCESS;
            contactStatus->environment = 0u;
            contactStatus->requiredConstraints = 1u;
            contactStatus->activeContacts = 1u;
            *contact = {};
            contact->bodyA = 0u;
            contact->bodyB = MR_INVALID_INDEX;
            contact->targetVelocityAndPreSolveNormal.w = -1.0f;
            contact->impulses.x = 0.1f;
            const auto beforePromotion = runtime.snapshot();
            require(beforePromotion.available,
                "adaptive promotion rollback baseline is unavailable");

            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
            require(commandBuffer != nil,
                "failed to allocate adaptive promotion command buffer");
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
            request.rigidContactConstraints = (__bridge void*)contacts;
            request.rigidContactStatuses = (__bridge void*)contactStatuses;
            request.rigidContactConstraintStride = 1u;
            request.controlStep = 30u;
            request.physicsSubstep = 0u;
            request.physicsSubsteps = 1u;
            request.timestepSeconds = runtime.timestepSeconds();
            request.phase = numi::matter::EncodePhase::preDynamics;
            request.runAdaptiveTransfer = false;
            auto encoded = runtime.encode(request);
            require(encoded.encoded,
                "adaptive promotion pre-dynamics encoding failed: " + encoded.message);
            if (rejectPromotion) {
                encodeRigidWorldFailure(
                    device,
                    commandBuffer,
                    statuses
                );
            }
            request.phase = numi::matter::EncodePhase::postCommit;
            request.runAdaptiveTransfer = true;
            encoded = runtime.encode(request);
            require(encoded.encoded,
                "adaptive promotion post-commit encoding failed: " + encoded.message);
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            require(commandBuffer.status == MTLCommandBufferStatusCompleted,
                "adaptive promotion command buffer did not complete");

            if (rejectPromotion) {
                require(matterStatus->code == NM_STATUS_RIGID_WORLD_FAILURE,
                    "rejected rigid contact did not latch Matter failure");
                const auto restored = runtime.snapshot();
                const auto equalBytes = [](const auto& left, const auto& right) {
                    return left.size() == right.size() &&
                        (left.empty() || std::memcmp(
                            left.data(), right.data(),
                            left.size() * sizeof(left.front())
                        ) == 0);
                };
                require(restored.available &&
                            restored.adaptive.size() == 1u &&
                            restored.schedulers.size() == 1u &&
                            equalBytes(restored.adaptive, beforePromotion.adaptive) &&
                            equalBytes(restored.schedulers, beforePromotion.schedulers) &&
                            restored.adaptive[0].activeRepresentation ==
                                NM_REPRESENTATION_RIGID &&
                            (scene->flagsAndIndices[3] &
                             MR_BODY_STATE_COLLISION_DISABLED) == 0u,
                    "rejected rigid contact did not restore adaptive rigid ownership");
                std::cout
                    << "{\"schema\":\"numi.matter.physics-probe.v1\""
                    << ",\"representation\":\"adaptive_promotion_rollback\""
                    << ",\"transaction_rollback\":true}\n";
                return;
            }
            require(matterStatus->code == NM_STATUS_SUCCESS,
                "adaptive promotion command buffer did not complete successfully");

            const auto promoted = runtime.snapshot();
            require(promoted.available && promoted.adaptive.size() == 1u &&
                        promoted.schedulers.size() == 1u,
                "adaptive promotion did not publish diagnostic state");
            require(promoted.adaptive[0].activeRepresentation ==
                        NM_REPRESENTATION_MPM &&
                        promoted.adaptive[0].requestedRepresentation ==
                            NM_REPRESENTATION_MPM &&
                        promoted.schedulers[0].physical.x >= 1.0f &&
                        (scene->flagsAndIndices[3] &
                         MR_BODY_STATE_COLLISION_DISABLED) != 0u,
                "rigid contact did not restore continuum ownership");
            std::cout
                << "{\"schema\":\"numi.matter.physics-probe.v1\""
                << ",\"representation\":\"adaptive_rigid_to_mpm\""
                << ",\"contact_speed\":"
                << promoted.schedulers[0].physical.x
                << ",\"continuum_collision_disabled\":true}\n";
            return;
        }
        std::cout
            << "{\"schema\":\"numi.matter.physics-probe.v1\""
            << ",\"representation\":\"adaptive_mpm_to_rigid\""
            << ",\"stable_frames\":" << adaptive.stableFrames
            << ",\"scene_x\":" << scene->position.x
            << ",\"inverse_inertia_diag\":["
            << adaptive.inverseInertiaRow0.x << ','
            << adaptive.inverseInertiaRow1.y << ','
            << adaptive.inverseInertiaRow2.z << ']'
            << "}\n";
    }
}

void runMetalWorldCoupling() {
    @autoreleasepool {
        constexpr std::uint32_t controlSteps = 1u;
        // This probe qualifies both directions of the borrowed transaction:
        // continuum reactions enter MetalWorld ABA, while real inverse-ABA
        // point-response columns return to Matter's contact CSR. One particle
        // activates several distinct grid-node contacts against each of two
        // independently articulated planes. This gives both a nontrivial
        // within-articulation off-diagonal oracle and a zero cross-articulation
        // oracle without turning the material CFL planner into the benchmark.
        const auto matterWorld = compileCase(
            numi::matter::Representation::mpm,
            true,
            true,
            false,
            false,
            false,
            true,
            1.0 / 480.0,
            1u,
            0u,
            0u,
            true
        );
        numi::matter::Runtime matter;
        const auto initialized = matter.initialize(
            matterWorld,
            {
                .metallib = NUMI_MATTER_METALLIB,
                .captureEvents = true,
                .captureDiagnostics = true,
                .automaticIdentification = false,
                .adaptiveTransfer = false,
            }
        );
        require(initialized.encoded && matter.valid(),
            "MetalWorld coupling could not initialize Matter: " + initialized.message);

        const metalrobo::EngineModel model =
            makeTwoFreeSphereEngineModel();
        metalrobo::CompiledWorld rigidWorld;
        const auto compiled = metalrobo::compileMetalWorld(model, 0u, rigidWorld);
        require(compiled.succeeded(),
            "MetalWorld coupling could not compile free body: " + compiled.message);
        std::vector<float> efforts(
            static_cast<std::size_t>(controlSteps) * rigidWorld.nv(),
            0.0f
        );
        const metalrobo::MetalWorldBatch batch{
            .environmentCount = 1u,
            .controlStepCount = controlSteps,
            .initialQ = model.defaultQ,
            .initialV = model.defaultV,
            .efforts = efforts,
        };
        metalrobo::MetalWorldStepConfig config{};
        config.timestepSeconds = 1.0f / 480.0f;
        config.physicsSubsteps = 1u;
        // Matter owns the continuum-plane contact in this probe. MetalWorld
        // only consumes the equal-and-opposite articulated body wrench, so
        // running the rigid contact compiler here is redundant and can
        // dominate qualification latency on shared Apple runners.
        config.solverMode = metalrobo::MetalWorldSolverMode::freeMotionABA;
        config.matrixFreeArticulatedContact = false;
        config.streamedArticulatedContactResponses = false;
        config.captureContactEvidence = false;
        config.devicePhysicsProgram =
            numi::matter::makeMetalWorldDevicePhysicsProgram(matter);
        require(config.devicePhysicsProgram.valid(),
            "Matter did not produce a valid MetalWorld adapter");
        require(
            (config.devicePhysicsProgram.flags &
             metalrobo::MetalWorldDevicePhysicsWritesBodyWrenches) != 0u &&
            (config.devicePhysicsProgram.flags &
             metalrobo::
                 MetalWorldDevicePhysicsRequiresRigidContactEvidence) == 0u,
            "non-adaptive Matter coupling published incorrect device-physics capabilities"
        );

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
            : " operator=(count=" + std::to_string(
                ran.layout.kinematicsDispatches.size()
            ) + ",factor_count=" + std::to_string(
                ran.layout.factorDispatches.size()
            ) + ",articulation=" + std::to_string(
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
        require(
            snapshot.contactSamples.size() == matterWorld.contact.pairs.size() &&
                snapshot.contactResponseRows.size() ==
                    snapshot.contactResponseColumns.size() &&
                snapshot.contactResponseRows.size() ==
                    snapshot.contactResponseValues.size(),
            "MetalWorld/Matter coupling did not publish complete contact CSR diagnostics"
        );
        float articulatedOffDiagonal = 0.0f;
        float articulatedTranspose = 0.0f;
        float independentArticulationResponse = 0.0f;
        std::uint32_t articulatedRow = NM_INVALID_INDEX;
        std::uint32_t articulatedColumn = NM_INVALID_INDEX;
        for (std::size_t entry = 0u;
             entry < snapshot.contactResponseValues.size();
             ++entry) {
            const std::uint32_t row = snapshot.contactResponseRows[entry];
            const std::uint32_t column =
                snapshot.contactResponseColumns[entry];
            if (row >= matterWorld.contact.pairs.size() ||
                column >= matterWorld.contact.pairs.size() || row == column ||
                (snapshot.contactSamples[row].identity.w & NM_CONTACT_VALID) == 0u ||
                (snapshot.contactSamples[column].identity.w & NM_CONTACT_VALID) == 0u ||
                matterWorld.contact.pairs[row].continuumNode ==
                    matterWorld.contact.pairs[column].continuumNode) {
                continue;
            }
            const float response = snapshot.contactResponseValues[entry];
            if (matterWorld.contact.pairs[row].rigidProxy !=
                    matterWorld.contact.pairs[column].rigidProxy) {
                independentArticulationResponse = std::max(
                    independentArticulationResponse,
                    std::abs(response)
                );
                continue;
            }
            if (std::abs(response) <= std::abs(articulatedOffDiagonal)) {
                continue;
            }
            for (std::size_t transpose = 0u;
                 transpose < snapshot.contactResponseValues.size();
                 ++transpose) {
                if (snapshot.contactResponseRows[transpose] == column &&
                    snapshot.contactResponseColumns[transpose] == row) {
                    articulatedOffDiagonal = response;
                    articulatedTranspose =
                        snapshot.contactResponseValues[transpose];
                    articulatedRow = row;
                    articulatedColumn = column;
                    break;
                }
            }
        }
        const float articulatedSymmetryError =
            std::abs(articulatedOffDiagonal - articulatedTranspose);
        require(
            articulatedRow != NM_INVALID_INDEX &&
                std::isfinite(articulatedOffDiagonal) &&
                std::abs(articulatedOffDiagonal) > 1.0e-6f,
            "borrowed inverse ABA did not populate an off-diagonal CSR response"
        );
        require(
            std::isfinite(articulatedTranspose) &&
                articulatedSymmetryError <=
                    2.0e-4f * std::max(1.0f, std::abs(articulatedOffDiagonal)),
            "borrowed inverse-ABA CSR response is not symmetric: forward=" +
                std::to_string(articulatedOffDiagonal) +
                " transpose=" + std::to_string(articulatedTranspose)
        );
        require(
            independentArticulationResponse <= 1.0e-6f,
            "independent MetalWorld articulations leaked into one another: response=" +
                std::to_string(independentArticulationResponse)
        );
        const float reactionZ = snapshot.reactions[0].impulseAndCount.z;
        require(snapshot.reactions.size() >= 2u,
            "multi-articulation coupling did not publish both rigid reactions");
        const float secondReactionZ =
            snapshot.reactions[1].impulseAndCount.z;
        // Matter sees a downward particle impact, so the equal-and-opposite
        // impulse applied to the body-backed plane is downward as well.
        require(reactionZ < -1.0e-6f,
            "continuum impact did not accumulate a rigid reaction: reaction_z=" +
                std::to_string(reactionZ) +
                " final_body_velocity_z=" +
                std::to_string(result.finalV[2]));
        require(
            result.finalV[2] < -1.0e-6f &&
                result.finalV.size() >= 12u &&
                result.finalV[8] < -1.0e-6f &&
                std::abs(result.finalV[2] - reactionZ) < 1.0e-5f &&
                std::abs(result.finalV[8] - secondReactionZ) < 1.0e-5f,
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
            << ",\"second_rigid_reaction_z\":" << secondReactionZ
            << ",\"second_body_velocity_z\":" << result.finalV[8]
            << ",\"articulated_response_row\":" << articulatedRow
            << ",\"articulated_response_column\":" << articulatedColumn
            << ",\"articulated_off_diagonal\":" << articulatedOffDiagonal
            << ",\"articulated_transpose\":" << articulatedTranspose
            << ",\"articulated_symmetry_error\":" << articulatedSymmetryError
            << ",\"independent_articulation_response\":"
            << independentArticulationResponse
            << ",\"gpu_milliseconds\":" << ran.gpuElapsedMilliseconds
            << "}\n";
    }
}

void runCoupledContactOracle() {
    @autoreleasepool {
        // Two separate unit-inverse-mass continuum nodes impact two colliders
        // attached to one unit-inverse-mass free body along the same normal.
        // The body-owned sparse response is W = [[2, 1], [1, 2]], so the
        // unique projected solution for unit closing speed is lambda =
        // [1/3, 1/3]. An independent-pair solve instead produces [1/2, 1/2].
        // This launches the production kernel directly on Metal with the
        // exact nested CSR incidence it receives from the runtime; no CPU
        // contact solve participates.
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "no Metal device is available");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        require(queue != nil, "failed to create coupled-contact queue");
        NSError* error = nil;
        id<MTLLibrary> library = [device
            newLibraryWithURL:[NSURL fileURLWithPath:@NUMI_MATTER_METALLIB]
                       error:&error];
        require(library != nil, "could not load Matter Metal library");
        id<MTLFunction> gatherFunction = [library newFunctionWithName:
            @"numi_matter_metal::nm_contact_gather_response"];
        require(gatherFunction != nil,
            "Matter Metal library is missing contact response gather");
        id<MTLComputePipelineState> gatherPipeline = [device
            newComputePipelineStateWithFunction:gatherFunction error:&error];
        require(gatherPipeline != nil,
            "could not create contact-response gather pipeline");
        id<MTLFunction> function = [library newFunctionWithName:
            @"numi_matter_metal::nm_contact_solve_coupled"];
        require(function != nil, "Matter Metal library is missing coupled contact");
        id<MTLComputePipelineState> pipeline = [device
            newComputePipelineStateWithFunction:function error:&error];
        require(pipeline != nil, "could not create coupled-contact pipeline");

        NMMatterDispatchGPU dispatch{};
        dispatch.abiVersion = NM_MATTER_ABI_VERSION;
        dispatch.environmentCount = 1u;
        dispatch.objectCount = 1u;
        dispatch.materialCount = 1u;
        dispatch.gridNodeCount = 2u;
        dispatch.rigidProxyCount = 2u;
        dispatch.contactPairCount = 2u;
        dispatch.coupledContactIterations = NM_COUPLED_CONTACT_ITERATIONS;
        dispatch.gravityAndTimestep.w = 1.0f;
        dispatch.numericalLimits = {0.0f, 0.0f, 0.0f, 1.0e6f};
        NMMicrostepGPU microstep{};

        NMContinuumObjectGPU object{};
        object.materialIndex = 0u;
        NMMaterialGPU material{};
        material.interfaceResponse.y = 1.0f;
        std::array<NMGridNodeStateGPU, 2u> nodes{};
        for (NMGridNodeStateGPU& node : nodes) {
            node.velocityAndInverseMass.w = 1.0f;
        }
        NMFEMNodeStateGPU unusedFEM{};
        std::array<NMRigidStateGPU, 2u> rigid{};
        for (NMRigidStateGPU& state : rigid) {
            state.linearVelocityAndInverseMass.w = 1.0f;
            state.bodyCenter.w = 1.0f;
        }
        std::array<NMRigidProxyGPU, 2u> proxies{};
        for (NMRigidProxyGPU& proxy : proxies) {
            proxy.bodyIndex = 0u;
        }
        const std::array<NMContactPairGPU, 2u> pairs{{
            {.continuumNode = 0u, .rigidProxy = 0u, .objectIndex = 0u,
             .materialInterface = 0u},
            {.continuumNode = 1u, .rigidProxy = 1u, .objectIndex = 0u,
             .materialInterface = 0u},
        }};
        const std::array<std::uint32_t, 4u> responseColumns{{0u, 1u, 0u, 1u}};
        const std::array<NMIncidenceRangeGPU, 2u> responseRanges{{
            {0u, 2u, 0u, 0u},
            {2u, 2u, 1u, 0u},
        }};
        std::array<float, 4u> responseValues{};
        const std::array<std::uint32_t, 2u> componentIncidence{{0u, 1u}};
        const NMIncidenceRangeGPU componentRange{0u, 2u, 0u, 0u};
        constexpr std::uint32_t responseEntryCount = 4u;
        constexpr std::uint32_t componentCount = 1u;
        NMSchedulerStateGPU scheduler{};
        NMMatterStatusGPU status{};
        status.code = NM_STATUS_SUCCESS;
        std::array<NMContactSampleGPU, 2u> samples{};
        for (std::uint32_t index = 0u; index < samples.size(); ++index) {
            NMContactSampleGPU& sample = samples[index];
            sample.identity = {index, 0u, 0u, NM_CONTACT_VALID};
            sample.normalAndVelocity = {0.0f, 0.0f, 1.0f, -1.0f};
        }

        const auto makeBuffer = [&](const void* bytes, const NSUInteger length,
                                    NSString* label) {
            id<MTLBuffer> buffer = [device newBufferWithBytes:bytes
                length:length options:MTLResourceStorageModeShared];
            require(buffer != nil, "failed to allocate coupled-contact buffer");
            buffer.label = label;
            return buffer;
        };
        id<MTLBuffer> objects = makeBuffer(&object, sizeof(object), @"oracle objects");
        id<MTLBuffer> materials = makeBuffer(&material, sizeof(material), @"oracle materials");
        id<MTLBuffer> mpmNodes = makeBuffer(nodes.data(), sizeof(nodes), @"oracle MPM nodes");
        id<MTLBuffer> femNodes = makeBuffer(&unusedFEM, sizeof(unusedFEM), @"oracle FEM nodes");
        id<MTLBuffer> proxyBuffer = makeBuffer(proxies.data(), sizeof(proxies), @"oracle proxies");
        id<MTLBuffer> rigidStates = makeBuffer(rigid.data(), sizeof(rigid), @"oracle rigid states");
        id<MTLBuffer> pairBuffer = makeBuffer(pairs.data(), sizeof(pairs), @"oracle pairs");
        id<MTLBuffer> responseColumnBuffer = makeBuffer(responseColumns.data(), sizeof(responseColumns), @"oracle response columns");
        id<MTLBuffer> responseRangeBuffer = makeBuffer(responseRanges.data(), sizeof(responseRanges), @"oracle response ranges");
        id<MTLBuffer> responseValueBuffer = makeBuffer(responseValues.data(), sizeof(responseValues), @"oracle response values");
        id<MTLBuffer> componentIncidenceBuffer = makeBuffer(componentIncidence.data(), sizeof(componentIncidence), @"oracle component incidence");
        id<MTLBuffer> componentRangeBuffer = makeBuffer(&componentRange, sizeof(componentRange), @"oracle component range");
        id<MTLBuffer> schedulers = makeBuffer(&scheduler, sizeof(scheduler), @"oracle schedulers");
        id<MTLBuffer> sampleBuffer = makeBuffer(samples.data(), sizeof(samples), @"oracle samples");
        id<MTLBuffer> statuses = makeBuffer(&status, sizeof(status), @"oracle statuses");

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        require(commandBuffer != nil && encoder != nil,
            "failed to encode coupled-contact oracle");
        [encoder setComputePipelineState:gatherPipeline];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBuffer:mpmNodes offset:0u atIndex:1u];
        [encoder setBuffer:femNodes offset:0u atIndex:2u];
        [encoder setBuffer:proxyBuffer offset:0u atIndex:3u];
        [encoder setBuffer:rigidStates offset:0u atIndex:4u];
        [encoder setBuffer:pairBuffer offset:0u atIndex:5u];
        [encoder setBuffer:sampleBuffer offset:0u atIndex:6u];
        [encoder setBuffer:responseRangeBuffer offset:0u atIndex:7u];
        [encoder setBuffer:responseColumnBuffer offset:0u atIndex:8u];
        [encoder setBuffer:responseValueBuffer offset:0u atIndex:9u];
        [encoder setBuffer:statuses offset:0u atIndex:10u];
        [encoder setBytes:&responseEntryCount length:sizeof(responseEntryCount) atIndex:11u];
        [encoder dispatchThreads:MTLSizeMake(2u, 1u, 1u)
          threadsPerThreadgroup:MTLSizeMake(gatherPipeline.threadExecutionWidth, 1u, 1u)];
        [encoder setComputePipelineState:pipeline];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&microstep length:sizeof(microstep) atIndex:1u];
        [encoder setBuffer:objects offset:0u atIndex:2u];
        [encoder setBuffer:materials offset:0u atIndex:3u];
        [encoder setBuffer:rigidStates offset:0u atIndex:4u];
        [encoder setBuffer:pairBuffer offset:0u atIndex:5u];
        [encoder setBuffer:schedulers offset:0u atIndex:6u];
        [encoder setBuffer:sampleBuffer offset:0u atIndex:7u];
        [encoder setBuffer:statuses offset:0u atIndex:8u];
        [encoder setBuffer:responseColumnBuffer offset:0u atIndex:9u];
        [encoder setBuffer:responseRangeBuffer offset:0u atIndex:10u];
        [encoder setBuffer:responseValueBuffer offset:0u atIndex:11u];
        [encoder setBuffer:componentIncidenceBuffer offset:0u atIndex:12u];
        [encoder setBuffer:componentRangeBuffer offset:0u atIndex:13u];
        [encoder setBytes:&responseEntryCount length:sizeof(responseEntryCount) atIndex:14u];
        [encoder setBytes:&componentCount length:sizeof(componentCount) atIndex:15u];
        [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
          threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1u, 1u)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        require(commandBuffer.status == MTLCommandBufferStatusCompleted,
            "coupled-contact Metal command did not complete");

        const auto* solved = static_cast<const NMContactSampleGPU*>(
            sampleBuffer.contents
        );
        const auto* gathered = static_cast<const float*>(
            responseValueBuffer.contents
        );
        require(solved != nullptr, "coupled-contact results are unavailable");
        require(gathered != nullptr &&
                    std::abs(gathered[0] - 2.0f) < 1.0e-6f &&
                    std::abs(gathered[1] - 1.0f) < 1.0e-6f &&
                    std::abs(gathered[2] - 1.0f) < 1.0e-6f &&
                    std::abs(gathered[3] - 2.0f) < 1.0e-6f,
            "device response gather did not assemble the shared-body Delassus block");
        const float first = solved[0].impulseAndNormal.w;
        const float second = solved[1].impulseAndNormal.w;
        const float symmetryError = std::abs(gathered[1] - gathered[2]);
        const float trace = gathered[0] + gathered[3];
        const float discriminant = std::sqrt(std::max(
            (gathered[0] - gathered[3]) *
                (gathered[0] - gathered[3]) +
                4.0f * gathered[1] * gathered[2],
            0.0f
        ));
        const float minimumEigenvalue = 0.5f * (trace - discriminant);
        const float residual = std::max(
            std::abs(2.0f * first + second - 1.0f),
            std::abs(first + 2.0f * second - 1.0f)
        );
        require(std::abs(first - 1.0f / 3.0f) < 2.0e-5f &&
                    std::abs(second - 1.0f / 3.0f) < 2.0e-5f &&
                    residual < 4.0e-5f && symmetryError < 1.0e-6f &&
                    minimumEigenvalue >= -1.0e-6f,
            "coupled-contact oracle did not solve the shared-rigid 1/3 impulse case");
        std::cout
            << "{\"schema\":\"numi.matter.physics-probe.v2\""
            << ",\"representation\":\"shared_rigid_coupled_contact\""
            << ",\"delassus\":[[" << gathered[0] << ',' << gathered[1]
            << "],[" << gathered[2] << ',' << gathered[3] << "]]"
            << ",\"impulses\":[" << first << ',' << second << ']'
            << ",\"residual\":" << residual
            << ",\"symmetry_error\":" << symmetryError
            << ",\"minimum_eigenvalue\":" << minimumEigenvalue
            << "}\n";
    }
}

Outcome runCase(
    const numi::matter::CompiledWorld& world,
    const char* label,
    const bool requireContact,
    const bool requireDescent,
    const std::uint32_t controlSteps,
    const bool forceRollback = false,
    const bool updateLearned = false
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
        std::vector<float> learnedUpdateValues = world.learnedWeights;
        for (float& value : learnedUpdateValues) value += 0.001f;
        id<MTLBuffer> learnedUpdate = learnedUpdateValues.empty()
            ? nil
            : [device newBufferWithBytes:learnedUpdateValues.data()
                length:learnedUpdateValues.size() * sizeof(float)
                options:MTLResourceStorageModeShared];

        Outcome outcome;
        float initialMinimumHeight = std::numeric_limits<float>::infinity();
        for (const NMParticleStateGPU& particle : world.mpm.particles) {
            initialMinimumHeight = std::min(
                initialMinimumHeight,
                particle.positionAndMass.z
            );
        }
        for (const NMFEMNodeStateGPU& node : world.fem.nodes) {
            initialMinimumHeight = std::min(
                initialMinimumHeight,
                node.positionAndMass.z
            );
        }
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
            for (const NMFEMFieldStateGPU& field : snapshot.femFields) {
                outcome.minimumTemperature = std::min(
                    outcome.minimumTemperature, field.primary.y
                );
                outcome.maximumTemperature = std::max(
                    outcome.maximumTemperature, field.primary.y
                );
                outcome.maximumElectricPotential = std::max(
                    outcome.maximumElectricPotential, field.primary.w
                );
                outcome.maximumActivation = std::max(
                    outcome.maximumActivation, field.secondary.x
                );
            }
            outcome.activeTetrahedra = 0u;
            for (const NMTetrahedronGPU& tetrahedron :
                 snapshot.femTopologyTetrahedra) {
                outcome.activeTetrahedra +=
                    (tetrahedron.identity.w & NM_OBJECT_ACTIVE) != 0u;
            }
            outcome.learnedRevision = snapshot.learnedWeightRevision;
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
            if (updateLearned && step == 0u) {
                request.learnedWeightUpdate = (__bridge void*)learnedUpdate;
                request.learnedWeightCount =
                    static_cast<std::uint32_t>(learnedUpdateValues.size());
                request.learnedWeightRevision = 1u;
            }
            auto encoded = runtime.encode(request);
            require(encoded.encoded, label + std::string(" pre-dynamics: ") + encoded.message);
            if (forceRollback) {
                // The ordered blit executes after the tentative continuum
                // update and before post-commit reconciliation. This verifies
                // real GPU rollback rather than pre-marking the transaction
                // failed before any Matter work executes.
                encodeRigidWorldFailure(
                    device,
                    commandBuffer,
                    worldStatuses
                );
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
                    equalBytes(
                        restored.particleMaterialState,
                        baseline.particleMaterialState
                    ) &&
                    equalBytes(
                        restored.femMaterialState,
                        baseline.femMaterialState
                    ) &&
                    equalBytes(restored.femFields, baseline.femFields) &&
                    equalBytes(restored.learnedWeights, baseline.learnedWeights) &&
                    equalBytes(
                        restored.femTopologyTetrahedra,
                        baseline.femTopologyTetrahedra
                    ) &&
                    equalBytes(restored.femTopologyNodes, baseline.femTopologyNodes) &&
                    equalBytes(restored.cohesiveFaces, baseline.cohesiveFaces) &&
                    equalBytes(restored.punctureChannels, baseline.punctureChannels) &&
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
            const float descentTolerance = std::max(
                1.0e-6f,
                std::abs(initialMinimumHeight) * 1.0e-6f
            );
            require(
                std::isfinite(initialMinimumHeight) &&
                    outcome.minimumHeight <
                        initialMinimumHeight - descentTolerance,
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
        const bool mixedOnly = argc == 2 && std::string_view(argv[1]) == "--mixed";
        const bool statefulMPM = argc == 2 &&
            std::string_view(argv[1]) == "--stateful-mpm";
        const bool statefulFEM = argc == 2 &&
            std::string_view(argv[1]) == "--stateful-fem";
        const bool mpmOnly = argc == 2 && std::string_view(argv[1]) == "--mpm";
        const bool mpmFree = argc == 2 && std::string_view(argv[1]) == "--mpm-free";
        const bool mpmSingle = argc == 2 && std::string_view(argv[1]) == "--mpm-single";
        const bool mpmSingleContact = argc == 2 && std::string_view(argv[1]) == "--mpm-single-contact";
        const bool mpmGentle = argc == 2 && std::string_view(argv[1]) == "--mpm-gentle-contact";
        const bool mpmRollback = argc == 2 && std::string_view(argv[1]) == "--mpm-rollback";
        const bool metalWorldCoupling = argc == 2 && std::string_view(argv[1]) == "--metal-world-coupling";
        const bool coupledContact = argc == 2 && std::string_view(argv[1]) == "--coupled-contact";
        const bool multiphysics = argc == 2 && std::string_view(argv[1]) == "--multiphysics";
        const bool topologyMutation = argc == 2 &&
            std::string_view(argv[1]) == "--topology-mutation";
        const bool topologyRollback = argc == 2 &&
            std::string_view(argv[1]) == "--topology-rollback";
        const bool learnedMaterial = argc == 2 &&
            std::string_view(argv[1]) == "--learned-material";
        const bool productionRollback = argc == 2 &&
            std::string_view(argv[1]) == "--production-rollback";
        const bool identification = argc == 2 && std::string_view(argv[1]) == "--identification";
        const bool adaptiveDemotion = argc == 2 && std::string_view(argv[1]) == "--adaptive-demotion";
        const bool adaptivePromotion = argc == 2 && std::string_view(argv[1]) == "--adaptive-promotion";
        const bool adaptivePromotionRollback = argc == 2 && std::string_view(argv[1]) == "--adaptive-promotion-rollback";
        const bool femFree = argc == 2 && std::string_view(argv[1]) == "--fem-free";
        const bool femHighRate = argc == 2 && std::string_view(argv[1]) == "--fem-high-rate";
        const bool femHighDrop = argc == 2 && std::string_view(argv[1]) == "--fem-high-drop";
        require(
            argc == 1 || mixedOnly || statefulMPM || statefulFEM ||
                femOnly || mpmOnly || mpmFree || mpmSingle ||
                mpmSingleContact || mpmGentle || mpmRollback ||
                metalWorldCoupling || coupledContact || multiphysics ||
                topologyMutation || topologyRollback || learnedMaterial ||
                productionRollback ||
                identification || adaptiveDemotion ||
                adaptivePromotion || adaptivePromotionRollback || femFree ||
                femHighRate || femHighDrop,
            "usage: metalrobo_matter_physics_probe [--mixed|--multiphysics|--topology-mutation|--topology-rollback|--learned-material|--production-rollback|--stateful-mpm|--stateful-fem|--mpm|--mpm-free|--mpm-single|--mpm-single-contact|--mpm-gentle-contact|--mpm-rollback|--metal-world-coupling|--coupled-contact|--identification|--adaptive-demotion|--adaptive-promotion|--adaptive-promotion-rollback|--fem|--fem-free|--fem-high-rate|--fem-high-drop]"
        );
        if (identification) {
            runIdentification();
        }
        if (adaptiveDemotion) {
            runAdaptiveTransfer(false);
        }
        if (adaptivePromotion) {
            runAdaptiveTransfer(true);
        }
        if (adaptivePromotionRollback) {
            runAdaptiveTransfer(true, true);
        }
        if (metalWorldCoupling) {
            runMetalWorldCoupling();
        }
        if (coupledContact) {
            runCoupledContactOracle();
        }
        if (multiphysics) {
            const auto outcome = runCase(
                compileCase(
                    numi::matter::Representation::fem,
                    false, false, false, false, false, false,
                    1.0 / 240.0, 1u, 0u, 0u, false, true
                ),
                "monolithic multiphysics", false, false, 2u
            );
            require(
                outcome.minimumTemperature >= 299.9f &&
                outcome.maximumTemperature >= 349.9f &&
                outcome.maximumElectricPotential >= 0.99f &&
                outcome.maximumActivation > 0.0f,
                "multiphysics fields did not diffuse and activate on the Metal timeline"
            );
            std::cout
                << "{\"schema\":\"numi.matter.physics-probe.v2\""
                << ",\"representation\":\"monolithic_multiphysics\""
                << ",\"temperature_min\":" << outcome.minimumTemperature
                << ",\"temperature_max\":" << outcome.maximumTemperature
                << ",\"electric_max\":" << outcome.maximumElectricPotential
                << ",\"activation_max\":" << outcome.maximumActivation
                << "}\n";
        }
        if (topologyMutation || topologyRollback) {
            const auto outcome = runCase(
                compileCase(
                    numi::matter::Representation::fem,
                    false, false, false, false, false, false,
                    1.0 / 240.0, 1u, 0u, 0u, false, false, true
                ),
                topologyRollback ? "topology rollback" : "topology mutation",
                false, false, 1u, topologyRollback
            );
            if (!topologyRollback) {
                require(outcome.activeTetrahedra == 0u,
                    "device mutation did not deactivate the target tetrahedron");
            }
            std::cout
                << "{\"schema\":\"numi.matter.physics-probe.v2\""
                << ",\"representation\":\"topology_mutation\""
                << ",\"rollback\":" << (topologyRollback ? "true" : "false")
                << ",\"active_tetrahedra\":" << outcome.activeTetrahedra
                << "}\n";
        }
        if (learnedMaterial) {
            const auto world = compileCase(
                numi::matter::Representation::fem,
                false, false, false, false, false, false,
                1.0 / 240.0, 1u, 0u, 0u, false, false, false, true
            );
            require(world.dispatch.learnedMaterialCount == 1u &&
                    world.dispatch.learnedWeightCount == 5u,
                "learned ICNN was not retained in the executable package");
            const auto outcome = runCase(
                world, "polyconvex ICNN", false, true, 2u, false, true
            );
            require(outcome.minimumDeterminant > 0.0f &&
                    outcome.learnedRevision == 1u,
                "learned ICNN failed its transactional weight update");
            std::cout
                << "{\"schema\":\"numi.matter.physics-probe.v2\""
                << ",\"representation\":\"polyconvex_icnn\""
                << ",\"weights\":" << world.dispatch.learnedWeightCount
                << ",\"revision\":" << outcome.learnedRevision
                << ",\"minimum_J\":" << outcome.minimumDeterminant
                << "}\n";
        }
        if (productionRollback) {
            const auto outcome = runCase(
                compileCase(
                    numi::matter::Representation::fem,
                    false, false, false, false, false, false,
                    1.0 / 240.0, 1u, 0u, 0u, false, true, true, true
                ),
                "production transaction rollback", false, false, 1u,
                true, true
            );
            require(outcome.activeTetrahedra == 1u &&
                    outcome.learnedRevision == 0u,
                "production rollback published topology or learned revision");
            std::cout
                << "{\"schema\":\"numi.matter.physics-probe.v2\""
                << ",\"representation\":\"production_transaction\""
                << ",\"rollback\":true"
                << ",\"active_tetrahedra\":" << outcome.activeTetrahedra
                << ",\"learned_revision\":" << outcome.learnedRevision
                << "}\n";
        }
        if (statefulMPM) {
            runStatefulMaterial(numi::matter::Representation::mpm);
        }
        if (statefulFEM) {
            runStatefulMaterial(numi::matter::Representation::fem);
        }
        if (mixedOnly) {
            const auto mixed = runCase(
                compileMixedCase(),
                "mixed MPM/FEM",
                false,
                true,
                3u
            );
            require(
                mixed.pcgIterations > 0u,
                "mixed MPM/FEM world did not execute FEM PCG"
            );
            std::cout
                << "{\"schema\":\"numi.matter.physics-probe.v2\""
                << ",\"representation\":\"mixed_mpm_fem\""
                << ",\"pcg_iterations\":" << mixed.pcgIterations
                << ",\"completed_microsteps\":"
                << mixed.completedMicrosteps
                << ",\"minimum_J\":" << mixed.minimumDeterminant
                << "}\n";
        }
        if (!identification && !adaptiveDemotion && !adaptivePromotion &&
            !adaptivePromotionRollback && !metalWorldCoupling && !coupledContact &&
            !multiphysics &&
            !topologyMutation && !topologyRollback &&
            !learnedMaterial &&
            !productionRollback &&
            !mixedOnly && !statefulMPM && !statefulFEM &&
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
        if (!mixedOnly && !statefulMPM && !statefulFEM &&
            !mpmOnly && !mpmFree && !mpmSingle && !mpmSingleContact &&
            !mpmGentle && !mpmRollback && !metalWorldCoupling && !coupledContact &&
            !multiphysics &&
            !topologyMutation && !topologyRollback &&
            !learnedMaterial &&
            !productionRollback &&
            !identification && !adaptiveDemotion && !adaptivePromotion &&
            !adaptivePromotionRollback) {
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
