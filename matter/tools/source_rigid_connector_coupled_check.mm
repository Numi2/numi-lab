#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef NUMI_MATTER_METALLIB
#define NUMI_MATTER_METALLIB ""
#endif
#ifndef NUMI_MATTER_FIXTURE_MATERIAL
#define NUMI_MATTER_FIXTURE_MATERIAL ""
#endif
#ifndef NUMI_MATTER_SOURCE_PRESTRAIN_MATERIAL
#define NUMI_MATTER_SOURCE_PRESTRAIN_MATERIAL ""
#endif

namespace {
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            using namespace numi::matter;
            bool withTie = true;
            bool withFEMSpring = false;
            bool quasiStatic = false;
            bool sourceContinuation = false;
            float sourceTime = 0.0f;
            float numericalInverseMass = 1.0f;
            for (int argument = 1; argument < argc; ++argument) {
                const std::string option = argv[argument];
                if (option == "--no-tie") withTie = false;
                else if (option == "--with-fem-spring") withFEMSpring = true;
                else if (option == "--quasistatic") quasiStatic = true;
                else if (option == "--source-continuation") {
                    require(++argument < argc,
                        "source continuation requires a time");
                    sourceContinuation = true;
                    sourceTime = std::stof(argv[argument]);
                }
                else if (option == "--numerical-inverse-mass") {
                    require(++argument < argc,
                        "numerical inverse mass requires a value");
                    numericalInverseMass = std::stof(argv[argument]);
                } else require(false,
                    "unsupported source connector test argument: " + option);
            }
            require(std::isfinite(numericalInverseMass) &&
                numericalInverseMass > 0.0f,
                "numerical inverse mass must be positive and finite");
            require(!withFEMSpring || quasiStatic,
                    "cross-tissue source spring check requires static mechanics");
            require(!sourceContinuation ||
                (quasiStatic && std::isfinite(sourceTime) &&
                 sourceTime >= 0.0f && sourceTime <= 2.0f),
                "source continuation check requires bounded static time");
            auto material = parseMatterFile(sourceContinuation
                ? NUMI_MATTER_SOURCE_PRESTRAIN_MATERIAL
                : NUMI_MATTER_FIXTURE_MATERIAL);
            require(material.succeeded(), "source connector runtime material did not parse");
            if (sourceContinuation)
                for (auto& parameter : material.material.parameters)
                    if (parameter.name == "initial_stretch")
                        parameter.defaultValue = 1.0;
            WorldSource source;
            source.frameTimestep = 1.0e-3;
            source.gravity = {0.0, 0.0, 0.0};
            source.mixedSolver.newtonIterations = quasiStatic ? 20u : 8u;
            if (quasiStatic) {
                source.mixedSolver.relativeResidual = 1.0e-6;
                source.mixedSolver.fgmresIterations = 128u;
            }
            source.materials.push_back(std::move(material.material));
            RigidProxySource moving;
            moving.shape = NM_RIGID_SPHERE;
            moving.bodyIndex = 0u;
            moving.sceneBodyIndex = 0u;
            moving.radiusOrOffset = 0.0;
            moving.frameOnly = true;
            moving.dynamic = true;
            moving.quasiStatic = quasiStatic;
            RigidProxySource anchor = moving;
            anchor.bodyIndex = 1u;
            anchor.sceneBodyIndex = NM_INVALID_INDEX;
            anchor.dynamic = false;
            anchor.quasiStatic = false;
            source.rigidProxies = {moving, anchor};
            ObjectSource tendon;
            tendon.name = "source_rigid_tied_continuum";
            tendon.materialIndex = 0u;
            tendon.representation = Representation::fem;
            tendon.mixedFEM = false;
            tendon.deformableContact = false;
            tendon.deformableSelfContact = false;
            tendon.quasiStatic = quasiStatic;
            tendon.characteristicLength = 0.01;
            tendon.femNodes = {{0.002, 0.001, 0.0},
                               {0.012, 0.001, 0.0},
                               {0.002, 0.011, 0.0},
                               {0.002, 0.001, 0.01}};
            tendon.femFixedNodes = quasiStatic
                ? std::vector<std::uint32_t>{0u, 1u, 2u}
                : std::vector<std::uint32_t>{0u, 1u, 2u, 3u};
            tendon.tetrahedra = {{{0u, 1u, 2u, 3u}}};
            source.objects.push_back(std::move(tendon));
            if (withFEMSpring) {
                ObjectSource meniscus = source.objects.front();
                meniscus.name = "source_spring_coupled_second_continuum";
                for (auto& node : meniscus.femNodes) node[0] += 0.03;
                meniscus.femFixedNodes = {0u, 1u, 2u};
                source.objects.push_back(std::move(meniscus));
            }
            auto cooked = compileWorld(source, {
                .maximumRateExponent = 0u, .emitSpecializedMetal = false});
            std::string errors;
            for (const auto& row : cooked.diagnostics) errors += row.message + "; ";
            require(cooked.succeeded(), "source connector world cook: " + errors);
            require(cooked.world.dispatch.rigidGeneralizedCapacity == 6u,
                "source connector free-body capacity changed");
            require(cooked.world.contact.pairs.empty(),
                "source frame-only bodies created invented contact geometry");

            NMSourceCylindricalJointGPU joint{};
            joint.indices = {0u, 1u, 0u, 0u};
            joint.referenceA = {0.0f, 0.0f, 0.0f, 0.0f};
            joint.referenceB = {0.02f, 0.0f, 0.0f, 0.0f};
            joint.origin = {0.01f, 0.0f, 0.0f, 0.0f};
            joint.axis = {1.0f, 0.0f, 0.0f, 0.0f};
            joint.parameters = {10000.0f, 3000000.0f, 0.0f, 0.0f};
            if (sourceContinuation) {
                joint.indices.w = 1u;
                joint.parameters.w = -0.2f;
                joint.axial.z = 1.0f;
            }
            NMSourceRigidSpringGPU spring{};
            spring.indices = {0u, 1u, 0u, 0u};
            spring.referenceA = joint.referenceA;
            spring.referenceB = joint.referenceB;
            spring.insertionA = joint.referenceA;
            spring.insertionB = joint.referenceB;
            spring.parameters = {1000.0f, 0.0f, 0.0f, 0.0f};
            NMSourceFEMRigidTieGPU tie{};
            tie.identity = {0u, 0u, 0u, 1u};
            tie.localPoint = {0.0f, 0.0f, 0.0f, 0.0f};
            NMSourceFEMSpringGPU femSpring{};
            NMSourcePrestrainGPU prestrain{};
            if (sourceContinuation) {
                const auto& parameters = source.materials[0].parameters;
                const auto found = std::find_if(parameters.begin(), parameters.end(),
                    [](const auto& parameter) {
                        return parameter.name == "initial_stretch";
                    });
                require(found != parameters.end(),
                    "source continuation material has no stretch parameter");
                prestrain.identity = {
                    cooked.world.materials[0].parameterOffset +
                        static_cast<std::uint32_t>(found - parameters.begin()),
                    0u, 3u, 0u};
                prestrain.stretch = {1.0f, 1.016f, 0.0f, 0.0f};
            }
            if (withFEMSpring) {
                femSpring.identity = {3u, 7u, 1u, 0u};
                femSpring.referenceA = {0.002f, 0.001f, 0.01f, 0.0f};
                femSpring.referenceB = {0.032f, 0.001f, 0.01f, 0.0f};
                femSpring.parameters = {1.0e4f, 0.0f, 0.0f, 0.0f};
            }

            RuntimeConfiguration configuration;
            configuration.metallib = NUMI_MATTER_METALLIB;
            configuration.captureDiagnostics = true;
            configuration.captureEvents = false;
            configuration.adaptiveTransfer = false;
            configuration.sourceCylindricalJoints = {&joint, 1u};
            configuration.sourceRigidSprings = {&spring, 1u};
            if (withTie) configuration.sourceFEMRigidTies = {&tie, 1u};
            if (withFEMSpring)
                configuration.sourceFEMSprings = {&femSpring, 1u};
            if (sourceContinuation)
                configuration.sourcePrestrain = {&prestrain, 1u};
            configuration.sourceRigidConnectorFingerprint =
                0x4e4d53434f4e4e31ull;
            Runtime runtime;
            const auto initialized = runtime.initialize(cooked.world, configuration);
            require(initialized.encoded, "source connector runtime init: " +
                initialized.message);
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "source connector runtime requires Metal");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue != nil, "source connector queue unavailable");
            MRBodyStateGPU bodies[2]{};
            for (auto& body : bodies) {
                body.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
                body.inverseInertiaWorldRow0 = {1.0f, 0.0f, 0.0f, 0.0f};
                body.inverseInertiaWorldRow1 = {0.0f, 1.0f, 0.0f, 0.0f};
                body.inverseInertiaWorldRow2 = {0.0f, 0.0f, 1.0f, 0.0f};
            }
            bodies[0].position = {0.002f, 0.001f, 0.0f, 0.0f};
            bodies[0].linearVelocityAndInverseMass.w = numericalInverseMass;
            bodies[0].flagsAndIndices[0] = MR_MOTION_DYNAMIC;
            bodies[1].position = {0.02f, 0.0f, 0.0f, 0.0f};
            bodies[1].flagsAndIndices[0] = MR_MOTION_STATIC;
            MRMetalWorldStatusGPU worldStatus{};
            worldStatus.code = MR_STEP_SUCCESS;
            id<MTLBuffer> current = [device newBufferWithBytes:bodies
                length:sizeof(bodies) options:MTLResourceStorageModeShared];
            id<MTLBuffer> scene = [device newBufferWithBytes:&bodies[0]
                length:sizeof(bodies[0]) options:MTLResourceStorageModeShared];
            id<MTLBuffer> wrenches = [device newBufferWithLength:
                2u * sizeof(MRABABodyWrenchGPU)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> statuses = [device newBufferWithBytes:&worldStatus
                length:sizeof(worldStatus) options:MTLResourceStorageModeShared];
            require(current != nil && scene != nil && wrenches != nil &&
                statuses != nil, "source connector Metal buffers unavailable");
            id<MTLCommandBuffer> command = [queue commandBuffer];
            EncodeRequest request{};
            request.commandBuffer = (__bridge void*)command;
            request.environmentStatuses = (__bridge void*)statuses;
            request.rigid.currentBodies = (__bridge void*)current;
            request.rigid.currentBodyCount = 2u;
            request.rigid.currentBodyStride = 2u;
            request.rigid.sceneBodies = (__bridge void*)scene;
            request.rigid.sceneBodyCount = 1u;
            request.rigid.sceneStride = 1u;
            request.rigid.bodyWrenches = (__bridge void*)wrenches;
            request.rigid.bodyWrenchCount = 2u;
            request.rigid.bodyWrenchStride = 2u;
            request.timestepSeconds = runtime.timestepSeconds();
            request.sourceContinuationTime = sourceTime;
            request.runAdaptiveTransfer = false;
            request.phase = EncodePhase::preDynamics;
            auto encoded = runtime.encode(request);
            require(encoded.encoded, "source connector preDynamics: " +
                encoded.message);
            request.phase = EncodePhase::postCommit;
            if (sourceContinuation) {
                request.sourceContinuationTime = sourceTime + 0.01f;
                const auto mismatched = runtime.encode(request);
                require(!mismatched.encoded &&
                    mismatched.message.find("source time differs") != std::string::npos,
                    "source transaction admitted a changed post-commit target");
                request.sourceContinuationTime = sourceTime;
            }
            encoded = runtime.encode(request);
            require(encoded.encoded, "source connector postCommit: " +
                encoded.message);
            [command commit];
            [command waitUntilCompleted];
            require(command.status == MTLCommandBufferStatusCompleted,
                "source connector Metal command failed");
            const auto state = runtime.snapshot();
            require(state.available && state.statuses.size() == 1u &&
                state.statuses[0].code == NM_STATUS_SUCCESS &&
                state.rigidGeneralizedCandidate.size() == 6u &&
                state.femNodes.size() == (withFEMSpring ? 8u : 4u) &&
                state.femNodes[0].restAndFixed.w == (withTie ? 3.0f : 1.0f),
                "source connector runtime did not accept the coupled candidate");
            const auto& v = state.rigidGeneralizedCandidate;
            require(std::isfinite(v[0]) && std::isfinite(v[1]) &&
                (quasiStatic || (v[0] < 0.0f && v[1] < 0.0f)),
                "source spring/joint did not restore the displaced free body: " +
                std::to_string(v[0]) + ", " + std::to_string(v[1]));
            if (sourceContinuation && sourceTime > 1.0f)
                require(std::abs(1.0e-3f * v[3] -
                            0.2f * std::min(sourceTime - 1.0f, 1.0f)) < 1.0e-4f,
                    "prescribed flexion did not enter the accepted coupled correction");
            const auto& tied = state.femNodes[0];
            if (withTie) require(std::abs(tied.positionAndMass.x -
                        (bodies[0].position.x + 1.0e-3f * v[0])) < 2.0e-6f &&
                    std::abs(tied.positionAndMass.y -
                        (bodies[0].position.y + 1.0e-3f * v[1])) < 2.0e-6f &&
                    std::abs(tied.velocityAndInverseMass.x - v[0]) < 2.0e-5f &&
                    std::abs(tied.velocityAndInverseMass.y - v[1]) < 2.0e-5f,
                "source FEM tie did not follow the same accepted rigid correction");
            if (withFEMSpring)
                require(std::isfinite(state.femNodes[7].positionAndMass.x) &&
                        std::abs(state.femNodes[7].positionAndMass.x - 0.032f) >
                            1.0e-10f,
                        "cross-tissue spring did not move its second continuum");
            std::cout << "source_connector_coupled=accepted"
                      << " free_increment_x=" << 1.0e-3f * v[0]
                      << " free_increment_y=" << 1.0e-3f * v[1]
                      << " free_angular_increment_x=" << 1.0e-3f * v[3]
                      << " tied_node_x=" << tied.positionAndMass.x
                      << " free_node_z=" << state.femNodes[3].positionAndMass.z
                      << " source_tie=" << (withTie ? "on" : "off")
                      << " fem_spring=" << (withFEMSpring ? "on" : "off")
                      << " quasistatic=" << (quasiStatic ? "on" : "off")
                      << " numerical_inverse_mass=" << numericalInverseMass
                      << " source_time=" << sourceTime
                      << " source_knee_equivalence=unqualified\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "source_connector_coupled=failed "
                      << error.what() << '\n';
            return 1;
        }
    }
}
