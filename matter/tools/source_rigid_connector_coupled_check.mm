#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"

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

namespace {
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
}

int main() {
    @autoreleasepool {
        try {
            using namespace numi::matter;
            auto material = parseMatterFile(NUMI_MATTER_FIXTURE_MATERIAL);
            require(material.succeeded(), "source connector runtime material did not parse");
            WorldSource source;
            source.frameTimestep = 1.0e-3;
            source.gravity = {0.0, 0.0, 0.0};
            source.mixedSolver.newtonIterations = 8u;
            source.materials.push_back(std::move(material.material));
            RigidProxySource moving;
            moving.shape = NM_RIGID_SPHERE;
            moving.bodyIndex = 0u;
            moving.sceneBodyIndex = 0u;
            moving.radiusOrOffset = 0.001;
            moving.dynamic = true;
            RigidProxySource anchor = moving;
            anchor.bodyIndex = 1u;
            anchor.sceneBodyIndex = NM_INVALID_INDEX;
            anchor.dynamic = false;
            source.rigidProxies = {moving, anchor};
            auto cooked = compileWorld(source, {
                .maximumRateExponent = 0u, .emitSpecializedMetal = false});
            std::string errors;
            for (const auto& row : cooked.diagnostics) errors += row.message + "; ";
            require(cooked.succeeded(), "source connector world cook: " + errors);
            require(cooked.world.dispatch.rigidGeneralizedCapacity == 6u,
                "source connector free-body capacity changed");

            NMSourceCylindricalJointGPU joint{};
            joint.indices = {0u, 1u, 0u, 0u};
            joint.referenceA = {0.0f, 0.0f, 0.0f, 0.0f};
            joint.referenceB = {0.02f, 0.0f, 0.0f, 0.0f};
            joint.origin = {0.01f, 0.0f, 0.0f, 0.0f};
            joint.axis = {1.0f, 0.0f, 0.0f, 0.0f};
            joint.parameters = {10000.0f, 3000000.0f, 0.0f, 0.0f};
            NMSourceRigidSpringGPU spring{};
            spring.indices = {0u, 1u, 0u, 0u};
            spring.referenceA = joint.referenceA;
            spring.referenceB = joint.referenceB;
            spring.insertionA = joint.referenceA;
            spring.insertionB = joint.referenceB;
            spring.parameters = {1000.0f, 0.0f, 0.0f, 0.0f};

            RuntimeConfiguration configuration;
            configuration.metallib = NUMI_MATTER_METALLIB;
            configuration.captureDiagnostics = true;
            configuration.captureEvents = false;
            configuration.adaptiveTransfer = false;
            configuration.sourceCylindricalJoints = {&joint, 1u};
            configuration.sourceRigidSprings = {&spring, 1u};
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
            bodies[0].linearVelocityAndInverseMass.w = 1.0f;
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
            request.runAdaptiveTransfer = false;
            request.phase = EncodePhase::preDynamics;
            auto encoded = runtime.encode(request);
            require(encoded.encoded, "source connector preDynamics: " +
                encoded.message);
            request.phase = EncodePhase::postCommit;
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
                state.rigidGeneralizedCandidate.size() == 6u,
                "source connector runtime did not accept the coupled candidate");
            const auto& v = state.rigidGeneralizedCandidate;
            require(std::isfinite(v[0]) && std::isfinite(v[1]) &&
                v[0] < 0.0f && v[1] < 0.0f,
                "source spring/joint did not restore the displaced free body");
            std::cout << "source_connector_coupled=accepted"
                      << " free_velocity_x=" << v[0]
                      << " free_velocity_y=" << v[1]
                      << " source_knee_equivalence=unqualified\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "source_connector_coupled=failed "
                      << error.what() << '\n';
            return 1;
        }
    }
}
