#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "cardboard_tooling.hpp"
#include "metalrobo/engine_types.h"
#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

using namespace numi::matter;

namespace {

constexpr double kMillimetre = 1.0e-3;
constexpr unsigned kEnvironmentCount = 2u;
constexpr unsigned kPunchBodyIndex = 0u;
constexpr unsigned kAnvilBodyIndex = 1u;
constexpr double kPadLength = 4.0 * kMillimetre;
constexpr double kPadWidth = 4.0 * kMillimetre;
constexpr double kPadCaliper = 2.0 * kMillimetre;
constexpr double kPunchRadius = 0.5 * kMillimetre;
constexpr double kBottomClearance = 20.0e-6;
constexpr double kInitialPunchGap = 10.0e-6;
constexpr double kContactSlop = 10.0e-6;
constexpr std::uint32_t kInteriorGripNode = 13u;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <class T>
id<MTLBuffer> makeBuffer(id<MTLDevice> device, const std::vector<T>& values) {
    const T zero{};
    const void* bytes = values.empty() ? static_cast<const void*>(&zero)
                                       : static_cast<const void*>(values.data());
    const NSUInteger count = std::max<std::size_t>(1u, values.size());
    return [device newBufferWithBytes:bytes
                               length:count * sizeof(T)
                              options:MTLResourceStorageModeShared];
}

double maximumNodeDisplacement(const RuntimeStateSnapshot& state,
                               const RuntimeStateSnapshot& initial,
                               unsigned environment,
                               std::uint32_t nodeCount);

template <class A, class B>
bool sameEnvironmentBytes(const A& a, const B& b, unsigned environment);

struct Fixture {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLBuffer> statuses = nil;
    id<MTLBuffer> bodies = nil;
    id<MTLBuffer> endpointBodies = nil;
    id<MTLBuffer> femKinematicTargets = nil;
    std::vector<std::array<float, 4>> femKinematicTargetValues;
    std::array<float, 3> temporaryGripTarget{};
    Runtime runtime;
    CompiledWorld world;
    std::vector<MRBodyStateGPU> authoredBodies;
    std::uint32_t femNodeCount = 0u;
    double timestep = 0.0;
    bool includeSupportBox = false;
    std::uint32_t bodyCountPerEnvironment = 1u;

    Fixture(const bool prescribed, const double punchSpeed, const bool armijo,
            const bool withSupportBox = false,
            const double supportSurfaceZ = -kBottomClearance)
        : includeSupportBox(withSupportBox),
          bodyCountPerEnvironment(withSupportBox ? 2u : 1u) {
        if (armijo && withSupportBox) {
            setenv("NM_FEM_CAPSULE_BOX_RESIDUAL_ARMIJO_ROOT", "0", 1);
            unsetenv("NM_FEM_CAPSULE_RESIDUAL_ARMIJO_ROOT");
        } else if (armijo) {
            setenv("NM_FEM_CAPSULE_RESIDUAL_ARMIJO_ROOT", "0", 1);
            unsetenv("NM_FEM_CAPSULE_BOX_RESIDUAL_ARMIJO_ROOT");
        } else {
            unsetenv("NM_FEM_CAPSULE_RESIDUAL_ARMIJO_ROOT");
            unsetenv("NM_FEM_CAPSULE_BOX_RESIDUAL_ARMIJO_ROOT");
        }
        device = MTLCreateSystemDefaultDevice();
        require(device != nil, "Metal device unavailable");
        require([[device name] rangeOfString:@"Apple"].location != NSNotFound &&
                    [[device name] rangeOfString:@"Paravirtual"].location == NSNotFound,
                "native tooling fixture requires physical Apple Metal");
        queue = [device newCommandQueue];
        require(queue != nil, "Metal command queue unavailable");

        auto parsed = parseMatter(R"(
            material synthetic_tool_pad {
                parameter density : kg/m^3 = 1000;
                parameter mu : Pa = 384615.3846153846;
                parameter lambda : Pa = 576923.0769230769;
                model neo_hookean;
                energy = neo_hookean(mu, lambda);
                valid = J() - 0.2;
                supports fem;
            }
        )");
        require(parsed.succeeded(), "synthetic pad material did not parse");
        parsed.material.staticFriction = 0.0;
        parsed.material.dynamicFriction = 0.0;

        WorldSource source;
        source.environmentCount = kEnvironmentCount;
        source.frameTimestep = 1.0e-4;
        source.gravity = {0.0, 0.0, 0.0};
        source.contactSlop = kContactSlop;
        source.mixedSolver.relativeResidual = 1.0e-7;
        source.mixedSolver.newtonIterations = 16u;
        source.mixedSolver.fgmresIterations = 64u;
        source.materials = {parsed.material};

        ObjectSource pad;
        pad.name = "synthetic_elastic_pad_for_kinematic_tool_contact";
        pad.materialIndex = 0u;
        pad.representation = Representation::fem;
        pad.mixedFEM = false;
        pad.adaptive = false;
        pad.deformableContact = false;
        pad.deformableSelfContact = false;
        pad.characteristicLength = 1.0 * kMillimetre;

        // A 3 x 3 x 3 nodal block creates a conforming six-tet split in each
        // of eight cells. The central upper and lower nodes are free; four
        // lower corner nodes remove rigid translation without prescribing the
        // tool-contact displacement. This is only a native contact fixture.
        const auto node = [](const unsigned i, const unsigned j,
                             const unsigned k) -> std::uint32_t {
            return static_cast<std::uint32_t>(i + 3u * j + 9u * k);
        };
        for (unsigned k = 0u; k < 3u; ++k) {
            const double z = kPadCaliper * static_cast<double>(k) / 2.0;
            for (unsigned j = 0u; j < 3u; ++j) {
                const double y = -0.5 * kPadWidth +
                    kPadWidth * static_cast<double>(j) / 2.0;
                for (unsigned i = 0u; i < 3u; ++i) {
                    const double x = -0.5 * kPadLength +
                        kPadLength * static_cast<double>(i) / 2.0;
                    pad.femNodes.push_back({x, y, z});
                    if (i == 0u || i == 2u || j == 0u || j == 2u ||
                        k == 0u || k == 2u)
                        pad.femContactNodes.push_back(node(i, j, k));
                }
            }
        }
        for (unsigned k = 0u; k < 2u; ++k) {
            for (unsigned j = 0u; j < 2u; ++j) {
                for (unsigned i = 0u; i < 2u; ++i) {
                    const std::array<std::uint32_t, 8> c = {
                        node(i, j, k), node(i + 1u, j, k),
                        node(i + 1u, j + 1u, k), node(i, j + 1u, k),
                        node(i, j, k + 1u), node(i + 1u, j, k + 1u),
                        node(i + 1u, j + 1u, k + 1u),
                        node(i, j + 1u, k + 1u),
                    };
                    constexpr std::array<std::array<unsigned, 4>, 6> split = {{
                        {{0u, 1u, 2u, 6u}}, {{0u, 2u, 3u, 6u}},
                        {{0u, 3u, 7u, 6u}}, {{0u, 7u, 4u, 6u}},
                        {{0u, 4u, 5u, 6u}}, {{0u, 5u, 1u, 6u}},
                    }};
                    for (const auto& local : split)
                        pad.tetrahedra.push_back({{
                            c[local[0]], c[local[1]],
                            c[local[2]], c[local[3]],
                        }});
                }
            }
        }
        // Clamp four lower perimeter nodes. The optional BOX case adds the
        // actual body-backed support geometry below; the capsule baseline keeps
        // the original capsule-only layout.
        for (const auto& [i, j] : std::array<std::array<unsigned, 2>, 4>{{
                 {{0u, 0u}}, {{2u, 0u}}, {{0u, 2u}}, {{2u, 2u}},
             }})
            pad.femFixedNodes.push_back(node(i, j, 0u));
        source.objects.push_back(std::move(pad));

        numi::cardboard::CreaseToolingSpec toolSpec;
        toolSpec.boardLength = kPadLength;
        toolSpec.boardWidth = kPadWidth;
        toolSpec.boardCaliper = kPadCaliper;
        toolSpec.creaseX = 0.0;
        toolSpec.supportSurfaceZ = supportSurfaceZ;
        toolSpec.punchRadius = kPunchRadius;
        toolSpec.punchEndOverhang = 0.1 * kMillimetre;
        toolSpec.supportEndOverhang = 0.25 * kMillimetre;
        toolSpec.supportThickness = 1.0 * kMillimetre;
        // The helper uses a reference plane to place the capsule. No box
        // anvil is included, and the resulting punch-to-pad top gap is 10 um.
        toolSpec.initialClearance = kBottomClearance + kInitialPunchGap;
        toolSpec.punchBodyIndex = kPunchBodyIndex;
        toolSpec.contactMaterialIndex = 0u;
        auto tools = numi::cardboard::makeCreaseTooling(toolSpec);
        tools.punch.prescribedEndPoseTranslation = prescribed;
        source.rigidProxies = {tools.punch};
        if (includeSupportBox) {
            tools.support.bodyIndex = kAnvilBodyIndex;
            source.rigidProxies.push_back(tools.support);
        }

        const auto compiled = compileWorld(source, {
            .maximumRateExponent = 0u,
            .emitSpecializedMetal = false,
        });
        for (const auto& diagnostic : compiled.diagnostics) {
            if (diagnostic.severity == Diagnostic::Severity::error)
                std::cerr << diagnostic.message << '\n';
        }
        require(compiled.succeeded(), "native tooling FEM fixture did not compile");
        world = compiled.world;
        require(world.contact.rigidProxies.size() == (includeSupportBox ? 2u : 1u) &&
                    world.contact.rigidProxies[0].shapeKind == NM_RIGID_CAPSULE &&
                    world.contact.rigidProxies[0].bodyIndex == kPunchBodyIndex &&
                    world.contact.rigidProxies[0].flags == NM_RIGID_PRESCRIBED_TRANSLATION,
                "compiled fixture must contain its endpoint-prescribed capsule");
        if (includeSupportBox)
            require(world.contact.rigidProxies[1].shapeKind == NM_RIGID_BOX &&
                        world.contact.rigidProxies[1].bodyIndex == kAnvilBodyIndex &&
                        world.contact.rigidProxies[1].flags == 0u &&
                        world.contact.rigidProxies[1].sceneBodyIndex == NM_INVALID_INDEX,
                    "support fixture must contain one unowned body-backed static BOX");
        require(world.dispatch.contactPairCount > 0u && world.dispatch.rigidQCapacity == 0u,
                "capsule fixture must cook source FEM contact pairs without rigid unknowns");
        require((world.objects[0].flags & (NM_OBJECT_DISABLE_SELF_CONTACT |
                    NM_OBJECT_DISABLE_DEFORMABLE_CONTACT)) ==
                    (NM_OBJECT_DISABLE_SELF_CONTACT | NM_OBJECT_DISABLE_DEFORMABLE_CONTACT),
                "fixture must exclude self/deformable contact while retaining rigid proxy pairs");
        femNodeCount = world.dispatch.femNodeCount;
        require(femNodeCount == 27u, "unexpected synthetic pad node count");
        timestep = world.dispatch.gravityAndTimestep.w;

        RuntimeConfiguration config;
        config.metallib = NUMI_MATTER_METALLIB;
        config.environmentCount = kEnvironmentCount;
        config.captureEvents = false;
        config.captureDiagnostics = true;
        config.adaptiveTransfer = false;
        const auto initialized = runtime.initialize(world, config);
        require(initialized.encoded, "native tooling runtime initialize: " + initialized.message);

        authoredBodies.resize(kEnvironmentCount * bodyCountPerEnvironment);
        for (unsigned environment = 0u; environment < kEnvironmentCount; ++environment) {
            MRBodyStateGPU body{};
            const auto pose = numi::cardboard::makeCreaseTooling(toolSpec).punchInitialPose;
            body.position = {
                static_cast<float>(pose.translation.x),
                static_cast<float>(pose.translation.y),
                static_cast<float>(pose.translation.z +
                    (environment == 0u ? 0.0 : 1.0 * kMillimetre)),
                0.0f,
            };
            body.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
            body.linearVelocityAndInverseMass = {
                0.0f,
                0.0f,
                static_cast<float>(environment == 0u ? -punchSpeed : 0.0),
                0.0f,
            };
            body.angularVelocity = {0.0f, 0.0f, 0.0f, 0.0f};
            body.inverseInertiaWorldRow0 = {0.0f, 0.0f, 0.0f, 0.0f};
            body.inverseInertiaWorldRow1 = {0.0f, 0.0f, 0.0f, 0.0f};
            body.inverseInertiaWorldRow2 = {0.0f, 0.0f, 0.0f, 0.0f};
            body.flagsAndIndices[0] = MR_MOTION_KINEMATIC;
            authoredBodies[environment * bodyCountPerEnvironment + kPunchBodyIndex] = body;
            if (includeSupportBox) {
                MRBodyStateGPU anvil{};
                anvil.position = {0.0f, 0.0f, 0.0f, 0.0f};
                anvil.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
                anvil.flagsAndIndices[0] = MR_MOTION_KINEMATIC;
                authoredBodies[environment * bodyCountPerEnvironment + kAnvilBodyIndex] = anvil;
            }
        }
        bodies = makeBuffer(device, authoredBodies);
        endpointBodies = makeBuffer(device, authoredBodies);
        statuses = [device newBufferWithLength:
            kEnvironmentCount * sizeof(MRMetalWorldStatusGPU)
            options:MTLResourceStorageModeShared];
        require(bodies != nil && endpointBodies != nil && statuses != nil,
                "native tooling borrowed body/status buffers unavailable");
    }

    RuntimeStateSnapshot snapshot() {
        auto value = runtime.snapshot();
        require(value.available, "native tooling snapshot: " + value.message);
        return value;
    }

    void prepareTemporaryGripTarget() {
        require(femNodeCount == 27u && kInteriorGripNode < femNodeCount,
                "temporary-grip fixture requires the manufactured 27-node pad");
        require(std::none_of(world.contact.pairs.begin(),
                             world.contact.pairs.end(),
                             [](const NMContactPairGPU& pair) {
                                 return pair.continuumNode == kInteriorGripNode;
                             }),
                "temporary grip must be on an interior node with no tool contact pair");
        femKinematicTargetValues.assign(
            kEnvironmentCount * femNodeCount, std::array<float, 4>{});
        const auto& initial = world.fem.nodes.at(kInteriorGripNode).positionAndMass;
        // The selected node lies one millimetre inside each tool-facing surface
        // in the 2 mm pad. Move it 10 um in x to prove the accepted Dirichlet
        // target position, while remaining outside all authored tool-pair rows.
        temporaryGripTarget = {initial.x + 10.0e-6f, initial.y, initial.z};
        femKinematicTargetValues[kInteriorGripNode] = {
            temporaryGripTarget[0], temporaryGripTarget[1],
            temporaryGripTarget[2], 3.0f};
        femKinematicTargets = makeBuffer(device, femKinematicTargetValues);
        require(femKinematicTargets != nil &&
                    femKinematicTargetValues.size() ==
                        kEnvironmentCount * femNodeCount &&
                    femKinematicTargets.length >=
                        femKinematicTargetValues.size() * 4u * sizeof(float),
                "temporary-grip target buffer must cover every environment and node");
    }

    void prepareEnvironmentGripTargets(
        const unsigned environment,
        const std::uint32_t movedNode = NM_INVALID_INDEX,
        const float movedNodeDeltaZ = 0.0f) {
        require(environment < kEnvironmentCount,
                "all-node grip environment is out of range");
        femKinematicTargetValues.assign(
            kEnvironmentCount * femNodeCount, std::array<float, 4>{});
        for (std::uint32_t node = 0u; node < femNodeCount; ++node) {
            const auto& initial = world.fem.nodes.at(node).positionAndMass;
            // The four authored supports are already static. Tag-3 acquisition
            // is valid only for free nodes, so leave those rows untouched.
            if (world.fem.nodes.at(node).restAndFixed.w == 1.0f) continue;
            const float deltaZ = node == movedNode ? movedNodeDeltaZ : 0.0f;
            femKinematicTargetValues[environment * femNodeCount + node] = {
                initial.x, initial.y, initial.z + deltaZ, 3.0f};
        }
        femKinematicTargets = makeBuffer(device, femKinematicTargetValues);
        require(femKinematicTargets != nil &&
                    femKinematicTargetValues.size() ==
                        kEnvironmentCount * femNodeCount &&
                    femKinematicTargets.length >=
                        femKinematicTargetValues.size() * 4u * sizeof(float),
                "all-node grip target buffer must cover every environment and node");
    }

    void prepareSingleEnvironmentGripTarget(
        const unsigned environment,
        const std::uint32_t movedNode,
        const float movedNodeDeltaZ) {
        require(environment < kEnvironmentCount && movedNode < femNodeCount,
                "single-node grip target is out of range");
        require(world.fem.nodes.at(movedNode).restAndFixed.w == 0.0f,
                "single-node grip target must acquire a free FEM node");
        femKinematicTargetValues.assign(
            kEnvironmentCount * femNodeCount, std::array<float, 4>{});
        const auto& initial = world.fem.nodes.at(movedNode).positionAndMass;
        femKinematicTargetValues[environment * femNodeCount + movedNode] = {
            initial.x, initial.y, initial.z + movedNodeDeltaZ, 3.0f};
        femKinematicTargets = makeBuffer(device, femKinematicTargetValues);
        require(femKinematicTargets != nil &&
                    femKinematicTargets.length >=
                        femKinematicTargetValues.size() * 4u * sizeof(float),
                "single-node grip target buffer must cover every environment and node");
    }

    RuntimeStateSnapshot step(const unsigned index,
                              const bool unsafeExternalEndpoint = false,
                              const bool expectedPreRejection = false,
                              const unsigned preFault = 0u,
                              const bool twoSubsteps = false,
                              const unsigned postFault = 0u) {
        const auto before = snapshot();
        auto nextBodies = authoredBodies;
        for (unsigned environment = 0u; environment < kEnvironmentCount; ++environment) {
            auto& punch = nextBodies[environment * bodyCountPerEnvironment + kPunchBodyIndex];
            punch.position.z += punch.linearVelocityAndInverseMass.z *
                static_cast<float>(runtime.timestepSeconds());
        }
        auto& firstPunch = nextBodies[kPunchBodyIndex];
        if (preFault == 1u) firstPunch.position.z -= 0.0001f;
        if (preFault == 2u) firstPunch.orientation = {0.0f, 0.0f, std::sin(0.2f), std::cos(0.2f)};
        if (preFault == 3u) firstPunch.linearVelocityAndInverseMass.w = 1.0f;
        if (includeSupportBox && preFault == 4u)
            nextBodies[kAnvilBodyIndex].position.z += 0.0001f;
        if (includeSupportBox && preFault == 5u)
            nextBodies[kAnvilBodyIndex].flagsAndIndices[0] = MR_MOTION_DYNAMIC;
        if (includeSupportBox && preFault == 6u)
            nextBodies[kAnvilBodyIndex].linearVelocityAndInverseMass.x = 0.001f;
        if (includeSupportBox && preFault == 7u)
            nextBodies[kAnvilBodyIndex].orientation = {
                0.0f, 0.0f, std::sin(0.2f), std::cos(0.2f)};
        if (includeSupportBox && preFault == 8u)
            nextBodies[kAnvilBodyIndex].angularVelocity.z = 0.001f;
        if (includeSupportBox && preFault == 9u)
            nextBodies[kAnvilBodyIndex].linearVelocityAndInverseMass.w = 0.001f;
        if (includeSupportBox && preFault == 10u)
            nextBodies[kAnvilBodyIndex].inverseInertiaWorldRow1.y = 0.001f;
        if (unsafeExternalEndpoint) {
            // Put the nose inside the authored top collision surface. A jump
            // through the entire pad would instead test swept CCD, which an
            // endpoint certificate alone cannot establish.
            firstPunch.position.z -= static_cast<float>(0.1 * kMillimetre);
            const double endNose = firstPunch.position.z - kPunchRadius;
            require(endNose < kPadCaliper &&
                    firstPunch.position.z > kPadCaliper,
                    "unsafe endpoint fixture does not intersect the pad top");
        }
        std::memcpy(endpointBodies.contents, nextBodies.data(),
                    nextBodies.size() * sizeof(MRBodyStateGPU));
        auto postBodies = nextBodies;
        if (postFault == 1u) postBodies[kPunchBodyIndex].inverseInertiaWorldRow0.x = 1.0f;
        if (postFault == 2u) postBodies[kPunchBodyIndex].flagsAndIndices[0] = MR_MOTION_DYNAMIC;
        if (includeSupportBox && postFault == 3u)
            postBodies[kAnvilBodyIndex].position.z += 0.0001f;
        if (includeSupportBox && postFault == 4u)
            postBodies[kAnvilBodyIndex].orientation = {
                0.0f, 0.0f, std::sin(0.2f), std::cos(0.2f)};
        id<MTLBuffer> plannedBodies = postFault != 0u ? makeBuffer(device, postBodies) : nil;
        auto* statusValues = static_cast<MRMetalWorldStatusGPU*>(statuses.contents);
        for (unsigned environment = 0u; environment < kEnvironmentCount; ++environment) {
            statusValues[environment] = {};
            statusValues[environment].environment = environment;
        }
        id<MTLCommandBuffer> command = [queue commandBuffer];
        require(command != nil, "native tooling command buffer unavailable");
        EncodeRequest request;
        request.commandBuffer = (__bridge void*)command;
        request.environmentStatuses = (__bridge void*)statuses;
        request.rigid.currentBodies = (__bridge void*)(unsafeExternalEndpoint ? bodies : endpointBodies);
        request.rigid.currentBodyCount = bodyCountPerEnvironment;
        request.rigid.currentBodyStride = bodyCountPerEnvironment;
        request.controlStep = twoSubsteps ? index / 2u : index;
        request.physicsSubsteps = twoSubsteps ? 2u : 1u;
        request.physicsSubstep = twoSubsteps ? index % 2u : 0u;
        request.timestepSeconds = runtime.timestepSeconds();
        request.runAdaptiveTransfer = false;
        if (femKinematicTargets != nil) {
            request.femKinematicTargets = (__bridge void*)femKinematicTargets;
            request.femKinematicTargetCount =
                kEnvironmentCount * femNodeCount;
        }
        request.phase = EncodePhase::preDynamics;
        auto encoded = runtime.encode(request);
        require(encoded.encoded, "native tooling preDynamics encode: " + encoded.message);
        // The negative case deliberately substitutes an invalid motion type
        // only after solving, exercising native postCommit rollback.
        request.rigid.currentBodies = (__bridge void*)(postFault != 0u ? plannedBodies : endpointBodies);
        request.phase = EncodePhase::postCommit;
        encoded = runtime.encode(request);
        require(encoded.encoded, "native tooling postCommit encode: " + encoded.message);
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
                "native tooling Metal command did not complete");
        const auto result = snapshot();
        require(result.statuses.size() == kEnvironmentCount,
                "native tooling status snapshot arity");
        if (unsafeExternalEndpoint || expectedPreRejection || postFault != 0u) {
            require(result.statuses[0].code != NM_STATUS_SUCCESS,
                    "invalid external endpoint was not rejected by native certification");
            require(result.statuses[1].code == NM_STATUS_SUCCESS,
                    "unsafe moving endpoint incorrectly rejected the separated control");
            const double rolledBackMotion = maximumNodeDisplacement(
                result, before, 0u, femNodeCount);
            require(rolledBackMotion <= 1.0e-9,
                    "postCommit endpoint rejection did not roll back the FEM state");
            require(sameEnvironmentBytes(result.contactHistories,
                                         before.contactHistories, 0u),
                    "rejected endpoint changed accepted contact history");
            require(sameEnvironmentBytes(result.femMaterialState,
                                         before.femMaterialState, 0u),
                    "rejected endpoint changed accepted FEM material history");
            require(sameEnvironmentBytes(result.rigidStates,
                                         before.rigidStates, 0u),
                    "rejected endpoint changed accepted capsule/BOX rigid state");
            require(std::abs(static_cast<double>(
                        static_cast<MRBodyStateGPU*>(bodies.contents)[kPunchBodyIndex].position.z) -
                    authoredBodies[kPunchBodyIndex].position.z) <= 1.0e-9,
                    "rejected external endpoint advanced the caller-owned tool pose");
            if ((world.contact.rigidProxies[0].flags & NM_RIGID_PRESCRIBED_TRANSLATION) != 0u)
                require(std::memcmp(&result.rigidStates[0], &before.rigidStates[0], sizeof(NMRigidStateGPU)) == 0,
                    "rejected prescribed tool did not restore exact prior native rigid pose");
            std::cout << std::setprecision(10)
                      << "unsafe_endpoint_regression=rejected"
                      << " status=" << result.statuses[0].code
                      << " rolledback_fem_displacement_m=" << rolledBackMotion
                      << " caller_pose_unchanged=true\n";
            return result;
        }
        for (unsigned environment = 0u; environment < kEnvironmentCount; ++environment) {
            require(result.statuses[environment].code == NM_STATUS_SUCCESS,
                    "native tooling environment " + std::to_string(environment) +
                    " rejected code=" + std::to_string(result.statuses[environment].code) +
                    " object=" + std::to_string(result.statuses[environment].objectIndex) +
                    " row=" + std::to_string(result.statuses[environment].failingIndex));
        }
        // Advance the caller-owned start pose to the exact end pose once, and
        // only after native endpoint certification accepts the transaction.
        authoredBodies = std::move(nextBodies);
        std::memcpy(bodies.contents, authoredBodies.data(),
                    authoredBodies.size() * sizeof(MRBodyStateGPU));
        return result;
    }
};

template <class T>
std::vector<T> readPrivateBuffer(id<MTLDevice> device,
                                id<MTLCommandQueue> queue,
                                void* borrowedBuffer,
                                const std::size_t count,
                                const std::string& label) {
    require(borrowedBuffer != nullptr && count > 0u,
            label + " readback source/count is invalid");
    id<MTLBuffer> source = (__bridge id<MTLBuffer>)borrowedBuffer;
    const NSUInteger byteCount = count * sizeof(T);
    require(source != nil && source.length >= byteCount,
            label + " readback source buffer is short");
    id<MTLBuffer> destination = [device newBufferWithLength:byteCount
        options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    require(destination != nil && command != nil && blit != nil,
            label + " readback allocation/encoder failed");
    [blit copyFromBuffer:source sourceOffset:0u toBuffer:destination
        destinationOffset:0u size:byteCount];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
            label + " readback blit did not complete");
    std::vector<T> values(count);
    std::memcpy(values.data(), destination.contents, byteCount);
    return values;
}

struct ContactSummary {
    std::uint32_t valid = 0u;
    double maxNormalImpulse = 0.0;
    double minSeparation = INFINITY;
};

ContactSummary contactsFor(const RuntimeStateSnapshot& state,
                           unsigned environment,
                           unsigned proxy,
                           std::uint32_t proxyCount);

void checkAcceptedConstraintReaction(
    Fixture& fixture,
    const RuntimeStateSnapshot& initial,
    const RuntimeStateSnapshot& accepted,
    const std::vector<unsigned>& requiredProxies,
    const std::string& label,
    const bool requireEmptyBulkChannel) {
    require(accepted.statuses.at(0).code == NM_STATUS_SUCCESS &&
                accepted.statuses.at(1).code == NM_STATUS_SUCCESS,
            label + " tag-3 constrained contact transaction was rejected");
    require(accepted.solverCertificates.at(0).nonlinear.x <=
                fixture.world.mixedSolver.residualTolerances.x,
            label + " tag-3 transaction missed the existing certificate");

    std::vector<std::uint32_t> constrainedNodes;
    constrainedNodes.reserve(fixture.femNodeCount);
    for (std::uint32_t node = 0u; node < fixture.femNodeCount; ++node) {
        const auto& state = accepted.femNodes.at(node);
        if (state.restAndFixed.w == 1.0f || state.restAndFixed.w == 3.0f)
            constrainedNodes.push_back(node);
    }
    require(!constrainedNodes.empty(),
            label + " reaction fixture has no prescribed FEM rows");

    const std::size_t totalNodes =
        kEnvironmentCount * static_cast<std::size_t>(fixture.femNodeCount);
    const auto contactForce = readPrivateBuffer<nm_float4>(
        fixture.device, fixture.queue,
        fixture.runtime.femConstraintContactForceBuffer(), totalNodes,
        "FEM contact support force");
    const auto bulkForce = readPrivateBuffer<nm_float4>(
        fixture.device, fixture.queue,
        fixture.runtime.femConstraintReactionBuffer(), totalNodes,
        "FEM bulk/cohesive support force");

    std::vector<std::array<double, 3>> expected(fixture.femNodeCount);
    std::vector<std::array<double, 3>> expectedProxyImpulse(
        fixture.world.contact.rigidProxies.size());
    std::vector<std::uint32_t> validProxySamples(
        fixture.world.contact.rigidProxies.size(), 0u);
    std::uint32_t validFreeNodeSamples = 0u;
    for (std::size_t pairIndex = 0u;
         pairIndex < fixture.world.contact.pairs.size(); ++pairIndex) {
        const NMContactPairGPU& pair = fixture.world.contact.pairs[pairIndex];
        if (pair.objectIndex != 0u ||
            pair.rigidProxy >= expectedProxyImpulse.size() ||
            pair.continuumNode < fixture.world.dispatch.gridNodeCount)
            continue;
        const std::uint32_t node =
            pair.continuumNode - fixture.world.dispatch.gridNodeCount;
        if (node >= fixture.femNodeCount) continue;
        const NMContactSampleGPU& sample = accepted.contactSamples.at(pairIndex);
        if ((sample.identity.w & NM_CONTACT_VALID) == 0u) continue;
        ++validProxySamples[pair.rigidProxy];
        if (accepted.femNodes.at(node).restAndFixed.w == 0.0f &&
            accepted.femNodes.at(node).positionAndMass.w > 0.0f)
            ++validFreeNodeSamples;
        for (unsigned axis = 0u; axis < 3u; ++axis) {
            const float impulse = axis == 0u ? sample.impulseAndNormal.x
                : axis == 1u ? sample.impulseAndNormal.y
                             : sample.impulseAndNormal.z;
            expected[node][axis] -= static_cast<double>(impulse) /
                                    fixture.timestep;
            expectedProxyImpulse[pair.rigidProxy][axis] += impulse;
        }
    }
    for (const unsigned proxyIndex : requiredProxies) {
        require(proxyIndex < validProxySamples.size() &&
                    validProxySamples[proxyIndex] > 0u,
                label + " produced no accepted sample for required rigid proxy " +
                    std::to_string(proxyIndex));
        const double normalImpulse = proxyIndex < fixture.world.contact.rigidProxies.size()
            ? contactsFor(accepted, 0u, proxyIndex,
                          static_cast<std::uint32_t>(fixture.world.contact.rigidProxies.size()))
                  .maxNormalImpulse
            : 0.0;
        require(normalImpulse > 0.0,
                label + " required rigid proxy has no positive native normal impulse");
    }

    double maximumNodeForceError = 0.0;
    double freeSupplementMaximum = 0.0;
    double bulkForceMagnitude = 0.0;
    for (std::uint32_t node = 0u; node < fixture.femNodeCount; ++node) {
        const nm_float4& actual = contactForce.at(node);
        const nm_float4& bulk = bulkForce.at(node);
        const double values[3] = {actual.x, actual.y, actual.z};
        const double bulkValues[3] = {bulk.x, bulk.y, bulk.z};
        const bool constrained =
            accepted.femNodes.at(node).restAndFixed.w == 1.0f ||
            accepted.femNodes.at(node).restAndFixed.w == 3.0f ||
            accepted.femNodes.at(node).positionAndMass.w <= 0.0f;
        for (unsigned axis = 0u; axis < 3u; ++axis) {
            if (constrained) {
                maximumNodeForceError = std::max(maximumNodeForceError,
                    std::abs(values[axis] - expected[node][axis]));
            } else {
                // The public supplement is specifically the support force on
                // prescribed/massless rows. Contact on a free FEM row remains
                // in the production residual and must not leak into it.
                freeSupplementMaximum = std::max(freeSupplementMaximum,
                    std::abs(values[axis]));
            }
            bulkForceMagnitude = std::max(bulkForceMagnitude,
                std::abs(bulkValues[axis]));
        }
    }
    double sampleFemContactForce[3] = {0.0, 0.0, 0.0};
    for (const auto& force : expected)
        for (unsigned axis = 0u; axis < 3u; ++axis)
            sampleFemContactForce[axis] += force[axis];
    require(maximumNodeForceError <= 2.0e-3,
            "native constrained contact supplement differs from constrained per-node proxy impulses");
    require(freeSupplementMaximum <= 2.0e-3,
            "free FEM contact force leaked into the constrained supplement");
    if (requireEmptyBulkChannel)
        require(bulkForceMagnitude <= 2.0e-3,
                label + " stationary fixture unexpectedly folded contact into bulk channel");
    double actionReactionError = 0.0;
    for (unsigned axis = 0u; axis < 3u; ++axis) {
        double proxyImpulse = 0.0;
        double sampleProxyImpulse = 0.0;
        for (std::size_t proxyIndex = 0u;
             proxyIndex < fixture.world.contact.rigidProxies.size(); ++proxyIndex) {
            const std::size_t reactionIndex = proxyIndex;
            require(reactionIndex < accepted.reactions.size(),
                    label + " reaction snapshot lacks environment-zero proxy record");
            const auto& reaction = accepted.reactions[reactionIndex];
            const double value = axis == 0u ? reaction.impulseAndCount.x
                : axis == 1u ? reaction.impulseAndCount.y
                             : reaction.impulseAndCount.z;
            proxyImpulse += value;
            sampleProxyImpulse += expectedProxyImpulse[proxyIndex][axis];
        }
        actionReactionError = std::max(actionReactionError,
            std::abs(sampleFemContactForce[axis] * fixture.timestep + proxyImpulse));
        actionReactionError = std::max(actionReactionError,
            std::abs(proxyImpulse - sampleProxyImpulse));
    }
    require(actionReactionError <= 2.0e-7,
            label + " accepted sampled FEM/proxy impulses do not balance under dt/sign");
    require(validFreeNodeSamples > 0u,
            label + " did not exercise a nonzero-mass free FEM contact row");
    for (std::uint32_t node = 0u; node < fixture.femNodeCount; ++node) {
        const nm_float4& control = contactForce.at(
            fixture.femNodeCount + node);
        require(std::abs(control.x) <= 1.0e-8f &&
                    std::abs(control.y) <= 1.0e-8f &&
                    std::abs(control.z) <= 1.0e-8f,
                label + " separated environment acquired a rigid reaction");
        const auto& before = initial.femNodes.at(
            fixture.femNodeCount + node).positionAndMass;
        const auto& after = accepted.femNodes.at(
            fixture.femNodeCount + node).positionAndMass;
        require(std::abs(after.x - before.x) <= 1.0e-9f &&
                    std::abs(after.y - before.y) <= 1.0e-9f &&
                    std::abs(after.z - before.z) <= 1.0e-9f,
                label + " separated environment moved during the tag-3 reaction check");
    }
    std::cout << std::setprecision(10)
              << "native_" << label << "_constraint_reaction=pass accepted=true"
              << " constrained_nodes=" << constrainedNodes.size()
              << " free_contact_samples=" << validFreeNodeSamples
              << " active_proxy_samples=";
    for (std::size_t i = 0u; i < validProxySamples.size(); ++i) {
        if (i != 0u) std::cout << ",";
        std::cout << validProxySamples[i];
    }
    std::cout
              << " max_per_node_force_error_N=" << maximumNodeForceError
              << " free_row_supplement_max_N=" << freeSupplementMaximum
              << " action_reaction_error_Ns=" << actionReactionError
              << " bulk_channel_max_N=" << bulkForceMagnitude
              << " reaction_qualification=free_contact_action_reaction_only"
              << " physical_validation=false\n";
}

void runAcceptedCapsuleConstraintReactionCase() {
    // Keep the actual free FEM pad: contact mechanics samples free positive-
    // inverse-mass nodes, while the readback supplement is reserved for
    // prescribed/massless rows. An all-tag3 case is recorded separately below
    // as unsupported for nonzero rigid-reaction qualification.
    constexpr double punchSpeed = 0.2;
    Fixture fixture(true, punchSpeed, true);
    const double initialGap =
        static_cast<double>(fixture.authoredBodies[kPunchBodyIndex].position.z) -
        kPunchRadius - kPadCaliper;
    const double endpointGap = initialGap - punchSpeed * fixture.timestep;
    require(initialGap > 0.0 && endpointGap < 0.0,
            "free-pad capsule fixture must advance from clearance into contact");
    const auto initial = fixture.snapshot();
    const auto accepted = fixture.step(0u);
    checkAcceptedConstraintReaction(
        fixture, initial, accepted, {0u}, "capsule", false);
    std::cout << std::setprecision(10)
              << "capsule_nominal_endpoint_gap_m=" << endpointGap
              << " accepted_min_separation_m="
              << contactsFor(accepted, 0u, 0u, 1u).minSeparation
              << " contact_slop_m=" << kContactSlop << '\n';
}

void runAllTag3ReactionLimitationCase() {
    constexpr double punchSpeed = 0.05;
    Fixture fixture(true, punchSpeed, false);
    const double initialGap =
        static_cast<double>(fixture.authoredBodies[kPunchBodyIndex].position.z) -
        kPunchRadius - kPadCaliper;
    const double endpointGap = initialGap - punchSpeed * fixture.timestep;
    require(initialGap > 0.0 && endpointGap > 0.0 &&
                endpointGap <= kContactSlop,
            "all-tag3 limitation geometry must end separated by 5 um within slop");
    fixture.prepareEnvironmentGripTargets(0u);
    const auto accepted = fixture.step(0u);
    require(accepted.statuses[0].code == NM_STATUS_SUCCESS &&
                accepted.statuses[1].code == NM_STATUS_SUCCESS,
            "all-tag3 separated geometry should remain an admissible prescribed state");
    std::uint32_t tag3Rows = 0u;
    for (std::uint32_t node = 0u; node < fixture.femNodeCount; ++node) {
        const auto& state = accepted.femNodes.at(node);
        if (state.restAndFixed.w == 3.0f) {
            ++tag3Rows;
            require(state.velocityAndInverseMass.w == 0.0f,
                    "all-tag3 row retained nonzero inverse mass");
        } else {
            require(state.restAndFixed.w == 1.0f,
                    "all-tag3 fixture left an untagged free row");
        }
    }
    require(tag3Rows > 0u && tag3Rows + 4u == fixture.femNodeCount,
            "all-tag3 fixture did not acquire every free FEM node");
    const ContactSummary contact = contactsFor(accepted, 0u, 0u, 1u);
    require(contact.valid == 0u,
            "all-tag3 zero-inverse-mass geometry unexpectedly produced rigid contact impulse");
    const auto contactForce = readPrivateBuffer<nm_float4>(
        fixture.device, fixture.queue,
        fixture.runtime.femConstraintContactForceBuffer(),
        kEnvironmentCount * static_cast<std::size_t>(fixture.femNodeCount),
        "all-tag3 contact support force");
    double supplementMaximum = 0.0;
    for (std::uint32_t node = 0u; node < fixture.femNodeCount; ++node) {
        const auto& value = contactForce[node];
        supplementMaximum = std::max(supplementMaximum,
            std::max({std::abs(static_cast<double>(value.x)),
                      std::abs(static_cast<double>(value.y)),
                      std::abs(static_cast<double>(value.z))}));
    }
    require(supplementMaximum == 0.0,
            "all-tag3 geometry unexpectedly published a contact reaction");
    std::cout << std::setprecision(10)
              << "all_tag3_rigid_contact=unsupported_absent"
              << " accepted=true initial_gap_m=" << initialGap
              << " endpoint_gap_m=" << endpointGap
              << " tag3_rows=" << tag3Rows
              << " rigid_samples=" << contact.valid
              << " contact_supplement_max_N=" << supplementMaximum
              << " reason=zero_inverse_mass_contact_rows_skipped"
              << " unsupported_for_nonzero_reaction_readback=true"
              << " physical_reaction_qualification=false\n";
}

void runAcceptedBoxConstraintReactionCase() {
    // The support face is 10 um below the board. A single bottom-center tag-3
    // target moves 5 um toward it during the accepted step, creating a bounded
    // native BOX contact without penetrating the source surface.
    constexpr double supportSurfaceZ = -10.0e-6;
    constexpr std::uint32_t bottomCenterNode = 4u;
    Fixture fixture(true, 0.2, false, true, supportSurfaceZ);
    fixture.prepareSingleEnvironmentGripTarget(
        0u, bottomCenterNode, -5.0e-6f);
    const auto initial = fixture.snapshot();
    const auto accepted = fixture.step(0u);
    checkAcceptedConstraintReaction(
        fixture, initial, accepted, {0u, 1u}, "capsule_box", false);
}

std::vector<nm_float4> dispatchDeformableReactionPublisher(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> pipeline,
    const NMMatterDispatchGPU& dispatch,
    const std::vector<NMFEMNodeStateGPU>& nodes,
    const std::vector<NMDeformableContactGPU>& contacts) {
    require(nodes.size() == dispatch.femNodeCount &&
                contacts.size() == dispatch.deformableContactCapacity,
            "synthetic deformable reaction fixture buffer sizes disagree with dispatch");
    std::vector<NMContactPairGPU> rigidPairs(1u);
    std::vector<std::uint32_t> rigidIncidence(1u, 0u);
    std::vector<NMIncidenceRangeGPU> rigidRanges(dispatch.femNodeCount);
    std::vector<NMContactSampleGPU> rigidSamples(1u);
    std::vector<std::uint32_t> deformableIncidence(dispatch.femNodeCount);
    std::vector<NMIncidenceRangeGPU> deformableRanges(dispatch.femNodeCount);
    std::vector<NMSchedulerStateGPU> schedulers(dispatch.objectCount);
    std::vector<NMMatterStatusGPU> statuses(dispatch.environmentCount);
    std::vector<nm_float4> zeroForces(dispatch.environmentCount * dispatch.femNodeCount);
    for (std::uint32_t node = 0u; node < dispatch.femNodeCount; ++node) {
        deformableIncidence[node] = node; // contact slot 0, corresponding local node
        deformableRanges[node] = {node, 1u, 0u, 0u};
    }

    std::array<id<MTLBuffer>, 12> buffers = {
        makeBuffer(device, std::vector<NMMatterDispatchGPU>{dispatch}),
        makeBuffer(device, nodes),
        makeBuffer(device, rigidPairs),
        makeBuffer(device, rigidIncidence),
        makeBuffer(device, rigidRanges),
        makeBuffer(device, rigidSamples),
        makeBuffer(device, contacts),
        makeBuffer(device, deformableIncidence),
        makeBuffer(device, deformableRanges),
        makeBuffer(device, schedulers),
        makeBuffer(device, statuses),
        makeBuffer(device, zeroForces),
    };
    for (std::size_t index = 0u; index < buffers.size(); ++index)
        require(buffers[index] != nil,
                "synthetic deformable reaction fixture Metal buffer allocation failed at " +
                    std::to_string(index));
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(command != nil && encoder != nil,
            "synthetic deformable reaction fixture command encoder unavailable");
    [encoder setComputePipelineState:pipeline];
    for (NSUInteger index = 0u; index < buffers.size(); ++index)
        [encoder setBuffer:buffers[index] offset:0u atIndex:index];
    [encoder dispatchThreads:MTLSizeMake(dispatch.femNodeCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(std::min<std::uint32_t>(
            dispatch.femNodeCount, 32u), 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
            "synthetic deformable reaction fixture kernel did not complete");
    const auto* output = static_cast<const nm_float4*>(buffers[11].contents);
    return std::vector<nm_float4>(output,
        output + dispatch.environmentCount * dispatch.femNodeCount);
}

void runNativeDeformableConstraintReactionKernelCase() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    require(device != nil, "native deformable reaction fixture requires Metal");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    require(queue != nil, "native deformable reaction fixture command queue unavailable");
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:
        [NSURL fileURLWithPath:@NUMI_MATTER_METALLIB] error:&error];
    require(library != nil,
            "native deformable reaction fixture could not load Matter metallib");
    id<MTLFunction> function = [library newFunctionWithName:
        @"numi_matter_metal::nm_contact_publish_fem_constraint_contact_force"];
    require(function != nil,
            "native deformable constraint reaction publisher is absent from metallib");
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:
        function error:&error];
    require(pipeline != nil,
            "native deformable constraint reaction publisher pipeline failed");

    constexpr float dt = 0.02f;
    constexpr float mollifierEnergyScale = 0.25f;
    const std::array<float, 4> weights = {1.0f, -0.2f, -0.3f, -0.5f};
    const std::array<std::array<float, 3>, 4> mollifierGradients = {{
        {{0.0010f, -0.0020f, 0.0005f}},
        {{-0.0004f, 0.0010f, -0.0002f}},
        {{-0.0003f, 0.0005f, -0.0001f}},
        {{-0.0003f, 0.0005f, -0.0002f}},
    }};
    const std::array<double, 3> contactImpulse = {0.001, 0.002, 0.007};

    NMMatterDispatchGPU dispatch{};
    dispatch.environmentCount = 1u;
    dispatch.objectCount = 1u;
    dispatch.femNodeCount = 4u;
    dispatch.deformableContactCapacity = 1u;
    dispatch.gravityAndTimestep.w = dt;
    std::vector<NMFEMNodeStateGPU> nodes(dispatch.femNodeCount);
    for (std::uint32_t node = 0u; node < dispatch.femNodeCount; ++node) {
        nodes[node].positionAndMass = {static_cast<float>(node), 0.0f, 0.0f, 1.0f};
        nodes[node].restAndFixed = {0.0f, 0.0f, 0.0f, 3.0f};
    }
    NMDeformableContactGPU contact{};
    contact.nodes = {0u, 1u, 2u, 3u};
    contact.identity = {0u, 0u, NM_DEFORMABLE_CONTACT_VERTEX_TRIANGLE,
                        NM_CONTACT_VALID};
    contact.weights = {weights[0], weights[1], weights[2], weights[3]};
    contact.timeScales = {1.0f, 1.0f, 1.0f, 1.0f};
    contact.normalAndTOI = {0.0f, 0.0f, 1.0f, 0.5f};
    // For +Z, the production frame is t0=+Y and t1=-X.
    contact.impulseAndFriction = {0.007f, 0.002f, -0.001f, 0.0f};
    contact.mollifier = {1.0f, mollifierEnergyScale, 0.0f, 1.0f};
    contact.mollifierGradientRow0 = {mollifierGradients[0][0],
        mollifierGradients[0][1], mollifierGradients[0][2], 0.0f};
    contact.mollifierGradientRow1 = {mollifierGradients[1][0],
        mollifierGradients[1][1], mollifierGradients[1][2], 0.0f};
    contact.mollifierGradientRow2 = {mollifierGradients[2][0],
        mollifierGradients[2][1], mollifierGradients[2][2], 0.0f};
    contact.mollifierGradientRow3 = {mollifierGradients[3][0],
        mollifierGradients[3][1], mollifierGradients[3][2], 0.0f};
    contact.extraNodes = {NM_INVALID_INDEX, NM_INVALID_INDEX,
                          NM_INVALID_INDEX, NM_INVALID_INDEX};
    const std::vector<NMDeformableContactGPU> contacts = {contact};

    const auto forces = dispatchDeformableReactionPublisher(
        device, queue, pipeline, dispatch, nodes, contacts);
    double maximumError = 0.0;
    std::array<double, 3> netForce{};
    for (std::size_t local = 0u; local < 4u; ++local) {
        const double expected[3] = {
            (weights[local] * contactImpulse[0] - mollifierEnergyScale *
                mollifierGradients[local][0]) / dt,
            (weights[local] * contactImpulse[1] - mollifierEnergyScale *
                mollifierGradients[local][1]) / dt,
            (weights[local] * contactImpulse[2] - mollifierEnergyScale *
                mollifierGradients[local][2]) / dt,
        };
        const double actual[3] = {
            forces[local].x, forces[local].y, forces[local].z};
        for (unsigned axis = 0u; axis < 3u; ++axis) {
            maximumError = std::max(maximumError,
                std::abs(actual[axis] - expected[axis]));
            netForce[axis] += actual[axis];
        }
        require(forces[local].w == 0.0f,
                "deformable reaction publisher populated the reserved force component");
    }
    double netForceMagnitude = 0.0;
    for (const double component : netForce)
        netForceMagnitude = std::max(netForceMagnitude, std::abs(component));
    require(maximumError <= 2.0e-5,
            "native deformable reaction differs from mapped-weight/mollifier residual formula");
    require(netForceMagnitude <= 2.0e-5,
            "native deformable mapped contact reaction violates total force balance");

    // Human tag 2 belongs to the separate Human J^T reaction channel. Setting
    // its inverse mass to zero reaches the explicit tag-2 exclusion in this
    // publisher rather than the ordinary free-node filter.
    nodes[0].restAndFixed.w = 2.0f;
    nodes[0].positionAndMass.w = 0.0f;
    const auto humanFiltered = dispatchDeformableReactionPublisher(
        device, queue, pipeline, dispatch, nodes, contacts);
    require(humanFiltered[0].x == 0.0f && humanFiltered[0].y == 0.0f &&
                humanFiltered[0].z == 0.0f,
            "Human tag-2 row leaked into the FEM constrained contact reaction channel");
    for (std::size_t local = 1u; local < 4u; ++local) {
        require(std::abs(humanFiltered[local].x - forces[local].x) <= 1.0e-7f &&
                    std::abs(humanFiltered[local].y - forces[local].y) <= 1.0e-7f &&
                    std::abs(humanFiltered[local].z - forces[local].z) <= 1.0e-7f,
                "Human tag-2 filtering changed unrelated deformable contact rows");
    }
    std::cout << std::setprecision(10)
              << "native_deformable_constraint_reaction=pass synthetic_kernel=true"
              << " max_mapped_force_error_N=" << maximumError
              << " net_force_error_N=" << netForceMagnitude
              << " human_tag2_excluded=true physical_validation=false\n";
}

ContactSummary contactsFor(const RuntimeStateSnapshot& state,
                           const unsigned environment,
                           const unsigned proxy,
                           const std::uint32_t proxyCount) {
    ContactSummary summary;
    const std::size_t capacity = state.contactSamples.size() / kEnvironmentCount;
    require(capacity * kEnvironmentCount == state.contactSamples.size(),
            "contact diagnostic array is not environment-major");
    for (std::size_t row = 0u; row < capacity; ++row) {
        const auto& sample = state.contactSamples[environment * capacity + row];
        if ((sample.identity.w & NM_CONTACT_VALID) == 0u ||
            sample.identity.y != proxy)
            continue;
        require(sample.identity.y < proxyCount,
                "contact sample references an unknown rigid proxy");
        ++summary.valid;
        summary.maxNormalImpulse = std::max(
            summary.maxNormalImpulse,
            static_cast<double>(sample.impulseAndNormal.w));
        summary.minSeparation = std::min(
            summary.minSeparation,
            static_cast<double>(sample.pointAndSeparation.w));
    }
    return summary;
}

double maximumNodeDisplacement(const RuntimeStateSnapshot& state,
                               const RuntimeStateSnapshot& initial,
                               const unsigned environment,
                               const std::uint32_t nodeCount) {
    double maximum = 0.0;
    const std::size_t base = static_cast<std::size_t>(environment) * nodeCount;
    for (std::uint32_t node = 0u; node < nodeCount; ++node) {
        const auto& a = state.femNodes[base + node].positionAndMass;
        const auto& b = initial.femNodes[base + node].positionAndMass;
        const double dx = static_cast<double>(a.x) - b.x;
        const double dy = static_cast<double>(a.y) - b.y;
        const double dz = static_cast<double>(a.z) - b.z;
        maximum = std::max(maximum, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    return maximum;
}

class StderrCapture {
public:
    StderrCapture() : file_(std::tmpfile()), saved_(::dup(STDERR_FILENO)) {
        require(file_ != nullptr && saved_ >= 0, "stderr capture setup failed");
        require(::dup2(::fileno(file_), STDERR_FILENO) >= 0,
                "stderr capture redirect failed");
    }
    StderrCapture(const StderrCapture&) = delete;
    StderrCapture& operator=(const StderrCapture&) = delete;
    ~StderrCapture() {
        if (saved_ >= 0) {
            std::fflush(stderr);
            (void)::dup2(saved_, STDERR_FILENO);
            (void)::close(saved_);
        }
        if (file_ != nullptr) std::fclose(file_);
    }
    std::string finish() {
        std::fflush(stderr);
        require(::dup2(saved_, STDERR_FILENO) >= 0,
                "stderr capture restore failed");
        (void)::close(saved_);
        saved_ = -1;
        std::rewind(file_);
        std::string text;
        char buffer[4096];
        for (;;) {
            const std::size_t count = std::fread(buffer, 1u, sizeof(buffer), file_);
            text.append(buffer, count);
            if (count != sizeof(buffer)) break;
        }
        return text;
    }
private:
    FILE* file_ = nullptr;
    int saved_ = -1;
};

bool sameBytes(const auto& a, const auto& b) {
    return a.size() == b.size() &&
        (a.empty() || std::memcmp(a.data(), b.data(),
                                  a.size() * sizeof(a.front())) == 0);
}

template <class A, class B>
bool sameEnvironmentBytes(const A& a, const B& b,
                          const unsigned environment) {
    if (a.size() != b.size() || a.size() % kEnvironmentCount != 0u)
        return false;
    const std::size_t count = a.size() / kEnvironmentCount;
    return count == 0u || std::memcmp(a.data() + environment * count,
        b.data() + environment * count, count * sizeof(a.front())) == 0;
}

struct CaseResult {
    std::uint32_t contactPairs = 0u;
    double displacement = 0.0;
    double impulse = 0.0;
    double residual = 0.0;
    RuntimeStateSnapshot state;
};

CaseResult runPhysicalCase(const bool armijo) {
    Fixture fixture(true, 0.2, armijo);
    const auto initial = fixture.snapshot();
    require(fixture.world.dispatch.environmentCount == kEnvironmentCount &&
                fixture.world.dispatch.contactPairCount > 0u &&
                fixture.world.contact.rigidProxies.size() == 1u,
            "manufactured capsule-contact source did not compile to expected dimensions");
    require(std::all_of(fixture.world.contact.pairs.begin(),
                        fixture.world.contact.pairs.end(),
                        [](const NMContactPairGPU& pair) {
                            return pair.objectIndex == 0u && pair.rigidProxy == 0u;
                        }),
            "fixture included non-FEM or non-capsule source contact pairs");
    const double perStepTravel = 0.2 * fixture.timestep;
    require(perStepTravel > 0.0 && perStepTravel < 0.1 * kMillimetre,
            "manufactured tool motion must remain below the capsule radius scale");
    std::string selectorLog;
    RuntimeStateSnapshot accepted;
    if (armijo) {
        StderrCapture capture;
        accepted = fixture.step(0u);
        selectorLog = capture.finish();
        require(selectorLog.find("fem_capsule_residual_armijo_enabled=") != std::string::npos,
                "native selector did not admit the capsule-only fixture: " + selectorLog);
        require(selectorLog.find("fem_capsule_residual_armijo_skipped=") == std::string::npos,
                "native capsule Armijo selector skipped the manufactured fixture: " + selectorLog);
    } else {
        StderrCapture capture;
        accepted = fixture.step(0u);
        selectorLog = capture.finish();
        require(selectorLog.find("fem_capsule_residual_armijo_enabled=") == std::string::npos,
                "legacy case unexpectedly enabled residual Armijo");
    }
    require(accepted.statuses.at(0).code == NM_STATUS_SUCCESS &&
                accepted.statuses.at(1).code == NM_STATUS_SUCCESS,
            "manufactured source capsule contact step was not accepted");
    const auto contact = contactsFor(accepted, 0u, 0u, 1u);
    require(contact.valid > 0u && contact.maxNormalImpulse > 0.0,
            "native source capsule produced no accepted contact impulse");
    require(accepted.solverCertificates.at(0).nonlinear.x <=
                fixture.world.mixedSolver.residualTolerances.x,
            "accepted capsule response is outside the unchanged native residual certificate");
    const double displacement = maximumNodeDisplacement(
        accepted, initial, 0u, fixture.femNodeCount);
    require(displacement > 0.0,
            "capsule source contact did not change the accepted FEM state");
    require(maximumNodeDisplacement(accepted, initial, 1u, fixture.femNodeCount) <= 1.0e-9,
            "spatially separated FEM control changed despite no capsule contact");
    return {fixture.world.dispatch.contactPairCount, displacement,
            contact.maxNormalImpulse, accepted.solverCertificates.at(0).nonlinear.x,
            std::move(accepted)};
}

void rollbackAfterArmijoCase() {
    Fixture fixture(true, 0.2, true);
    const auto accepted = fixture.step(0u);
    require(accepted.statuses.at(0).code == NM_STATUS_SUCCESS,
            "precondition for post-solve rollback did not accept the physical capsule step");
    const auto beforeFailure = fixture.snapshot();
    require(!beforeFailure.contactHistories.empty(),
            "runtime snapshot omitted capsule contact history needed by rollback check");
    setenv("NM_FEM_CAPSULE_RESIDUAL_ARMIJO_ROOT", "1", 1);
    StderrCapture capture;
    const auto rejected = fixture.step(1u, false, false, 0u, false, 2u);
    const std::string selectorLog = capture.finish();
    require(selectorLog.find("fem_capsule_residual_armijo_enabled=") != std::string::npos,
            "post-solve rollback attempt did not enter capsule Armijo path: " + selectorLog);
    require(rejected.statuses.at(0).code != NM_STATUS_SUCCESS,
            "postCommit motion-type mutation was not rejected");
    require(rejected.statuses.at(1).code == NM_STATUS_SUCCESS,
            "environment-isolated postCommit fault rejected the control environment");
    const auto afterFailure = fixture.snapshot();
    // Only environment zero is rejected. Environment one's successful step
    // may publish new solver metadata, so compare the rejected environment's
    // complete native state rather than requiring the successful control to
    // remain byte-identical to its previous step.
    require(sameEnvironmentBytes(beforeFailure.femNodes, afterFailure.femNodes, 0u),
            "rejected environment changed accepted FEM nodes");
    require(sameEnvironmentBytes(beforeFailure.contactHistories, afterFailure.contactHistories, 0u),
            "rejected environment changed accepted capsule contact history");
    require(sameEnvironmentBytes(beforeFailure.femMaterialState,
                                 afterFailure.femMaterialState, 0u),
            "rejected environment changed accepted FEM material history");
    require(sameEnvironmentBytes(beforeFailure.rigidStates, afterFailure.rigidStates, 0u),
            "rejected environment changed accepted prescribed rigid state");
    require(maximumNodeDisplacement(afterFailure, beforeFailure, 1u,
                                    fixture.femNodeCount) <= 1.0e-9,
            "separated successful control acquired motion during rejection");
    std::cout << "capsule_armijo_rollback=pass postcommit_rejected=true "
              << "accepted_fem_history_rigid_state_unchanged=true\n";
}

void runStaticBoxArmijoPair() {
    Fixture legacyFixture(true, 0.2, false, true);
    const auto legacyInitial = legacyFixture.snapshot();
    require(legacyFixture.world.contact.rigidProxies.size() == 2u &&
                legacyFixture.world.contact.rigidProxies[0].shapeKind == NM_RIGID_CAPSULE &&
                legacyFixture.world.contact.rigidProxies[1].shapeKind == NM_RIGID_BOX,
            "static-BOX fixture source layout mismatch");
    require(std::all_of(legacyFixture.world.contact.pairs.begin(),
                        legacyFixture.world.contact.pairs.end(),
                        [](const NMContactPairGPU& pair) {
                            return pair.objectIndex == 0u && pair.rigidProxy < 2u;
                        }),
            "BOX fixture includes unsupported contact ownership");
    require(std::any_of(legacyFixture.world.contact.pairs.begin(),
                        legacyFixture.world.contact.pairs.end(),
                        [](const NMContactPairGPU& pair) { return pair.rigidProxy == 1u; }),
            "BOX fixture did not cook any FEM-to-anvil candidate pairs");
    const auto legacy = legacyFixture.step(0u);
    require(legacy.statuses[0].code == NM_STATUS_SUCCESS &&
                legacy.statuses[1].code == NM_STATUS_SUCCESS &&
                maximumNodeDisplacement(legacy, legacyInitial, 0u,
                    legacyFixture.femNodeCount) > 0.0,
            "legacy static-BOX control did not accept prescribed capsule contact");
    const double legacyDisplacement = maximumNodeDisplacement(
        legacy, legacyInitial, 0u, legacyFixture.femNodeCount);
    const auto legacyContact = contactsFor(legacy, 0u, 0u, 2u);
    require(legacyContact.valid > 0u && legacyContact.maxNormalImpulse > 0.0,
            "legacy static-BOX control produced no capsule contact impulse");
    require(legacy.solverCertificates.at(0).nonlinear.x <=
                legacyFixture.world.mixedSolver.residualTolerances.x,
            "legacy static-BOX control missed the native nonlinear certificate");

    Fixture armijoFixture(true, 0.2, true, true);
    const auto initial = armijoFixture.snapshot();
    std::string selectorLog;
    RuntimeStateSnapshot accepted;
    {
        setenv("NM_FEM_CAPSULE_BOX_RESIDUAL_ARMIJO_ROOT", "all", 1);
        StderrCapture capture;
        accepted = armijoFixture.step(0u);
        selectorLog = capture.finish();
    }
    require(selectorLog.find("fem_capsule_box_residual_armijo_enabled=") != std::string::npos,
            "static-BOX selector did not admit stationary body endpoint: " + selectorLog);
    require(selectorLog.find("fem_capsule_box_residual_armijo_skipped=") == std::string::npos,
            "static-BOX selector unexpectedly skipped its manufactured fixture: " + selectorLog);
    require(accepted.statuses[0].code == NM_STATUS_SUCCESS &&
                accepted.statuses[1].code == NM_STATUS_SUCCESS,
            "native prescribed capsule + static BOX step was rejected");
    const auto contact = contactsFor(accepted, 0u, 0u, 2u);
    require(contact.valid > 0u && contact.maxNormalImpulse > 0.0,
            "static-BOX fixture produced no accepted capsule contact impulse");
    const double armijoDisplacement = maximumNodeDisplacement(
        accepted, initial, 0u, armijoFixture.femNodeCount);
    require(accepted.solverCertificates.at(0).nonlinear.x <=
                armijoFixture.world.mixedSolver.residualTolerances.x,
            "static-BOX accepted response bypassed the unchanged nonlinear certificate");
    require(armijoDisplacement > 0.0 &&
            maximumNodeDisplacement(accepted, initial, 1u,
                armijoFixture.femNodeCount) <= 1.0e-9,
            "static-BOX physical/control response was not environment-local");
    require(sameEnvironmentBytes(legacy.rigidStates, accepted.rigidStates, 1u),
            "legacy and Armijo BOX fixtures disagreed on the stationary separated rigid records");
    const auto& anvilState = accepted.rigidStates.at(1u);
    require(anvilState.centerAndRadius.z ==
                legacyFixture.world.contact.rigidProxies[1].localCenterAndRadius.z &&
            anvilState.linearVelocityAndInverseMass.w == 0.0f,
            "accepted BOX endpoint did not retain the authored static transform");
    std::cout << "capsule_static_box_armijo=pass contact_pairs="
              << armijoFixture.world.dispatch.contactPairCount
              << std::setprecision(10)
              << " legacy_max_displacement_m=" << legacyDisplacement
              << " armijo_max_displacement_m=" << armijoDisplacement
              << " legacy_capsule_impulse_Ns=" << legacyContact.maxNormalImpulse
              << " armijo_capsule_impulse_Ns=" << contact.maxNormalImpulse
              << " legacy_residual=" << legacy.solverCertificates.at(0).nonlinear.x
              << " armijo_residual=" << accepted.solverCertificates.at(0).nonlinear.x
              << " accepted_capsule_contact=true stationary_box_endpoint=true\n";
}

void runStaticBoxTemporaryGripCase() {
    Fixture fixture(true, 0.2, true, true);
    fixture.prepareTemporaryGripTarget();
    const auto initial = fixture.snapshot();
    setenv("NM_FEM_CAPSULE_BOX_RESIDUAL_ARMIJO_ROOT", "all", 1);
    std::string selectorLog;
    RuntimeStateSnapshot accepted;
    {
        StderrCapture capture;
        accepted = fixture.step(0u);
        selectorLog = capture.finish();
    }
    require(selectorLog.find("fem_capsule_box_residual_armijo_enabled=") !=
                std::string::npos &&
                selectorLog.find("fem_capsule_box_residual_armijo_skipped=") ==
                std::string::npos,
            "BOX Armijo did not admit exact-count temporary-grip targets: " + selectorLog);
    require(accepted.statuses[0].code == NM_STATUS_SUCCESS &&
                accepted.statuses[1].code == NM_STATUS_SUCCESS,
            "BOX temporary-grip step did not accept both environments");
    require(accepted.solverCertificates.at(0).nonlinear.x <=
                fixture.world.mixedSolver.residualTolerances.x,
            "BOX temporary-grip response missed the unchanged nonlinear certificate");

    const auto& grip = accepted.femNodes.at(kInteriorGripNode);
    const auto& ungrippedControl = accepted.femNodes.at(
        fixture.femNodeCount + kInteriorGripNode);
    require(grip.restAndFixed.w == 3.0f &&
                grip.velocityAndInverseMass.w == 0.0f,
            "accepted target did not acquire a zero-inverse-mass temporary grip");
    require(std::abs(grip.positionAndMass.x - fixture.temporaryGripTarget[0]) <= 2.0e-7f &&
                std::abs(grip.positionAndMass.y - fixture.temporaryGripTarget[1]) <= 2.0e-7f &&
                std::abs(grip.positionAndMass.z - fixture.temporaryGripTarget[2]) <= 2.0e-7f &&
                grip.positionAndMass.w == initial.femNodes.at(
                    kInteriorGripNode).positionAndMass.w,
            "accepted temporary grip position or nodal mass differs from its authored target");
    require(ungrippedControl.restAndFixed.w ==
                initial.femNodes.at(fixture.femNodeCount +
                    kInteriorGripNode).restAndFixed.w &&
                ungrippedControl.velocityAndInverseMass.w ==
                initial.femNodes.at(fixture.femNodeCount + kInteriorGripNode)
                    .velocityAndInverseMass.w,
            "separated environment acquired the other environment's temporary grip");

    const auto beforeFailure = fixture.snapshot();
    setenv("NM_FEM_CAPSULE_BOX_RESIDUAL_ARMIJO_ROOT", "all", 1);
    RuntimeStateSnapshot rejected;
    {
        StderrCapture capture;
        rejected = fixture.step(1u, false, false, 0u, false, 2u);
        selectorLog = capture.finish();
    }
    require(selectorLog.find("fem_capsule_box_residual_armijo_enabled=") !=
                std::string::npos &&
                selectorLog.find("\"root\":1") != std::string::npos,
            "BOX Armijo selector all did not exercise the second target/control step: " +
                selectorLog);
    require(rejected.statuses.at(0).code != NM_STATUS_SUCCESS &&
                rejected.statuses.at(1).code == NM_STATUS_SUCCESS,
            "post-commit target rollback did not reject only the faulted environment");
    const auto afterFailure = fixture.snapshot();
    require(sameEnvironmentBytes(beforeFailure.femNodes,
                                 afterFailure.femNodes, 0u) &&
                sameEnvironmentBytes(beforeFailure.contactHistories,
                                     afterFailure.contactHistories, 0u) &&
                sameEnvironmentBytes(beforeFailure.femMaterialState,
                                     afterFailure.femMaterialState, 0u) &&
                sameEnvironmentBytes(beforeFailure.rigidStates,
                                     afterFailure.rigidStates, 0u),
            "rejected temporary-grip environment changed exact accepted node/history/rigid bytes");
    require(maximumNodeDisplacement(afterFailure, beforeFailure, 1u,
                                    fixture.femNodeCount) <= 1.0e-9,
            "separated environment moved beyond tolerance during target rollback");
    std::cout << "static_box_temporary_grip_armijo=pass"
              << " validated_target_count=" << fixture.femKinematicTargetValues.size()
              << " target_node=" << kInteriorGripNode
              << " tag3=true accepted_position_and_mass=true"
              << " postcommit_env0_exact_rollback=true env1_success=true\n";
}

void rejectStaticBoxPreEndpoint(const unsigned fault, const char* label) {
    Fixture fixture(true, 0.2, true, true);
    const auto prior = fixture.step(0u);
    require(prior.statuses[0].code == NM_STATUS_SUCCESS &&
                prior.statuses[1].code == NM_STATUS_SUCCESS,
            "static-BOX pre-endpoint negative fixture failed to establish an accepted prior state");
    std::memcpy(fixture.bodies.contents, fixture.authoredBodies.data(),
                fixture.authoredBodies.size() * sizeof(MRBodyStateGPU));
    const auto before = fixture.snapshot();
    require(fixture.authoredBodies.at(kAnvilBodyIndex).flagsAndIndices[0] ==
                MR_MOTION_KINEMATIC,
            "moved-BOX pre-endpoint fixture must use a kinematic body");
    // Exercise every-step selection after an accepted prior control step.
    // The body-backed kinematic BOX endpoint is deliberately moved below.
    setenv("NM_FEM_CAPSULE_BOX_RESIDUAL_ARMIJO_ROOT", "all", 1);
    std::string selectorLog;
    RuntimeStateSnapshot rejected;
    {
        StderrCapture capture;
        rejected = fixture.step(1u, false, true, fault);
        selectorLog = capture.finish();
    }
    require(selectorLog.find("fem_capsule_box_residual_armijo_enabled=") != std::string::npos &&
                selectorLog.find("\"root\":1") != std::string::npos,
            std::string("pre-BOX endpoint fixture did not enter the all-selector bounded gate: ") + label + selectorLog);
    require(rejected.statuses[0].code != NM_STATUS_SUCCESS &&
                rejected.statuses[1].code == NM_STATUS_SUCCESS,
            std::string("invalid pre-BOX endpoint did not fail only its environment: ") + label);
    require(maximumNodeDisplacement(rejected, before, 0u,
                fixture.femNodeCount) <= 1.0e-9 &&
                sameEnvironmentBytes(before.femNodes, rejected.femNodes, 0u),
            std::string("invalid pre-BOX endpoint changed accepted FEM state: ") + label);
    require(sameEnvironmentBytes(before.contactHistories,
                                 rejected.contactHistories, 0u) &&
                sameEnvironmentBytes(before.femMaterialState,
                                     rejected.femMaterialState, 0u) &&
                sameEnvironmentBytes(before.rigidStates,
                                     rejected.rigidStates, 0u),
            std::string("invalid pre-BOX endpoint changed accepted history/rigid state: ") + label);
    std::cout << "static_box_pre_endpoint_reject=pass case=" << label
              << " selector=all exact_accepted_state_rollback=true\n";
}

void rejectStaticBoxPostEndpoint(const unsigned fault, const char* label) {
    Fixture fixture(true, 0.2, true, true);
    const auto accepted = fixture.step(0u);
    require(accepted.statuses[0].code == NM_STATUS_SUCCESS,
            "static-BOX precondition did not accept initial contact step");
    const auto beforeFailure = fixture.snapshot();
    require(fixture.authoredBodies.at(kAnvilBodyIndex).flagsAndIndices[0] ==
                MR_MOTION_KINEMATIC,
            "moved-BOX post-endpoint fixture must use a kinematic body");
    // Mutate the kinematic BOX endpoint after the accepted first step; the
    // "all" selector must keep post-commit endpoint certification active.
    setenv("NM_FEM_CAPSULE_BOX_RESIDUAL_ARMIJO_ROOT", "all", 1);
    std::string selectorLog;
    RuntimeStateSnapshot rejected;
    {
        StderrCapture capture;
        rejected = fixture.step(1u, false, false, 0u, false, fault);
        selectorLog = capture.finish();
    }
    require(selectorLog.find("fem_capsule_box_residual_armijo_enabled=") != std::string::npos &&
                selectorLog.find("\"root\":1") != std::string::npos,
            "post-BOX rollback attempt did not enter all-selector BOX Armijo: " + selectorLog);
    require(rejected.statuses[0].code != NM_STATUS_SUCCESS &&
                rejected.statuses[1].code == NM_STATUS_SUCCESS,
            std::string("post-commit BOX endpoint mutation was not isolated and rejected: ") +
                label);
    const auto afterFailure = fixture.snapshot();
    require(sameEnvironmentBytes(beforeFailure.femNodes, afterFailure.femNodes, 0u) &&
            sameEnvironmentBytes(beforeFailure.contactHistories,
                                 afterFailure.contactHistories, 0u) &&
            sameEnvironmentBytes(beforeFailure.femMaterialState,
                                 afterFailure.femMaterialState, 0u) &&
            sameEnvironmentBytes(beforeFailure.rigidStates,
                                 afterFailure.rigidStates, 0u),
            "rejected BOX endpoint changed the rejected environment's accepted state");
    require(maximumNodeDisplacement(afterFailure, beforeFailure, 1u,
                fixture.femNodeCount) <= 1.0e-9,
            "post-BOX rejection changed separated control environment");
    std::cout << "static_box_post_endpoint_rollback=pass case=" << label
              << " accepted_state_restored=true\n";
}

void runCheck() {
    const CaseResult legacy = runPhysicalCase(false);
    const CaseResult armijo = runPhysicalCase(true);
    std::cout << std::setprecision(10)
              << "capsule_armijo_pair=source_only_prescribed_capsule"
              << " contact_pairs=" << armijo.contactPairs
              << " legacy_displacement_m=" << legacy.displacement
              << " armijo_displacement_m=" << armijo.displacement
              << " legacy_impulse_Ns=" << legacy.impulse
              << " armijo_impulse_Ns=" << armijo.impulse
              << " legacy_residual=" << legacy.residual
              << " armijo_residual=" << armijo.residual
              << " both_native_certified=true\n";
    rollbackAfterArmijoCase();
    runStaticBoxArmijoPair();
    runStaticBoxTemporaryGripCase();
    runAcceptedCapsuleConstraintReactionCase();
    runAcceptedBoxConstraintReactionCase();
    runNativeDeformableConstraintReactionKernelCase();
    rejectStaticBoxPreEndpoint(4u, "pre_pose_changed");
    rejectStaticBoxPreEndpoint(5u, "dynamic_motion_type");
    rejectStaticBoxPreEndpoint(6u, "nonzero_tool_velocity");
    rejectStaticBoxPreEndpoint(7u, "pre_orientation_changed");
    rejectStaticBoxPreEndpoint(8u, "nonzero_angular_velocity");
    rejectStaticBoxPreEndpoint(9u, "nonzero_inverse_mass");
    rejectStaticBoxPreEndpoint(10u, "nonzero_inverse_inertia");
    rejectStaticBoxPostEndpoint(3u, "post_position_changed");
    rejectStaticBoxPostEndpoint(4u, "post_orientation_changed");
}

} // namespace

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        try {
            if (argc == 2 && std::string(argv[1]) == "--reaction-publisher-only") {
                runNativeDeformableConstraintReactionKernelCase();
            } else if (argc == 2 &&
                       std::string(argv[1]) == "--constraint-reactions-only") {
                runAcceptedCapsuleConstraintReactionCase();
                runAcceptedBoxConstraintReactionCase();
                runAllTag3ReactionLimitationCase();
            } else {
                require(argc == 1,
                    "usage: cardboard-capsule-armijo-check "
                    "[--reaction-publisher-only|--constraint-reactions-only]");
                runCheck();
            }
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "native cardboard tooling fixture failed: "
                      << error.what() << '\n';
            return 1;
        }
    }
}
