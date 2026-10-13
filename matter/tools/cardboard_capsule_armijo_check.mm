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
            const bool withSupportBox = false)
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
        toolSpec.supportSurfaceZ = -kBottomClearance;
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

struct ContactSummary {
    std::uint32_t valid = 0u;
    double maxNormalImpulse = 0.0;
    double minSeparation = INFINITY;
};

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

int main() {
    @autoreleasepool {
        try {
            runCheck();
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "native cardboard tooling fixture failed: "
                      << error.what() << '\n';
            return 1;
        }
    }
}
