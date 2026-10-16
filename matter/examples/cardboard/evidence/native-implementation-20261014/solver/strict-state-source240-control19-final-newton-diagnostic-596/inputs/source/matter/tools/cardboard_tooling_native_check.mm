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
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace numi::matter;

namespace {

constexpr double kMillimetre = 1.0e-3;
constexpr unsigned kEnvironmentCount = 2u;
constexpr unsigned kPunchBodyIndex = 0u;
constexpr unsigned kPunchProxyIndex = 0u;
constexpr unsigned kSupportProxyIndex = 1u;
constexpr double kPadLength = 4.0 * kMillimetre;
constexpr double kPadWidth = 4.0 * kMillimetre;
constexpr double kPadCaliper = 2.0 * kMillimetre;
constexpr double kPunchRadius = 0.5 * kMillimetre;
constexpr double kBottomClearance = 20.0e-6;
constexpr double kInitialPunchGap = 10.0e-6;
constexpr double kPunchSpeed = 0.02;
constexpr double kContactSlop = 10.0e-6;
constexpr unsigned kSteps = 3u;

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

void validateRigidProxyShapeRegressions(const CompiledWorld& validWorld) {
    require(validWorld.contact.rigidProxies.size() == 2u,
            "validator fixture must contain one capsule and one box");

    // Box contact consumes only its three positive half extents. Its packed
    // radius slot is unused, so zero is a valid canonical value.
    auto zeroRadiusBox = validWorld;
    zeroRadiusBox.contact.rigidProxies[kSupportProxyIndex]
        .localCenterAndRadius.w = 0.0f;
    zeroRadiusBox.fingerprint = compiledWorldFingerprint(zeroRadiusBox);
    std::string error;
    require(validateCompiledWorldLayout(zeroRadiusBox, &error),
            "validator rejected a zero-radius box: " + error);

    auto zeroExtentBox = zeroRadiusBox;
    zeroExtentBox.contact.rigidProxies[kSupportProxyIndex].localExtent.x = 0.0f;
    zeroExtentBox.fingerprint = compiledWorldFingerprint(zeroExtentBox);
    error.clear();
    require(!validateCompiledWorldLayout(zeroExtentBox, &error),
            "validator accepted a box with a zero half extent");
    require(error.find("box half extent is nonpositive") != std::string::npos,
            "box extent rejection reported an unrelated invariant: " + error);

    auto zeroRadiusCapsule = validWorld;
    zeroRadiusCapsule.contact.rigidProxies[kPunchProxyIndex]
        .localCenterAndRadius.w = 0.0f;
    zeroRadiusCapsule.fingerprint = compiledWorldFingerprint(zeroRadiusCapsule);
    error.clear();
    require(!validateCompiledWorldLayout(zeroRadiusCapsule, &error),
            "validator accepted a zero-radius capsule");
    require(error.find("curved shape radius is nonpositive") != std::string::npos,
            "capsule radius rejection reported an unrelated invariant: " + error);

    std::cout << "rigid_proxy_validator=pass zero_radius_box=accepted "
              << "zero_extent_box=rejected zero_radius_capsule=rejected\n";
}

double maximumNodeDisplacement(const RuntimeStateSnapshot& state,
                               const RuntimeStateSnapshot& initial,
                               unsigned environment,
                               std::uint32_t nodeCount);

struct Fixture {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLBuffer> statuses = nil;
    id<MTLBuffer> bodies = nil;
    id<MTLBuffer> endpointBodies = nil;
    Runtime runtime;
    CompiledWorld world;
    std::vector<MRBodyStateGPU> authoredBodies;
    std::array<double, kEnvironmentCount> initialBodyZ{};
    std::uint32_t femNodeCount = 0u;
    std::uint32_t punchProxyIndex = kPunchProxyIndex;
    std::uint32_t supportProxyIndex = kSupportProxyIndex;
    double timestep = 0.0;

    Fixture(const bool prescribed = false, const double punchSpeed = kPunchSpeed) {
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
        pad.deformableContact = true;
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
        // Clamp the lower perimeter only. The lower centre remains free and
        // faces the finite support anvil across a declared 20 um initial gap.
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
        // The board starts 20 um above the helper's support reference plane,
        // so add that offset to the punch's nominal clearance. The actual
        // punch-to-board top gap remains exactly 10 um.
        toolSpec.initialClearance = kBottomClearance + kInitialPunchGap;
        toolSpec.punchBodyIndex = kPunchBodyIndex;
        toolSpec.contactMaterialIndex = 0u;
        auto tools = numi::cardboard::makeCreaseTooling(toolSpec);
        tools.punch.prescribedEndPoseTranslation = prescribed;
        source.rigidProxies = {tools.punch, tools.support};

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
        validateRigidProxyShapeRegressions(world);
        require(world.contact.rigidProxies.size() == 2u &&
                    world.contact.rigidProxies[kPunchProxyIndex].shapeKind == NM_RIGID_CAPSULE &&
                    world.contact.rigidProxies[kPunchProxyIndex].bodyIndex == kPunchBodyIndex &&
                    world.contact.rigidProxies[kSupportProxyIndex].shapeKind == NM_RIGID_BOX &&
                    world.contact.rigidProxies[kSupportProxyIndex].bodyIndex == NM_INVALID_INDEX,
                "compiled native proxies lost capsule/anvil provenance");
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

        authoredBodies.resize(kEnvironmentCount);
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
            authoredBodies[environment] = body;
            initialBodyZ[environment] = body.position.z;
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

    RuntimeStateSnapshot step(const unsigned index,
                              const bool unsafeExternalEndpoint = false,
                              const bool expectedPreRejection = false,
                              const unsigned preFault = 0u,
                              const bool twoSubsteps = false,
                              const unsigned postFault = 0u) {
        const auto before = snapshot();
        auto nextBodies = authoredBodies;
        for (unsigned environment = 0u; environment < kEnvironmentCount; ++environment) {
            nextBodies[environment].position.z +=
                nextBodies[environment].linearVelocityAndInverseMass.z *
                static_cast<float>(runtime.timestepSeconds());
        }
        if (preFault == 1u) nextBodies[0].position.z -= 0.0001f;
        if (preFault == 2u) nextBodies[0].orientation = {0.0f, 0.0f, std::sin(0.2f), std::cos(0.2f)};
        if (preFault == 3u) nextBodies[0].linearVelocityAndInverseMass.w = 1.0f;
        if (unsafeExternalEndpoint) {
            // Put the nose inside the authored top collision surface. A jump
            // through the entire pad would instead test swept CCD, which an
            // endpoint certificate alone cannot establish.
            nextBodies[0].position.z -= static_cast<float>(0.1 * kMillimetre);
            const double endNose = nextBodies[0].position.z - kPunchRadius;
            require(endNose < kPadCaliper &&
                    nextBodies[0].position.z > kPadCaliper,
                    "unsafe endpoint fixture does not intersect the pad top");
        }
        id<MTLBuffer> plannedBodies = postFault != 0u ? makeBuffer(device, nextBodies) : nil;
        if (postFault == 1u) nextBodies[0].inverseInertiaWorldRow0.x = 1.0f;
        if (postFault == 2u) nextBodies[0].flagsAndIndices[0] = MR_MOTION_DYNAMIC;
        std::memcpy(endpointBodies.contents, nextBodies.data(),
                    nextBodies.size() * sizeof(MRBodyStateGPU));
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
        request.rigid.currentBodies = (__bridge void*)(postFault != 0u ? plannedBodies :
            (unsafeExternalEndpoint ? bodies : endpointBodies));
        request.rigid.currentBodyCount = 1u;
        request.rigid.currentBodyStride = 1u;
        request.controlStep = twoSubsteps ? index / 2u : index;
        request.physicsSubsteps = twoSubsteps ? 2u : 1u;
        request.physicsSubstep = twoSubsteps ? index % 2u : 0u;
        request.timestepSeconds = runtime.timestepSeconds();
        request.runAdaptiveTransfer = false;
        request.phase = EncodePhase::preDynamics;
        auto encoded = runtime.encode(request);
        require(encoded.encoded, "native tooling preDynamics encode: " + encoded.message);
        // The normal solve and certificate use the identical endpoint. The
        // negative case deliberately substitutes an unexpected external endpoint
        // after solving, exercising the native rejection and rollback path.
        request.rigid.currentBodies = (__bridge void*)endpointBodies;
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
                    "unsafe external end pose was not rejected by postCommit certification");
            require(result.statuses[1].code == NM_STATUS_SUCCESS,
                    "unsafe moving endpoint incorrectly rejected the separated control");
            const double rolledBackMotion = maximumNodeDisplacement(
                result, before, 0u, femNodeCount);
            require(rolledBackMotion <= 1.0e-9,
                    "postCommit endpoint rejection did not roll back the FEM state");
            require(std::abs(static_cast<double>(
                        static_cast<MRBodyStateGPU*>(bodies.contents)[0].position.z) -
                    authoredBodies[0].position.z) <= 1.0e-9,
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

void runCheck() {
    Fixture fixture;
    const auto initial = fixture.snapshot();
    require(initial.femNodes.size() ==
                static_cast<std::size_t>(kEnvironmentCount) * fixture.femNodeCount,
            "initial synthetic FEM state arity");
    require(fixture.timestep > 0.0, "native tooling timestep must be positive");
    const double perStepTravel = kPunchSpeed * fixture.timestep;
    require(std::abs(perStepTravel) <= 0.25 * kContactSlop,
            "punch trajectory exceeds the declared per-step contact floor budget");

    RuntimeStateSnapshot last;
    for (unsigned step = 0u; step < kSteps; ++step) last = fixture.step(step);

    const std::uint32_t proxyCount =
        static_cast<std::uint32_t>(fixture.world.contact.rigidProxies.size());
    const auto movingPunch = contactsFor(
        last, 0u, fixture.punchProxyIndex, proxyCount);
    const auto stationaryPunch = contactsFor(
        last, 1u, fixture.punchProxyIndex, proxyCount);
    const auto movingSupport = contactsFor(
        last, 0u, fixture.supportProxyIndex, proxyCount);
    require(movingPunch.valid > 0u && movingPunch.maxNormalImpulse > 0.0,
            "advancing kinematic punch produced no native contact impulse");
    require(stationaryPunch.valid == 0u,
            "separated stationary punch unexpectedly contacted the pad");

    const double loadedMotion = maximumNodeDisplacement(
        last, initial, 0u, fixture.femNodeCount);
    const double controlMotion = maximumNodeDisplacement(
        last, initial, 1u, fixture.femNodeCount);
    require(loadedMotion > 0.0,
            "native punch contact did not change the accepted pad state");
    require(controlMotion <= 1.0e-9,
            "separated stationary-punch control changed its pad state");

    const auto* bodies = static_cast<const MRBodyStateGPU*>(fixture.bodies.contents);
    const double prescribedBodyZ = fixture.initialBodyZ[0] -
        kSteps * perStepTravel;
    require(std::abs(static_cast<double>(bodies[0].position.z) - prescribedBodyZ) < 2.0e-7,
            "kinematic body trajectory did not advance by its declared velocity and timestep");
    require(std::abs(static_cast<double>(bodies[1].position.z) -
                     fixture.initialBodyZ[1]) <= 1.0e-9,
            "stationary control punch pose changed");
    require(last.rigidStates.size() ==
                static_cast<std::size_t>(kEnvironmentCount) * proxyCount,
            "projected native rigid-state diagnostics missing");
    const auto& projectedPunch = last.rigidStates[kPunchProxyIndex];
    require(std::abs(static_cast<double>(projectedPunch.centerAndRadius.z) -
                     (static_cast<double>(bodies[0].position.z) +
                      fixture.world.contact.rigidProxies[kPunchProxyIndex].localCenterAndRadius.z)) < 5.0e-7,
            "published capsule state disagrees with the certified end kinematic pose");

    std::cout << std::setprecision(10)
              << "native_tooling_fixture=synthetic_elastic_pad"
              << " accepted_steps=" << kSteps
              << " timestep_s=" << fixture.timestep
              << " contact_slop_m=" << kContactSlop
              << " initial_punch_gap_m=" << kInitialPunchGap
              << " support_gap_m=" << kBottomClearance
              << " punch_step_travel_m=" << perStepTravel
              << " moving_punch_valid_contacts=" << movingPunch.valid
              << " moving_punch_max_normal_impulse=" << movingPunch.maxNormalImpulse
              << " moving_punch_min_separation_m=" << movingPunch.minSeparation
              << " stationary_punch_valid_contacts=" << stationaryPunch.valid
              << " moving_support_valid_contacts=" << movingSupport.valid
              << " moving_support_min_separation_m=" << movingSupport.minSeparation
              << " accepted_pad_displacement_m=" << loadedMotion
              << " stationary_control_displacement_m=" << controlMotion
              << " kinematic_body_pose_advanced=true"
              << " postcommit_end_pose_certified=true"
              << " cardboard_qualification=false\n";

    Fixture rollbackFixture;
    const auto rejected = rollbackFixture.step(0u, true);
    require(rejected.statuses[0].code != NM_STATUS_SUCCESS,
            "unsafe endpoint regression unexpectedly committed");
    // This endpoint starts 10 um inside the previous pad surface. A feasible
    // native initial guess must still converge under the original 1e-7 gate.
    Fixture prescribedFixture(true, 0.2);
    const auto beforePredictor = prescribedFixture.snapshot();
    const auto predicted = prescribedFixture.step(0u);
    const auto predictedContacts = contactsFor(predicted, 0u, 0u, 2u);
    require(predictedContacts.valid > 0u && predictedContacts.maxNormalImpulse > 0.0,
            "prescribed endpoint predictor produced no physical contact response");
    require(predicted.solverCertificates.at(0).nonlinear.x <= prescribedFixture.world.mixedSolver.residualTolerances.x,
            "feasible predictor was published without native equilibrium");
    require(maximumNodeDisplacement(predicted, beforePredictor, 1u, prescribedFixture.femNodeCount) <= 1e-9,
            "prescribed initial guess moved stationary control");
    std::cout << "prescribed_predictor=pass endpoint_travel_m=0.00002 residual="
        << predicted.solverCertificates.at(0).nonlinear.x << " native_contact_impulse="
        << predictedContacts.maxNormalImpulse << '\n';

    // A capsule translating through the pad can end outside the top surface;
    // the swept path must reject it even when an endpoint-only check could pass.
    Fixture sweepFixture(true, 20.0);
    const auto sweptRejection = sweepFixture.step(0u, false, true);
    require(sweptRejection.statuses[0].code == NM_STATUS_CONTACT_FAILURE,
            "through-surface tool trajectory did not fail the swept contact gate");
    std::cout << "prescribed_swept_capsule=rejected rollback=true\n";

    Fixture changedEndpointFixture(true);
    const auto changedEndpoint = changedEndpointFixture.step(0u, true);
    require(changedEndpoint.statuses[0].code == NM_STATUS_CONTACT_FAILURE,
            "changed prescribed endpoint was not rejected");

    for (unsigned fault = 1u; fault <= 3u; ++fault) {
        Fixture continuityFixture(true);
        const auto accepted = continuityFixture.step(0u);
        const auto restored = continuityFixture.runtime.restore(accepted);
        require(restored.encoded, "prescribed tool accepted-pose restore failed: " + restored.message);
        const auto inconsistent = continuityFixture.step(1u, false, true, fault);
        require(inconsistent.statuses[0].code != NM_STATUS_SUCCESS,
                "inconsistent start/rotation/inverse mass was accepted");
    }
    for (unsigned fault = 1u; fault <= 2u; ++fault) {
        Fixture postFaultFixture(true);
        (void)postFaultFixture.step(0u);
        (void)postFaultFixture.step(1u, false, false, 0u, false, fault);
    }
    Fixture substepFixture(true);
    (void)substepFixture.step(0u, false, false, 0u, true);
    (void)substepFixture.step(1u, false, false, 0u, true);
    std::cout << "prescribed_pose_continuity=pass invalid_start_rotation_mass=rejected post_inertia_motion_type=rejected snapshot_restore=pass multi_substep=pass\n";

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
