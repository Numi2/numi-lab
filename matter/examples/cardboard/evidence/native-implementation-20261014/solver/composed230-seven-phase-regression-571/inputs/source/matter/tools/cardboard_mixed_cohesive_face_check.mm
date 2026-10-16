#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef NUMI_MATTER_METALLIB
#define NUMI_MATTER_METALLIB ""
#endif

namespace {
using namespace numi::matter;

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Vec3 operator+(const Vec3 a, const Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
Vec3 operator-(const Vec3 a, const Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
Vec3 operator*(const double s, const Vec3 a) {
    return {s * a.x, s * a.y, s * a.z};
}
Vec3 cross(const Vec3 a, const Vec3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}
double dot(const Vec3 a, const Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
double norm(const Vec3 a) { return std::sqrt(dot(a, a)); }

constexpr double kHalfEdgeM = 0.01;
constexpr double kActivationToleranceM = 5.0e-5;
constexpr double kShearM = 2.0e-5;
constexpr double kOpeningM = 4.0e-5;
constexpr double kFDStepM = 5.0e-7;
constexpr double kFrameTimestepS = 1.0e-6;
unsigned checks = 0u;
std::uint64_t nativeFEMTransactionsExecuted = 0u;

void require(const bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

std::string diagnostics(const std::vector<Diagnostic>& values) {
    std::string result;
    for (const Diagnostic& value : values) result += value.message + "; ";
    return result;
}

Vec3 rotateY(const Vec3 p, const double angle) {
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    return {c * p.x + s * p.z, p.y, -s * p.x + c * p.z};
}

std::array<std::uint32_t, 4> positiveTet(
    std::array<std::uint32_t, 4> tet, const std::vector<Vec3>& nodes) {
    const Vec3 a = nodes.at(tet[0]);
    const Vec3 b = nodes.at(tet[1]);
    const Vec3 c = nodes.at(tet[2]);
    const Vec3 d = nodes.at(tet[3]);
    const double volume6 = dot(b - a, cross(c - a, d - a));
    if (volume6 < 0.0) std::swap(tet[1], tet[2]);
    require(dot(nodes.at(tet[1]) - a,
                cross(nodes.at(tet[2]) - a, nodes.at(tet[3]) - a)) > 0.0,
        "manufactured paired tetrahedron is degenerate");
    return tet;
}

struct FixtureSource {
    WorldSource world;
    std::vector<Vec3> nodes;
};

FixtureSource makeSource(const MaterialProgram& bulk,
                         const bool withInterface,
                         const double rigidRotationY = 0.0,
                         const double initialGap = 0.0,
                         const bool mortarPatch = false,
                         const bool duplicateMortarCapacity = false) {
    FixtureSource result;
    const double h = kHalfEdgeM;
    const std::array<Vec3, 8> unrotated{{
        {0.0, 0.0, 0.0}, {h, 0.0, 0.0}, {0.0, h, 0.0},
        {0.0, 0.0, -h},
        mortarPatch ? Vec3{-h, 0.0, initialGap} : Vec3{0.0, 0.0, initialGap},
        mortarPatch ? Vec3{2.0 * h, 0.0, initialGap} : Vec3{h, 0.0, initialGap},
        mortarPatch ? Vec3{0.0, 2.0 * h, initialGap} : Vec3{0.0, h, initialGap},
        {0.0, 0.0, h + initialGap},
    }};
    result.nodes.reserve(unrotated.size());
    for (const Vec3 node : unrotated)
        result.nodes.push_back(rotateY(node, rigidRotationY));

    WorldSource& source = result.world;
    source.environmentCount = 1u;
    source.frameTimestep = kFrameTimestepS;
    source.gravity = {0.0, 0.0, 0.0};
    source.materials = {bulk};
    source.cohesiveMaterials = {{
        .name = "synthetic_mixed_mode_fixture_only",
        .normalPenaltyPaPerM = 1.0e9,
        .shearPenaltyPaPerM = 8.0e8,
        .compressionPenaltyPaPerM = 2.0e9,
        .normalStrengthPa = 1.0e6,
        .shearStrengthPa = 8.0e5,
        .fractureEnergyModeIJPerM2 = 1000.0,
        .fractureEnergyModeIIJPerM2 = 2000.0,
        .bkExponent = 1.5,
    }};
    source.mixedSolver.newtonIterations = 16u;
    source.mixedSolver.fgmresIterations = 64u;
    source.mixedSolver.relativeResidual = 1.0e-7;

    ObjectSource object;
    object.name = withInterface ? "synthetic_mixed_cohesive_face"
                                : "synthetic_mixed_cohesive_control";
    object.representation = Representation::fem;
    object.mixedFEM = false;
    object.materialIndex = 0u;
    object.characteristicLength = h;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.femReferenceDisplacementGradient = true;
    object.femReferenceSourceIdentity = {
        0x7a1c2946e5b30fd1ull, 0xb4076a18239de5c2ull,
        0x4fe288ca7619530bull, 0x91d2a4b6c8e0375full,
    };
    for (const Vec3 node : result.nodes) {
        const std::array<double, 3> p{{node.x, node.y, node.z}};
        object.femReferenceNodes.push_back(p);
        object.femNodes.push_back(p);
    }
    const auto tetA = positiveTet({{0u, 2u, 1u, 3u}}, result.nodes);
    const auto tetB = positiveTet({{4u, 5u, 6u, 7u}}, result.nodes);
    object.tetrahedra.push_back({tetA});
    object.tetrahedra.push_back({tetB});
    // Reference-displacement FEM deliberately requires explicit per-cell
    // material ownership, even when this synthetic fixture uses one law.
    object.femMaterialIndices = {0u, 0u};
    object.femMaterialSourceIdentity = {
        0x315b907ae6c42df8ull, 0x8042c1f76ab395deull,
        0x5df0a18372e64b9cull, 0xc2946e0fb3157a82ull,
    };
    if (withInterface) {
        if (mortarPatch) {
            CohesiveMortarPatchSource face;
            face.sideANodes = {{0u, 1u, 2u}};
            face.sideBNodes = {{4u, 5u, 6u}};
            face.sideAVertexBarycentric = {{
                {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}
            }};
            // The integration patch is the smaller side-A triangle. These
            // rows map its vertices into the larger side-B parent triangle.
            face.sideBVertexBarycentric = {{
                {{2.0 / 3.0, 1.0 / 3.0, 0.0}},
                {{1.0 / 3.0, 2.0 / 3.0, 0.0}},
                {{1.0 / 3.0, 1.0 / 6.0, 1.0 / 2.0}}
            }};
            face.firstTetrahedron = 0u;
            face.secondTetrahedron = 1u;
            face.materialIndex = 0u;
            face.stableIdentifier = 17u;
            face.cohesiveMaterialIndex = 0u;
            face.activationGapToleranceM = kActivationToleranceM;
            object.femCohesiveMortarPatches.push_back(face);
            if (duplicateMortarCapacity) {
                // A second compiled dormant patch raises the admitted command
                // capacity to two. The duplicate-command test below can then
                // reach the transactional GPU duplicate-identity validator
                // without bypassing the host capacity bound.
                face.stableIdentifier = 18u;
                object.femCohesiveMortarPatches.push_back(face);
            }
        } else {
            CohesiveInterfaceSource face;
            face.sideANodes = {{0u, 1u, 2u}};
            face.sideBNodes = {{4u, 5u, 6u}};
            face.firstTetrahedron = 0u;
            face.secondTetrahedron = 1u;
            face.materialIndex = 0u;
            face.stableIdentifier = 1u;
            face.lawKind = CohesiveLawKind::mixedModeBilinear;
            face.cohesiveMaterialIndex = 0u;
            face.activationControlStep = 0u;
            face.activationGapToleranceM = kActivationToleranceM;
            object.femCohesiveInterfaces.push_back(face);
        }
    }
    source.objects.push_back(std::move(object));
    return result;
}

CompiledWorld compile(const WorldSource& source) {
    CompileOptions options;
    options.maximumRateExponent = 0u;
    options.emitSpecializedMetal = false;
    options.localMaterialNewtonIterations = 16u;
    const CompileResult result = compileWorld(source, options);
    require(result.succeeded(), "cohesive fixture compile failed: " +
        diagnostics(result.diagnostics));
    return result.world;
}

template<class T>
id<MTLBuffer> makeBuffer(id<MTLDevice> device,
                         const std::vector<T>& values) {
    const T zero{};
    const void* data = values.empty() ? static_cast<const void*>(&zero)
                                      : static_cast<const void*>(values.data());
    return [device newBufferWithBytes:data
        length:std::max(std::size_t{1u}, values.size()) * sizeof(T)
        options:MTLResourceStorageModeShared];
}

struct StepResult {
    RuntimeStateSnapshot snapshot;
    std::vector<nm_float4> reactions;
};

struct Run {
    Runtime runtime;
    CompiledWorld world;
    std::vector<Vec3> nodes;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLBuffer> statuses = nil;
    std::uint32_t nextControlStep = 0u;

    Run(CompiledWorld compiled, std::vector<Vec3> sourceNodes)
        : world(std::move(compiled)), nodes(std::move(sourceNodes)) {
        device = MTLCreateSystemDefaultDevice();
        require(device != nil, "native Metal device unavailable");
        const std::string deviceName([[device name] UTF8String]);
        require(deviceName.find("Apple") != std::string::npos &&
                deviceName.find("Paravirtual") == std::string::npos,
            "a native Apple Metal device is required");
        queue = [device newCommandQueue];
        require(queue != nil, "Metal command queue unavailable");
        RuntimeConfiguration configuration;
        configuration.metallib = NUMI_MATTER_METALLIB;
        configuration.environmentCount = world.dispatch.environmentCount;
        configuration.captureEvents = false;
        configuration.captureDiagnostics = true;
        configuration.adaptiveTransfer = false;
        const RuntimeDiagnostics initialized = runtime.initialize(
            world, configuration);
        require(initialized.encoded, "runtime initialize failed: " +
            initialized.message);
        statuses = makeBuffer(device, std::vector<MRMetalWorldStatusGPU>(
            world.dispatch.environmentCount));
    }

    RuntimeStateSnapshot state() {
        const RuntimeStateSnapshot result = runtime.snapshot();
        require(result.available, "runtime snapshot failed: " + result.message);
        return result;
    }

    void restore(const RuntimeStateSnapshot& snapshot) {
        const RuntimeDiagnostics restored = runtime.restore(snapshot);
        require(restored.encoded, "checkpoint restore failed: " + restored.message);
        nextControlStep = snapshot.controlStep + 1u;
    }

    std::vector<nm_float4> targets(const Vec3 upperTranslation) const {
        require(nodes.size() == world.dispatch.femNodeCount,
            "fixture source mesh does not match compiled FEM nodes");
        std::vector<nm_float4> values(nodes.size());
        for (std::size_t i = 0u; i < nodes.size(); ++i) {
            const Vec3 p = nodes[i] + (i >= 4u ? upperTranslation : Vec3{});
            values[i] = {static_cast<float>(p.x), static_cast<float>(p.y),
                         static_cast<float>(p.z), 3.0f};
        }
        return values;
    }

    StepResult step(
        const std::vector<nm_float4>& targetValues,
        const std::vector<NMCohesiveActivationCommandGPU>& activationCommands = {}) {
        require(targetValues.size() == world.dispatch.femNodeCount,
            "kinematic target count differs from the compiled mesh");
        auto* statusValues = static_cast<MRMetalWorldStatusGPU*>(statuses.contents);
        for (std::uint32_t env = 0u; env < world.dispatch.environmentCount; ++env) {
            statusValues[env] = {};
            statusValues[env].environment = env;
        }
        id<MTLBuffer> targetBuffer = makeBuffer(device, targetValues);
        id<MTLCommandBuffer> command = [queue commandBuffer];
        require(command != nil, "Metal command buffer unavailable");
        EncodeRequest request;
        request.commandBuffer = (__bridge void*)command;
        request.environmentStatuses = (__bridge void*)statuses;
        request.femKinematicTargets = (__bridge void*)targetBuffer;
        request.femKinematicTargetCount =
            static_cast<std::uint32_t>(targetValues.size());
        id<MTLBuffer> activationBuffer = nil;
        if (!activationCommands.empty()) {
            activationBuffer = makeBuffer(device, activationCommands);
            request.cohesiveActivationCommands =
                (__bridge void*)activationBuffer;
            request.cohesiveActivationCommandCount =
                static_cast<std::uint32_t>(activationCommands.size());
            request.cohesiveActivationCommandStride =
                sizeof(NMCohesiveActivationCommandGPU);
        }
        request.controlStep = nextControlStep++;
        request.physicsSubsteps = 1u;
        request.timestepSeconds = runtime.timestepSeconds();
        request.runAdaptiveTransfer = false;
        request.phase = EncodePhase::preDynamics;
        RuntimeDiagnostics encoded = runtime.encode(request);
        require(encoded.encoded, "preDynamics encode failed: " + encoded.message);

        id<MTLBuffer> reactionSource =
            (__bridge id<MTLBuffer>)runtime.femConstraintReactionBuffer();
        const std::size_t reactionBytes = targetValues.size() * sizeof(nm_float4);
        require(reactionSource != nil && reactionSource.length >= reactionBytes,
            "native FEM constraint reaction buffer is unavailable");
        id<MTLBuffer> reactionReadback = [device
            newBufferWithLength:reactionBytes options:MTLResourceStorageModeShared];
        require(reactionReadback != nil, "reaction readback allocation failed");
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        require(blit != nil, "reaction readback blit unavailable");
        [blit copyFromBuffer:reactionSource sourceOffset:0u
                    toBuffer:reactionReadback destinationOffset:0u
                        size:reactionBytes];
        [blit endEncoding];

        request.phase = EncodePhase::postCommit;
        request.cohesiveActivationCommands = nullptr;
        request.cohesiveActivationCommandCount = 0u;
        request.cohesiveActivationCommandStride = 0u;
        encoded = runtime.encode(request);
        require(encoded.encoded, "postCommit encode failed: " + encoded.message);
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
            "native cohesive command buffer failed");
        ++nativeFEMTransactionsExecuted;
        StepResult result;
        result.snapshot = state();
        const auto* data = static_cast<const nm_float4*>(reactionReadback.contents);
        result.reactions.assign(data, data + targetValues.size());
        return result;
    }
};

bool hasFlag(const NMCohesiveFaceGPU& face, const std::uint32_t flag) {
    return (face.adjacency.w & flag) != 0u;
}

void checkCertificateAccepted(const RuntimeStateSnapshot& snapshot) {
    require(snapshot.statuses.size() == 1u &&
            snapshot.solverCertificates.size() == 1u,
        "single-object fixture has unexpected status/certificate layout");
    require(snapshot.statuses[0].code == NM_STATUS_SUCCESS,
        "native cohesive step rejected with status " +
        std::to_string(snapshot.statuses[0].code));
    require(snapshot.solverCertificates[0].validity.w > 0.5f,
        "accepted native cohesive step lacks a positive production certificate");
}

Vec3 subtractForceOnB(const StepResult& bonded,
                      const StepResult& control) {
    Vec3 force{};
    for (const std::uint32_t node : {4u, 5u, 6u}) {
        const nm_float4 a = bonded.reactions.at(node);
        const nm_float4 b = control.reactions.at(node);
        force.x -= double(a.x) - double(b.x);
        force.y -= double(a.y) - double(b.y);
        force.z -= double(a.z) - double(b.z);
    }
    return force;
}

struct LoadSummary {
    Vec3 forceOnB{};
    double maximumNetReactionN = 0.0;
    double maximumReactionTorqueNm = 0.0;
    double workJ = 0.0;
    double areaM2 = 0.0;
    RuntimeStateSnapshot bondedSnapshot;
    RuntimeStateSnapshot controlSnapshot;
};

LoadSummary runSubcriticalPath(const MaterialProgram& bulk,
                               const double rotationY,
                               const Vec3 baseShearNormal) {
    const FixtureSource bondedSource = makeSource(bulk, true, rotationY);
    const FixtureSource controlSource = makeSource(bulk, false, rotationY);
    const CompiledWorld bondedWorld = compile(bondedSource.world);
    const CompiledWorld controlWorld = compile(controlSource.world);
    require(bondedWorld.dispatch.cohesiveFaceCount == 1u,
        "one authored mixed paired face did not compile to one native face");
    require(controlWorld.dispatch.cohesiveFaceCount == 0u,
        "no-face control unexpectedly contains cohesive faces");
    const double area = double(bondedWorld.fem.cohesiveFaces[0].geometry.w);
    require(area > 0.0, "compiled mixed face has nonpositive reference area");

    Run bonded(compile(bondedSource.world), bondedSource.nodes);
    Run control(compile(controlSource.world), controlSource.nodes);
    const RuntimeStateSnapshot initial = bonded.state();
    require(initial.cohesiveFaces.size() == 1u &&
            hasFlag(initial.cohesiveFaces[0], NM_TOPOLOGY_COHESIVE_DORMANT),
        "mixed interface must begin dormant pending its scheduled activation");
    const StepResult activated = bonded.step(bonded.targets({}));
    const StepResult controlActivated = control.step(control.targets({}));
    checkCertificateAccepted(activated.snapshot);
    checkCertificateAccepted(controlActivated.snapshot);
    require(!hasFlag(activated.snapshot.cohesiveFaces[0],
                     NM_TOPOLOGY_COHESIVE_DORMANT),
        "accepted activation did not activate the paired mixed interface");
    for (const nm_float4 history : activated.snapshot.cohesiveFaces[0].mixedPointHistory)
        require(history.x == 0.0f && history.y == 0.0f &&
                history.z == 0.0f && history.w == 0.0f,
            "virgin accepted activation did not initialize all quadrature histories");

    constexpr unsigned steps = 8u;
    Vec3 previousDisplacement{};
    Vec3 previousForce{};
    double maximumNetReaction = 0.0;
    double maximumReactionTorque = 0.0;
    double work = 0.0;
    StepResult lastBonded = activated;
    StepResult lastControl = controlActivated;
    for (unsigned step = 1u; step <= steps; ++step) {
        const double fraction = double(step) / double(steps);
        const Vec3 displacement = fraction * baseShearNormal;
        const StepResult nextBonded = bonded.step(bonded.targets(displacement));
        const StepResult nextControl = control.step(control.targets(displacement));
        checkCertificateAccepted(nextBonded.snapshot);
        checkCertificateAccepted(nextControl.snapshot);
        const Vec3 forceOnB = subtractForceOnB(nextBonded, nextControl);
        work += 0.5 * dot(previousForce + forceOnB,
                          displacement - previousDisplacement);
        previousDisplacement = displacement;
        previousForce = forceOnB;

        Vec3 stepNetReaction{};
        Vec3 stepReactionTorque{};
        for (std::size_t node = 0u; node < nextBonded.reactions.size(); ++node) {
            const nm_float4 rb = nextBonded.reactions[node];
            const nm_float4 rc = nextControl.reactions[node];
            // The common prescribed-motion reaction is removed. The negative
            // reaction difference is the assembled cohesive force.
            const Vec3 force{-double(rb.x - rc.x), -double(rb.y - rc.y),
                             -double(rb.z - rc.z)};
            const nm_float4 p = nextBonded.snapshot.femNodes[node].positionAndMass;
            stepNetReaction = stepNetReaction + force;
            stepReactionTorque = stepReactionTorque + cross(
                {double(p.x), double(p.y), double(p.z)}, force);
        }
        maximumNetReaction = std::max(maximumNetReaction, norm(stepNetReaction));
        maximumReactionTorque = std::max(maximumReactionTorque,
                                         norm(stepReactionTorque));
        lastBonded = nextBonded;
        lastControl = nextControl;
    }

    const Vec3 worldNormal = rotateY({0.0, 0.0, 1.0}, rotationY);
    const Vec3 worldTangent = rotateY({1.0, 0.0, 0.0}, rotationY);
    const Vec3 expectedForce = area * (
        8.0e8 * dot(baseShearNormal, worldTangent) * worldTangent +
        1.0e9 * dot(baseShearNormal, worldNormal) * worldNormal);
    const Vec3 measuredForce = subtractForceOnB(lastBonded, lastControl);
    const double forceScale = std::max(1.0, norm(expectedForce));
    require(norm(measuredForce - expectedForce) < 0.025 * forceScale + 2.0e-4,
        "native mixed cohesive resultant differs from its subcritical penalty law");
    const double expectedWork = 0.5 * area * (
        8.0e8 * std::pow(dot(baseShearNormal, worldTangent), 2) +
        1.0e9 * std::pow(dot(baseShearNormal, worldNormal), 2));
    require(std::abs(work - expectedWork) <
            0.04 * std::max(expectedWork, 1.0e-9) + 2.0e-7,
        "native incremental reaction work differs from the subcritical interface energy");
    const double forceReference = std::max(norm(expectedForce), 1.0e-8);
    require(maximumNetReaction < 2.0e-3 * forceReference + 2.0e-5,
        "native cohesive pair violates net force balance");
    const double torqueScale = forceReference * 2.0 * kHalfEdgeM;
    require(maximumReactionTorque <
            3.0e-3 * std::max(torqueScale, 1.0e-9) + 2.0e-7,
        "native mixed cohesive frame-conjugate forces violate torque balance");
    for (const NMCohesiveFaceGPU& face : lastBonded.snapshot.cohesiveFaces) {
        require(face.state.x == 0.0f,
            "subcritical native mixed path unexpectedly accumulated damage");
        for (const nm_float4 history : face.mixedPointHistory)
            require(history.y == 0.0f && history.w == 0.0f,
                "subcritical native mixed path crossed initiation");
    }

    return {measuredForce, maximumNetReaction, maximumReactionTorque, work, area,
            lastBonded.snapshot, lastControl.snapshot};
}

void checkCheckpointReplay(const MaterialProgram& bulk,
                           const LoadSummary& center,
                           const Vec3 nextDisplacement) {
    const FixtureSource source = makeSource(bulk, true);
    Run uninterrupted(compile(source.world), source.nodes);
    uninterrupted.restore(center.bondedSnapshot);
    Run resumed(compile(source.world), source.nodes);
    resumed.restore(center.bondedSnapshot);
    const StepResult a = uninterrupted.step(uninterrupted.targets(nextDisplacement));
    const StepResult b = resumed.step(resumed.targets(nextDisplacement));
    checkCertificateAccepted(a.snapshot);
    checkCertificateAccepted(b.snapshot);
    require(a.snapshot.controlStep == b.snapshot.controlStep &&
            a.snapshot.physicsSubstep == b.snapshot.physicsSubstep,
        "checkpoint replay changed the native accepted control cursor");
    require(a.snapshot.cohesiveFaces.size() == b.snapshot.cohesiveFaces.size() &&
            std::memcmp(a.snapshot.cohesiveFaces.data(),
                b.snapshot.cohesiveFaces.data(),
                a.snapshot.cohesiveFaces.size() * sizeof(NMCohesiveFaceGPU)) == 0,
        "checkpoint restore did not reproduce mixed face activation/history exactly");
    require(a.snapshot.femNodes.size() == b.snapshot.femNodes.size() &&
            std::memcmp(a.snapshot.femNodes.data(), b.snapshot.femNodes.data(),
                a.snapshot.femNodes.size() * sizeof(NMFEMNodeStateGPU)) == 0,
        "checkpoint restore did not reproduce FEM accepted state exactly");
}

void checkRejectedTransactions(const MaterialProgram& bulk) {
    const FixtureSource source = makeSource(bulk, true);
    {
        Run rejected(compile(source.world), source.nodes);
        const RuntimeStateSnapshot before = rejected.state();
        const Vec3 tooWideGap = 2.5 * kActivationToleranceM *
            rotateY({0.0, 0.0, 1.0}, 0.0);
        const StepResult result = rejected.step(rejected.targets(tooWideGap));
        require(result.snapshot.statuses[0].code != NM_STATUS_SUCCESS,
            "over-tolerance activation gap unexpectedly committed");
        require(result.snapshot.solverCertificates.size() == 1u &&
                result.snapshot.solverCertificates[0].validity.w <= 0.5f,
            "late activation rejection retained an accepted production certificate bit");
        require(result.snapshot.controlStep == before.controlStep &&
                result.snapshot.physicsSubstep == before.physicsSubstep,
            "rejected activation advanced the accepted control cursor");
        require(std::memcmp(result.snapshot.femNodes.data(), before.femNodes.data(),
                before.femNodes.size() * sizeof(NMFEMNodeStateGPU)) == 0,
            "rejected activation changed accepted FEM nodes");
        require(std::memcmp(result.snapshot.cohesiveFaces.data(),
                before.cohesiveFaces.data(),
                before.cohesiveFaces.size() * sizeof(NMCohesiveFaceGPU)) == 0,
            "rejected activation changed dormant cohesive history");
        require(hasFlag(result.snapshot.cohesiveFaces[0],
                        NM_TOPOLOGY_COHESIVE_DORMANT),
            "rejected activation was published as an active bond");
    }
    {
        Run rejected(compile(source.world), source.nodes);
        const StepResult activated = rejected.step(rejected.targets({}));
        checkCertificateAccepted(activated.snapshot);
        const RuntimeStateSnapshot before = activated.snapshot;
        std::vector<nm_float4> singular = rejected.targets({});
        singular[1] = singular[0];
        const StepResult result = rejected.step(singular);
        require(result.snapshot.statuses[0].code != NM_STATUS_SUCCESS,
            "singular endpoint unexpectedly committed");
        require(result.snapshot.solverCertificates[0].validity.w <= 0.5f,
            "rejected FEM endpoint retained an accepted certificate bit");
        require(result.snapshot.controlStep == before.controlStep &&
                result.snapshot.physicsSubstep == before.physicsSubstep,
            "rejected FEM endpoint advanced the accepted control cursor");
        require(std::memcmp(result.snapshot.femNodes.data(), before.femNodes.data(),
                before.femNodes.size() * sizeof(NMFEMNodeStateGPU)) == 0 &&
                std::memcmp(result.snapshot.cohesiveFaces.data(),
                    before.cohesiveFaces.data(),
                    before.cohesiveFaces.size() * sizeof(NMCohesiveFaceGPU)) == 0,
            "failed FEM endpoint did not roll back nodes and cohesive history together");
    }
}

void checkExplicitMortarActivation(const MaterialProgram& bulk) {
    const double initialGap = 0.5 * kActivationToleranceM;
    const FixtureSource source = makeSource(
        bulk, true, 0.0, initialGap, true);
    const CompiledWorld compiled = compile(source.world);
    require(compiled.dispatch.cohesiveFaceCount == 1u &&
            hasFlag(compiled.fem.cohesiveFaces[0], NM_TOPOLOGY_COHESIVE_MORTAR) &&
            hasFlag(compiled.fem.cohesiveFaces[0], NM_TOPOLOGY_COHESIVE_DORMANT) &&
            compiled.fem.cohesiveFaces[0].lawIdentity.w ==
                NM_COHESIVE_ACTIVATION_EXPLICIT_COMMAND,
        "mortar source did not compile as a dormant explicit-command patch");

    Run run(compile(source.world), source.nodes);
    const RuntimeStateSnapshot initial = run.state();
    require(initial.cohesiveFaces.size() == 1u &&
            hasFlag(initial.cohesiveFaces[0], NM_TOPOLOGY_COHESIVE_DORMANT),
        "explicit mortar patch was active before a command");

    // An accepted close step moves side B from the source-separated pose to
    // side A. No bond command is present during this step.
    const Vec3 closeTranslation{0.0, 0.0, -initialGap};
    const StepResult closed = run.step(run.targets(closeTranslation));
    checkCertificateAccepted(closed.snapshot);
    require(hasFlag(closed.snapshot.cohesiveFaces[0],
                    NM_TOPOLOGY_COHESIVE_DORMANT),
        "mortar patch activated without an explicit command");
    require(closed.snapshot.controlStep >= initial.controlStep,
        "accepted closure step moved the transaction cursor backwards");

    NMCohesiveActivationCommandGPU command{};
    command.identity = {0u, 0u, 17u, 0u};
    const StepResult activated = run.step(
        run.targets(closeTranslation), {command});
    checkCertificateAccepted(activated.snapshot);
    require(activated.snapshot.controlStep > closed.snapshot.controlStep,
        "accepted activation command did not advance the transaction cursor");
    const NMCohesiveFaceGPU& active = activated.snapshot.cohesiveFaces[0];
    require(!hasFlag(active, NM_TOPOLOGY_COHESIVE_DORMANT) &&
            hasFlag(active, NM_TOPOLOGY_COHESIVE_MORTAR),
        "explicit accepted-endpoint command did not activate its mortar patch");
    const double expectedPatchArea = 0.5 * kHalfEdgeM * kHalfEdgeM;
    require(std::abs(double(active.geometry.w) - expectedPatchArea) <
                1.0e-5 * expectedPatchArea,
        "activation did not latch the accepted side-A patch area");
    for (const nm_float4 localGap : active.activationLocalGap)
        require(std::sqrt(double(localGap.x) * localGap.x +
                          double(localGap.y) * localGap.y +
                          double(localGap.z) * localGap.z) < 1.0e-7,
            "closed accepted endpoint did not latch a near-zero local gap");
    for (const nm_float4 history : active.mixedPointHistory)
        require(history.x == 0.0f && history.y == 0.0f &&
                history.z == 0.0f && history.w == 0.0f,
            "explicit activation did not start with virgin quadrature history");

    // The latched patch reference and its accepted history must round-trip.
    Run uninterrupted(compile(source.world), source.nodes);
    Run resumed(compile(source.world), source.nodes);
    uninterrupted.restore(activated.snapshot);
    resumed.restore(activated.snapshot);
    const StepResult nextA = uninterrupted.step(
        uninterrupted.targets(closeTranslation));
    const StepResult nextB = resumed.step(resumed.targets(closeTranslation));
    checkCertificateAccepted(nextA.snapshot);
    checkCertificateAccepted(nextB.snapshot);
    require(nextA.snapshot.controlStep == nextB.snapshot.controlStep &&
            nextA.snapshot.physicsSubstep == nextB.snapshot.physicsSubstep &&
            nextA.snapshot.femNodes.size() == nextB.snapshot.femNodes.size() &&
            std::memcmp(nextA.snapshot.femNodes.data(), nextB.snapshot.femNodes.data(),
                nextA.snapshot.femNodes.size() * sizeof(NMFEMNodeStateGPU)) == 0 &&
            std::memcmp(nextA.snapshot.cohesiveFaces.data(),
                nextB.snapshot.cohesiveFaces.data(),
                nextA.snapshot.cohesiveFaces.size() * sizeof(NMCohesiveFaceGPU)) == 0,
        "explicit mortar checkpoint replay changed accepted state");

    const auto assertCommandRejected = [&] (
        const FixtureSource& rejectedSource,
        const std::vector<NMCohesiveActivationCommandGPU>& commands,
        const std::string& label) {
        Run rejected(compile(rejectedSource.world), rejectedSource.nodes);
        const RuntimeStateSnapshot before = rejected.state();
        const StepResult result = rejected.step(rejected.targets({}), commands);
        require(result.snapshot.statuses.size() == 1u &&
                result.snapshot.statuses[0].code != NM_STATUS_SUCCESS,
            label + " unexpectedly accepted");
        require(result.snapshot.solverCertificates.size() == 1u &&
                result.snapshot.solverCertificates[0].validity.w <= 0.5f,
            label + " retained an accepted production certificate bit");
        require(result.snapshot.controlStep == before.controlStep &&
                result.snapshot.physicsSubstep == before.physicsSubstep,
            label + " advanced the accepted transaction cursor");
        require(result.snapshot.femNodes.size() == before.femNodes.size() &&
                std::memcmp(result.snapshot.femNodes.data(), before.femNodes.data(),
                    before.femNodes.size() * sizeof(NMFEMNodeStateGPU)) == 0 &&
                result.snapshot.cohesiveFaces.size() == before.cohesiveFaces.size() &&
                std::memcmp(result.snapshot.cohesiveFaces.data(),
                    before.cohesiveFaces.data(),
                    before.cohesiveFaces.size() * sizeof(NMCohesiveFaceGPU)) == 0 &&
                result.snapshot.femMaterialState.size() ==
                    before.femMaterialState.size() &&
                std::memcmp(result.snapshot.femMaterialState.data(),
                    before.femMaterialState.data(),
                    before.femMaterialState.size() * sizeof(float)) == 0,
            label + " changed accepted nodes, face history, or bulk material state");
        require(std::ranges::all_of(result.snapshot.cohesiveFaces,
                [](const NMCohesiveFaceGPU& face) {
                    return hasFlag(face, NM_TOPOLOGY_COHESIVE_DORMANT);
                }),
            label + " published a bond despite transaction rejection");
    };

    const FixtureSource tooFar = makeSource(
        bulk, true, 0.0, 2.0 * kActivationToleranceM, true);
    assertCommandRejected(tooFar, {command}, "over-tolerance activation command");
    const FixtureSource duplicateCapacity = makeSource(
        bulk, true, 0.0, initialGap, true, true);
    const CompiledWorld duplicateCapacityWorld = compile(duplicateCapacity.world);
    require(duplicateCapacityWorld.dispatch.cohesiveFaceCount == 2u,
        "duplicate-command fixture did not compile capacity for two commands");
    assertCommandRejected(duplicateCapacity, {command, command},
        "duplicate activation command within compiled capacity");
}

void cpuAdmissionChecks(const MaterialProgram& bulk) {
    const FixtureSource authored = makeSource(bulk, true);
    const CompiledWorld compiled = compile(authored.world);
    require(compiled.dispatch.cohesiveFaceCount == 1u &&
            compiled.cohesiveMaterials.size() == 1u,
        "mixed interface did not compile into one face and one explicit law");
    const NMCohesiveFaceGPU& face = compiled.fem.cohesiveFaces[0];
    require(hasFlag(face, NM_TOPOLOGY_COHESIVE_MIXED_MODE) &&
            hasFlag(face, NM_TOPOLOGY_COHESIVE_DORMANT),
        "mixed interface did not compile dormant with the authored schedule");
    require(face.lawIdentity.x == NM_COHESIVE_LAW_MIXED_MODE_BILINEAR &&
            face.lawIdentity.y == 0u && face.lawIdentity.z == 0u &&
            face.activationControl.x == float(kActivationToleranceM) &&
            face.geometry.w > 0.0f,
        "compiled mixed face law, activation, or reference area is incorrect");
    require(compiled.cohesiveMaterials[0].stiffnessAndNormalStrength.x == 1.0e9f &&
            compiled.cohesiveMaterials[0].stiffnessAndNormalStrength.y == 8.0e8f,
        "compiled synthetic cohesive parameters changed unexpectedly");
    auto invalid = authored.world;
    invalid.cohesiveMaterials[0].normalStrengthPa = 0.0;
    const CompileResult rejected = compileWorld(invalid);
    require(!rejected.succeeded(),
        "invalid mixed cohesive law was admitted by the compiler");

    const FixtureSource mortar = makeSource(
        bulk, true, 0.0, 0.5 * kActivationToleranceM, true);
    const CompiledWorld mortarCompiled = compile(mortar.world);
    const NMCohesiveFaceGPU& mortarFace = mortarCompiled.fem.cohesiveFaces[0];
    require(hasFlag(mortarFace, NM_TOPOLOGY_COHESIVE_MORTAR) &&
            hasFlag(mortarFace, NM_TOPOLOGY_COHESIVE_DORMANT) &&
            mortarFace.lawIdentity.w == NM_COHESIVE_ACTIVATION_EXPLICIT_COMMAND &&
            std::abs(double(mortarFace.geometry.w) -
                0.5 * kHalfEdgeM * kHalfEdgeM) <
                1.0e-5 * 0.5 * kHalfEdgeM * kHalfEdgeM &&
            std::abs(double(mortarFace.mortarSideBVertexWeights[2].x) - 1.0 / 3.0) <
                1.0e-6,
        "nonmatching mortar patch maps or dormant explicit activation did not compile");
}

MaterialProgram readMaterial(const std::string& path) {
    const ParseResult parsed = parseMatterFile(path);
    require(parsed.succeeded(), "bulk material parse failed: " + path + ": " +
        diagnostics(parsed.diagnostics));
    return parsed.material;
}

} // namespace

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        try {
            require(argc == 2 || (argc == 3 &&
                    std::string(argv[2]) == "--cpu-only"),
                "usage: cardboard-mixed-cohesive-face-check BULK.nmatter [--cpu-only]");
            const MaterialProgram bulk = readMaterial(argv[1]);
            cpuAdmissionChecks(bulk);
            if (argc == 3) {
                std::cout << "PASS mixed cohesive compiler admission; native_steps=0; "
                    << "synthetic interface parameters; physical starch qualification=false\n";
                return 0;
            }

            const Vec3 displacement{kShearM, 0.0, kOpeningM};
            const LoadSummary base = runSubcriticalPath(bulk, 0.0, displacement);
            const double angle = 0.63;
            const LoadSummary rotated = runSubcriticalPath(
                bulk, angle, rotateY(displacement, angle));
            const Vec3 expectedRotatedForce = rotateY(base.forceOnB, angle);
            require(norm(rotated.forceOnB - expectedRotatedForce) <
                    0.025 * std::max(1.0, norm(expectedRotatedForce)) + 2.0e-4,
                "native mixed cohesive traction is not objective under rigid face rotation");
            require(std::abs(rotated.workJ - base.workJ) <
                    0.03 * std::max(1.0e-9, std::abs(base.workJ)) + 2.0e-7,
                "native mixed cohesive work changed under rigid face rotation");

            // Centered finite difference of the complete Runtime reaction
            // path, including the native co-rotating frame force correction.
            const FixtureSource source = makeSource(bulk, true);
            const FixtureSource controlSource = makeSource(bulk, false);
            Run plus(compile(source.world), source.nodes);
            Run minus(compile(source.world), source.nodes);
            Run plusControl(compile(controlSource.world), controlSource.nodes);
            Run minusControl(compile(controlSource.world), controlSource.nodes);
            plus.restore(base.bondedSnapshot);
            minus.restore(base.bondedSnapshot);
            plusControl.restore(base.controlSnapshot);
            minusControl.restore(base.controlSnapshot);
            const StepResult plusResult = plus.step(plus.targets(
                displacement + Vec3{kFDStepM, 0.0, 0.0}));
            const StepResult minusResult = minus.step(minus.targets(
                displacement - Vec3{kFDStepM, 0.0, 0.0}));
            const StepResult plusControlResult = plusControl.step(
                plusControl.targets(displacement + Vec3{kFDStepM, 0.0, 0.0}));
            const StepResult minusControlResult = minusControl.step(
                minusControl.targets(displacement - Vec3{kFDStepM, 0.0, 0.0}));
            checkCertificateAccepted(plusResult.snapshot);
            checkCertificateAccepted(minusResult.snapshot);
            const Vec3 fPlus = subtractForceOnB(plusResult, plusControlResult);
            const Vec3 fMinus = subtractForceOnB(minusResult, minusControlResult);
            const double measuredTangent = (fPlus.x - fMinus.x) /
                (2.0 * kFDStepM);
            const double expectedTangent = base.areaM2 * 8.0e8;
            const double tangentRelativeError = std::abs(measuredTangent - expectedTangent) /
                std::max(1.0, std::abs(expectedTangent));
            require(tangentRelativeError < 0.05,
                "native mixed cohesive reaction tangent fails centered FD on intact branch");

            checkRejectedTransactions(bulk);
            checkExplicitMortarActivation(bulk);
            checkCheckpointReplay(bulk, base, displacement +
                Vec3{0.25 * kFDStepM, 0.0, 0.25 * kFDStepM});
            std::cout << "PASS native mixed cohesive fixture checks=" << checks
                << " native_fem_transactions_executed="
                << nativeFEMTransactionsExecuted
                << " physical_validation=false synthetic_law=true"
                << " base_work_J=" << base.workJ
                << " force_N=" << norm(base.forceOnB)
                << " force_balance_N=" << base.maximumNetReactionN
                << " torque_balance_Nm=" << base.maximumReactionTorqueNm
                << " rotated_force_error_N=" <<
                    norm(rotated.forceOnB - expectedRotatedForce)
                << " intact_tangent_relative_error=" << tangentRelativeError
                << " activation_rejection_rollback=true"
                << " failed_endpoint_rollback=true checkpoint_replay=true"
                << " explicit_mortar_activation=true command_rejection_rollback=true"
                << " physical_starch_qualification=false\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "cardboard mixed cohesive native fixture failed: "
                << error.what() << '\n';
            return 1;
        }
    }
}
