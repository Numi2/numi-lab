#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/engine_types.h"
#include "numi/matter/matter.hpp"
#include "cardboard_solver_reference.hpp"
#include "cardboard_paper_projection_scaling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef NUMI_MATTER_METALLIB
#error "CMake must define NUMI_MATTER_METALLIB"
#endif

namespace {
using namespace numi::matter;
using Vec = std::array<double, 3>;
using Mat = std::array<double, 9>; // row-major

void need(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Vec add(const Vec& a, const Vec& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
Vec sub(const Vec& a, const Vec& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
Vec scale(const Vec& a, const double s) {
    return {a[0] * s, a[1] * s, a[2] * s};
}
double dot(const Vec& a, const Vec& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Mat transpose(const Mat& a) {
    Mat r{};
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j) r[3*i+j] = a[3*j+i];
    return r;
}
Mat multiply(const Mat& a, const Mat& b) {
    Mat r{};
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j)
            for (unsigned k = 0; k < 3; ++k)
                r[3*i+j] += a[3*i+k] * b[3*k+j];
    return r;
}
Mat add(const Mat& a, const Mat& b) {
    Mat r{};
    for (unsigned i=0;i<9;++i) r[i]=a[i]+b[i];
    return r;
}
Mat scale(const Mat& a, const double s) {
    Mat r=a;
    for (double& v:r) v*=s;
    return r;
}
double determinant(const Mat& a) {
    return a[0]*(a[4]*a[8]-a[5]*a[7]) -
           a[1]*(a[3]*a[8]-a[5]*a[6]) +
           a[2]*(a[3]*a[7]-a[4]*a[6]);
}
Mat inverse(const Mat& a) {
    const double d=determinant(a);
    need(std::isfinite(d)&&std::abs(d)>1e-40,"singular FP64 matrix");
    return {
        (a[4]*a[8]-a[5]*a[7])/d, (a[2]*a[7]-a[1]*a[8])/d, (a[1]*a[5]-a[2]*a[4])/d,
        (a[5]*a[6]-a[3]*a[8])/d, (a[0]*a[8]-a[2]*a[6])/d, (a[2]*a[3]-a[0]*a[5])/d,
        (a[3]*a[7]-a[4]*a[6])/d, (a[1]*a[6]-a[0]*a[7])/d, (a[0]*a[4]-a[1]*a[3])/d};
}
Vec from4(const nm_float4& x) { return {x.x,x.y,x.z}; }

std::uint32_t regionOf(const NMTetrahedronGPU& tet) {
    return (tet.identity.w & NM_TETRAHEDRON_PRECONDITIONER_REGION_MASK) >>
        NM_TETRAHEDRON_PRECONDITIONER_REGION_SHIFT;
}

WorldSource source(const MaterialProgram& material,
                   const FEMPreconditionerMode mode,
                   const bool useTangentCache = false,
                   const std::uint32_t environments = 1u) {
    WorldSource world;
    world.environmentCount = environments;
    world.frameTimestep = 2.0e-5;
    world.gravity = {0.0, 0.0, 0.0};
    world.mixedSolver.femPreconditioner = mode;
    world.mixedSolver.newtonIterations = 12u;
    world.mixedSolver.fgmresRestart = 16u;
    world.mixedSolver.fgmresIterations = 64u;
    world.mixedSolver.cacheFEMAlgorithmicTangent = useTangentCache;
    // The inherited default comparison (case regional-rap-invariance-021)
    // admitted very different accepted Newton residuals: 5.05793e-5 for the
    // scalar selector and 1.67512e-8 for regional RAP, with a 2.99% relative
    // displacement difference. Use one explicit tighter fixture tolerance
    // for both selectors so the preconditioner comparison is not dominated
    // by the default 1e-4 stopping envelope. Production defaults are unchanged.
    world.mixedSolver.relativeResidual = 1.0e-7;
    world.materials = {material};

    ObjectSource object;
    object.name = "regional_galerkin_native_two_region_starch_control";
    object.representation = Representation::fem;
    object.materialIndex = 0u;
    object.mixedFEM = false;
    object.twoWayCoupling = false;
    object.adaptive = false;
    object.identifiable = false;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.femReferenceDisplacementGradient = true;
    object.femReferenceNodes = {
        {{0.0,0.0,0.0}}, {{0.01,0.0,0.0}}, {{0.0,0.01,0.0}},
        {{0.0,0.0,0.01}}, {{0.0,0.0,-0.01}}};
    // These identities bind the checker-authored synthetic reference and
    // uniform material map. They are fixture provenance, not literature data.
    object.femReferenceSourceIdentity = {
        0x8ab0e838d7654b21ull, 0xcab1f2a4d4c90137ull,
        0x2a81cc8a0f3e906bull, 0x72d419ce15b67403ull};
    object.femMaterialIndices = {0u, 0u};
    object.femMaterialSourceIdentity = {
        0x75e3914b0a6c2f18ull, 0x6ff8d2c4be139a05ull,
        0x93ad0fe17742c65bull, 0x2f18a763c0de594bull};
    object.femNodes = object.femReferenceNodes;
    object.femNodes[3][0] += 1.8e-4;
    object.femNodes[3][1] += 0.7e-4;
    object.femNodes[3][2] += 0.9e-4;
    object.femNodes[4][0] -= 0.9e-4;
    object.femNodes[4][1] += 1.2e-4;
    object.femNodes[4][2] -= 1.1e-4;
    object.femFixedNodes = {0u,1u,2u};
    object.tetrahedra = {{{0u,1u,2u,3u}}, {{0u,2u,1u,4u}}};
    if (mode == FEMPreconditionerMode::regionalGalerkin ||
        mode == FEMPreconditionerMode::regionalGalerkinBlockJacobi ||
        mode == FEMPreconditionerMode::regionalGalerkinContact ||
        mode == FEMPreconditionerMode::regionalGalerkinGraphSchwarz)
        object.femPreconditionerRegionIds = {0u,1u};
    world.objects = {std::move(object)};
    return world;
}

using PaperParameters = numi_cardboard_paper::Parameters;
using PaperState = numi_cardboard_paper::State;

double requiredParameter(const MaterialProgram& material,
                         const std::string_view name) {
    for (const Parameter& parameter : material.parameters) {
        if (parameter.name == name) return parameter.defaultValue;
    }
    throw std::runtime_error("paper oracle parameter missing: " +
                             std::string(name));
}

PaperParameters paperParameters(const MaterialProgram& material) {
    using namespace numi_cardboard_paper;
    need(material.internalState.size() == 13u,
         "native paper regional oracle only admits the authored 13-state ideal-Hill layout");
    constexpr std::array<std::string_view, 13> expectedStates{
        "ep11", "ep22", "ep33", "ep23", "ep13", "ep12",
        "sbar11", "sbar22", "sbar33", "sbar23", "sbar13", "sbar12",
        "accumulated_multiplier"};
    for (std::size_t index = 0u; index < expectedStates.size(); ++index) {
        need(material.internalState[index].name == expectedStates[index],
             "native paper regional oracle state order differs from its 13-state source reference");
    }
    PaperParameters result{};
    result.name = material.name;
    result.density = requiredParameter(material, "density");
    constexpr double kPascalPerMegapascal = 1.0e6;
    result.c = {
        requiredParameter(material, "c11") / kPascalPerMegapascal,
        requiredParameter(material, "c22") / kPascalPerMegapascal,
        requiredParameter(material, "c33") / kPascalPerMegapascal,
        requiredParameter(material, "c12") / kPascalPerMegapascal,
        requiredParameter(material, "c13") / kPascalPerMegapascal,
        requiredParameter(material, "c23") / kPascalPerMegapascal};
    result.shear = {
        requiredParameter(material, "g12") / kPascalPerMegapascal,
        requiredParameter(material, "g13") / kPascalPerMegapascal,
        requiredParameter(material, "g23") / kPascalPerMegapascal};
    result.sigmaReference = requiredParameter(material, "sigma_ref") /
        kPascalPerMegapascal;
    result.hill = {
        requiredParameter(material, "hill_F"),
        requiredParameter(material, "hill_G"),
        requiredParameter(material, "hill_H"),
        requiredParameter(material, "hill_L"),
        requiredParameter(material, "hill_M"),
        requiredParameter(material, "hill_N")};
    result.qFloor = requiredParameter(material, "q_floor");
    result.fischerBurmeisterEpsilon = requiredParameter(material, "fb_epsilon");
    return result;
}

WorldSource paperSource(MaterialProgram material,
                        const FEMPreconditionerMode mode,
                        const double displacementScale,
                        const bool rotateMaterialAxes = false,
                        const bool useTangentCache = false) {
    need(material.internalState.size() == 13u,
         "paper regional fixture requires a 13-state source program");
    // The caller authors the projection policy before world construction. Do
    // not replace an explicitly requested initializer in this fixture helper.
    for (Parameter& parameter : material.parameters)
        parameter.identifiable = false;
    WorldSource world = source(material, mode);
    world.mixedSolver.cacheFEMAlgorithmicTangent = useTangentCache;
    world.gravity = {0.0, 0.0, -9.81};
    ObjectSource& object = world.objects.front();
    object.name = "source_bound_haj_ali_ideal_hill_native_regional_fixture";
    if (rotateMaterialAxes) {
        // A single uniform 30-degree in-plane material rotation. This is a
        // numerical frame-invariance fixture, not a measured paper orientation.
        object.femMaterialFrameRotations = {
            {{0.0, 0.0, 0.25881904510252074, 0.9659258262890683}},
            {{0.0, 0.0, 0.25881904510252074, 0.9659258262890683}}};
        object.femMaterialFrameSourceIdentity = {
            0x4f97be12a8c630d5ull, 0x8a42d1e6f937b05cull,
            0x7c19e5a20b6438dfull, 0xd34a0f6182be759cull};
    }
    for (std::size_t node = 3u; node < object.femNodes.size(); ++node) {
        for (unsigned axis = 0u; axis < 3u; ++axis) {
            object.femNodes[node][axis] = object.femReferenceNodes[node][axis] +
                displacementScale *
                    (object.femNodes[node][axis] -
                     object.femReferenceNodes[node][axis]);
        }
    }
    return world;
}

WorldSource cohesiveOperatorSource(
    const MaterialProgram& material,
    const bool includeMixedMortar
) {
    WorldSource world = source(
        material, FEMPreconditionerMode::regionalGalerkinContact);
    ObjectSource& object = world.objects.front();
    object.name = includeMixedMortar
        ? "regional_contact_active_mortar_operator_fixture"
        : "regional_contact_mortar_operator_control";

    // Split only the interface nodes of the two-tet control mesh. The source
    // deformation and all other reference coordinates remain identical in
    // both worlds, so the AP difference isolates the active face operator.
    const std::array<std::uint32_t, 3> duplicatedNodes{0u, 2u, 1u};
    std::array<std::uint32_t, 3> sideB{};
    for (std::size_t slot = 0u; slot < duplicatedNodes.size(); ++slot) {
        const std::uint32_t original = duplicatedNodes[slot];
        sideB[slot] = static_cast<std::uint32_t>(object.femNodes.size());
        object.femNodes.push_back(object.femNodes[original]);
        object.femReferenceNodes.push_back(object.femReferenceNodes[original]);
    }
    // Preserve the orientation of the original second tetrahedron
    // (0,2,1,4) while giving it an independent coincident interface.
    object.tetrahedra[1] = {{sideB[0], sideB[1], sideB[2], 4u}};
    object.femMaterialIndices = {0u, 0u};
    object.femPreconditionerRegionIds = {0u, 1u};
    object.femFixedNodes = {0u, 1u, 2u};
    object.femReferenceDisplacementGradient = true;

    if (includeMixedMortar) {
        CohesiveMaterialSource cohesive;
        cohesive.name = "synthetic_native_preconditioner_operator_only";
        cohesive.normalPenaltyPaPerM = 1.0e9;
        cohesive.shearPenaltyPaPerM = 1.0e9;
        cohesive.compressionPenaltyPaPerM = 2.0e9;
        cohesive.normalStrengthPa = 1.0e6;
        cohesive.shearStrengthPa = 1.0e6;
        cohesive.fractureEnergyModeIJPerM2 = 1.0e3;
        cohesive.fractureEnergyModeIIJPerM2 = 1.5e3;
        cohesive.bkExponent = 1.5;
        world.cohesiveMaterials.push_back(cohesive);

        CohesiveMortarPatchSource patch;
        patch.sideANodes = {0u, 2u, 1u};
        patch.sideBNodes = sideB;
        patch.sideAVertexBarycentric = {{{1.0, 0.0, 0.0},
                                          {0.0, 1.0, 0.0},
                                          {0.0, 0.0, 1.0}}};
        patch.sideBVertexBarycentric = patch.sideAVertexBarycentric;
        patch.firstTetrahedron = 0u;
        patch.secondTetrahedron = 1u;
        patch.materialIndex = 0u;
        patch.stableIdentifier = 771u;
        patch.cohesiveMaterialIndex = 0u;
        patch.activationGapToleranceM = 1.0e-6;
        object.femCohesiveMortarPatches.push_back(patch);
    }
    return world;
}

CompiledWorld cook(const WorldSource& source,
                    const std::uint32_t maximumRateExponent = 0u) {
    const auto result = compileWorld(source,
        {.maximumRateExponent=maximumRateExponent,.emitSpecializedMetal=false});
    std::string diagnostics;
    for (const auto& item:result.diagnostics) diagnostics += item.message + "; ";
    need(result.succeeded(),"regional native fixture compile failed: " + diagnostics);
    return result.world;
}

WorldSource scheduledBlockSource(const MaterialProgram& material,
                                 const bool cacheFallback) {
    WorldSource world = source(
        material, FEMPreconditionerMode::coupledFEMTangentBlockJacobi);
    constexpr double smallCharacteristicLength = 1.0e-2;
    double density = 0.0;
    double modulus = 0.0;
    for (const Parameter& parameter : material.parameters) {
        if (parameter.name == "density") density = parameter.defaultValue;
        if (parameter.dimension == kPressure)
            modulus = std::max(modulus, std::abs(parameter.upper));
    }
    need(density > 0.0 && modulus > 0.0,
         "scheduled block fixture requires positive material density and stiffness");
    const double waveSpeed = std::sqrt(modulus / density);
    // Select a timestep 1.5 times the smaller body's authored CFL limit.
    // With a 2:1 geometry/characteristic-length ratio this yields exponents
    // 1 and 0 while keeping both compiled rates consistent with their meshes.
    world.frameTimestep = 1.5 * 0.35 * smallCharacteristicLength / waveSpeed;
    world.gravity = {0.0, 0.0, -9.81};

    ObjectSource slow = world.objects.front();
    slow.name = "scheduled_block_non_due_slow_fem";
    slow.characteristicLength = smallCharacteristicLength;
    slow.femNodes = slow.femReferenceNodes;
    for (auto& point : slow.femNodes) point[0] += 0.05;
    for (auto& point : slow.femReferenceNodes) point[0] += 0.05;

    ObjectSource& fast = world.objects.front();
    fast.name = "scheduled_block_due_fast_fem";
    fast.characteristicLength = 2.0 * smallCharacteristicLength;
    for (auto& point : fast.femReferenceNodes)
        for (double& coordinate : point) coordinate *= 2.0;
    fast.femNodes = fast.femReferenceNodes;

    // Both immutable objects are separated spatially and have deformable
    // contact disabled; this fixture isolates scheduler gating with physical
    // scale metadata and nonzero gravity loading at the reference pose.
    world.objects.push_back(std::move(slow));
    world.mixedSolver.cacheFEMAlgorithmicTangent = false;
    world.mixedSolver.cacheFEMAlgorithmicTangentWithDirectFallback =
        cacheFallback;
    return world;
}

struct RunResult {
    CompiledWorld world;
    RuntimeStateSnapshot snapshot;
    std::uint64_t executionFingerprint = 0u;
};

class NativeRun {
public:
    NativeRun(const CompiledWorld& world, const std::filesystem::path& metallib)
        : world_(world) {
        device_ = MTLCreateSystemDefaultDevice();
        need(device_ != nil,"Metal device unavailable");
        const std::string deviceName([[device_ name] UTF8String]);
        need(deviceName.find("Apple") != std::string::npos &&
             deviceName.find("Paravirtual") == std::string::npos,
             "physical Apple GPU required for native test");
        queue_ = [device_ newCommandQueue];
        need(queue_ != nil,"Metal command queue unavailable");
        RuntimeConfiguration config;
        config.metallib = metallib;
        config.environmentCount = world_.dispatch.environmentCount;
        config.captureEvents = false;
        config.captureDiagnostics = true;
        config.adaptiveTransfer = false;
        const auto initialized = runtime_.initialize(world_,config);
        need(initialized.encoded,"Runtime initialize failed: "+initialized.message);
        statuses_ = [device_ newBufferWithLength:
            sizeof(MRMetalWorldStatusGPU)*world_.dispatch.environmentCount
            options:MTLResourceStorageModeShared];
        need(statuses_ != nil,"cannot allocate native environment status");
    }

    RunResult run(
        const std::uint32_t controlStep = 0u,
        const std::vector<nm_float4>& kinematicTargets = {},
        const std::vector<NMCohesiveActivationCommandGPU>& activationCommands = {}) {
        auto* status = static_cast<MRMetalWorldStatusGPU*>(statuses_.contents);
        for (std::uint32_t environment = 0u;
             environment < world_.dispatch.environmentCount; ++environment) {
            status[environment] = {};
            status[environment].environment = environment;
        }
        id<MTLBuffer> targets = nil;
        if (!kinematicTargets.empty()) {
            const std::size_t expectedCount =
                std::size_t(world_.dispatch.environmentCount) *
                world_.dispatch.femNodeCount;
            need(kinematicTargets.size() == expectedCount,
                 "regional grip target count does not cover every FEM node");
            targets = [device_ newBufferWithLength:
                kinematicTargets.size() * sizeof(nm_float4)
                options:MTLResourceStorageModeShared];
            need(targets != nil,
                 "cannot allocate native FEM kinematic-target arena");
            std::memcpy(targets.contents, kinematicTargets.data(),
                        kinematicTargets.size() * sizeof(nm_float4));
        }
        id<MTLBuffer> activationBuffer = nil;
        if (!activationCommands.empty()) {
            activationBuffer = [device_ newBufferWithLength:
                activationCommands.size() *
                    sizeof(NMCohesiveActivationCommandGPU)
                options:MTLResourceStorageModeShared];
            need(activationBuffer != nil,
                 "cannot allocate cohesive activation command arena");
            std::memcpy(activationBuffer.contents, activationCommands.data(),
                activationCommands.size() *
                    sizeof(NMCohesiveActivationCommandGPU));
        }
        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        need(command != nil,"cannot create native command buffer");
        EncodeRequest request;
        request.commandBuffer = (__bridge void*)command;
        request.environmentStatuses = (__bridge void*)statuses_;
        request.controlStep = controlStep;
        request.physicsSubsteps = 1u;
        request.timestepSeconds = runtime_.timestepSeconds();
        request.runAdaptiveTransfer = false;
        if (targets != nil) {
            request.femKinematicTargets = (__bridge void*)targets;
            request.femKinematicTargetCount = static_cast<std::uint32_t>(
                kinematicTargets.size());
        }
        if (activationBuffer != nil) {
            request.cohesiveActivationCommands =
                (__bridge void*)activationBuffer;
            request.cohesiveActivationCommandCount =
                static_cast<std::uint32_t>(activationCommands.size());
            request.cohesiveActivationCommandStride =
                sizeof(NMCohesiveActivationCommandGPU);
        }
        request.phase = EncodePhase::preDynamics;
        auto encoded = runtime_.encode(request);
        need(encoded.encoded,"native preDynamics encode failed: "+encoded.message);
        request.phase = EncodePhase::postCommit;
        request.cohesiveActivationCommands = nullptr;
        request.cohesiveActivationCommandCount = 0u;
        request.cohesiveActivationCommandStride = 0u;
        encoded = runtime_.encode(request);
        need(encoded.encoded,"native postCommit encode failed: "+encoded.message);
        [command commit];
        [command waitUntilCompleted];
        need(command.status == MTLCommandBufferStatusCompleted,
             command.error == nil ? "native command failed" :
             std::string([[command.error.localizedDescription description] UTF8String]));
        RuntimeStateSnapshot snapshot = runtime_.snapshot();
        need(snapshot.available,"native snapshot failed: "+snapshot.message);
        if (snapshot.statuses.size() != world_.dispatch.environmentCount ||
            std::any_of(snapshot.statuses.begin(), snapshot.statuses.end(),
                [](const NMMatterStatusGPU& value) {
                    return value.code != NM_STATUS_SUCCESS;
                })) {
            std::string message = "native FEM control_step=" +
                std::to_string(controlStep) + " status=" +
                (snapshot.statuses.empty()
                    ? std::string("missing")
                    : std::to_string(snapshot.statuses[0].code));
            const auto failed = std::find_if(
                snapshot.statuses.begin(), snapshot.statuses.end(),
                [](const NMMatterStatusGPU& value) {
                    return value.code != NM_STATUS_SUCCESS;
                });
            if (failed != snapshot.statuses.end()) {
                const auto& status = *failed;
                message += " object=" + std::to_string(status.objectIndex) +
                    " environment=" + std::to_string(status.environment) +
                    " failing_index=" + std::to_string(status.failingIndex) +
                    " completed_microsteps=" +
                        std::to_string(status.completedMicrosteps) +
                    " fgmres_iterations=" +
                        std::to_string(status.fgmresIterations) +
                    " diagnostics=[" + std::to_string(status.diagnostics.x) +
                        "," + std::to_string(status.diagnostics.y) +
                        "," + std::to_string(status.diagnostics.z) +
                        "," + std::to_string(status.diagnostics.w) + "]";
            }
            throw std::runtime_error(message);
        }
        const std::size_t expectedCertificateCount =
            std::size_t(world_.dispatch.environmentCount) *
            world_.dispatch.objectCount;
        bool allContinuumCertificatesValid =
            snapshot.solverCertificates.size() == expectedCertificateCount;
        if (allContinuumCertificatesValid) {
            for (std::uint32_t environment = 0u;
                 environment < world_.dispatch.environmentCount; ++environment) {
                for (std::uint32_t objectIndex = 0u;
                     objectIndex < world_.dispatch.objectCount; ++objectIndex) {
                    const auto representation =
                        world_.objects[objectIndex].representation;
                    if (representation != NM_REPRESENTATION_FEM &&
                        representation != NM_REPRESENTATION_MPM)
                        continue;
                    const std::size_t index =
                        std::size_t(environment) * world_.dispatch.objectCount +
                        objectIndex;
                    allContinuumCertificatesValid =
                        allContinuumCertificatesValid &&
                        snapshot.solverCertificates[index].validity.w > 0.5f;
                }
            }
        }
        need(allContinuumCertificatesValid,
             "native solve lacks valid per-object continuum certificates: actual=" +
                 std::to_string(snapshot.solverCertificates.size()) +
                 " expected=" + std::to_string(expectedCertificateCount));
        return {world_,std::move(snapshot),runtime_.deviceProgramFingerprint()};
    }

private:
    CompiledWorld world_;
    Runtime runtime_;
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLBuffer> statuses_ = nil;
};

struct BasisResult {
    std::uint32_t modes=0u;
    std::vector<Vec> vectors; // mode-major, node-major xyz
};

BasisResult fp64Basis(const CompiledWorld& world) {
    constexpr std::uint32_t maxRegions=5u;
    constexpr std::uint32_t perRegion=6u;
    const std::size_t nodes=world.dispatch.femNodeCount;
    std::array<std::vector<double>,maxRegions> nodalVolume;
    std::array<double,maxRegions> totalVolume{};
    std::array<Vec,maxRegions> center{};
    std::array<double,maxRegions> radiusMoment{};
    std::uint32_t regions=0u;
    for(auto& x:nodalVolume)x.assign(nodes,0.0);
    for(const auto& tet:world.fem.tetrahedra){
        const auto r=regionOf(tet);
        need(r<maxRegions,"region label exceeds reference bound");
        regions=std::max(regions,r+1u);
        const double volume=tet.inverseRestRow0.w;
        totalVolume[r]+=volume;
        const std::array<std::uint32_t,4> ids{tet.nodes.x,tet.nodes.y,tet.nodes.z,tet.nodes.w};
        Vec c{};
        for(auto id:ids){c=add(c,scale(from4(world.fem.nodes[id].positionAndMass),.25));nodalVolume[r][id]+=.25*volume;}
        center[r]=add(center[r],scale(c,volume));
    }
    for(std::uint32_t r=0;r<regions;++r){need(totalVolume[r]>0,"empty reference region");center[r]=scale(center[r],1.0/totalVolume[r]);}
    for(const auto& tet:world.fem.tetrahedra){const auto r=regionOf(tet);const double v=tet.inverseRestRow0.w;
        for(auto id:{tet.nodes.x,tet.nodes.y,tet.nodes.z,tet.nodes.w}){
            const Vec d=sub(from4(world.fem.nodes[id].positionAndMass),center[r]);radiusMoment[r]+=.25*v*dot(d,d);}}
    std::vector<double> nodalTotal(nodes,0.0);
    for(std::uint32_t r=0;r<regions;++r)for(std::size_t n=0;n<nodes;++n)nodalTotal[n]+=nodalVolume[r][n];
    std::vector<std::vector<double>> q;
    for(std::uint32_t r=0;r<regions;++r){const double radius=std::sqrt(radiusMoment[r]/totalVolume[r]);
        for(std::uint32_t local=0;local<perRegion;++local){
            std::vector<double> candidate(nodes*3u,0.0);
            for(std::size_t n=0;n<nodes;++n){
                if(nodalTotal[n]<=0.0||world.fem.nodes[n].restAndFixed.w!=0.0f)continue;
                const double weight=nodalVolume[r][n]/nodalTotal[n];if(weight<=0.0)continue;
                const Vec p=from4(world.fem.nodes[n].positionAndMass);const double x=p[0]-center[r][0],y=p[1]-center[r][1],z=p[2]-center[r][2];
                Vec value{};
                if(local<3u)value[local]=weight;
                else if(radius>1e-15){const double s=weight/radius;
                    if(local==3u)value={0.0,-s*z,s*y};
                    if(local==4u)value={s*z,0.0,-s*x};
                    if(local==5u)value={-s*y,s*x,0.0};}
                for(unsigned a=0;a<3;++a)candidate[n*3u+a]=value[a];}
            double raw2=0.0;for(double x:candidate)raw2+=x*x;if(!(raw2>0.0))continue;
            for(unsigned pass=0;pass<2u;++pass)for(const auto& basis:q){double projection=0.0;
                for(std::size_t i=0;i<candidate.size();++i)projection+=basis[i]*candidate[i];
                for(std::size_t i=0;i<candidate.size();++i)candidate[i]-=projection*basis[i];}
            double norm2=0.0;for(double x:candidate)norm2+=x*x;const double norm=std::sqrt(norm2);
            if(!(norm>1e-12*std::sqrt(raw2))||!std::isfinite(norm))continue;
            for(double& x:candidate)x/=norm;q.push_back(std::move(candidate));}}
    // Selectors 3 and 4 use the same enriched trial space: after the authored
    // regional translations/rotations, append up to eight global bending
    // fields. Reconstruct those fields independently here so the native
    // export check compares against the actual selector-4 basis, rather than
    // accidentally comparing it to selector 2's six-per-region basis.
    if(world.mixedSolver.executionBudgets.z>=3u && q.size()<32u){
        std::array<double,3> minimum{INFINITY,INFINITY,INFINITY};
        std::array<double,3> maximum{-INFINITY,-INFINITY,-INFINITY};
        for(std::size_t n=0;n<nodes;++n){
            const Vec p=from4(world.fem.nodes[n].positionAndMass);
            for(unsigned axis=0;axis<3u;++axis){
                minimum[axis]=std::min(minimum[axis],p[axis]);
                maximum[axis]=std::max(maximum[axis],p[axis]);}}
        std::array<double,3> center{},halfExtent{};
        std::array<unsigned,3> order{0u,1u,2u};
        for(unsigned axis=0;axis<3u;++axis){
            center[axis]=.5*(minimum[axis]+maximum[axis]);
            halfExtent[axis]=.5*(maximum[axis]-minimum[axis]);}
        std::stable_sort(order.begin(),order.end(),[&](unsigned left,unsigned right){
            return halfExtent[left]>halfExtent[right];});
        const unsigned xAxis=order[0],yAxis=order[1],normalAxis=order[2];
        if(halfExtent[yAxis]>1e-15){
            constexpr std::array<std::array<unsigned,2>,8> powers{{
                {2u,0u},{3u,0u},{0u,2u},{1u,1u},
                {2u,1u},{1u,2u},{0u,3u},{4u,0u}}};
            const auto integerPower=[](double base,unsigned exponent){
                double value=1.0;
                for(unsigned index=0;index<exponent;++index)value*=base;
                return value;};
            for(unsigned mode=0;mode<8u&&q.size()<32u;++mode){
                std::vector<double> candidate(nodes*3u,0.0);
                const unsigned px=powers[mode][0],py=powers[mode][1];
                for(std::size_t n=0;n<nodes;++n){
                    if(nodalTotal[n]<=0.0||
                       world.fem.nodes[n].restAndFixed.w!=0.0f)continue;
                    const Vec p=from4(world.fem.nodes[n].positionAndMass);
                    const double x=(p[xAxis]-center[xAxis])/halfExtent[xAxis];
                    const double y=(p[yAxis]-center[yAxis])/halfExtent[yAxis];
                    const double z=p[normalAxis]-center[normalAxis];
                    const double w=integerPower(x,px)*integerPower(y,py);
                    const double dx=px==0u?0.0:
                        double(px)*integerPower(x,px-1u)*integerPower(y,py)/
                            halfExtent[xAxis];
                    const double dy=py==0u?0.0:
                        double(py)*integerPower(x,px)*integerPower(y,py-1u)/
                            halfExtent[yAxis];
                    candidate[3u*n+normalAxis]=w;
                    candidate[3u*n+xAxis]=-z*dx;
                    candidate[3u*n+yAxis]=-z*dy;
                }
                double raw2=0.0;
                for(double value:candidate)raw2+=value*value;
                if(!(raw2>0.0))continue;
                for(unsigned pass=0;pass<2u;++pass)
                    for(const auto& basis:q){double projection=0.0;
                        for(std::size_t i=0;i<candidate.size();++i)
                            projection+=basis[i]*candidate[i];
                        for(std::size_t i=0;i<candidate.size();++i)
                            candidate[i]-=projection*basis[i];}
                double norm2=0.0;
                for(double value:candidate)norm2+=value*value;
                const double norm=std::sqrt(norm2);
                if(!(norm>1e-12*std::sqrt(raw2))||!std::isfinite(norm))continue;
                for(double& value:candidate)value/=norm;
                q.push_back(std::move(candidate));
            }
        }
    }
    BasisResult result;result.modes=static_cast<std::uint32_t>(q.size());
    result.vectors.resize(q.size()*nodes);
    for(std::size_t m=0;m<q.size();++m)for(std::size_t n=0;n<nodes;++n)
        result.vectors[m*nodes+n]={q[m][3*n],q[m][3*n+1],q[m][3*n+2]};
    return result;
}

BasisResult projectBasisThroughSnapshotMask(
    const BasisResult& authored,
    const RuntimeStateSnapshot& snapshot
) {
    const std::size_t nodes = snapshot.femRegionalGalerkinLinearizationNodes.size();
    need(nodes != 0u && authored.vectors.size() % nodes == 0u,
         "regional candidate mask does not match the authored P node layout");
    const std::size_t modeCount = authored.vectors.size() / nodes;
    BasisResult effective = authored;
    effective.modes = static_cast<std::uint32_t>(modeCount);
    for (std::size_t node = 0u; node < nodes; ++node) {
        const float marker =
            snapshot.femRegionalGalerkinLinearizationNodes[node]
                .restAndFixed.w;
        need(std::isfinite(marker),
             "regional linearization contains a nonfinite constraint marker");
        const bool constrained = marker == 1.0f || marker == 3.0f;
        const bool acceptedConstrained =
            node < snapshot.femNodes.size() &&
            (snapshot.femNodes[node].restAndFixed.w == 1.0f ||
             snapshot.femNodes[node].restAndFixed.w == 3.0f);
        need(node < snapshot.femNodes.size() &&
                 constrained == acceptedConstrained,
             "regional effective P was not built from the committed FEM mask");
        if (!constrained) continue;
        for (std::size_t mode = 0u; mode < modeCount; ++mode)
            effective.vectors[mode * nodes + node] = {};
    }
    return effective;
}

std::uint32_t numericalBasisRank(const BasisResult& basis) {
    if (basis.modes == 0u || basis.vectors.size() % basis.modes != 0u)
        return 0u;
    const std::size_t nodes = basis.vectors.size() / basis.modes;
    std::vector<std::vector<double>> orthogonal;
    for (std::uint32_t mode = 0u; mode < basis.modes; ++mode) {
        std::vector<double> candidate(nodes * 3u, 0.0);
        for (std::size_t node = 0u; node < nodes; ++node) {
            const Vec& value = basis.vectors[std::size_t(mode) * nodes + node];
            for (std::size_t axis = 0u; axis < 3u; ++axis)
                candidate[node * 3u + axis] = value[axis];
        }
        double rawSquared = 0.0;
        for (const double value : candidate) rawSquared += value * value;
        if (!(rawSquared > 0.0) || !std::isfinite(rawSquared)) continue;
        for (unsigned pass = 0u; pass < 2u; ++pass) {
            for (const auto& q : orthogonal) {
                double projection = 0.0;
                for (std::size_t index = 0u; index < candidate.size(); ++index)
                    projection += q[index] * candidate[index];
                for (std::size_t index = 0u; index < candidate.size(); ++index)
                    candidate[index] -= projection * q[index];
            }
        }
        double normSquared = 0.0;
        for (const double value : candidate) normSquared += value * value;
        const double norm = std::sqrt(normSquared);
        if (!(norm > 1.0e-8 * std::sqrt(rawSquared)) || !std::isfinite(norm))
            continue;
        for (double& value : candidate) value /= norm;
        orthogonal.push_back(std::move(candidate));
    }
    return static_cast<std::uint32_t>(orthogonal.size());
}

Mat inverseRest(const NMTetrahedronGPU& t) {
    return {t.inverseRestRow0.x,t.inverseRestRow0.y,t.inverseRestRow0.z,
            t.inverseRestRow1.x,t.inverseRestRow1.y,t.inverseRestRow1.z,
            t.inverseRestRow2.x,t.inverseRestRow2.y,t.inverseRestRow2.z};
}
Mat gradientFrom(const std::array<Vec,4>& values,const Mat& inverseRest) {
    Mat dp{};
    for(unsigned row=0;row<3;++row){dp[3*row]=values[1][row]-values[0][row];
        dp[3*row+1]=values[2][row]-values[0][row];dp[3*row+2]=values[3][row]-values[0][row];}
    return multiply(dp,inverseRest);
}

Mat materialFrameMatrix(const NMTetrahedronGPU& tetrahedron) {
    const double x = tetrahedron.materialFrameRotation.x;
    const double y = tetrahedron.materialFrameRotation.y;
    const double z = tetrahedron.materialFrameRotation.z;
    const double w = tetrahedron.materialFrameRotation.w;
    const double inverseNorm = 1.0 / std::sqrt(x*x + y*y + z*z + w*w);
    const double qx = x * inverseNorm, qy = y * inverseNorm;
    const double qz = z * inverseNorm, qw = w * inverseNorm;
    return {
        1.0 - 2.0*(qy*qy+qz*qz), 2.0*(qx*qy-qz*qw),
            2.0*(qx*qz+qy*qw),
        2.0*(qx*qy+qz*qw), 1.0 - 2.0*(qx*qx+qz*qz),
            2.0*(qy*qz-qx*qw),
        2.0*(qx*qz-qy*qw), 2.0*(qy*qz+qx*qw),
            1.0 - 2.0*(qx*qx+qy*qy)};
}

Mat neoHookeanDirectional(const Mat& f,const Mat& d,const double mu,const double lambda) {
    const double j=determinant(f);need(j>0,"oracle deformation inverted");
    const Mat inv=inverse(f), invT=transpose(inv), dT=transpose(d);
    double trace=0.0;
    const Mat invD=multiply(inv,d);
    trace=invD[0]+invD[4]+invD[8];
    const double coefficient=lambda*std::log(j)-mu;
    return add(add(scale(d,mu),scale(invT,lambda*trace)),
               scale(multiply(multiply(invT,dT),invT),-coefficient));
}

std::vector<Vec> fp64AP(const CompiledWorld& world,
                        const RuntimeStateSnapshot& snapshot,
                        const BasisResult& basis) {
    const std::size_t nodes=world.dispatch.femNodeCount;
    need(snapshot.femRegionalGalerkinLinearizationNodes.size() == nodes,
         "FP64 AP oracle requires one captured candidate node per FEM node");
    const double dt=world.dispatch.gravityAndTimestep.w;
    const double mu=world.materials.at(0).bulk.z;
    const double lambda=world.materials.at(0).bulk.w;
    std::vector<Vec> result(std::size_t(basis.modes)*nodes);
    for(std::uint32_t mode=0;mode<basis.modes;++mode){
        std::vector<Vec> accumulated(nodes);
        for(const auto& tet:world.fem.tetrahedra){
            const std::array<std::uint32_t,4> ids{tet.nodes.x,tet.nodes.y,tet.nodes.z,tet.nodes.w};
            std::array<Vec,4> x{},p{};
            for(unsigned a=0;a<4;++a){
                x[a]=from4(snapshot.femRegionalGalerkinLinearizationNodes[ids[a]].positionAndMass);
                p[a]=basis.vectors[std::size_t(mode)*nodes+ids[a]];}
            const Mat invR=inverseRest(tet);
            const Mat f=gradientFrom(x,invR),d=gradientFrom(p,invR);
            const Mat dp=neoHookeanDirectional(f,d,mu,lambda);
            const Mat h=scale(multiply(dp,transpose(invR)),-double(tet.inverseRestRow0.w));
            std::array<Vec,4> local{};
            for(unsigned axis=0;axis<3;++axis){
                local[1][axis]=h[3*axis];local[2][axis]=h[3*axis+1];local[3][axis]=h[3*axis+2];
                local[0][axis]=-(local[1][axis]+local[2][axis]+local[3][axis]);}
            for(unsigned a=0;a<4;++a)accumulated[ids[a]]=sub(accumulated[ids[a]],scale(local[a],dt*dt));
        }
        for(std::size_t n=0;n<nodes;++n){
            const NMFEMNodeStateGPU& candidate =
                snapshot.femRegionalGalerkinLinearizationNodes[n];
            const float marker = candidate.restAndFixed.w;
            const bool fixedVelocity = marker == 1.0f || marker == 3.0f;
            if(!fixedVelocity)
                accumulated[n]=add(accumulated[n],scale(
                    basis.vectors[std::size_t(mode)*nodes+n],
                    candidate.positionAndMass.w));
            else accumulated[n]={};
            result[std::size_t(mode)*nodes+n]=accumulated[n];}
    }
    return result;
}

std::vector<Vec> fp64PaperAP(const CompiledWorld& world,
                             const RuntimeStateSnapshot& snapshot,
                             const BasisResult& basis,
                             const PaperParameters& parameters,
                             double* maximumPositionVsReferenceFInputDifference = nullptr,
                             double* maximumReconstructedVsCapturedLocalFWordDifference = nullptr,
                             std::uint32_t* referenceDisplacementTetCount = nullptr,
                             std::uint32_t* minimumSchedulerExponent = nullptr,
                             std::uint32_t* maximumSchedulerExponent = nullptr,
                             double* minimumOperatorTimestep = nullptr,
                             double* maximumOperatorTimestep = nullptr,
                             std::vector<Vec>* tangentOnlyOutput = nullptr,
                             std::vector<Vec>* massOnlyOutput = nullptr,
                             const bool useCapturedLocalF = false) {
    using namespace numi_cardboard_solver;
    const std::size_t nodes = world.dispatch.femNodeCount;
    const std::size_t tetrahedra = world.fem.tetrahedra.size();
    const std::size_t stateStride = snapshot.materialStateStride;
    need(snapshot.femRegionalGalerkinLinearizationNodes.size() == nodes,
         "paper FP64 AP oracle requires the captured regional linearization nodes");
    need(stateStride == world.dispatch.materialStateStride && stateStride == 13u &&
         snapshot.femRegionalGalerkinLinearizationMaterialState.size() ==
             tetrahedra * stateStride,
         "paper FP64 AP oracle requires the exact pre-AP accepted 13-state rows");
    const double frameDt = world.dispatch.gravityAndTimestep.w;
    constexpr double kPascalPerMegapascal = 1.0e6;
    double maximumPositionVsReferenceFInputDifferenceLocal = 0.0;
    double maximumReconstructedVsCapturedLocalFWordDifferenceLocal = 0.0;
    std::uint32_t referenceDisplacementTetCountLocal = 0u;
    std::uint32_t minimumSchedulerExponentLocal =
        std::numeric_limits<std::uint32_t>::max();
    std::uint32_t maximumSchedulerExponentLocal = 0u;
    double minimumOperatorTimestepLocal = std::numeric_limits<double>::infinity();
    double maximumOperatorTimestepLocal = 0.0;
    need(snapshot.femRegionalGalerkinLinearizationSchedulerExponent.size() ==
             tetrahedra &&
         snapshot.femRegionalGalerkinLinearizationLocalFHigh.size() ==
             tetrahedra * 3u &&
         snapshot.femRegionalGalerkinLinearizationLocalFLow.size() ==
             tetrahedra * 3u,
         "paper FP64 AP oracle requires the exact pre-AP exponent and local-F words");
    std::vector<Vec> result(std::size_t(basis.modes) * nodes);
    std::vector<Vec> tangentOnly(std::size_t(basis.modes) * nodes);
    std::vector<Vec> massOnly(std::size_t(basis.modes) * nodes);
    for (std::uint32_t mode = 0u; mode < basis.modes; ++mode) {
        std::vector<Vec> accumulated(nodes);
        for (std::size_t tetIndex = 0u; tetIndex < tetrahedra; ++tetIndex) {
            const NMTetrahedronGPU& tet = world.fem.tetrahedra[tetIndex];
            need(tet.identity.y < world.dispatch.objectCount,
                 "paper AP tetrahedron references an invalid object scheduler");
            const std::uint32_t activeExponent =
                snapshot.femRegionalGalerkinLinearizationSchedulerExponent[tetIndex];
            need(activeExponent <= NM_MAX_RATE_EXPONENT,
                 "paper AP snapshot scheduler exponent is out of range");
            const double dt = std::ldexp(frameDt,
                -static_cast<int>(activeExponent));
            minimumSchedulerExponentLocal = std::min(
                minimumSchedulerExponentLocal, activeExponent);
            maximumSchedulerExponentLocal = std::max(
                maximumSchedulerExponentLocal, activeExponent);
            minimumOperatorTimestepLocal = std::min(
                minimumOperatorTimestepLocal, dt);
            maximumOperatorTimestepLocal = std::max(
                maximumOperatorTimestepLocal, dt);
            const std::array<std::uint32_t, 4> ids{
                tet.nodes.x, tet.nodes.y, tet.nodes.z, tet.nodes.w};
            std::array<Vec, 4> x{};
            std::array<Vec, 4> displacement{};
            std::array<Vec, 4> direction{};
            for (unsigned local = 0u; local < 4u; ++local) {
                const NMFEMNodeStateGPU& node =
                    snapshot.femRegionalGalerkinLinearizationNodes[ids[local]];
                x[local] = from4(node.positionAndMass);
                displacement[local] =
                    from4(node.referenceDisplacementAndMode);
                direction[local] = basis.vectors[
                    std::size_t(mode) * nodes + ids[local]];
            }
            const Mat invR = inverseRest(tet);
            const Mat positionFWorld = gradientFrom(x, invR);
            Mat fWorld = positionFWorld;
            if ((tet.identity.w &
                 NM_TETRAHEDRON_REFERENCE_DISPLACEMENT_GRADIENT) != 0u) {
                // Reference-displacement FEM uses the stored FP32 u as the
                // strain unknown. Reconstructing F from materialized X loses
                // small components when X0 + u rounds back to X0. Widening
                // the stored float u and compiled inverse-rest matrix before
                // evaluating I + grad(u) gives the independent FP64 value
                // represented by production's compensated high/low F input.
                for (const std::uint32_t id : ids) {
                    need(snapshot.femRegionalGalerkinLinearizationNodes[id]
                                 .referenceDisplacementAndMode.w == 1.0f,
                         "reference-displacement tetrahedron has a node without authoritative u");
                }
                const Mat gradU = gradientFrom(displacement, invR);
                fWorld = gradU;
                fWorld[0] += 1.0;
                fWorld[4] += 1.0;
                fWorld[8] += 1.0;
                for (std::size_t component = 0u; component < fWorld.size(); ++component) {
                    maximumPositionVsReferenceFInputDifferenceLocal = std::max(
                        maximumPositionVsReferenceFInputDifferenceLocal,
                        std::abs(positionFWorld[component] - fWorld[component]));
                }
                ++referenceDisplacementTetCountLocal;
            }
            const Mat dFWorld = gradientFrom(direction, invR);
            const bool framed =
                (world.objects.at(tet.identity.y).flags &
                 NM_OBJECT_FEM_MATERIAL_FRAME) != 0u;
            const Mat q = materialFrameMatrix(tet);
            const Mat f = framed ? multiply(fWorld, q) : fWorld;
            const Mat dF = framed ? multiply(dFWorld, q) : dFWorld;
            Mat capturedLocalF{};
            const std::size_t fWordBase = tetIndex * 3u;
            for (unsigned row = 0u; row < 3u; ++row) {
                const nm_float4 high = snapshot
                    .femRegionalGalerkinLinearizationLocalFHigh[fWordBase + row];
                const nm_float4 low = snapshot
                    .femRegionalGalerkinLinearizationLocalFLow[fWordBase + row];
                capturedLocalF[row * 3u] = double(high.x) + double(low.x);
                capturedLocalF[row * 3u + 1u] = double(high.y) + double(low.y);
                capturedLocalF[row * 3u + 2u] = double(high.z) + double(low.z);
            }
            for (std::size_t component = 0u; component < f.size(); ++component)
                maximumReconstructedVsCapturedLocalFWordDifferenceLocal = std::max(
                    maximumReconstructedVsCapturedLocalFWordDifferenceLocal,
                    std::abs(f[component] - capturedLocalF[component]));
            PaperState accepted{};
            for (std::size_t state = 0u; state < accepted.size(); ++state) {
                accepted[state] =
                    snapshot.femRegionalGalerkinLinearizationMaterialState[
                        tetIndex * stateStride + state];
                need(std::isfinite(accepted[state]),
                     "paper snapshot contains a nonfinite accepted state");
            }
            // Keep the end-to-end reconstruction as the primary oracle, and
            // admit a separate same-center-input comparison for isolating
            // local constitutive error from kinematic/frame rounding.
            const Mat& materialF = useCapturedLocalF ? capturedLocalF : f;
            const auto projected = project(parameters, materialF, accepted);
            Mat dP = directionalTangent(parameters, materialF, dF,
                                        accepted, projected.state);
            for (double& value : dP) value *= kPascalPerMegapascal;
            if (framed) dP = multiply(dP, transpose(q));
            const Mat h = scale(multiply(dP, transpose(invR)),
                                -double(tet.inverseRestRow0.w));
            std::array<Vec, 4> local{};
            for (unsigned axis = 0u; axis < 3u; ++axis) {
                local[1][axis] = h[3u * axis];
                local[2][axis] = h[3u * axis + 1u];
                local[3][axis] = h[3u * axis + 2u];
                local[0][axis] = -(local[1][axis] + local[2][axis] +
                                   local[3][axis]);
            }
            for (unsigned localNode = 0u; localNode < 4u; ++localNode)
                accumulated[ids[localNode]] = add(
                    accumulated[ids[localNode]],
                    // tetrahedronForce returns the (negative) internal-force
                    // derivative assembled by the production element kernel.
                    // nm_fgmres_gather_nodes forms M*v - dt^2*dF_internal,
                    // so the oracle action must subtract this element vector.
                    scale(local[localNode], -dt * dt));
        }
        for (std::size_t node = 0u; node < nodes; ++node) {
            const NMFEMNodeStateGPU& candidate =
                snapshot.femRegionalGalerkinLinearizationNodes[node];
            const float marker = candidate.restAndFixed.w;
            const bool fixedVelocity = marker == 1.0f || marker == 3.0f;
            if (!fixedVelocity) {
                massOnly[std::size_t(mode) * nodes + node] = scale(
                    basis.vectors[std::size_t(mode) * nodes + node],
                    candidate.positionAndMass.w);
            } else {
                accumulated[node] = {};
                massOnly[std::size_t(mode) * nodes + node] = {};
            }
            tangentOnly[std::size_t(mode) * nodes + node] = accumulated[node];
            result[std::size_t(mode) * nodes + node] = add(
                accumulated[node], massOnly[std::size_t(mode) * nodes + node]);
        }
    }
    if (maximumPositionVsReferenceFInputDifference != nullptr)
        *maximumPositionVsReferenceFInputDifference =
            maximumPositionVsReferenceFInputDifferenceLocal;
    if (maximumReconstructedVsCapturedLocalFWordDifference != nullptr)
        *maximumReconstructedVsCapturedLocalFWordDifference =
            maximumReconstructedVsCapturedLocalFWordDifferenceLocal;
    if (referenceDisplacementTetCount != nullptr)
        *referenceDisplacementTetCount = referenceDisplacementTetCountLocal;
    if (minimumSchedulerExponent != nullptr)
        *minimumSchedulerExponent = minimumSchedulerExponentLocal;
    if (maximumSchedulerExponent != nullptr)
        *maximumSchedulerExponent = maximumSchedulerExponentLocal;
    if (minimumOperatorTimestep != nullptr)
        *minimumOperatorTimestep = minimumOperatorTimestepLocal;
    if (maximumOperatorTimestep != nullptr)
        *maximumOperatorTimestep = maximumOperatorTimestepLocal;
    if (tangentOnlyOutput != nullptr)
        *tangentOnlyOutput = std::move(tangentOnly);
    if (massOnlyOutput != nullptr)
        *massOnlyOutput = std::move(massOnly);
    return result;
}

std::vector<double> fp64RAP(const BasisResult& p,const std::vector<Vec>& ap,
                            const std::size_t nodes) {
    std::vector<double> matrix(std::size_t(p.modes)*p.modes,0.0);
    for(std::uint32_t i=0;i<p.modes;++i)for(std::uint32_t j=0;j<p.modes;++j)
        for(std::size_t n=0;n<nodes;++n)
            matrix[std::size_t(i)*p.modes+j]+=dot(p.vectors[std::size_t(i)*nodes+n],ap[std::size_t(j)*nodes+n]);
    return matrix;
}

std::vector<double> reconstructRAP(const RuntimeStateSnapshot& snapshot,
                                  const std::uint32_t environment) {
    const std::uint32_t k=snapshot.femRegionalGalerkinModeCount;
    const std::size_t base=std::size_t(environment)*k*k;
    const std::size_t pivotBase=std::size_t(environment)*k;
    if (k==0u || environment>=snapshot.femRegionalGalerkinFactorValid.size() ||
        snapshot.femRegionalGalerkinFactorValid[environment]!=1u ||
        snapshot.femRegionalGalerkinPivotedRAP.size()<base+std::size_t(k)*k ||
        snapshot.femRegionalGalerkinPivots.size()<pivotBase+k) return {};
    std::vector<double> permuted(std::size_t(k)*k,0.0),original(permuted.size(),0.0);
    const auto& lu=snapshot.femRegionalGalerkinPivotedRAP;
    const auto& piv=snapshot.femRegionalGalerkinPivots;
    for(std::uint32_t i=0;i<k;++i)for(std::uint32_t j=0;j<k;++j){
        double sum=0.0;for(std::uint32_t m=0;m<k;++m){
            const double l=(i==m)?1.0:(i>m?lu[base+std::size_t(i)*k+m]:0.0);
            const double u=(m<=j)?lu[base+std::size_t(m)*k+j]:0.0;
            sum+=l*u;}
        permuted[std::size_t(i)*k+j]=sum;}
    for(std::uint32_t i=0;i<k;++i){const auto originalRow=piv[pivotBase+i];
        need(originalRow<k,"RAP pivot is out of range");
        for(std::uint32_t j=0;j<k;++j)original[std::size_t(originalRow)*k+j]=permuted[std::size_t(i)*k+j];}
    return original;
}

double relativeError(const std::vector<double>& a,const std::vector<double>& b) {
    if(a.size()!=b.size())return std::numeric_limits<double>::quiet_NaN();
    double e=0.0,s=0.0;for(std::size_t i=0;i<a.size();++i){
        if(!std::isfinite(a[i])||!std::isfinite(b[i]))
            return std::numeric_limits<double>::quiet_NaN();
        e=std::max(e,std::abs(a[i]-b[i]));s=std::max(s,std::abs(b[i]));}
    return e/std::max(s,1e-30);
}
std::vector<double> flatten(const std::vector<nm_float4>& values) {
    std::vector<double> r;r.reserve(values.size()*3u);
    for(const auto& x:values){r.push_back(x.x);r.push_back(x.y);r.push_back(x.z);}return r;
}
std::vector<double> flatten(const std::vector<Vec>& values) {
    std::vector<double> r;r.reserve(values.size()*3u);
    for(const auto& x:values)for(double v:x)r.push_back(v);return r;
}
std::vector<double> flattenLinearization(
    const std::vector<NMFEMNodeStateGPU>& nodes) {
    std::vector<double> r;r.reserve(nodes.size()*20u);
    for(const auto& node:nodes) {
        for(const nm_float4 value:{node.positionAndMass,
                                   node.velocityAndInverseMass,
                                   node.restAndFixed,
                                   node.deltaVelocity,
                                   node.referenceDisplacementAndMode}) {
            r.push_back(value.x);r.push_back(value.y);
            r.push_back(value.z);r.push_back(value.w);
        }
    }
    return r;
}
std::vector<double> flattenFEMNodes(
    const std::vector<NMFEMNodeStateGPU>& nodes) {
    std::vector<double> r;
    r.reserve(nodes.size() * 20u);
    for (const auto& node : nodes) {
        for (const nm_float4 value : {
                 node.positionAndMass,
                 node.velocityAndInverseMass,
                 node.restAndFixed,
                 node.deltaVelocity,
                 node.referenceDisplacementAndMode}) {
            r.push_back(value.x); r.push_back(value.y);
            r.push_back(value.z); r.push_back(value.w);
        }
    }
    return r;
}
std::vector<double> flattenCertificate(
    const NMSolverCertificateGPU& certificate) {
    std::vector<double> r;
    r.reserve(16u);
    for (const nm_float4 value : {
             certificate.nonlinear,
             certificate.contact,
             certificate.transport,
             certificate.validity}) {
        r.push_back(value.x); r.push_back(value.y);
        r.push_back(value.z); r.push_back(value.w);
    }
    return r;
}
void emitJSONArray(const char* key,const std::vector<double>& values) {
    std::cout<<",\""<<key<<"\":[";
    for(std::size_t i=0;i<values.size();++i){
        if(i)std::cout<<',';
        if(std::isfinite(values[i]))
            std::cout<<std::setprecision(12)<<values[i];
        else
            std::cout<<"null";
    }
    std::cout<<']';
}
void emitJSONExactArray(const char* key,const std::vector<double>& values) {
    std::cout<<",\""<<key<<"\":[";
    for(std::size_t i=0;i<values.size();++i){
        if(i)std::cout<<',';
        if(std::isfinite(values[i]))
            std::cout<<std::setprecision(17)<<values[i];
        else
            std::cout<<"null";
    }
    std::cout<<']';
}
void emitSelectorReadback(const char* selector,const RunResult& run) {
    const auto& snapshot=run.snapshot;
    const auto nodeState=flattenFEMNodes(snapshot.femNodes);
    const bool statusAvailable=!snapshot.statuses.empty();
    const bool certificateAvailable=!snapshot.solverCertificates.empty();
    std::cout<<std::setprecision(12)
        <<"{\"schema\":1,\"record\":\"selector_accepted_state\",\"physical_validation\":false"
        <<",\"selector\":\""<<selector<<"\""
        <<",\"world_fingerprint\":"<<run.world.fingerprint
        <<",\"device_program_fingerprint\":"<<run.executionFingerprint
        <<",\"relative_residual_tolerance\":"
        <<run.world.mixedSolver.residualTolerances.x
        <<",\"node_count\":"<<snapshot.femNodes.size()
        <<",\"status_available\":"<<(statusAvailable?"true":"false")
        <<",\"status_code\":";
    if(statusAvailable)std::cout<<snapshot.statuses[0].code;
    else std::cout<<"null";
    std::cout<<",\"status_failing_index\":";
    if(statusAvailable)std::cout<<snapshot.statuses[0].failingIndex;
    else std::cout<<"null";
    std::cout<<",\"completed_microsteps\":";
    if(statusAvailable)std::cout<<snapshot.statuses[0].completedMicrosteps;
    else std::cout<<"null";
    std::cout<<",\"fgmres_iterations\":";
    if(statusAvailable)std::cout<<snapshot.statuses[0].fgmresIterations;
    else std::cout<<"null";
    std::cout<<",\"contact_count\":";
    if(statusAvailable)std::cout<<snapshot.statuses[0].contactCount;
    else std::cout<<"null";
    std::cout<<",\"event_count\":";
    if(statusAvailable)std::cout<<snapshot.statuses[0].eventCount;
    else std::cout<<"null";
    std::cout<<",\"status_diagnostics\":";
    if(statusAvailable) {
        const std::vector<double> statusDiagnostics{
            snapshot.statuses[0].diagnostics.x,
            snapshot.statuses[0].diagnostics.y,
            snapshot.statuses[0].diagnostics.z,
            snapshot.statuses[0].diagnostics.w};
        std::cout<<'[';
        for(std::size_t i=0u;i<statusDiagnostics.size();++i) {
            if(i)std::cout<<',';
            if(std::isfinite(statusDiagnostics[i]))
                std::cout<<std::setprecision(12)<<statusDiagnostics[i];
            else
                std::cout<<"null";
        }
        std::cout<<']';
    } else {
        std::cout<<"null";
    }
    // The public snapshot exposes iteration count and final status residual,
    // but not the private per-environment NMFGMRESStateGPU vector.
    std::cout<<",\"fgmres_private_state_available\":false"
        <<",\"certificate_available\":"
        <<(certificateAvailable?"true":"false")
        <<",\"fem_node_state_float_count\":"<<nodeState.size();
    if(certificateAvailable) {
        emitJSONArray("solver_certificate",
            flattenCertificate(snapshot.solverCertificates[0]));
    } else {
        std::cout<<",\"solver_certificate\":null";
    }
    emitJSONArray("accepted_fem_node_state",nodeState);
    std::cout<<"}\n";
}

void checkRegionalExport(const RunResult& regional) {
    const auto& world=regional.world;const auto& snapshot=regional.snapshot;
    need(snapshot.femRegionalGalerkinDiagnosticsAvailable,
        "native RAP vectors unavailable: "+snapshot.femRegionalGalerkinDiagnosticsMessage);
    const BasisResult expectedP=fp64Basis(world);
    const BasisResult expectedEffectiveP =
        projectBasisThroughSnapshotMask(expectedP, snapshot);
    const std::size_t nodes=world.dispatch.femNodeCount;
    std::vector<double> nativeP=flatten(snapshot.femRegionalGalerkinBasisP);
    std::vector<double> nativeEffectiveP=
        flatten(snapshot.femRegionalGalerkinEffectiveBasisP);
    std::vector<double> expectedPFlat=flatten(expectedP.vectors);
    std::vector<double> expectedEffectivePFlat=
        flatten(expectedEffectiveP.vectors);
    const double pError=relativeError(nativeP,expectedPFlat);
    const double effectivePError=
        relativeError(nativeEffectiveP,expectedEffectivePFlat);
    const auto expectedAP=fp64AP(world,snapshot,expectedEffectiveP);
    const std::vector<double> nativeAP=flatten(snapshot.femRegionalGalerkinOperatorAP);
    const std::vector<double> expectedAPFlat=flatten(expectedAP);
    const double apError=relativeError(nativeAP,expectedAPFlat);
    const std::vector<double> expectedMatrix=fp64RAP(
        expectedEffectiveP,expectedAP,nodes);
    const std::uint32_t effectiveRank = numericalBasisRank(expectedEffectiveP);
    const bool rapFactorValid=
        !snapshot.femRegionalGalerkinFactorValid.empty() &&
        snapshot.femRegionalGalerkinFactorValid[0u]==1u;
    const auto recovered=reconstructRAP(snapshot,0u);
    const double rapError=rapFactorValid
        ? relativeError(recovered,expectedMatrix)
        : std::numeric_limits<double>::quiet_NaN();
    const bool statusAvailable=!snapshot.statuses.empty();
    const bool certificateAvailable=!snapshot.solverCertificates.empty();
    const std::vector<double> linearizationState=flattenLinearization(
        snapshot.femRegionalGalerkinLinearizationNodes);
    std::cout<<std::setprecision(12)
        <<"{\"schema\":1,\"record\":\"regional_rap_export\",\"physical_validation\":false"
        <<",\"modes\":"<<expectedP.modes
        <<",\"native_mode_count\":"<<snapshot.femRegionalGalerkinModeCount
        <<",\"status_available\":"<<(statusAvailable?"true":"false")
        <<",\"status_code\":";
    if (statusAvailable) std::cout<<snapshot.statuses[0].code;
    else std::cout<<"null";
    std::cout<<",\"fgmres_iterations\":";
    if (statusAvailable) std::cout<<snapshot.statuses[0].fgmresIterations;
    else std::cout<<"null";
    std::cout<<",\"certificate_available\":"
        <<(certificateAvailable?"true":"false")
        <<",\"certificate_nonlinear_residual\":";
    if (certificateAvailable && std::isfinite(snapshot.solverCertificates[0].nonlinear.x))
        std::cout<<snapshot.solverCertificates[0].nonlinear.x;
    else std::cout<<"null";
    std::cout
        <<",\"nodes\":"<<nodes<<",\"native_P_value_count\":"<<nativeP.size()
        <<",\"native_effective_P_value_count\":"<<nativeEffectiveP.size()
        <<",\"fp64_P_value_count\":"<<expectedPFlat.size()
        <<",\"effective_basis_rank\":"<<effectiveRank
        <<",\"native_AP_value_count\":"<<nativeAP.size()
        <<",\"fp64_AP_value_count\":"<<expectedAPFlat.size()
        <<",\"P_relative_error\":";
    if (std::isfinite(pError)) std::cout<<pError;
    else std::cout<<"null";
    std::cout<<",\"AP_relative_error\":";
    if (std::isfinite(apError)) std::cout<<apError;
    else std::cout<<"null";
    std::cout<<",\"effective_P_relative_error\":";
    if (std::isfinite(effectivePError)) std::cout<<effectivePError;
    else std::cout<<"null";
    std::cout
        <<",\"RAP_relative_error\":";
    if (std::isfinite(rapError)) std::cout<<rapError;
    else std::cout<<"null";
    std::cout<<",\"rap_factor_valid\":"
             <<(rapFactorValid?"true":"false");
    emitJSONArray("native_P",nativeP);
    emitJSONArray("fp64_P",expectedPFlat);
    emitJSONArray("native_effective_P",nativeEffectiveP);
    emitJSONArray("fp64_effective_P",expectedEffectivePFlat);
    emitJSONArray("native_AP",nativeAP);
    emitJSONArray("fp64_AP",expectedAPFlat);
    emitJSONArray("native_RAP_reconstructed",recovered);
    emitJSONArray("fp64_RAP",expectedMatrix);
    std::vector<double> nativeRAPLU;
    nativeRAPLU.reserve(snapshot.femRegionalGalerkinPivotedRAP.size());
    for(const auto value:snapshot.femRegionalGalerkinPivotedRAP)
        nativeRAPLU.push_back(static_cast<double>(value));
    emitJSONArray("native_RAP_pivoted_LU",nativeRAPLU);
    std::vector<double> nativePivots;
    nativePivots.reserve(snapshot.femRegionalGalerkinPivots.size());
    for(const auto pivot:snapshot.femRegionalGalerkinPivots)
        nativePivots.push_back(static_cast<double>(pivot));
    emitJSONArray("native_RAP_pivots",nativePivots);
    emitJSONArray("linearization_node_state",linearizationState);
    std::cout<<"}\n";
    need(expectedP.modes==snapshot.femRegionalGalerkinModeCount,
         "native and FP64 region basis mode counts differ");
    need(std::isfinite(pError)&&pError<2e-6,
         "native source-region P differs from independent FP64 construction: "+std::to_string(pError));
    need(std::isfinite(effectivePError)&&effectivePError<2e-6,
         "native effective P differs from candidate-mask projection of authored P: "+std::to_string(effectivePError));
    need(std::isfinite(apError)&&apError<2e-3,
         "native AP differs from FP64 production-material operator: "+std::to_string(apError));
    if (effectiveRank < expectedEffectiveP.modes) {
        need(!rapFactorValid,
             "rank-deficient candidate-masked basis did not fail closed to the fine preconditioner");
    } else {
        need(rapFactorValid && rapError<3e-3,
             "native pivoted RAP does not reconstruct FP64 P^TAP: "+std::to_string(rapError));
    }

    if (world.mixedSolver.executionBudgets.z == 3u ||
        world.mixedSolver.executionBudgets.z == 4u) {
        const auto& packed = snapshot.femRegionalGalerkinNodeBlockInverse;
        need(packed.size() == nodes * 3u,
             "selector-3 snapshot omitted its packed nodal inverse blocks");
        double maximumInverseRelativeError = 0.0;
        double maximumIdentityError = 0.0;
        std::size_t validBlocks = 0u;
        for (std::size_t node = 0u; node < nodes; ++node) {
            const float marker = snapshot.femRegionalGalerkinLinearizationNodes[
                node].restAndFixed.w;
            const bool fixed = marker > 0.5f;
            const nm_float4 packedRow0 = packed[node * 3u];
            if (fixed) {
                need(packedRow0.w == 0.0f &&
                     packed[node * 3u + 1u].w == 0.0f &&
                     packed[node * 3u + 2u].w == 0.0f &&
                     packed[node * 3u + 1u].x == 0.0f &&
                     packed[node * 3u + 1u].y == 0.0f &&
                     packed[node * 3u + 1u].z == 0.0f &&
                     packed[node * 3u + 2u].x == 0.0f &&
                     packed[node * 3u + 2u].y == 0.0f &&
                     packed[node * 3u + 2u].z == 0.0f,
                     "fixed node received a nonzero block-Jacobi correction");
                continue;
            }
            if (packedRow0.w <= 0.5f) continue; // documented scalar fallback
            need(packed[node * 3u + 1u].w == 0.0f &&
                 packed[node * 3u + 2u].w == 0.0f,
                 "selector-3 inverse block corrupted its row metadata words");
            ++validBlocks;

            // Form the exact three-column same-node action from the independent
            // FP64 production-material operator, then invert it in FP64. These
            // are three unit-axis probes at one free node, not a diagonal proxy.
            Mat nodalOperator{};
            for (std::size_t column = 0u; column < 3u; ++column) {
                BasisResult probe;
                probe.modes = 1u;
                probe.vectors.assign(nodes, Vec{});
                probe.vectors[node][column] = 1.0;
                const auto action = fp64AP(world, snapshot, probe);
                const Vec value = action[node];
                for (std::size_t row = 0u; row < 3u; ++row)
                    nodalOperator[row * 3u + column] = value[row];
            }
            const Mat expectedInverse = inverse(nodalOperator);
            const Mat nativeInverse{
                packed[node * 3u].x, packed[node * 3u].y,
                packed[node * 3u].z,
                packed[node * 3u + 1u].x, packed[node * 3u + 1u].y,
                packed[node * 3u + 1u].z,
                packed[node * 3u + 2u].x, packed[node * 3u + 2u].y,
                packed[node * 3u + 2u].z};
            std::vector<double> expectedValues(expectedInverse.begin(),
                                                expectedInverse.end());
            std::vector<double> nativeValues(nativeInverse.begin(),
                                              nativeInverse.end());
            const double inverseError = relativeError(nativeValues,
                                                       expectedValues);
            maximumInverseRelativeError = std::max(
                maximumInverseRelativeError, inverseError);
            for (std::size_t row = 0u; row < 3u; ++row)
                for (std::size_t column = 0u; column < 3u; ++column) {
                    double value = 0.0;
                    for (std::size_t k = 0u; k < 3u; ++k)
                        value += nodalOperator[row * 3u + k] *
                            nativeInverse[k * 3u + column];
                    maximumIdentityError = std::max(
                        maximumIdentityError,
                        std::abs(value - (row == column ? 1.0 : 0.0)));
                }
        }
        need(validBlocks > 0u,
             "selector-3 fixture did not produce a valid free-node block");
        std::cout << std::setprecision(12)
            << "{\"schema\":1,\"record\":\"regional_block_jacobi_export\",\"physical_validation\":false"
            << ",\"three_axis_fp64_nodal_actions\":true"
            << ",\"valid_free_node_blocks\":" << validBlocks
            << ",\"maximum_inverse_relative_error\":"
            << maximumInverseRelativeError
            << ",\"maximum_A_times_inverse_identity_error\":"
            << maximumIdentityError << "}\n";
        need(maximumInverseRelativeError < 5.0e-3 &&
             maximumIdentityError < 5.0e-3,
             "native selector-3 block inverse differs from exact FP64 nodal action");
    }
}

void checkPaperRegionalExport(const RunResult& regional,
                              const PaperParameters& parameters,
                              const double displacementScale,
                              const std::string_view fixtureCase) {
    const auto& world = regional.world;
    const auto& snapshot = regional.snapshot;
    need(snapshot.femRegionalGalerkinDiagnosticsAvailable,
         "native paper AP export unavailable: " +
             snapshot.femRegionalGalerkinDiagnosticsMessage);
    const BasisResult expectedP = fp64Basis(world);
    const BasisResult effectiveP =
        projectBasisThroughSnapshotMask(expectedP, snapshot);
    const std::size_t nodes = world.dispatch.femNodeCount;
    BasisResult nativeP;
    nativeP.modes = snapshot.femRegionalGalerkinModeCount;
    BasisResult nativeEffectiveP;
    nativeEffectiveP.modes = nativeP.modes;
    need(snapshot.femRegionalGalerkinBasisP.size() ==
             std::size_t(nativeP.modes) * nodes &&
         snapshot.femRegionalGalerkinEffectiveBasisP.size() ==
             std::size_t(nativeP.modes) * nodes,
         "native paper AP export has incomplete P/effective-P vectors");
    nativeP.vectors.resize(std::size_t(nativeP.modes) * nodes);
    nativeEffectiveP.vectors.resize(std::size_t(nativeP.modes) * nodes);
    for (std::size_t index = 0u; index < nativeP.vectors.size(); ++index) {
        const nm_float4 p = snapshot.femRegionalGalerkinBasisP[index];
        const nm_float4 effective =
            snapshot.femRegionalGalerkinEffectiveBasisP[index];
        nativeP.vectors[index] = {p.x, p.y, p.z};
        nativeEffectiveP.vectors[index] = {
            effective.x, effective.y, effective.z};
    }
    double maximumPositionVsReferenceFInputDifference = 0.0;
    double maximumReconstructedVsCapturedLocalFWordDifference = 0.0;
    std::uint32_t referenceDisplacementTetCount = 0u;
    std::uint32_t minimumSchedulerExponent = 0u;
    std::uint32_t maximumSchedulerExponent = 0u;
    double minimumOperatorTimestep = 0.0;
    double maximumOperatorTimestep = 0.0;
    const auto expectedAP = fp64PaperAP(
        world, snapshot, effectiveP, parameters,
        &maximumPositionVsReferenceFInputDifference,
        &maximumReconstructedVsCapturedLocalFWordDifference,
        &referenceDisplacementTetCount,
        &minimumSchedulerExponent,
        &maximumSchedulerExponent,
        &minimumOperatorTimestep,
        &maximumOperatorTimestep);
    std::vector<Vec> fp64TangentWithNativeBasis;
    std::vector<Vec> fp64MassWithNativeBasis;
    const auto fp64APWithNativeBasis = fp64PaperAP(
        world, snapshot, nativeEffectiveP, parameters,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        &fp64TangentWithNativeBasis, &fp64MassWithNativeBasis);
    const auto expectedAPFlat = flatten(expectedAP);
    const auto fp64APWithNativeBasisFlat = flatten(fp64APWithNativeBasis);
    const auto fp64TangentWithNativeBasisFlat =
        flatten(fp64TangentWithNativeBasis);
    const auto fp64MassWithNativeBasisFlat =
        flatten(fp64MassWithNativeBasis);
    const auto nativeAP = flatten(snapshot.femRegionalGalerkinOperatorAP);
    const double apError = relativeError(nativeAP, expectedAPFlat);
    const double pError = relativeError(flatten(nativeP.vectors),
                                        flatten(expectedP.vectors));
    const double effectivePError = relativeError(
        flatten(nativeEffectiveP.vectors), flatten(effectiveP.vectors));

    // Reproduce the production float32 mass multiply independently. Subtract
    // it from native AP so a basis mismatch cannot be mistaken for a stiffness
    // or constitutive-operator mismatch.
    std::vector<Vec> nativeMassFloat32(nativeEffectiveP.vectors.size());
    std::vector<Vec> nativeAPWithoutMass(nativeEffectiveP.vectors.size());
    need(snapshot.femRegionalGalerkinOperatorAP.size() ==
             nativeEffectiveP.vectors.size(),
         "native paper AP vector is incomplete");
    for (std::uint32_t mode = 0u; mode < nativeP.modes; ++mode) {
        for (std::size_t node = 0u; node < nodes; ++node) {
            const std::size_t index = std::size_t(mode) * nodes + node;
            const auto& state =
                snapshot.femRegionalGalerkinLinearizationNodes.at(node);
            const bool fixed = state.restAndFixed.w == 1.0f ||
                               state.restAndFixed.w == 3.0f;
            const nm_float4 p =
                snapshot.femRegionalGalerkinEffectiveBasisP[index];
            const nm_float4 ap =
                snapshot.femRegionalGalerkinOperatorAP[index];
            const float mass = fixed ? 0.0f : state.positionAndMass.w;
            const Vec massTerm{
                fixed ? 0.0 : double(p.x * mass),
                fixed ? 0.0 : double(p.y * mass),
                fixed ? 0.0 : double(p.z * mass)};
            nativeMassFloat32[index] = massTerm;
            nativeAPWithoutMass[index] = {
                double(ap.x) - massTerm[0],
                double(ap.y) - massTerm[1],
                double(ap.z) - massTerm[2]};
        }
    }
    const auto nativeMassFlat = flatten(nativeMassFloat32);
    const auto nativeAPWithoutMassFlat = flatten(nativeAPWithoutMass);
    const double nativeMassError = relativeError(
        nativeMassFlat, fp64MassWithNativeBasisFlat);
    const double nativeStiffnessError = relativeError(
        nativeAPWithoutMassFlat, fp64TangentWithNativeBasisFlat);
    need(world.materials.size() == 1u && world.constitutive.size() == 1u &&
         snapshot.environmentParameters.size() == world.dispatch.parameterCount,
         "same-center paper reference requires the captured single-environment parameter overlay");
    MaterialProgram capturedParameterMaterial = world.constitutive.front().material;
    const std::size_t parameterOffset = world.materials.front().parameterOffset;
    std::vector<double> capturedParameterValues;
    for (std::size_t index = 0u;
         index < capturedParameterMaterial.parameters.size(); ++index) {
        need(parameterOffset + index < snapshot.environmentParameters.size(),
             "same-center paper reference parameter overlay is incomplete");
        const double value = snapshot.environmentParameters[parameterOffset + index];
        capturedParameterMaterial.parameters[index].defaultValue = value;
        capturedParameterValues.push_back(value);
    }
    const auto capturedParameters = paperParameters(capturedParameterMaterial);
    std::vector<Vec> capturedCenterTangent;
    const auto capturedCenterAP = fp64PaperAP(
        world, snapshot, nativeEffectiveP, capturedParameters,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        &capturedCenterTangent, nullptr, true);
    const auto capturedCenterAPFlat = flatten(capturedCenterAP);
    const auto capturedCenterTangentFlat = flatten(capturedCenterTangent);
    // Cross the two input changes instead of attributing the combined result
    // to either one. A near-yield return map can amplify a tiny F or parameter
    // rounding change even when the tangent at each exact input is correct.
    std::vector<Vec> capturedFSourceParameterTangent;
    const auto capturedFSourceParameterAP = fp64PaperAP(
        world, snapshot, nativeEffectiveP, parameters,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        &capturedFSourceParameterTangent, nullptr, true);
    std::vector<Vec> reconstructedFNativeParameterTangent;
    const auto reconstructedFNativeParameterAP = fp64PaperAP(
        world, snapshot, nativeEffectiveP, capturedParameters,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        &reconstructedFNativeParameterTangent, nullptr, false);
    const auto capturedFSourceParameterAPFlat =
        flatten(capturedFSourceParameterAP);
    const auto capturedFSourceParameterTangentFlat =
        flatten(capturedFSourceParameterTangent);
    const auto reconstructedFNativeParameterAPFlat =
        flatten(reconstructedFNativeParameterAP);
    const auto reconstructedFNativeParameterTangentFlat =
        flatten(reconstructedFNativeParameterTangent);
    std::vector<double> perModeNativeAPError(nativeP.modes, 0.0);
    std::vector<double> perModeStiffnessError(nativeP.modes, 0.0);
    for (std::uint32_t mode = 0u; mode < nativeP.modes; ++mode) {
        const std::size_t first = std::size_t(mode) * nodes * 3u;
        const std::size_t last = first + nodes * 3u;
        const std::vector<double> nativeMode(nativeAP.begin() + first,
                                             nativeAP.begin() + last);
        const std::vector<double> expectedMode(
            fp64APWithNativeBasisFlat.begin() + first,
            fp64APWithNativeBasisFlat.begin() + last);
        const std::vector<double> nativeStiffnessMode(
            nativeAPWithoutMassFlat.begin() + first,
            nativeAPWithoutMassFlat.begin() + last);
        const std::vector<double> expectedStiffnessMode(
            fp64TangentWithNativeBasisFlat.begin() + first,
            fp64TangentWithNativeBasisFlat.begin() + last);
        perModeNativeAPError[mode] = relativeError(nativeMode, expectedMode);
        perModeStiffnessError[mode] = relativeError(
            nativeStiffnessMode, expectedStiffnessMode);
    }
    double maximumPlasticStrainState = 0.0;
    double maximumMultiplierState = 0.0;
    need(snapshot.materialStateStride == 13u,
         "paper regional snapshot did not preserve its 13-state material stride");
    for (std::size_t tet = 0u; tet < world.fem.tetrahedra.size(); ++tet) {
        for (std::size_t state = 0u; state < 6u; ++state) {
            maximumPlasticStrainState = std::max(
                maximumPlasticStrainState,
                std::abs(double(snapshot.femRegionalGalerkinLinearizationMaterialState[
                    tet * 13u + state])));
        }
        maximumMultiplierState = std::max(
            maximumMultiplierState,
            std::abs(double(snapshot.femRegionalGalerkinLinearizationMaterialState[
                tet * 13u + 12u])));
    }
    std::cout << std::setprecision(12)
        << "{\"schema\":1,\"record\":\"native_paper_regional_AP\""
        << ",\"physical_validation\":false"
        << ",\"source_material\":\"" << parameters.name << "\""
        << ",\"selector\":" << world.mixedSolver.executionBudgets.z
        << ",\"fixture_case\":\"" << fixtureCase << "\""
        << ",\"material_frame_enabled\":"
        << ((world.objects.front().flags & NM_OBJECT_FEM_MATERIAL_FRAME) != 0u
                ? "true" : "false")
        << ",\"displacement_scale\":" << displacementScale
        << ",\"material_state_stride\":" << snapshot.materialStateStride
        << ",\"maximum_accepted_plastic_strain_state\":"
        << maximumPlasticStrainState
        << ",\"maximum_accepted_multiplier_state\":"
        << maximumMultiplierState
        << ",\"reference_displacement_gradient_tet_count\":"
        << referenceDisplacementTetCount
        << ",\"maximum_position_vs_reference_F_input_difference\":"
        << maximumPositionVsReferenceFInputDifference
        << ",\"maximum_reconstructed_vs_captured_local_F_word_difference\":"
        << maximumReconstructedVsCapturedLocalFWordDifference
        << ",\"frame_timestep_seconds\":"
        << world.dispatch.gravityAndTimestep.w
        << ",\"minimum_accepted_scheduler_exponent\":"
        << minimumSchedulerExponent
        << ",\"maximum_accepted_scheduler_exponent\":"
        << maximumSchedulerExponent
        << ",\"minimum_operator_timestep_seconds\":"
        << minimumOperatorTimestep
        << ",\"maximum_operator_timestep_seconds\":"
        << maximumOperatorTimestep
        << ",\"native_AP_relative_error\":" << apError
        << ",\"native_P_relative_error\":" << pError
        << ",\"native_effective_P_relative_error\":" << effectivePError
        << ",\"native_AP_vs_FP64_with_native_P_relative_error\":"
        << relativeError(nativeAP, fp64APWithNativeBasisFlat)
        << ",\"native_mass_relative_error\":" << nativeMassError
        << ",\"native_stiffness_after_mass_subtraction_relative_error\":"
        << nativeStiffnessError
        << ",\"native_AP_vs_FP64_captured_center_F_relative_error\":"
        << relativeError(nativeAP, capturedCenterAPFlat)
        << ",\"native_stiffness_vs_FP64_captured_center_F_relative_error\":"
        << relativeError(nativeAPWithoutMassFlat, capturedCenterTangentFlat)
        << ",\"native_AP_vs_FP64_captured_F_source_parameters_relative_error\":"
        << relativeError(nativeAP, capturedFSourceParameterAPFlat)
        << ",\"native_stiffness_vs_FP64_captured_F_source_parameters_relative_error\":"
        << relativeError(nativeAPWithoutMassFlat,
                         capturedFSourceParameterTangentFlat)
        << ",\"native_AP_vs_FP64_reconstructed_F_native_parameters_relative_error\":"
        << relativeError(nativeAP, reconstructedFNativeParameterAPFlat)
        << ",\"native_stiffness_vs_FP64_reconstructed_F_native_parameters_relative_error\":"
        << relativeError(nativeAPWithoutMassFlat,
                         reconstructedFNativeParameterTangentFlat)
        << ",\"captured_center_F_comparison_scope\":\"exact captured local F high+low and native FP32 parameter overlay; independent FP64 direction/frame/force transformations retained; raw layer comparison does not replace primary gate\""
        << ",\"FP64_oracle\":\"source-parameterized 13-state Haj-Ali ideal-Hill StVK, analytic implicit tangent, F reconstructed from authoritative reference displacement when flagged\""
        << ",\"AP_includes_lumped_nodal_mass_and_element_tangent\":true"
        << ",\"mass_action\":\"candidate.positionAndMass.w times mode direction, diagonal lumped nodal mass; zeroed for static/temporary-grip velocity rows\""
        << ",\"stiffness_action\":\"negative objectTimestep squared times tangent force; no diagonal regularization or damping in the production gather operator\"";
    emitJSONExactArray("native_P", flatten(nativeP.vectors));
    emitJSONExactArray("fp64_P", flatten(expectedP.vectors));
    emitJSONExactArray("native_effective_P",
                       flatten(nativeEffectiveP.vectors));
    emitJSONExactArray("fp64_effective_P", flatten(effectiveP.vectors));
    emitJSONExactArray("native_AP", nativeAP);
    emitJSONExactArray("fp64_AP_expected_basis", expectedAPFlat);
    emitJSONExactArray("fp64_AP_native_basis", fp64APWithNativeBasisFlat);
    emitJSONExactArray("native_mass_float32", nativeMassFlat);
    emitJSONExactArray("native_AP_minus_native_mass",
                       nativeAPWithoutMassFlat);
    emitJSONExactArray("fp64_tangent_native_basis",
                       fp64TangentWithNativeBasisFlat);
    emitJSONExactArray("fp64_AP_captured_center_F_native_basis",
                       capturedCenterAPFlat);
    emitJSONExactArray("fp64_tangent_captured_center_F_native_basis",
                       capturedCenterTangentFlat);
    emitJSONExactArray("fp64_AP_captured_F_source_parameters_native_basis",
                       capturedFSourceParameterAPFlat);
    emitJSONExactArray("fp64_tangent_captured_F_source_parameters_native_basis",
                       capturedFSourceParameterTangentFlat);
    emitJSONExactArray("fp64_AP_reconstructed_F_native_parameters_native_basis",
                       reconstructedFNativeParameterAPFlat);
    emitJSONExactArray("fp64_tangent_reconstructed_F_native_parameters_native_basis",
                       reconstructedFNativeParameterTangentFlat);
    emitJSONExactArray("captured_native_parameter_values",
                       capturedParameterValues);
    emitJSONExactArray("fp64_mass_native_basis",
                       fp64MassWithNativeBasisFlat);
    emitJSONExactArray("per_mode_native_AP_relative_error",
                       perModeNativeAPError);
    emitJSONExactArray("per_mode_native_stiffness_relative_error",
                       perModeStiffnessError);
    std::vector<double> schedulerExponents;
    schedulerExponents.reserve(
        snapshot.femRegionalGalerkinLinearizationSchedulerExponent.size());
    for (const auto exponent :
         snapshot.femRegionalGalerkinLinearizationSchedulerExponent)
        schedulerExponents.push_back(double(exponent));
    std::vector<double> acceptedState(
        snapshot.femRegionalGalerkinLinearizationMaterialState.begin(),
        snapshot.femRegionalGalerkinLinearizationMaterialState.end());
    std::vector<double> highWords;
    std::vector<double> lowWords;
    for (const auto value :
         snapshot.femRegionalGalerkinLinearizationLocalFHigh)
        highWords.insert(highWords.end(), {double(value.x), double(value.y),
                                            double(value.z)});
    for (const auto value :
         snapshot.femRegionalGalerkinLinearizationLocalFLow)
        lowWords.insert(lowWords.end(), {double(value.x), double(value.y),
                                         double(value.z)});
    emitJSONExactArray("captured_scheduler_exponent", schedulerExponents);
    emitJSONExactArray("captured_pre_AP_accepted_material_state", acceptedState);
    emitJSONExactArray("captured_local_F_high", highWords);
    emitJSONExactArray("captured_local_F_low", lowWords);
    emitJSONExactArray("captured_linearization_nodes",
                       flattenLinearization(
                           snapshot.femRegionalGalerkinLinearizationNodes));
    std::cout
        << ",\"AP_components_layout\":\"mode-major, node-major xyz; AP=mass+stiffness; native_mass_float32 uses captured effective-P and captured node mass with float32 products\""
        << ",\"operator_error_layer\":\"P/effective-P errors distinguish basis construction from operator action; native AP minus reproduced mass is compared against FP64 stiffness using native effective-P\""
        << "}\n";
    need(std::isfinite(apError) && apError < 5.0e-3,
         "native regional AP differs from independent source-parameterized paper tangent: " +
             std::to_string(apError));
}

void checkInvariance(const RunResult& left, const RunResult& right);
std::vector<nm_float4> singleNodeTarget(
    const RuntimeStateSnapshot& snapshot,
    std::uint32_t nodeIndex,
    float command);

void checkPaperRegionalNative(const MaterialProgram& parsedMaterial,
                              const bool historyOnly = false) {
    const PaperParameters parameters = paperParameters(parsedMaterial);
    constexpr double lowDisplacementScale = 1.0e-3;
    if (!historyOnly) {
    const WorldSource regionalSource = paperSource(
        parsedMaterial, FEMPreconditionerMode::regionalGalerkin,
        lowDisplacementScale);
    const WorldSource enrichedSource = paperSource(
        parsedMaterial, FEMPreconditionerMode::regionalGalerkinContact,
        lowDisplacementScale);
    const CompiledWorld regionalWorld = cook(regionalSource);
    const CompiledWorld enrichedWorld = cook(enrichedSource);
    need(regionalWorld.dispatch.materialStateStride == 13u &&
         enrichedWorld.dispatch.materialStateStride == 13u,
         "paper native fixture compiled to an unexpected internal-state stride");
    need(regionalWorld.mixedSolver.residualTolerances.x ==
             enrichedWorld.mixedSolver.residualTolerances.x,
         "paper selector comparison changed the nonlinear residual tolerance");
    NativeRun regionalRun(regionalWorld, NUMI_MATTER_METALLIB);
    NativeRun enrichedRun(enrichedWorld, NUMI_MATTER_METALLIB);
    const RunResult regional = regionalRun.run();
    const RunResult enriched = enrichedRun.run();
    checkPaperRegionalExport(regional, parameters, lowDisplacementScale,
                             "low_F_regional_scalar_fine");
    checkPaperRegionalExport(enriched, parameters, lowDisplacementScale,
                             "low_F_regional_contact_enriched");
    checkInvariance(regional, enriched);

    // Compare the opt-in per-Newton tangent cache through the same regional
    // FGMRES/RAP solve. The cache is conditioning/operator scratch only; the
    // final production certificate remains evaluated by the direct path.
    const CompiledWorld cachedRegionalWorld = cook(paperSource(
        parsedMaterial, FEMPreconditionerMode::regionalGalerkin,
        lowDisplacementScale, false, true));
    NativeRun cachedRegionalRun(cachedRegionalWorld, NUMI_MATTER_METALLIB);
    const RunResult cachedRegional = cachedRegionalRun.run();
    checkPaperRegionalExport(cachedRegional, parameters, lowDisplacementScale,
                             "low_F_regional_cached_tangent");
    checkInvariance(regional, cachedRegional);

    const CompiledWorld rotatedRegionalWorld = cook(paperSource(
        parsedMaterial, FEMPreconditionerMode::regionalGalerkin,
        lowDisplacementScale, true));
    const CompiledWorld rotatedEnrichedWorld = cook(paperSource(
        parsedMaterial, FEMPreconditionerMode::regionalGalerkinContact,
        lowDisplacementScale, true));
    NativeRun rotatedRegionalRun(rotatedRegionalWorld, NUMI_MATTER_METALLIB);
    NativeRun rotatedEnrichedRun(rotatedEnrichedWorld, NUMI_MATTER_METALLIB);
    const RunResult rotatedRegional = rotatedRegionalRun.run();
    const RunResult rotatedEnriched = rotatedEnrichedRun.run();
    checkPaperRegionalExport(rotatedRegional, parameters, lowDisplacementScale,
                             "rotated_low_F_regional_scalar_fine");
    checkPaperRegionalExport(rotatedEnriched, parameters, lowDisplacementScale,
                             "rotated_low_F_regional_contact_enriched");
    checkInvariance(rotatedRegional, rotatedEnriched);
    const CompiledWorld rotatedCachedRegionalWorld = cook(paperSource(
        parsedMaterial, FEMPreconditionerMode::regionalGalerkin,
        lowDisplacementScale, true, true));
    NativeRun rotatedCachedRegionalRun(
        rotatedCachedRegionalWorld, NUMI_MATTER_METALLIB);
    const RunResult rotatedCachedRegional = rotatedCachedRegionalRun.run();
    checkPaperRegionalExport(rotatedCachedRegional, parameters,
                             lowDisplacementScale,
                             "rotated_low_F_regional_cached_tangent");
    checkInvariance(rotatedRegional, rotatedCachedRegional);
    }

    const auto runCompressedHistory = [&](const FEMPreconditionerMode mode,
                                          const bool useTangentCache = false) {
        WorldSource compressedSource = paperSource(
            parsedMaterial, mode, 0.0, true, useTangentCache);
        compressedSource.gravity = {0.0, 0.0, 0.0};
        NativeRun compressedRun(cook(compressedSource), NUMI_MATTER_METALLIB);
        const RunResult initial = compressedRun.run(0u);
        need(initial.snapshot.femNodes.size() == 5u,
             "paper history fixture changed the two-tet node layout");
        // Node 1 is an authored fixed node in this two-tet fixture. Move it
        // with the static-target command (tag 1), rather than attempting to
        // acquire a temporary grip (tag 3), which is only valid for a free or
        // already-temporary node.
        auto target = singleNodeTarget(initial.snapshot, 1u, 1.0f);
        // A 50 um shortening of the MD edge is a manufactured source-history
        // probe, not a paper specimen loading claim. Hold the accepted static
        // target for a second transaction so regional AP is built from a
        // nonzero accepted 13-state history.
        target[1].x -= 5.0e-5f;
        const RunResult loaded = compressedRun.run(1u, target);
        const RunResult held = compressedRun.run(2u, target);
        need(held.snapshot.femNodes[1u].restAndFixed.w == 1.0f,
             "paper history fixture did not retain its authored static target");
        double plasticHistory = 0.0;
        for (std::size_t tet = 0u;
             tet < compressedSource.objects.front().tetrahedra.size(); ++tet) {
            for (std::size_t state = 0u; state < 6u; ++state) {
                plasticHistory = std::max(plasticHistory, std::abs(double(
                    held.snapshot.femMaterialState[tet * 13u + state])));
            }
        }
        need(plasticHistory > 1.0e-10,
             "manufactured MD compression did not produce accepted paper plastic history");
        checkPaperRegionalExport(held, parameters, 0.0,
            mode == FEMPreconditionerMode::regionalGalerkin
                ? (useTangentCache
                    ? "accepted_history_regional_cached_tangent"
                    : "accepted_history_regional_scalar_fine")
                : "accepted_history_regional_contact_enriched");
        return std::pair<RunResult, double>{held, plasticHistory};
    };
    if (historyOnly) {
        const auto history = runCompressedHistory(
            FEMPreconditionerMode::regionalGalerkin);
        std::cout << "PASS focused source-bound paper plastic-history fixture"
                  << " source_material=" << parsedMaterial.name
                  << " selector=2 history_transactions=3"
                  << " max_accepted_plastic_strain=" << history.second
                  << " physical_validation=false" << '\n';
        return;
    }
    const auto regionalHistory = runCompressedHistory(
        FEMPreconditionerMode::regionalGalerkin);
    const auto cachedRegionalHistory = runCompressedHistory(
        FEMPreconditionerMode::regionalGalerkin, true);
    const auto enrichedHistory = runCompressedHistory(
        FEMPreconditionerMode::regionalGalerkinContact);
    checkInvariance(regionalHistory.first, enrichedHistory.first);
    checkInvariance(regionalHistory.first, cachedRegionalHistory.first);
    std::cout << "PASS native source-bound 13-state paper regional RAP low-F oracle"
              << " source_material=" << parsedMaterial.name
              << " low_F_selector2_and4_transactions=2"
              << " rotated_low_F_selector2_and4_transactions=2"
              << " rotated_accepted_history_selector2_and4_transactions=6"
              << " cached_tangent_parity_cases=3"
              << " cached_tangent_additional_transactions=5"
              << " max_accepted_plastic_strain="
              << std::max(regionalHistory.second, enrichedHistory.second)
              << " physical_validation=false" << '\n';
}

const char* selectorName(const RunResult& run) {
    switch (run.world.mixedSolver.executionBudgets.z) {
        case 0u: return "scalar_diagonal";
        case 2u: return "regional_galerkin_scalar_fine";
        case 3u: return "regional_galerkin_block_jacobi_fine";
        case 4u: return "regional_galerkin_contact_enriched";
        case 5u: return "regional_galerkin_graph_schwarz_potential_node";
        default: return "unsupported_selector";
    }
}

void checkInvariance(const RunResult& left,const RunResult& right) {
    const auto& a=left.snapshot;const auto& b=right.snapshot;
    // Emit both complete accepted FEM state vectors and all public residual,
    // certificate, and FGMRES-count fields before any invariance assertion.
    emitSelectorReadback(selectorName(left),left);
    emitSelectorReadback(selectorName(right),right);
    need(a.femNodes.size()==b.femNodes.size(),"solution node arity differs");
    double maxPosition=0.0,maxVelocity=0.0,maxDisplacementDifference=0.0;
    double displacementScale=0.0,velocityScale=0.0;
    bool allSolutionFieldsFinite=true;
    bool massAndInverseMassMatch=true;
    for(std::size_t i=0;i<a.femNodes.size();++i){
        const Vec pa=from4(a.femNodes[i].positionAndMass),pb=from4(b.femNodes[i].positionAndMass);
        const Vec va=from4(a.femNodes[i].velocityAndInverseMass),vb=from4(b.femNodes[i].velocityAndInverseMass);
        const Vec x0=from4(left.world.fem.nodes[i].positionAndMass);
        for(unsigned j=0;j<3;++j){
            allSolutionFieldsFinite=allSolutionFieldsFinite&&
                std::isfinite(pa[j])&&std::isfinite(pb[j])&&
                std::isfinite(va[j])&&std::isfinite(vb[j])&&std::isfinite(x0[j]);
            if(!std::isfinite(pa[j])||!std::isfinite(pb[j])||
               !std::isfinite(va[j])||!std::isfinite(vb[j])||
               !std::isfinite(x0[j]))continue;
            maxPosition=std::max(maxPosition,std::abs(pa[j]-pb[j]));
            maxVelocity=std::max(maxVelocity,std::abs(va[j]-vb[j]));
            maxDisplacementDifference=std::max(maxDisplacementDifference,
                std::abs((pa[j]-x0[j])-(pb[j]-x0[j])));
            displacementScale=std::max({displacementScale,std::abs(pa[j]-x0[j]),
                std::abs(pb[j]-x0[j])});
            velocityScale=std::max({velocityScale,std::abs(va[j]),std::abs(vb[j])});}
        massAndInverseMassMatch=massAndInverseMassMatch&&
            a.femNodes[i].positionAndMass.w==b.femNodes[i].positionAndMass.w&&
            a.femNodes[i].velocityAndInverseMass.w==b.femNodes[i].velocityAndInverseMass.w;
    }
    const double displacementRel=maxDisplacementDifference/
        std::max(displacementScale,1e-12);
    const double velocityRel=maxVelocity/std::max(velocityScale,1e-12);
    const float residualToleranceA=left.world.mixedSolver.residualTolerances.x;
    const float residualToleranceB=right.world.mixedSolver.residualTolerances.x;
    const auto& ca=a.solverCertificates.at(0u);const auto& cb=b.solverCertificates.at(0u);
    const std::array<float,16> certA{
        ca.nonlinear.x,ca.nonlinear.y,ca.nonlinear.z,ca.nonlinear.w,
        ca.contact.x,ca.contact.y,ca.contact.z,ca.contact.w,
        ca.transport.x,ca.transport.y,ca.transport.z,ca.transport.w,
        ca.validity.x,ca.validity.y,ca.validity.z,ca.validity.w};
    const std::array<float,16> certB{
        cb.nonlinear.x,cb.nonlinear.y,cb.nonlinear.z,cb.nonlinear.w,
        cb.contact.x,cb.contact.y,cb.contact.z,cb.contact.w,
        cb.transport.x,cb.transport.y,cb.transport.z,cb.transport.w,
        cb.validity.x,cb.validity.y,cb.validity.z,cb.validity.w};
    double certDelta=0.0,certScale=1.0;
    for(std::size_t i=0;i<certA.size();++i){
        need(std::isfinite(certA[i])&&std::isfinite(certB[i]),
             "nonfinite solver certificate field");
        certDelta=std::max(certDelta,std::abs(double(certA[i])-certB[i]));
        certScale=std::max({certScale,std::abs(double(certA[i])),std::abs(double(certB[i]))});}
    const auto& statusA=a.statuses.at(0u);const auto& statusB=b.statuses.at(0u);
    std::cout<<std::setprecision(12)
        <<"{\"schema\":1,\"record\":\"selector_invariance\",\"physical_validation\":false"
        <<",\"selector_a\":\""<<selectorName(left)<<"\""
        <<",\"selector_b\":\""<<selectorName(right)<<"\""
        <<",\"world_fingerprint_a\":"<<left.world.fingerprint
        <<",\"world_fingerprint_b\":"<<right.world.fingerprint
        <<",\"device_program_fingerprint_a\":"<<left.executionFingerprint
        <<",\"device_program_fingerprint_b\":"<<right.executionFingerprint
        <<",\"relative_residual_tolerance_a\":"<<residualToleranceA
        <<",\"relative_residual_tolerance_b\":"<<residualToleranceB
        <<",\"relative_residual_tolerances_match\":"
        <<(residualToleranceA==residualToleranceB?"true":"false")
        <<",\"position_max_m\":"<<maxPosition
        <<",\"displacement_difference_max_m\":"<<maxDisplacementDifference
        <<",\"displacement_difference_relative\":"<<displacementRel
        <<",\"velocity_difference_max_m_s\":"<<maxVelocity
        <<",\"velocity_difference_relative\":"<<velocityRel
        <<",\"certificate_a_nonlinear\":"<<ca.nonlinear.x
        <<",\"certificate_b_nonlinear\":"<<cb.nonlinear.x
        <<",\"certificate_max_abs_delta\":"<<certDelta
        <<",\"certificate_scale\":"<<certScale
        <<",\"fgmres_iterations_a\":"<<statusA.fgmresIterations
        <<",\"fgmres_iterations_b\":"<<statusB.fgmresIterations
        <<",\"fgmres_iteration_delta\":"
        <<(static_cast<std::int64_t>(statusB.fgmresIterations)-
           static_cast<std::int64_t>(statusA.fgmresIterations))
        <<",\"algorithmic_tangent_cache_a\":"
        <<(left.world.mixedSolver.executionBudgets.w==1u?"true":"false")
        <<",\"algorithmic_tangent_cache_b\":"
        <<(right.world.mixedSolver.executionBudgets.w==1u?"true":"false")
        <<",\"all_solution_fields_finite\":"
        <<(allSolutionFieldsFinite?"true":"false")
        <<",\"mass_and_inverse_mass_match\":"
        <<(massAndInverseMassMatch?"true":"false")
        <<",\"position_bound_pass\":"<<(maxPosition<1e-7?"true":"false")
        <<",\"displacement_bound_pass\":"<<(displacementRel<3e-3?"true":"false")
        <<",\"velocity_bound_pass\":"<<(velocityRel<3e-3?"true":"false")
        <<",\"certificate_bound_pass\":"
        <<(certDelta<=2.0e-6*certScale?"true":"false")<<"}\n";
    need(allSolutionFieldsFinite,"nonfinite native solution field");
    need(massAndInverseMassMatch,"preconditioner changed mass/inverse mass");
    need(residualToleranceA==residualToleranceB,
         "selector comparison did not use the same nonlinear residual tolerance");
    need(ca.validity.w>0.5f&&cb.validity.w>0.5f,
         "selector comparison lacks accepted certificates");
    need(maxPosition<1e-7&&displacementRel<3e-3&&velocityRel<3e-3,
        "selector changed accepted physical solution beyond the FP32 comparison bound");
    need(certDelta<=2.0e-6*certScale,
        "selector changed the final nonlinear certificate materially");
}

void checkScheduledBlockPreconditioner(const MaterialProgram& material) {
    const CompiledWorld directWorld = cook(
        scheduledBlockSource(material, false), 1u);
    const CompiledWorld cachedWorld = cook(
        scheduledBlockSource(material, true), 1u);
    WorldSource genericCacheSource = scheduledBlockSource(material, false);
    genericCacheSource.mixedSolver.cacheFEMAlgorithmicTangent = true;
    const CompiledWorld genericCacheWorld = cook(genericCacheSource, 1u);
    need(directWorld.dispatch.maximumRateExponent == 1u &&
         cachedWorld.dispatch.maximumRateExponent == 1u &&
         genericCacheWorld.dispatch.maximumRateExponent == 1u,
         "scheduled block fixture did not compile two microticks");
    need(directWorld.objects.size() == 2u && cachedWorld.objects.size() == 2u &&
         genericCacheWorld.objects.size() == 2u &&
         directWorld.objects[0].solver.x == 0u &&
         directWorld.objects[1].solver.x == 1u &&
         cachedWorld.objects[0].solver.x == 0u &&
         cachedWorld.objects[1].solver.x == 1u &&
         genericCacheWorld.objects[0].solver.x == 0u &&
         genericCacheWorld.objects[1].solver.x == 1u,
         "scheduled block fixture did not separate due and non-due object exponents");
    need(directWorld.mixedSolver.executionBudgets.z == 6u &&
         cachedWorld.mixedSolver.executionBudgets.z == 6u &&
         genericCacheWorld.mixedSolver.executionBudgets.z == 6u &&
         directWorld.mixedSolver.executionBudgets.w == 0u &&
         cachedWorld.mixedSolver.executionBudgets.w == 2u &&
         genericCacheWorld.mixedSolver.executionBudgets.w == 1u,
         "scheduled block cache policies were not compiled as requested");

    NativeRun directRun(directWorld, NUMI_MATTER_METALLIB);
    NativeRun cachedRun(cachedWorld, NUMI_MATTER_METALLIB);
    NativeRun genericCacheRun(genericCacheWorld, NUMI_MATTER_METALLIB);
    const RunResult direct = directRun.run();
    const RunResult cached = cachedRun.run();
    const RunResult genericCache = genericCacheRun.run();
    checkInvariance(direct, cached);
    checkInvariance(genericCache, cached);
    need(cached.snapshot.femRegionalGalerkinNodeBlockInverse.size() ==
             genericCache.snapshot.femRegionalGalerkinNodeBlockInverse.size() &&
         !cached.snapshot.femRegionalGalerkinNodeBlockInverse.empty() &&
         std::memcmp(
             cached.snapshot.femRegionalGalerkinNodeBlockInverse.data(),
             genericCache.snapshot.femRegionalGalerkinNodeBlockInverse.data(),
             cached.snapshot.femRegionalGalerkinNodeBlockInverse.size() *
                 sizeof(nm_float4)) == 0,
         "cached-only coupled FEM blocks differ bitwise from the generic valid-cache path");

    const auto checkNonDueMassBlock = [](const RunResult& run,
                                         const char* policy) {
        const auto& snapshot = run.snapshot;
        const auto& world = run.world;
        const std::size_t nodes = world.dispatch.femNodeCount;
        need(snapshot.femRegionalGalerkinNodeBlockInverse.size() ==
                 nodes * 3u,
             std::string(policy) +
                 " scheduled block fixture lacks native block rows");
        need(snapshot.schedulers.size() == world.objects.size() &&
             snapshot.schedulers[0].activeExponent == 0u &&
             snapshot.schedulers[1].activeExponent == 1u,
             std::string(policy) +
                 " scheduler exponents changed before the final microtick");
        double maximumRelativeError = 0.0;
        std::size_t checkedFreeNodes = 0u;
        for (std::size_t node = 0u; node < nodes; ++node) {
            if (world.fem.nodeRanges[node].objectIndex != 0u) continue;
            const auto& state = snapshot.femNodes[node];
            if (state.restAndFixed.w > 0.5f) continue;
            const nm_float4 row0 =
                snapshot.femRegionalGalerkinNodeBlockInverse[node * 3u];
            const nm_float4 row1 =
                snapshot.femRegionalGalerkinNodeBlockInverse[node * 3u + 1u];
            const nm_float4 row2 =
                snapshot.femRegionalGalerkinNodeBlockInverse[node * 3u + 2u];
            need(row0.w == 1.0f && row1.w == 0.0f && row2.w == 0.0f,
                 std::string(policy) +
                     " non-due free node did not receive a finite block");
            const double inverseMass = 1.0 / state.positionAndMass.w;
            const std::array<double, 9> actual{
                row0.x, row0.y, row0.z,
                row1.x, row1.y, row1.z,
                row2.x, row2.y, row2.z};
            const std::array<double, 9> expected{
                inverseMass, 0.0, 0.0,
                0.0, inverseMass, 0.0,
                0.0, 0.0, inverseMass};
            double error = 0.0;
            double scale = 0.0;
            for (std::size_t entry = 0u; entry < actual.size(); ++entry) {
                error = std::max(error,
                    std::abs(actual[entry] - expected[entry]));
                scale = std::max(scale, std::abs(expected[entry]));
            }
            maximumRelativeError = std::max(maximumRelativeError,
                error / std::max(scale, 1.0e-12));
            ++checkedFreeNodes;
        }
        need(checkedFreeNodes == 2u,
             std::string(policy) +
                 " scheduled fixture did not check both non-due free nodes");
        need(maximumRelativeError < 2.0e-5,
             std::string(policy) +
                 " non-due block retained stiffness instead of mass-only inverse");
        std::cout << std::setprecision(12)
            << "{\"schema\":1,\"record\":\"scheduled_block_mass_only\""
            << ",\"policy\":\"" << policy << "\""
            << ",\"non_due_object_exponent\":0"
            << ",\"checked_free_nodes\":" << checkedFreeNodes
            << ",\"maximum_relative_mass_block_error\":"
            << maximumRelativeError
            << ",\"physical_validation\":false}\n";
    };
    const auto checkDueStiffnessBlock = [](const RunResult& run,
                                           const char* policy) {
        const auto& snapshot = run.snapshot;
        const auto& world = run.world;
        const std::size_t nodes = world.dispatch.femNodeCount;
        double maximumRelativeDeviation = 0.0;
        std::size_t checkedFreeNodes = 0u;
        for (std::size_t node = 0u; node < nodes; ++node) {
            if (world.fem.nodeRanges[node].objectIndex != 1u) continue;
            const auto& state = snapshot.femNodes[node];
            if (state.restAndFixed.w > 0.5f) continue;
            const nm_float4 row0 =
                snapshot.femRegionalGalerkinNodeBlockInverse[node * 3u];
            const nm_float4 row1 =
                snapshot.femRegionalGalerkinNodeBlockInverse[node * 3u + 1u];
            const nm_float4 row2 =
                snapshot.femRegionalGalerkinNodeBlockInverse[node * 3u + 2u];
            const double inverseMass = 1.0 / state.positionAndMass.w;
            const std::array<double, 9> actual{
                row0.x, row0.y, row0.z,
                row1.x, row1.y, row1.z,
                row2.x, row2.y, row2.z};
            const std::array<double, 9> massOnly{
                inverseMass, 0.0, 0.0,
                0.0, inverseMass, 0.0,
                0.0, 0.0, inverseMass};
            double deviation = 0.0;
            for (std::size_t entry = 0u; entry < actual.size(); ++entry)
                deviation = std::max(deviation,
                    std::abs(actual[entry] - massOnly[entry]));
            maximumRelativeDeviation = std::max(maximumRelativeDeviation,
                deviation / std::max(inverseMass, 1.0e-12));
            ++checkedFreeNodes;
        }
        need(checkedFreeNodes == 2u,
             std::string(policy) +
                 " scheduled fixture did not check both due free nodes");
        need(maximumRelativeDeviation > 2.0e-5,
             std::string(policy) +
                 " due-object block lost its FEM stiffness contribution");
        std::cout << std::setprecision(12)
            << "{\"schema\":1,\"record\":\"scheduled_block_due_stiffness\""
            << ",\"policy\":\"" << policy << "\""
            << ",\"due_object_exponent\":1"
            << ",\"checked_free_nodes\":" << checkedFreeNodes
            << ",\"maximum_relative_stiffness_deviation\":"
            << maximumRelativeDeviation
            << ",\"physical_validation\":false}\n";
    };
    checkNonDueMassBlock(direct, "direct");
    checkNonDueMassBlock(cached, "cache_fallback");
    checkNonDueMassBlock(genericCache, "generic_cache");
    checkDueStiffnessBlock(direct, "direct");
    checkDueStiffnessBlock(cached, "cache_fallback");
    checkDueStiffnessBlock(genericCache, "generic_cache");
    std::cout << "PASS scheduled FEM block preconditioner due/non-due cache fallback"
              << " direct_vs_cache_fallback_solution_parity=true"
              << " cached_only_vs_generic_cached_block_bitwise=true"
              << " physical_validation=false\n";
}

void checkTangentCacheInvariance(const MaterialProgram& material) {
    const CompiledWorld directWorld = cook(source(
        material, FEMPreconditionerMode::regionalGalerkin, false));
    const CompiledWorld cachedWorld = cook(source(
        material, FEMPreconditionerMode::regionalGalerkin, true));
    need(directWorld.dispatch.femNodeCount == cachedWorld.dispatch.femNodeCount &&
         directWorld.fem.tetrahedra.size() == cachedWorld.fem.tetrahedra.size() &&
         directWorld.mixedSolver.residualTolerances.x ==
             cachedWorld.mixedSolver.residualTolerances.x,
         "tangent-cache comparison changed the physical fixture or tolerance");
    need(directWorld.mixedSolver.executionBudgets.w == 0u &&
         cachedWorld.mixedSolver.executionBudgets.w == 1u,
         "tangent-cache comparison did not compile the requested policies");

    NativeRun directRun(directWorld, NUMI_MATTER_METALLIB);
    NativeRun cachedRun(cachedWorld, NUMI_MATTER_METALLIB);
    const RunResult direct = directRun.run();
    const RunResult cached = cachedRun.run();
    need(direct.snapshot.statuses.at(0u).fgmresIterations > 0u &&
         cached.snapshot.statuses.at(0u).fgmresIterations > 0u,
         "tangent-cache comparison did not execute nonzero FGMRES work");
    checkInvariance(direct, cached);
    std::cout << std::setprecision(12)
        << "{\"schema\":1,\"record\":\"regional_algorithmic_tangent_cache\""
        << ",\"physical_validation\":false"
        << ",\"direct_fgmres_iterations\":"
        << direct.snapshot.statuses.at(0u).fgmresIterations
        << ",\"cached_fgmres_iterations\":"
        << cached.snapshot.statuses.at(0u).fgmresIterations
        << ",\"direct_final_certificate_valid\":"
        << (direct.snapshot.solverCertificates.at(0u).validity.w > 0.5f
                ? "true" : "false")
        << ",\"cached_final_certificate_valid\":"
        << (cached.snapshot.solverCertificates.at(0u).validity.w > 0.5f
                ? "true" : "false")
        << ",\"direct_world_fingerprint\":" << direct.world.fingerprint
        << ",\"cached_world_fingerprint\":" << cached.world.fingerprint
        << "}\n";
}

std::vector<nm_float4> singleNodeTarget(
    const RuntimeStateSnapshot& snapshot,
    const std::uint32_t nodeIndex,
    const float command
) {
    need(nodeIndex < snapshot.femNodes.size(),
         "regional grip fixture node index is out of range");
    std::vector<nm_float4> targets(snapshot.femNodes.size());
    const nm_float4 position = snapshot.femNodes[nodeIndex].positionAndMass;
    targets[nodeIndex] = {position.x, position.y, position.z, command};
    return targets;
}

std::uint32_t constrainedNodeCount(const RuntimeStateSnapshot& snapshot) {
    return static_cast<std::uint32_t>(std::count_if(
        snapshot.femNodes.begin(), snapshot.femNodes.end(),
        [](const NMFEMNodeStateGPU& node) {
            return node.restAndFixed.w > 0.5f;
        }));
}

void checkGripMaskTransition(const MaterialProgram& material) {
    const CompiledWorld world =
        cook(source(material, FEMPreconditionerMode::regionalGalerkin));
    NativeRun run(world, NUMI_MATTER_METALLIB);

    const RunResult threeFixed = run.run(0u);
    checkRegionalExport(threeFixed);
    need(constrainedNodeCount(threeFixed.snapshot) == 3u,
         "regional grip fixture did not start with the three authored fixed nodes");
    const std::vector<double> baseP =
        flatten(threeFixed.snapshot.femRegionalGalerkinBasisP);
    bool authoredModeUsesGripNode = false;
    const std::size_t nodes = world.dispatch.femNodeCount;
    for (std::size_t mode = 0u;
         mode < threeFixed.snapshot.femRegionalGalerkinModeCount; ++mode) {
        const nm_float4 p =
            threeFixed.snapshot.femRegionalGalerkinBasisP[mode * nodes + 3u];
        authoredModeUsesGripNode = authoredModeUsesGripNode ||
            std::abs(p.x) + std::abs(p.y) + std::abs(p.z) > 1.0e-8f;
    }
    need(authoredModeUsesGripNode,
         "new grip node is absent from immutable authored P, so mask projection is untested");

    const auto acquire = singleNodeTarget(threeFixed.snapshot, 3u, 3.0f);
    const RunResult fourFixed = run.run(1u, acquire);
    checkRegionalExport(fourFixed);
    need(constrainedNodeCount(fourFixed.snapshot) == 4u &&
         fourFixed.snapshot.femNodes[3u].restAndFixed.w == 3.0f,
         "accepted temporary grip did not change the FEM candidate constraint mask");
    need(relativeError(baseP,
             flatten(fourFixed.snapshot.femRegionalGalerkinBasisP)) < 1.0e-12,
         "runtime grip mutated immutable source-authored P");
    const std::size_t modes =
        fourFixed.snapshot.femRegionalGalerkinModeCount;
    for (std::size_t mode = 0u; mode < modes; ++mode) {
        const nm_float4 p = fourFixed.snapshot
            .femRegionalGalerkinEffectiveBasisP[mode * nodes + 3u];
        need(p.x == 0.0f && p.y == 0.0f && p.z == 0.0f,
             "effective P retained a constrained temporary-grip row");
    }

    const auto release = singleNodeTarget(fourFixed.snapshot, 3u, 2.0f);
    const RunResult released = run.run(2u, release);
    checkRegionalExport(released);
    need(constrainedNodeCount(released.snapshot) == 3u &&
         released.snapshot.femNodes[3u].restAndFixed.w == 0.0f,
         "accepted grip release did not restore the FEM candidate constraint mask");
    need(relativeError(baseP,
             flatten(released.snapshot.femRegionalGalerkinBasisP)) < 1.0e-12,
         "grip release changed immutable source-authored P");
    need(relativeError(
             flatten(threeFixed.snapshot.femRegionalGalerkinEffectiveBasisP),
             flatten(released.snapshot.femRegionalGalerkinEffectiveBasisP)) <
             1.0e-12,
         "effective P did not return to its initial three-fixed-node projection");

    std::cout << std::setprecision(12)
        << "{\"schema\":1,\"record\":\"regional_kinematic_grip_transition\",\"physical_validation\":false"
        << ",\"source_authored_P_immutable\":true"
        << ",\"fixed_nodes_before\":3,\"fixed_nodes_after_acquire\":4"
        << ",\"fixed_nodes_after_release\":3"
        << ",\"grip_node\":3"
        << ",\"initial_to_acquired_effective_P_delta\":"
        << relativeError(
            flatten(threeFixed.snapshot.femRegionalGalerkinEffectiveBasisP),
            flatten(fourFixed.snapshot.femRegionalGalerkinEffectiveBasisP))
        << ",\"initial_to_released_effective_P_relative_error\":"
        << relativeError(
            flatten(threeFixed.snapshot.femRegionalGalerkinEffectiveBasisP),
            flatten(released.snapshot.femRegionalGalerkinEffectiveBasisP))
        << "}\n";
}

struct EnvironmentGripCommand {
    std::uint32_t environment;
    std::uint32_t node;
    float command;
};

std::vector<nm_float4> makeEnvironmentGripTargets(
    const RuntimeStateSnapshot& snapshot,
    const std::uint32_t environmentCount,
    const std::uint32_t nodesPerEnvironment,
    const std::vector<EnvironmentGripCommand>& commands
) {
    need(snapshot.femNodes.size() ==
             std::size_t(environmentCount) * nodesPerEnvironment,
         "dynamic graph fixture snapshot does not have environment-major FEM nodes");
    std::vector<nm_float4> targets(snapshot.femNodes.size());
    for (const EnvironmentGripCommand& command : commands) {
        need(command.environment < environmentCount &&
                 command.node < nodesPerEnvironment &&
                 (command.command == 2.0f || command.command == 3.0f),
             "dynamic graph fixture authored an invalid grip command");
        const std::size_t index =
            std::size_t(command.environment) * nodesPerEnvironment + command.node;
        const nm_float4 position = snapshot.femNodes[index].positionAndMass;
        targets[index] = {position.x, position.y, position.z, command.command};
    }
    return targets;
}

std::uint32_t constrainedNodeCount(
    const RuntimeStateSnapshot& snapshot,
    const std::uint32_t environment,
    const std::uint32_t nodesPerEnvironment
) {
    const std::size_t first = std::size_t(environment) * nodesPerEnvironment;
    need(first + nodesPerEnvironment <= snapshot.femNodes.size(),
         "dynamic graph fixture environment slice is outside the FEM snapshot");
    return static_cast<std::uint32_t>(std::count_if(
        snapshot.femNodes.begin() + static_cast<std::ptrdiff_t>(first),
        snapshot.femNodes.begin() + static_cast<std::ptrdiff_t>(first + nodesPerEnvironment),
        [](const NMFEMNodeStateGPU& node) {
            return node.restAndFixed.w == 1.0f || node.restAndFixed.w == 3.0f;
        }));
}

void compareDynamicMaskRuns(const RunResult& scalar, const RunResult& graph,
                            const std::uint32_t transaction) {
    const auto& a = scalar.snapshot;
    const auto& b = graph.snapshot;
    need(a.femNodes.size() == b.femNodes.size() &&
             a.statuses.size() == b.statuses.size() &&
             a.solverCertificates.size() == b.solverCertificates.size(),
         "dynamic graph comparison changed environment or state arity");
    const std::uint32_t nodes = scalar.world.dispatch.femNodeCount;
    double maximumPosition = 0.0;
    double maximumVelocity = 0.0;
    double velocityScale = 0.0;
    double maximumCertificate = 0.0;
    bool massesAndInverseMassesMatch = true;
    for (std::size_t index = 0u; index < a.femNodes.size(); ++index) {
        const auto& leftNode = a.femNodes[index];
        const auto& rightNode = b.femNodes[index];
        for (const auto [left, right] : {
                 std::pair{leftNode.positionAndMass.x,
                           rightNode.positionAndMass.x},
                 std::pair{leftNode.positionAndMass.y,
                           rightNode.positionAndMass.y},
                 std::pair{leftNode.positionAndMass.z,
                           rightNode.positionAndMass.z}})
            maximumPosition = std::max(maximumPosition,
                                       std::abs(double(left) - right));
        for (const auto [left, right] : {
                 std::pair{leftNode.velocityAndInverseMass.x,
                           rightNode.velocityAndInverseMass.x},
                 std::pair{leftNode.velocityAndInverseMass.y,
                           rightNode.velocityAndInverseMass.y},
                 std::pair{leftNode.velocityAndInverseMass.z,
                           rightNode.velocityAndInverseMass.z}}) {
            maximumVelocity = std::max(maximumVelocity,
                                       std::abs(double(left) - right));
            velocityScale = std::max({velocityScale, std::abs(double(left)),
                                      std::abs(double(right))});
        }
        massesAndInverseMassesMatch = massesAndInverseMassesMatch &&
            leftNode.positionAndMass.w == rightNode.positionAndMass.w &&
            leftNode.velocityAndInverseMass.w ==
                rightNode.velocityAndInverseMass.w;
    }
    for (std::size_t environment = 0u;
         environment < a.solverCertificates.size(); ++environment) {
        const auto& ca = a.solverCertificates[environment];
        const auto& cb = b.solverCertificates[environment];
        for (const auto [left, right] : {
                 std::pair{ca.nonlinear.x, cb.nonlinear.x},
                 std::pair{ca.nonlinear.y, cb.nonlinear.y},
                 std::pair{ca.nonlinear.z, cb.nonlinear.z},
                 std::pair{ca.nonlinear.w, cb.nonlinear.w}})
            maximumCertificate = std::max(maximumCertificate,
                                           std::abs(double(left) - right));
    }
    const double relativeVelocity = maximumVelocity /
        std::max(velocityScale, 1.0e-12);
    std::cout << std::setprecision(12)
        << "{\"schema\":1,\"record\":\"dynamic_graph_selector_invariance\""
        << ",\"physical_validation\":false,\"transaction\":" << transaction
        << ",\"environment_count\":" << scalar.world.dispatch.environmentCount
        << ",\"node_count_per_environment\":" << nodes
        << ",\"position_max_m\":" << maximumPosition
        << ",\"velocity_relative\":" << relativeVelocity
        << ",\"certificate_nonlinear_max_abs\":" << maximumCertificate
        << ",\"mass_and_inverse_mass_match\":"
        << (massesAndInverseMassesMatch ? "true" : "false") << "}\n";
    need(massesAndInverseMassesMatch,
         "selector-5 dynamic graph changed node mass or inverse mass");
    need(maximumPosition < 1.0e-7 && relativeVelocity < 3.0e-3 &&
             maximumCertificate <= 2.0e-6,
         "selector-5 dynamic graph changed accepted solution or certificate");
}

void checkEffectiveBasisUsesPerEnvironmentMask(
    const RunResult& run,
    const std::uint32_t firstGripNode,
    const std::uint32_t secondGripNode,
    const bool requireDistinctMasks
) {
    const auto& snapshot = run.snapshot;
    const std::uint32_t nodes = run.world.dispatch.femNodeCount;
    const std::uint32_t environments = run.world.dispatch.environmentCount;
    const std::uint32_t modes = snapshot.femRegionalGalerkinModeCount;
    need(modes > 0u &&
             snapshot.femRegionalGalerkinBasisP.size() ==
                 std::size_t(modes) * nodes &&
             snapshot.femRegionalGalerkinEffectiveBasisP.size() ==
                 std::size_t(environments) * modes * nodes,
         "dynamic graph fixture lacks its source/effective regional basis arrays");
    bool firstSourceCovered = false;
    bool secondSourceCovered = false;
    for (std::uint32_t mode = 0u; mode < modes; ++mode) {
        const nm_float4 firstSource =
            snapshot.femRegionalGalerkinBasisP[std::size_t(mode) * nodes + firstGripNode];
        const nm_float4 secondSource =
            snapshot.femRegionalGalerkinBasisP[std::size_t(mode) * nodes + secondGripNode];
        firstSourceCovered = firstSourceCovered ||
            std::abs(firstSource.x) + std::abs(firstSource.y) +
                std::abs(firstSource.z) > 1.0e-8f;
        secondSourceCovered = secondSourceCovered ||
            std::abs(secondSource.x) + std::abs(secondSource.y) +
                std::abs(secondSource.z) > 1.0e-8f;
    }
    need(firstSourceCovered && secondSourceCovered,
         "dynamic graph fixture grip nodes are absent from immutable authored P");
    for (std::uint32_t environment = 0u; environment < environments; ++environment) {
        for (std::uint32_t mode = 0u; mode < modes; ++mode) {
            const std::size_t base =
                (std::size_t(environment) * modes + mode) * nodes;
            for (std::uint32_t node = 0u; node < nodes; ++node) {
                const nm_float4 effective =
                    snapshot.femRegionalGalerkinEffectiveBasisP[base + node];
                const float tag = snapshot.femNodes[
                    std::size_t(environment) * nodes + node].restAndFixed.w;
                if (tag == 1.0f || tag == 3.0f)
                    need(effective.x == 0.0f && effective.y == 0.0f &&
                             effective.z == 0.0f,
                         "effective regional P retained a constrained row");
            }
        }
    }
    const float firstTag = snapshot.femNodes[firstGripNode].restAndFixed.w;
    const float secondEnvironmentFirstTag =
        snapshot.femNodes[std::size_t(nodes) + firstGripNode].restAndFixed.w;
    const float firstEnvironmentSecondTag =
        snapshot.femNodes[secondGripNode].restAndFixed.w;
    const float secondTag = snapshot.femNodes[
        std::size_t(nodes) + secondGripNode].restAndFixed.w;
    if (requireDistinctMasks)
        need(firstTag != secondEnvironmentFirstTag ||
                 firstEnvironmentSecondTag != secondTag,
             "dynamic graph fixture did not exercise distinct per-environment masks");
}

void checkDynamicGraphGripTransitions(const MaterialProgram& material) {
    constexpr std::uint32_t environments = 2u;
    const auto scalarWorld = cook(source(material,
        FEMPreconditionerMode::scalarDiagonal, false, environments));
    const auto graphWorld = cook(source(material,
        FEMPreconditionerMode::regionalGalerkinGraphSchwarz, false,
        environments));
    need(graphWorld.dispatch.femNodeCount == 5u &&
             graphWorld.dispatch.environmentCount == environments &&
             graphWorld.mixedSolver.executionBudgets.z == 5u,
         "dynamic graph fixture did not compile its two-environment selector-5 mesh");
    NativeRun scalarRun(scalarWorld, NUMI_MATTER_METALLIB);
    NativeRun graphRun(graphWorld, NUMI_MATTER_METALLIB);
    RunResult scalar = scalarRun.run(0u);
    RunResult graph = graphRun.run(0u);
    compareDynamicMaskRuns(scalar, graph, 0u);
    need(constrainedNodeCount(graph.snapshot, 0u, 5u) == 3u &&
             constrainedNodeCount(graph.snapshot, 1u, 5u) == 3u,
         "dynamic graph fixture did not start with 3 fixed nodes in each environment");
    for (std::uint32_t environment = 0u; environment < environments; ++environment)
        for (std::uint32_t node = 0u; node < 3u; ++node)
            need(graph.snapshot.femNodes[
                     std::size_t(environment) * 5u + node].restAndFixed.w == 1.0f,
                 "dynamic graph fixture source-fixed node was not authored fixed");

    const auto acquireDistinct = makeEnvironmentGripTargets(
        scalar.snapshot, environments, 5u,
        {{0u, 0u, 2.0f}, {0u, 3u, 3.0f},
         {1u, 1u, 2.0f}, {1u, 4u, 3.0f}});
    scalar = scalarRun.run(1u, acquireDistinct);
    graph = graphRun.run(1u, acquireDistinct);
    compareDynamicMaskRuns(scalar, graph, 1u);
    need(constrainedNodeCount(graph.snapshot, 0u, 5u) == 3u &&
             constrainedNodeCount(graph.snapshot, 1u, 5u) == 3u &&
             graph.snapshot.femNodes[0u].restAndFixed.w == 0.0f &&
             graph.snapshot.femNodes[5u + 1u].restAndFixed.w == 0.0f &&
             graph.snapshot.femNodes[0u].velocityAndInverseMass.w > 0.0f &&
             graph.snapshot.femNodes[5u + 1u].velocityAndInverseMass.w > 0.0f &&
             graph.snapshot.femNodes[3u].restAndFixed.w == 3.0f &&
             graph.snapshot.femNodes[5u + 4u].restAndFixed.w == 3.0f,
         "selector-5 graph did not release source-fixed nodes and acquire per-environment grips");
    checkEffectiveBasisUsesPerEnvironmentMask(graph, 3u, 4u, true);

    const auto env0Release = makeEnvironmentGripTargets(
        scalar.snapshot, environments, 5u, {{0u, 3u, 2.0f}});
    scalar = scalarRun.run(2u, env0Release);
    graph = graphRun.run(2u, env0Release);
    compareDynamicMaskRuns(scalar, graph, 2u);
    need(constrainedNodeCount(graph.snapshot, 0u, 5u) == 2u &&
             constrainedNodeCount(graph.snapshot, 1u, 5u) == 3u &&
             graph.snapshot.femNodes[3u].restAndFixed.w == 0.0f &&
             graph.snapshot.femNodes[5u + 4u].restAndFixed.w == 3.0f,
         "selector-5 graph did not accept a one-environment grip release");
    checkEffectiveBasisUsesPerEnvironmentMask(graph, 3u, 4u, true);

    const auto swapGrips = makeEnvironmentGripTargets(
        scalar.snapshot, environments, 5u,
        {{0u, 4u, 3.0f}, {1u, 4u, 2.0f}, {1u, 1u, 3.0f}});
    scalar = scalarRun.run(3u, swapGrips);
    graph = graphRun.run(3u, swapGrips);
    compareDynamicMaskRuns(scalar, graph, 3u);
    need(graph.snapshot.femNodes[4u].restAndFixed.w == 3.0f &&
             graph.snapshot.femNodes[5u + 4u].restAndFixed.w == 0.0f &&
             graph.snapshot.femNodes[5u + 1u].restAndFixed.w == 3.0f &&
             constrainedNodeCount(graph.snapshot, 0u, 5u) == 3u &&
             constrainedNodeCount(graph.snapshot, 1u, 5u) == 3u,
         "selector-5 graph did not support environment-specific grip exchange");
    checkEffectiveBasisUsesPerEnvironmentMask(graph, 3u, 4u, true);

    const auto releaseEnv0 = makeEnvironmentGripTargets(
        scalar.snapshot, environments, 5u,
        {{0u, 4u, 2.0f}, {1u, 1u, 2.0f}});
    scalar = scalarRun.run(4u, releaseEnv0);
    graph = graphRun.run(4u, releaseEnv0);
    compareDynamicMaskRuns(scalar, graph, 4u);
    need(constrainedNodeCount(graph.snapshot, 0u, 5u) == 2u &&
             constrainedNodeCount(graph.snapshot, 1u, 5u) == 2u &&
             graph.snapshot.femNodes[5u + 1u].restAndFixed.w == 0.0f,
         "selector-5 graph did not release the environment-specific grips");
    checkEffectiveBasisUsesPerEnvironmentMask(graph, 3u, 4u, false);

    std::cout << "{\"schema\":1,\"record\":\"selector5_dynamic_grip_transition_fixture\""
        << ",\"physical_validation\":false,\"accepted_transactions_per_selector\":5"
        << ",\"environment_count\":2,\"initial_fixed_nodes_per_environment\":3"
        << ",\"acquire_node_env0\":3,\"acquire_node_env1\":4"
        << ",\"released_source_fixed_nodes\":true"
        << ",\"released_and_swapped_masks\":true"
        << ",\"selector0_state_certificate_comparison\":true"
        << ",\"contact_free_immutable_FEM_scope\":true}\n";
}

double maxAbsoluteDifference(
    const std::vector<nm_float4>& a,
    const std::vector<nm_float4>& b
) {
    need(a.size() == b.size(), "cohesive AP comparison changed vector size");
    double maximum = 0.0;
    for (std::size_t index = 0u; index < a.size(); ++index) {
        maximum = std::max(maximum, std::abs(double(a[index].x) - b[index].x));
        maximum = std::max(maximum, std::abs(double(a[index].y) - b[index].y));
        maximum = std::max(maximum, std::abs(double(a[index].z) - b[index].z));
    }
    return maximum;
}

void checkActiveCohesiveAP(const MaterialProgram& material) {
    const CompiledWorld controlWorld = cook(
        cohesiveOperatorSource(material, false));
    const CompiledWorld activeWorld = cook(
        cohesiveOperatorSource(material, true));
    need(controlWorld.dispatch.femNodeCount == activeWorld.dispatch.femNodeCount &&
         controlWorld.fem.tetrahedra.size() == activeWorld.fem.tetrahedra.size() &&
         controlWorld.mixedSolver.executionBudgets.z == 4u &&
         activeWorld.mixedSolver.executionBudgets.z == 4u,
         "cohesive operator control and active source do not share selector-4 mesh scope");
    need(controlWorld.fem.cohesiveFaces.empty() &&
         activeWorld.fem.cohesiveFaces.size() == 1u &&
         (activeWorld.fem.cohesiveFaces[0].adjacency.w &
             NM_TOPOLOGY_COHESIVE_DORMANT) != 0u,
         "native active-face fixture must begin with one dormant mixed mortar face");

    NativeRun controlRun(controlWorld, NUMI_MATTER_METALLIB);
    NativeRun activeRun(activeWorld, NUMI_MATTER_METALLIB);
    const RunResult control = controlRun.run(0u);
    checkRegionalExport(control);
    const NMCohesiveActivationCommandGPU command{
        {0u, 0u, 771u, 0u}};
    const RunResult activated = activeRun.run(0u, {}, {command});
    need(activated.snapshot.cohesiveFaces.size() == 1u,
         "native activation snapshot omitted its mixed mortar face");
    const NMCohesiveFaceGPU& face = activated.snapshot.cohesiveFaces.front();
    need((face.adjacency.w & NM_TOPOLOGY_COHESIVE_DORMANT) == 0u &&
         (face.adjacency.w & NM_TOPOLOGY_ACTIVE) != 0u &&
         face.geometry.w > 0.0f,
         "native activation command did not commit an active, positive-area mortar face");
    need(activated.snapshot.femRegionalGalerkinDiagnosticsAvailable &&
         activated.snapshot.femRegionalGalerkinModeCount ==
             control.snapshot.femRegionalGalerkinModeCount &&
         activated.snapshot.femRegionalGalerkinOperatorAP.size() ==
             control.snapshot.femRegionalGalerkinOperatorAP.size(),
         "active and control regional AP exports do not match");
    const double apDelta = maxAbsoluteDifference(
        activated.snapshot.femRegionalGalerkinOperatorAP,
        control.snapshot.femRegionalGalerkinOperatorAP);
    const double apRelativeDelta = relativeError(
        flatten(activated.snapshot.femRegionalGalerkinOperatorAP),
        flatten(control.snapshot.femRegionalGalerkinOperatorAP));
    need(std::isfinite(apDelta) && std::isfinite(apRelativeDelta) &&
         apRelativeDelta > 1.0e-7,
         "activating the mixed mortar face did not change production AP");

    const std::size_t nodes = activeWorld.dispatch.femNodeCount;
    BasisResult effective;
    effective.modes = activated.snapshot.femRegionalGalerkinModeCount;
    effective.vectors.resize(std::size_t(effective.modes) * nodes);
    for (std::size_t mode = 0u; mode < effective.modes; ++mode) {
        for (std::size_t node = 0u; node < nodes; ++node) {
            const nm_float4 p = activated.snapshot
                .femRegionalGalerkinEffectiveBasisP[mode * nodes + node];
            effective.vectors[mode * nodes + node] = {p.x, p.y, p.z};
        }
    }
    std::vector<Vec> observedAP(effective.modes * nodes);
    for (std::size_t mode = 0u; mode < effective.modes; ++mode) {
        for (std::size_t node = 0u; node < nodes; ++node) {
            const nm_float4 value = activated.snapshot
                .femRegionalGalerkinOperatorAP[mode * nodes + node];
            observedAP[mode * nodes + node] = {value.x, value.y, value.z};
        }
    }
    const std::vector<double> matrixFromAP = fp64RAP(
        effective, observedAP, nodes);
    const std::vector<double> recovered = reconstructRAP(
        activated.snapshot, 0u);
    need(!recovered.empty() && relativeError(recovered, matrixFromAP) < 2.0e-4,
         "cohesive AP contribution was not carried into the factored coarse RAP");

    std::cout << std::setprecision(12)
        << "{\"schema\":1,\"record\":\"regional_active_mortar_AP\",\"physical_validation\":false"
        << ",\"selector\":4,\"fixture_law\":\"synthetic mixed-mode bilinear\""
        << ",\"face_stable_id\":771,\"face_active_after_native_acceptance\":true"
        << ",\"face_reference_area_m2\":" << face.geometry.w
        << ",\"cohesive_face_count\":1"
        << ",\"max_active_vs_dormant_AP_delta\":" << apDelta
        << ",\"relative_active_vs_dormant_AP_delta\":" << apRelativeDelta
        << ",\"factored_RAP_matches_PtAP\":true"
        << ",\"AP_includes_active_cohesive_tangent\":true}\n";
}


void checkSixModeRAPFactorSolve(const std::filesystem::path& metallibPath) {
    constexpr std::uint32_t modes = 6u;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    need(device != nil, "six-mode RAP fixture requires a Metal device");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    need(queue != nil, "six-mode RAP fixture cannot create command queue");
    NSError* error = nil;
    NSString* path = [NSString stringWithUTF8String:metallibPath.string().c_str()];
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
    need(library != nil, error == nil ? "cannot load six-mode RAP metallib" :
        std::string("cannot load six-mode RAP metallib: ") +
            [[error.localizedDescription description] UTF8String]);
    auto pipeline = [&](NSString* name) {
        id<MTLFunction> function = [library newFunctionWithName:name];
        need(function != nil, std::string("missing RAP kernel ") + [name UTF8String]);
        NSError* pipelineError = nil;
        id<MTLComputePipelineState> result =
            [device newComputePipelineStateWithFunction:function error:&pipelineError];
        need(result != nil, pipelineError == nil ? "cannot compile RAP kernel" :
            std::string("cannot compile RAP kernel: ") +
                [[pipelineError.localizedDescription description] UTF8String]);
        return result;
    };
    id<MTLComputePipelineState> factor = pipeline(@"numi_matter_metal::nm_fgmres_regional_factor_rap");
    id<MTLComputePipelineState> solve = pipeline(@"numi_matter_metal::nm_fgmres_regional_solve_rap");
    NMMatterDispatchGPU dispatch{};
    dispatch.environmentCount = 1u;
    auto buffer = [&](const void* source, const std::size_t bytes) {
        id<MTLBuffer> result = [device newBufferWithLength:bytes
            options:MTLResourceStorageModeShared];
        need(result != nil, "cannot allocate six-mode RAP fixture buffer");
        if (source != nullptr) std::memcpy(result.contents, source, bytes);
        return result;
    };
    id<MTLBuffer> dispatchBuffer = buffer(&dispatch, sizeof(dispatch));
    std::array<float, modes * modes> matrix{};
    for (std::uint32_t row = 0u; row < modes; ++row) {
        for (std::uint32_t column = 0u; column < modes; ++column) {
            const int residue = int((row * 7u + column * 3u) % 9u) - 4;
            matrix[row * modes + column] = row == column
                ? 5.0f + 0.1f * row : 0.017f * residue;
        }
    }
    std::array<float, modes> rhs{};
    for (std::uint32_t row = 0u; row < modes; ++row)
        rhs[row] = std::sin(0.31f * row) + 0.07f * row;
    id<MTLBuffer> matrixBuffer = buffer(matrix.data(), sizeof(matrix));
    id<MTLBuffer> pivotsBuffer = buffer(nullptr, modes * sizeof(std::uint32_t));
    id<MTLBuffer> validBuffer = buffer(nullptr, sizeof(std::uint32_t));
    id<MTLBuffer> rhsBuffer = buffer(rhs.data(), sizeof(rhs));
    id<MTLBuffer> coefficientsBuffer = buffer(nullptr, sizeof(rhs));
    const std::uint32_t modeCount = modes;
    auto run = [&](id<MTLComputePipelineState> pso, auto bind) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        need(command != nil, "cannot create six-mode RAP command buffer");
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        need(encoder != nil, "cannot create six-mode RAP encoder");
        [encoder setComputePipelineState:pso];
        bind(encoder);
        [encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        need(command.status == MTLCommandBufferStatusCompleted,
            command.error == nil ? "six-mode RAP command failed" :
                [[command.error.localizedDescription description] UTF8String]);
    };
    auto factorAndSolve = [&] {
        run(factor, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
            [encoder setBuffer:matrixBuffer offset:0u atIndex:1u];
            [encoder setBuffer:pivotsBuffer offset:0u atIndex:2u];
            [encoder setBuffer:validBuffer offset:0u atIndex:3u];
            [encoder setBytes:&modeCount length:sizeof(modeCount) atIndex:4u];
        });
        run(solve, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
            [encoder setBuffer:matrixBuffer offset:0u atIndex:1u];
            [encoder setBuffer:pivotsBuffer offset:0u atIndex:2u];
            [encoder setBuffer:validBuffer offset:0u atIndex:3u];
            [encoder setBuffer:rhsBuffer offset:0u atIndex:4u];
            [encoder setBuffer:coefficientsBuffer offset:0u atIndex:5u];
            [encoder setBytes:&modeCount length:sizeof(modeCount) atIndex:6u];
        });
    };
    factorAndSolve();
    need(*static_cast<const std::uint32_t*>(validBuffer.contents) == 1u,
         "six-mode nonsymmetric RAP factor did not validate");
    std::array<double, modes * modes> oracle{};
    std::array<double, modes> expected{};
    for (std::size_t i = 0; i < matrix.size(); ++i) oracle[i] = matrix[i];
    for (std::uint32_t i = 0u; i < modes; ++i) expected[i] = rhs[i];
    for (std::uint32_t pivotColumn = 0u; pivotColumn < modes; ++pivotColumn) {
        std::uint32_t pivotRow = pivotColumn;
        for (std::uint32_t row = pivotColumn + 1u; row < modes; ++row)
            if (std::abs(oracle[row * modes + pivotColumn]) >
                std::abs(oracle[pivotRow * modes + pivotColumn])) pivotRow = row;
        need(std::abs(oracle[pivotRow * modes + pivotColumn]) > 1.0e-12,
             "six-mode RAP FP64 oracle is singular");
        if (pivotRow != pivotColumn) {
            for (std::uint32_t column = 0u; column < modes; ++column)
                std::swap(oracle[pivotColumn * modes + column],
                          oracle[pivotRow * modes + column]);
            std::swap(expected[pivotColumn], expected[pivotRow]);
        }
        for (std::uint32_t row = pivotColumn + 1u; row < modes; ++row) {
            const double scale = oracle[row * modes + pivotColumn] /
                oracle[pivotColumn * modes + pivotColumn];
            for (std::uint32_t column = pivotColumn + 1u; column < modes; ++column)
                oracle[row * modes + column] -= scale *
                    oracle[pivotColumn * modes + column];
            expected[row] -= scale * expected[pivotColumn];
        }
    }
    for (int reverse = int(modes) - 1; reverse >= 0; --reverse) {
        const auto row = static_cast<std::uint32_t>(reverse);
        double value = expected[row];
        for (std::uint32_t column = row + 1u; column < modes; ++column)
            value -= oracle[row * modes + column] * expected[column];
        expected[row] = value / oracle[row * modes + row];
    }
    const auto* actual = static_cast<const float*>(coefficientsBuffer.contents);
    double maxError = 0.0;
    for (std::uint32_t i = 0u; i < modes; ++i) {
        maxError = std::max(maxError, std::abs(double(actual[i]) - expected[i]));
        need(std::abs(double(actual[i]) - expected[i]) < 2e-5,
             "six-mode RAP solve differs from independent FP64 oracle");
    }
    std::fill(matrix.begin(), matrix.end(), 0.0f);
    std::memcpy(matrixBuffer.contents, matrix.data(), sizeof(matrix));
    factorAndSolve();
    need(*static_cast<const std::uint32_t*>(validBuffer.contents) == 0u,
         "singular six-mode RAP did not fail closed");
    const auto* fallback = static_cast<const float*>(coefficientsBuffer.contents);
    for (std::uint32_t i = 0u; i < modes; ++i)
        need(fallback[i] == 0.0f,
             "invalid six-mode RAP solve did not return zero correction");
    std::cout << std::setprecision(9)
        << "PASS selector6_global_coarse_rap_factor_solve modes=6 nonsymmetric=1"
        << " singular_fallback=1 max_oracle_error=" << maxError << std::endl;
}

} // namespace

int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            if (argc == 2 && std::string_view(argv[1]) == "--global-coarse-only") {
                checkSixModeRAPFactorSolve(
                    std::filesystem::path(NUMI_MATTER_METALLIB));
                return 0;
            }
            const bool scheduledBlockOnly = argc >= 3 &&
                std::string_view(argv[1]) == "--scheduled-block-only";
            if (scheduledBlockOnly) {
                need(argc == 3,
                     "usage: fem-regional-galerkin-native-check --scheduled-block-only <starch-material.nmatter>");
                const auto parsed = parseMatterFile(argv[2]);
                need(parsed.succeeded(),
                     "cannot parse scheduled-block finite-elastic starch material");
                MaterialProgram material = parsed.material;
                need(material.hint == ConstitutiveHint::neoHookean &&
                     material.internalState.empty(),
                     "scheduled block fixture requires stateless finite-elastic Neo-Hookean material");
                for (auto& parameter : material.parameters)
                    parameter.identifiable = false;
                checkScheduledBlockPreconditioner(material);
                return 0;
            }
            const bool paperOnly = argc >= 3 &&
                std::string_view(argv[1]) == "--paper-only";
            bool paperHistoryOnly = false;
            bool accuratePaper = false;
            if (paperOnly) {
                for (int argument = 3; argument < argc; ++argument) {
                    const std::string_view option(argv[argument]);
                    if (option == "--plastic-history-only" && !paperHistoryOnly)
                        paperHistoryOnly = true;
                    else if (option == "--accurate-paper" && !accuratePaper)
                        accuratePaper = true;
                    else throw std::runtime_error("unknown or duplicate paper checker option: " +
                                                  std::string(option));
                }
                const auto paperParsed = parseMatterFile(argv[2]);
                need(paperParsed.succeeded(),
                     "cannot parse source-bound 13-state paper material");
                MaterialProgram paperMaterial = paperParsed.material;
                need(paperMaterial.internalState.size() == 13u,
                     "--paper-only requires the independently referenced 13-state ideal-Hill material");
                // Retain the original checker policy unless the explicit
                // scaled, compensated comparison was selected.
                paperMaterial.paperHillScalarInitializer = true;
                paperMaterial.paperCompensatedArithmetic = true;
                paperMaterial.paperHillAccurateProjection = true;
                if (accuratePaper) {
                    paperMaterial.paperHillScalarInitializer = false;
                    paperMaterial.paperCompensatedArithmetic = true;
                    paperMaterial.paperHillAccurateProjection = true;
                    paperMaterial = numi::cardboard::scalePaperProjectionResiduals(
                        std::move(paperMaterial));
                }
                std::cout << "{\"record\":\"native_paper_regional_policy\","
                             "\"flags_authored_before_compile\":true,"
                             "\"accurate_scaled_compensated\":"
                          << (accuratePaper ? "true" : "false")
                          << ",\"scalar_initializer\":"
                          << (paperMaterial.paperHillScalarInitializer ? "true" : "false")
                          << ",\"compensated_arithmetic\":"
                          << (paperMaterial.paperCompensatedArithmetic ? "true" : "false")
                          << ",\"accurate_projection\":"
                          << (paperMaterial.paperHillAccurateProjection ? "true" : "false")
                          << ",\"global_tolerance_changed\":false}\n";
                checkPaperRegionalNative(paperMaterial, paperHistoryOnly);
                return 0;
            }
            need(argc==2,"usage: fem-regional-galerkin-native-check <starch-material.nmatter> | --paper-only <13-state-paper-material.nmatter> [--plastic-history-only] [--accurate-paper]");
            const auto parsed=parseMatterFile(argv[1]);
            need(parsed.succeeded(),"cannot parse elastic starch material");
            MaterialProgram material=parsed.material;
            need(material.hint==ConstitutiveHint::neoHookean &&
                 material.internalState.empty(),
                 "native regional checker requires the finite elastic Neo-Hookean material law");
            for(auto& p:material.parameters)p.identifiable=false;
            const auto scalarWorld=cook(source(material,FEMPreconditionerMode::scalarDiagonal));
            const auto regionalWorld=cook(source(material,FEMPreconditionerMode::regionalGalerkin));
            const auto blockWorld=cook(source(material,FEMPreconditionerMode::regionalGalerkinBlockJacobi));
            need(scalarWorld.dispatch.femNodeCount==regionalWorld.dispatch.femNodeCount&&
                 scalarWorld.dispatch.femNodeCount==blockWorld.dispatch.femNodeCount&&
                 scalarWorld.fem.tetrahedra.size()==regionalWorld.fem.tetrahedra.size()&&
                 scalarWorld.fem.tetrahedra.size()==blockWorld.fem.tetrahedra.size(),
                 "selector comparison changed source mesh layout");
            need(scalarWorld.mixedSolver.residualTolerances.x==
                     regionalWorld.mixedSolver.residualTolerances.x&&
                 scalarWorld.mixedSolver.residualTolerances.x==
                     blockWorld.mixedSolver.residualTolerances.x,
                 "selector comparison did not preserve one strict residual tolerance");
            NativeRun scalarRun(scalarWorld,NUMI_MATTER_METALLIB);
            NativeRun regionalRun(regionalWorld,NUMI_MATTER_METALLIB);
            NativeRun blockRun(blockWorld,NUMI_MATTER_METALLIB);
            const RunResult scalar=scalarRun.run();
            const RunResult regional=regionalRun.run();
            const RunResult block=blockRun.run();
            checkRegionalExport(regional);
            checkRegionalExport(block);
            checkInvariance(scalar,regional);
            checkInvariance(scalar,block);
            checkInvariance(regional,block);
            checkGripMaskTransition(material);
            checkDynamicGraphGripTransitions(material);
            checkActiveCohesiveAP(material);
            checkTangentCacheInvariance(material);
            std::cout<<"PASS regional RAP/block-Jacobi source-bound FP64 and selector invariance"
                     <<" native_transactions=8 physical_validation=false"
                     <<" grip_transition_transactions=3"
                     <<" graph_dynamic_mask_transactions=10"
                     <<" active_cohesive_transactions=2 tangent_cache_transactions=2"
                     <<" scope=synthetic-elastic-and-active-mortar-operator-and-pure-FEM-cache"<<'\n';
            return 0;
        } catch(const std::exception& e) {
            std::cerr<<"FAIL regional RAP native checker: "<<e.what()<<'\n';
            return 1;
        }
    }
}
