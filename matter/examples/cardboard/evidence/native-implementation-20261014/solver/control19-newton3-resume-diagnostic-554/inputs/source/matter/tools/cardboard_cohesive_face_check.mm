#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"
#include "cardboard_bleached_delamination_reference.hpp"

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
constexpr double kHalfEdge = 0.01;
constexpr double kInitialSpeed = 0.1;
unsigned checks = 0u;

using Point = std::array<double, 3>;

struct MeshSpec {
    std::vector<Point> referenceNodes;
    std::vector<std::uint32_t> lowerNodes;
    std::vector<std::uint32_t> upperNodes;
    std::vector<std::array<std::uint32_t, 3>> lowerFaces;
    std::vector<std::array<std::uint32_t, 3>> upperFaces;
    std::vector<std::array<std::uint32_t, 4>> lowerTetrahedra;
    std::vector<std::array<std::uint32_t, 4>> upperTetrahedra;
};

double signedTetrahedronVolume6(const std::array<std::uint32_t, 4>& tet,
                                const std::vector<Point>& nodes) {
    const Point& a = nodes.at(tet[0]);
    const Point& b = nodes.at(tet[1]);
    const Point& c = nodes.at(tet[2]);
    const Point& d = nodes.at(tet[3]);
    const double ab[3]{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const double ac[3]{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    const double ad[3]{d[0] - a[0], d[1] - a[1], d[2] - a[2]};
    return ab[0] * (ac[1] * ad[2] - ac[2] * ad[1]) -
           ab[1] * (ac[0] * ad[2] - ac[2] * ad[0]) +
           ab[2] * (ac[0] * ad[1] - ac[1] * ad[0]);
}

std::array<std::uint32_t, 4> positiveTetrahedron(
    std::array<std::uint32_t, 4> tet, const std::vector<Point>& nodes) {
    const double volume6 = signedTetrahedronVolume6(tet, nodes);
    if (volume6 < 0.0) std::swap(tet[1], tet[2]);
    if (!(signedTetrahedronVolume6(tet, nodes) > 0.0))
        throw std::runtime_error("manufactured cohesive mesh has a degenerate tetrahedron");
    return tet;
}

MeshSpec meshSpec(const bool subdivided) {
    MeshSpec mesh;
    if (!subdivided) {
        mesh.referenceNodes = {
            {{0.0, 0.0, 0.0}}, {{kHalfEdge, 0.0, 0.0}},
            {{0.0, kHalfEdge, 0.0}}, {{0.0, 0.0, -kHalfEdge}},
            {{0.0, 0.0, 0.0}}, {{kHalfEdge, 0.0, 0.0}},
            {{0.0, kHalfEdge, 0.0}}, {{0.0, 0.0, kHalfEdge}},
        };
        mesh.lowerNodes = {0u, 1u, 2u, 3u};
        mesh.upperNodes = {4u, 5u, 6u, 7u};
        mesh.lowerFaces = {{{0u, 1u, 2u}}};
        mesh.upperFaces = {{{4u, 5u, 6u}}};
        mesh.lowerTetrahedra = {{{0u, 2u, 1u, 3u}}};
        mesh.upperTetrahedra = {{{4u, 5u, 6u, 7u}}};
        return mesh;
    }

    // Split the planar paired face at its centroid. Each side remains one
    // rigidly translated tetrahedral star under the Mode-I target path.
    mesh.referenceNodes = {
        {{0.0, 0.0, 0.0}}, {{kHalfEdge, 0.0, 0.0}},
        {{0.0, kHalfEdge, 0.0}}, {{0.0, 0.0, -kHalfEdge}},
        {{kHalfEdge / 3.0, kHalfEdge / 3.0, 0.0}},
        {{0.0, 0.0, 0.0}}, {{kHalfEdge, 0.0, 0.0}},
        {{0.0, kHalfEdge, 0.0}}, {{0.0, 0.0, kHalfEdge}},
        {{kHalfEdge / 3.0, kHalfEdge / 3.0, 0.0}},
    };
    mesh.lowerNodes = {0u, 1u, 2u, 3u, 4u};
    mesh.upperNodes = {5u, 6u, 7u, 8u, 9u};
    mesh.lowerFaces = {{{0u, 1u, 4u}}, {{1u, 2u, 4u}}, {{2u, 0u, 4u}}};
    mesh.upperFaces = {{{5u, 6u, 9u}}, {{6u, 7u, 9u}}, {{7u, 5u, 9u}}};
    for (const auto& face : mesh.lowerFaces)
        mesh.lowerTetrahedra.push_back(positiveTetrahedron(
            {{face[0], face[1], face[2], 3u}}, mesh.referenceNodes));
    for (const auto& face : mesh.upperFaces)
        mesh.upperTetrahedra.push_back(positiveTetrahedron(
            {{face[0], face[1], face[2], 8u}}, mesh.referenceNodes));
    return mesh;
}

void require(const bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

std::string diagnostics(const std::vector<Diagnostic>& values) {
    std::string text;
    for (const Diagnostic& value : values) text += value.message + "; ";
    return text;
}

double parameter(const MaterialProgram& material, const std::string& name) {
    for (const Parameter& value : material.parameters)
        if (value.name == name) return value.defaultValue;
    throw std::runtime_error("missing parameter " + name);
}

MaterialProgram readMaterial(const std::string& path) {
    const ParseResult parsed = parseMatterFile(path);
    require(parsed.succeeded(), "material parse failed: " + path + ": " +
        diagnostics(parsed.diagnostics));
    return parsed.material;
}

double sourceSeedOpening(const MaterialProgram& material) {
    const double kappa = material.internalState.at(0u).initialValue;
    const double ratio = kappa / (1.0 - kappa);
    return parameter(material, "normal_length") *
        parameter(material, "shape_factor_xc") *
        std::pow(ratio, parameter(material, "shape"));
}

WorldSource worldSource(const MaterialProgram& bulk,
                        const MaterialProgram& cohesive,
                        const bool withInterface,
                        const bool subdivided = false) {
    WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = 1.0e-7;
    source.gravity = {0.0, 0.0, 0.0};
    source.materials = {bulk, cohesive};
    source.mixedSolver.relativeResidual = 1.0e-7;
    source.mixedSolver.newtonIterations = 16u;
    source.mixedSolver.fgmresIterations = 64u;

    const double opening = sourceSeedOpening(cohesive);
    ObjectSource object;
    object.name = withInterface ? "paired_mode_i_native_fixture" :
        "paired_mode_i_no_interface_control";
    object.representation = Representation::fem;
    object.mixedFEM = false;
    object.materialIndex = 0u;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.characteristicLength = kHalfEdge;
    object.femReferenceDisplacementGradient = true;
    object.femReferenceSourceIdentity = {
        0x2fce4490a2e36411ull, 0x51b087f670a35b18ull,
        0x91d716e32563aef0ull, 0x4a8b31c62d9085feull,
    };
    if (subdivided) object.femReferenceSourceIdentity = {
        0x2fce4490a2e36412ull, 0x51b087f670a35b18ull,
        0x91d716e32563aef0ull, 0x4a8b31c62d9085feull,
    };
    object.femMaterialIndices = {0u, 0u};
    object.femMaterialSourceIdentity = {
        0x3172a9ff012bc4f1ull, 0x59d15ea4c802773bull,
        0xaece2011a3f7d446ull, 0x080654ca2271ef95ull,
    };
    if (subdivided) object.femMaterialSourceIdentity = {
        0x3172a9ff012bc4f2ull, 0x59d15ea4c802773bull,
        0xaece2011a3f7d446ull, 0x080654ca2271ef95ull,
    };
    const MeshSpec mesh = meshSpec(subdivided);
    object.femReferenceNodes = mesh.referenceNodes;
    object.femNodes = object.femReferenceNodes;
    for (const std::uint32_t node : mesh.lowerNodes)
        object.femNodes[node][2] -= 0.5 * opening;
    for (const std::uint32_t node : mesh.upperNodes)
        object.femNodes[node][2] += 0.5 * opening;
    for (const auto& tet : mesh.lowerTetrahedra) object.tetrahedra.push_back({tet});
    for (const auto& tet : mesh.upperTetrahedra) object.tetrahedra.push_back({tet});
    object.femMaterialIndices.assign(object.tetrahedra.size(), 0u);
    if (withInterface) {
        for (std::uint32_t face = 0u; face < mesh.lowerFaces.size(); ++face) {
            CohesiveInterfaceSource interface;
            interface.sideANodes = mesh.lowerFaces[face];
            interface.sideBNodes = mesh.upperFaces[face];
            interface.firstTetrahedron = face;
            interface.secondTetrahedron = static_cast<std::uint32_t>(
                mesh.lowerTetrahedra.size()) + face;
            interface.materialIndex = 1u;
            interface.stableIdentifier = face + 1u;
            object.femCohesiveInterfaces.push_back(std::move(interface));
        }
    }
    source.objects.push_back(std::move(object));
    return source;
}

CompiledWorld compile(const WorldSource& source) {
    CompileOptions options;
    options.maximumRateExponent = 0;
    options.emitSpecializedMetal = false;
    options.localMaterialNewtonIterations = 16u;
    const CompileResult result = compileWorld(source, options);
    require(result.succeeded(), "cohesive fixture compile failed: " +
        diagnostics(result.diagnostics));
    return result.world;
}

template<class T>
id<MTLBuffer> buffer(id<MTLDevice> device, const std::vector<T>& values) {
    const T zero{};
    const void* data = values.empty() ? static_cast<const void*>(&zero) :
        static_cast<const void*>(values.data());
    return [device newBufferWithBytes:data
        length:std::max(std::size_t{1u}, values.size()) * sizeof(T)
        options:MTLResourceStorageModeShared];
}

struct Run {
    Runtime runtime;
    CompiledWorld world;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLBuffer> statuses = nil;
    std::vector<Point> referenceNodes;
    std::vector<std::uint32_t> lowerNodes;
    std::vector<std::uint32_t> upperNodes;
    std::uint32_t nextControlStep = 0u;

    explicit Run(CompiledWorld compiled,
                 const double relativeOpeningSpeed = kInitialSpeed,
                 const MeshSpec* prescribedMesh = nullptr)
        : world(std::move(compiled)) {
        device = MTLCreateSystemDefaultDevice();
        require(device != nil, "Metal device unavailable");
        require([[device name] rangeOfString:@"Apple"].location != NSNotFound &&
            [[device name] rangeOfString:@"Paravirtual"].location == NSNotFound,
            "native Apple Metal device required");
        queue = [device newCommandQueue];
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
        statuses = buffer(device, std::vector<MRMetalWorldStatusGPU>(
            world.dispatch.environmentCount));

        if (prescribedMesh != nullptr) {
            referenceNodes = prescribedMesh->referenceNodes;
            lowerNodes = prescribedMesh->lowerNodes;
            upperNodes = prescribedMesh->upperNodes;
            require(referenceNodes.size() == world.dispatch.femNodeCount &&
                    lowerNodes.size() + upperNodes.size() == referenceNodes.size(),
                "prescribed cohesive mesh does not cover the compiled FEM node arena");
        }

        RuntimeStateSnapshot initial = state();
        // Matched opening velocity is a manufactured positive Mode-I load.
        // Each tetrahedron begins in rigid translation, so its initial F is I.
        for (std::uint32_t node = 0u; node < 4u; ++node) {
            initial.femNodes[node].velocityAndInverseMass.z =
                -static_cast<float>(relativeOpeningSpeed);
        }
        for (std::uint32_t node = 4u; node < 8u; ++node) {
            initial.femNodes[node].velocityAndInverseMass.z =
                static_cast<float>(relativeOpeningSpeed);
        }
        const RuntimeDiagnostics restored = runtime.restore(initial);
        require(restored.encoded, "manufactured initial velocity restore failed: " +
            restored.message);
    }

    RuntimeStateSnapshot state() {
        const RuntimeStateSnapshot snapshot = runtime.snapshot();
        require(snapshot.available, "runtime snapshot failed: " + snapshot.message);
        return snapshot;
    }

    RuntimeStateSnapshot step(const bool expectedSuccess = true,
                              const std::vector<std::uint32_t>& episodeReset = {},
                              const std::vector<nm_float4>& targets = {}) {
        auto* status = static_cast<MRMetalWorldStatusGPU*>(statuses.contents);
        for (std::uint32_t env = 0u; env < world.dispatch.environmentCount; ++env) {
            status[env] = {};
            status[env].environment = env;
        }
        id<MTLCommandBuffer> command = [queue commandBuffer];
        require(command != nil, "command buffer unavailable");
        EncodeRequest request;
        request.commandBuffer = (__bridge void*)command;
        request.environmentStatuses = (__bridge void*)statuses;
        request.controlStep = nextControlStep++;
        id<MTLBuffer> resetBuffer = nil;
        if (!episodeReset.empty()) {
            require(episodeReset.size() == world.dispatch.environmentCount,
                "episode reset mask must cover each environment");
            std::vector<std::uint32_t> masks(
                std::size_t(request.controlStep + 1u) * episodeReset.size(), 0u);
            std::copy(episodeReset.begin(), episodeReset.end(),
                masks.begin() + std::size_t(request.controlStep) * episodeReset.size());
            resetBuffer = buffer(device, masks);
            request.resetMasks = (__bridge void*)resetBuffer;
            request.resetMaskStepStride = world.dispatch.environmentCount;
        }
        id<MTLBuffer> targetBuffer = nil;
        if (!targets.empty()) {
            require(targets.size() == std::size_t(world.dispatch.environmentCount) *
                    world.dispatch.femNodeCount, "target stream must cover each environment");
            targetBuffer = buffer(device, targets);
            request.femKinematicTargets = (__bridge void*)targetBuffer;
            request.femKinematicTargetCount = static_cast<std::uint32_t>(targets.size());
        }
        request.physicsSubsteps = 1u;
        request.timestepSeconds = runtime.timestepSeconds();
        request.runAdaptiveTransfer = false;
        request.phase = EncodePhase::preDynamics;
        RuntimeDiagnostics encoded = runtime.encode(request);
        require(encoded.encoded, "preDynamics encode failed: " + encoded.message);
        request.phase = EncodePhase::postCommit;
        encoded = runtime.encode(request);
        require(encoded.encoded, "postCommit encode failed: " + encoded.message);
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
            "native cohesive command failed");
        RuntimeStateSnapshot after = state();
        if (expectedSuccess) {
            require(after.statuses.at(0u).code == NM_STATUS_SUCCESS,
                "native cohesive step rejected, status=" +
                    std::to_string(after.statuses.at(0u).code));
        } else {
            require(after.statuses.at(0u).code != NM_STATUS_SUCCESS,
                "native cohesive closure unexpectedly committed");
        }
        return after;
    }

    struct KinematicStep {
        RuntimeStateSnapshot snapshot;
        std::vector<nm_float4> reactions;
    };

    KinematicStep stepToOpening(const double openingMeters,
                                const bool expectedSuccess = true) {
        require(referenceNodes.size() == world.dispatch.femNodeCount &&
                !lowerNodes.empty() && !upperNodes.empty(),
            "stepToOpening requires a complete paired-face mesh description");
        require(std::isfinite(openingMeters) && openingMeters > 0.0,
            "kinematic Mode-I opening must be finite and positive");
        std::vector<nm_float4> targets(referenceNodes.size());
        for (const std::uint32_t node : lowerNodes) {
            const Point& ref = referenceNodes.at(node);
            targets.at(node) = {static_cast<float>(ref[0]),
                static_cast<float>(ref[1]),
                static_cast<float>(ref[2] - 0.5 * openingMeters), 3.0f};
        }
        for (const std::uint32_t node : upperNodes) {
            const Point& ref = referenceNodes.at(node);
            targets.at(node) = {static_cast<float>(ref[0]),
                static_cast<float>(ref[1]),
                static_cast<float>(ref[2] + 0.5 * openingMeters), 3.0f};
        }
        id<MTLBuffer> targetBuffer = buffer(device, targets);
        auto* status = static_cast<MRMetalWorldStatusGPU*>(statuses.contents);
        *status = {};
        status->environment = 0u;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        require(command != nil, "kinematic target command buffer unavailable");
        EncodeRequest request;
        request.commandBuffer = (__bridge void*)command;
        request.environmentStatuses = (__bridge void*)statuses;
        request.femKinematicTargets = (__bridge void*)targetBuffer;
        request.femKinematicTargetCount = static_cast<std::uint32_t>(targets.size());
        request.controlStep = nextControlStep++;
        request.physicsSubsteps = 1u;
        request.timestepSeconds = runtime.timestepSeconds();
        request.runAdaptiveTransfer = false;
        request.phase = EncodePhase::preDynamics;
        const RuntimeDiagnostics pre = runtime.encode(request);
        require(pre.encoded, "cohesive target preDynamics encode failed: " +
            pre.message);

        id<MTLBuffer> reactionSource =
            (__bridge id<MTLBuffer>)runtime.femConstraintReactionBuffer();
        require(reactionSource != nil && reactionSource.length >=
                targets.size() * sizeof(nm_float4),
            "native FEM constraint reactions are unavailable for force readback");
        id<MTLBuffer> reactionReadback = [device
            newBufferWithLength:targets.size() * sizeof(nm_float4)
                        options:MTLResourceStorageModeShared];
        require(reactionReadback != nil,
            "native FEM constraint reaction staging allocation failed");
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        require(blit != nil, "cohesive reaction readback encoder unavailable");
        [blit copyFromBuffer:reactionSource sourceOffset:0u
                    toBuffer:reactionReadback destinationOffset:0u
                        size:targets.size() * sizeof(nm_float4)];
        [blit endEncoding];

        request.phase = EncodePhase::postCommit;
        const RuntimeDiagnostics post = runtime.encode(request);
        require(post.encoded, "cohesive target postCommit encode failed: " +
            post.message);
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
            "native cohesive target command failed");
        KinematicStep result;
        result.snapshot = state();
        if (expectedSuccess) {
            require(result.snapshot.statuses.at(0u).code == NM_STATUS_SUCCESS,
                "native kinematic cohesive step rejected, status=" +
                    std::to_string(result.snapshot.statuses.at(0u).code));
        } else {
            require(result.snapshot.statuses.at(0u).code != NM_STATUS_SUCCESS,
                "expected kinematic cohesive rejection committed");
        }
        const auto* values = static_cast<const nm_float4*>(
            reactionReadback.contents);
        result.reactions.assign(values, values + targets.size());
        return result;
    }

    void restore(const RuntimeStateSnapshot& snapshot) {
        const RuntimeDiagnostics restored = runtime.restore(snapshot);
        require(restored.encoded,
            "cohesive tangent-branch snapshot restore failed: " + restored.message);
        nextControlStep = snapshot.controlStep + 1u;
    }
};

double pairedOpening(const RuntimeStateSnapshot& state) {
    double value = 0.0;
    constexpr std::array<std::array<std::uint32_t, 2>, 3> pairs{{
        {{0u, 4u}}, {{1u, 5u}}, {{2u, 6u}},
    }};
    for (const auto& pair : pairs) {
        value += static_cast<double>(state.femNodes[pair[1]].positionAndMass.z) -
            state.femNodes[pair[0]].positionAndMass.z;
    }
    return value / 3.0;
}

double faceOpening(const RuntimeStateSnapshot& state,
                   const NMCohesiveFaceGPU& face) {
    double opening = 0.0;
    const std::array<std::uint32_t, 3> sideA{{
        face.nodesAndFirst.x, face.nodesAndFirst.y, face.nodesAndFirst.z}};
    const std::array<std::uint32_t, 3> sideB{{
        face.separatedNodes.x, face.separatedNodes.y, face.separatedNodes.z}};
    for (std::size_t node = 0u; node < sideA.size(); ++node)
        opening += state.femNodes.at(sideB[node]).positionAndMass.z -
                   state.femNodes.at(sideA[node]).positionAndMass.z;
    return opening / 3.0;
}

double sumFaceArea(const std::vector<NMCohesiveFaceGPU>& faces) {
    double area = 0.0;
    for (const NMCohesiveFaceGPU& face : faces) area += face.geometry.w;
    return area;
}

double topReactionZ(const std::vector<nm_float4>& reactions,
                    const std::vector<std::uint32_t>& upperNodes) {
    double total = 0.0;
    for (const std::uint32_t node : upperNodes)
        total += reactions.at(node).z;
    return total;
}

void cpuMeshChecks(const MaterialProgram& bulk,
                   const MaterialProgram& cohesive) {
    const MeshSpec coarse = meshSpec(false);
    const MeshSpec refined = meshSpec(true);
    const CompiledWorld coarseWorld = compile(
        worldSource(bulk, cohesive, true, false));
    const CompiledWorld refinedWorld = compile(
        worldSource(bulk, cohesive, true, true));
    const double expectedArea = 0.5 * kHalfEdge * kHalfEdge;
    require(coarseWorld.dispatch.cohesiveFaceCount == 1u &&
            coarseWorld.dispatch.tetrahedronCount == 2u,
        "coarse cohesive mesh did not retain one paired face and two tetrahedra");
    require(refinedWorld.dispatch.cohesiveFaceCount == 3u &&
            refinedWorld.dispatch.tetrahedronCount == 6u,
        "centroid-subdivided cohesive mesh did not create three paired faces and six tetrahedra");
    require(std::abs(sumFaceArea(coarseWorld.fem.cohesiveFaces) - expectedArea) <
                2.0e-6 * expectedArea &&
            std::abs(sumFaceArea(refinedWorld.fem.cohesiveFaces) - expectedArea) <
                1.0e-6 * expectedArea,
        "facet subdivision changed total reference cohesive area");
    auto checkPartition = [](const MeshSpec& mesh) {
        std::vector<bool> used(mesh.referenceNodes.size(), false);
        for (const std::uint32_t node : mesh.lowerNodes) {
            require(node < used.size() && !used[node],
                "lower Mode-I rigid group contains an invalid or repeated node");
            used[node] = true;
        }
        for (const std::uint32_t node : mesh.upperNodes) {
            require(node < used.size() && !used[node],
                "upper Mode-I rigid group overlaps or repeats a lower-group node");
            used[node] = true;
        }
        require(std::all_of(used.begin(), used.end(), [](bool value) { return value; }),
            "Mode-I rigid groups do not cover every mesh node");
    };
    checkPartition(coarse);
    checkPartition(refined);
    for (const NMCohesiveFaceGPU& face : refinedWorld.fem.cohesiveFaces) {
        require(std::abs(double(face.geometry.w) - expectedArea / 3.0) <
                    2.0e-6 * expectedArea,
            "refined interface facet does not carry one third of the same area");
        require(face.state.w == 0.0f,
            "refined interface facet did not preserve zero incremental work at the source seed");
    }
}

struct CohesiveMeshResult {
    double area = 0.0;
    double peakOpeningM = 0.0;
    double maximumForceError = 0.0;
    double maximumTractionRelativeError = 0.0;
    double finalForceN = 0.0;
    double finalTractionPa = 0.0;
    double finalWorkJ = 0.0;
    double finalWorkErrorJ = 0.0;
    double unloadingTangentRelativeError = 0.0;
};

void appendLinearPath(std::vector<double>& path, const double start,
                      const double finish, const unsigned steps) {
    require(steps > 0u, "kinematic cohesive path segment must have steps");
    for (unsigned index = 1u; index <= steps; ++index)
        path.push_back(start + (finish - start) *
            static_cast<double>(index) / static_cast<double>(steps));
}

double sourceOpeningMM(const MaterialProgram& material) {
    return sourceSeedOpening(material) * 1000.0;
}

void verifyTrydingReferenceBinding(const MaterialProgram& cohesive) {
    using namespace numi_cardboard::bleached_delamination;
    require(std::abs(cohesive.internalState.at(0u).initialValue - 0.01) < 1.0e-12,
        "native cohesive coupon no longer starts from the source kappa=0.01 seed");
    require(std::abs(parameter(cohesive, "normal_strength") / 1.0e6 -
                     bleachedPaperboard.normalStrengthMPa) < 1.0e-12,
        "native cohesive normal strength no longer matches the independent source oracle");
    require(std::abs(parameter(cohesive, "normal_length") * 1000.0 -
                     bleachedPaperboard.normalLengthMM) < 1.0e-12,
        "native cohesive characteristic length no longer matches the independent source oracle");
    require(std::abs(parameter(cohesive, "shape") - bleachedPaperboard.shape) <
                    1.0e-12 &&
            std::abs(parameter(cohesive, "shape_factor_xc") - xc()) < 1.0e-12,
        "native cohesive envelope shape no longer matches the independent source oracle");
    require(std::abs(parameter(cohesive, "unloading_scale") / 1.0e9 -
                     bleachedPaperboard.normalUnloadingScaleMPaPerMM) < 1.0e-12 &&
            std::abs(parameter(cohesive, "unloading_exponent") -
                     bleachedPaperboard.unloadingExponent) < 1.0e-12,
        "native cohesive unloading parameters no longer match the independent source oracle");
    require(std::abs(parameter(cohesive, "thickness") - 0.000410) < 1.0e-12,
        "native evaluator thickness no longer matches the source specimen thickness");
}

CohesiveMeshResult runCohesiveHistory(const MaterialProgram& bulk,
                                      const MaterialProgram& cohesive,
                                      const bool subdivided) {
    using namespace numi_cardboard::bleached_delamination;
    verifyTrydingReferenceBinding(cohesive);
    const MeshSpec mesh = meshSpec(subdivided);
    WorldSource coupledSource = worldSource(bulk, cohesive, true, subdivided);
    WorldSource controlSource = worldSource(bulk, cohesive, false, subdivided);
    CompiledWorld coupledWorld = compile(coupledSource);
    const double area = sumFaceArea(coupledWorld.fem.cohesiveFaces);
    const std::vector<NMCohesiveFaceGPU> faceLayout =
        coupledWorld.fem.cohesiveFaces;
    Run coupled(std::move(coupledWorld), 0.0, &mesh);
    Run control(compile(controlSource), 0.0, &mesh);

    State oracle;
    const double seedMM = sourceOpeningMM(cohesive);
    oracle.maximumOpeningMM = seedMM;
    oracle.openingMM = seedMM;
    oracle.tractionMPa = envelopeTractionMPa(seedMM);
    const double peakMM = 0.020;
    const double unloadMM = 0.016;
    const double reloadMM = 0.020;
    const double secondPeakMM = 0.022;
    std::vector<double> path;
    appendLinearPath(path, seedMM * 1.0e-3, peakMM * 1.0e-3, 32u);
    appendLinearPath(path, peakMM * 1.0e-3, unloadMM * 1.0e-3, 8u);
    appendLinearPath(path, unloadMM * 1.0e-3, reloadMM * 1.0e-3, 8u);
    appendLinearPath(path, reloadMM * 1.0e-3, secondPeakMM * 1.0e-3, 8u);

    CohesiveMeshResult summary;
    summary.area = area;
    summary.peakOpeningM = secondPeakMM * 1.0e-3;
    for (const double openingM : path) {
        const Response expected = advance(oracle, openingM * 1000.0);
        const Run::KinematicStep actual = coupled.stepToOpening(openingM);
        const Run::KinematicStep noInterface = control.stepToOpening(openingM);
        require(actual.snapshot.cohesiveFaces.size() == faceLayout.size(),
            "accepted native history changed the authored cohesive face count");
        double nativeWork = 0.0;
        double weightedOpening = 0.0;
        double weightedTraction = 0.0;
        for (std::size_t faceIndex = 0u;
             faceIndex < actual.snapshot.cohesiveFaces.size(); ++faceIndex) {
            const NMCohesiveFaceGPU& face = actual.snapshot.cohesiveFaces[faceIndex];
            const double faceArea = faceLayout[faceIndex].geometry.w;
            nativeWork += face.state.w;
            weightedOpening += faceArea * faceOpening(actual.snapshot,
                                                       faceLayout[faceIndex]);
            weightedTraction += faceArea * face.state.z;
            const double damageError = std::abs(double(face.state.x) - expected.damage);
            const double tractionPa = expected.tractionMPa * 1.0e6;
            const double tractionError = std::abs(double(face.state.z) - tractionPa);
            const double maxOpeningM = expected.maximumOpeningMM * 1.0e-3;
            require(damageError < 3.0e-5,
                "native cohesive damage differs from the independent source history oracle");
            require(std::abs(double(face.state.y) - maxOpeningM) < 3.0e-8,
                "native max-opening history differs from the source law");
            require(tractionError < 0.003 * tractionPa + 5.0,
                "native cohesive traction differs from the independent source law");
            summary.maximumTractionRelativeError = std::max(
                summary.maximumTractionRelativeError,
                tractionError / std::max(tractionPa, 1.0));
        }
        const double nativeOpening = weightedOpening / area;
        require(std::abs(nativeOpening - openingM) < 4.0e-8,
            "native kinematic grip did not realize the prescribed opening");
        const double expectedWork = area * expected.cumulativeWorkJPerM2;
        const double workError = std::abs(nativeWork - expectedWork);
        require(workError < 2.0e-7 + 0.003 * std::abs(expectedWork),
            "native cumulative interface work differs from the independent load/unload integral");

        // The production FEM reaction buffer accumulates the cohesive nodal
        // forces on temporary kinematic grips. Subtract the matched no-face
        // control to remove any constraint/inertial contribution unrelated to
        // the interface, then compare the integrated reaction to A * traction.
        const double nativeForce = -(topReactionZ(actual.reactions, mesh.upperNodes) -
                                     topReactionZ(noInterface.reactions, mesh.upperNodes));
        const double expectedForce = area * expected.tractionMPa * 1.0e6;
        const double forceError = std::abs(nativeForce - expectedForce);
        require(forceError < 2.0e-4 + 0.004 * std::abs(expectedForce),
            "native FEM cohesive reaction differs from the source traction times reference area");
        summary.maximumForceError = std::max(summary.maximumForceError, forceError);
        summary.finalForceN = nativeForce;
        summary.finalTractionPa = weightedTraction / area;
        summary.finalWorkJ = nativeWork;
        summary.finalWorkErrorJ = workError;
        require(std::abs(weightedTraction / area - expected.tractionMPa * 1.0e6) <
                    0.003 * expected.tractionMPa * 1.0e6 + 5.0,
            "area-weighted native traction changed under facet subdivision");
    }

    // A matrix-free operator is internal to Runtime and has no public apply
    // API. This finite difference of the assembled native constraint reaction
    // checks the force-coupling derivative on a smooth, positive unloading
    // branch against the independently calibrated Eq. (43) tangent.
    const RuntimeStateSnapshot coupledPeak = coupled.state();
    const RuntimeStateSnapshot controlPeak = control.state();
    const double centerM = 0.018e-3;
    const double epsilonM = 0.20e-6;
    auto branchForce = [&](const double targetM) {
        Run branch(compile(worldSource(bulk, cohesive, true, subdivided)),
                   0.0, &mesh);
        branch.restore(coupledPeak);
        Run branchControl(compile(worldSource(bulk, cohesive, false, subdivided)),
                          0.0, &mesh);
        branchControl.restore(controlPeak);
        const auto native = branch.stepToOpening(targetM);
        const auto controlResult = branchControl.stepToOpening(targetM);
        const double reaction = -(topReactionZ(native.reactions, mesh.upperNodes) -
            topReactionZ(controlResult.reactions, mesh.upperNodes));
        const double measuredOpening = faceOpening(native.snapshot,
                                                    faceLayout.front());
        const double sourceOpening = measuredOpening * 1000.0;
        const double branchExpectedTraction = responseTractionMPa(
            sourceOpening, secondPeakMM);
        for (const NMCohesiveFaceGPU& face : native.snapshot.cohesiveFaces)
            require(std::abs(double(face.state.z) -
                    branchExpectedTraction * 1.0e6) <
                    0.004 * branchExpectedTraction * 1.0e6 + 5.0,
                "native unloading traction differs from the source branch");
        return std::array<double, 2>{{reaction, measuredOpening}};
    };
    const std::array<double, 2> minus = branchForce(centerM - epsilonM);
    const std::array<double, 2> plus = branchForce(centerM + epsilonM);
    const double nativeTangent = (plus[0] - minus[0]) / (plus[1] - minus[1]);
    const double sourceModulus = unloadingModulusMPaPerMM(secondPeakMM);
    const double expectedTangent = area * sourceModulus * 1.0e9;
    summary.unloadingTangentRelativeError = std::abs(nativeTangent - expectedTangent) /
        std::max(std::abs(expectedTangent), 1.0);
    require(summary.unloadingTangentRelativeError < 0.08,
        "finite-difference native FEM reaction tangent differs from the source unloading modulus");
    return summary;
}

void cpuAdmissionChecks(const WorldSource& valid) {
    const MaterialProgram& cohesive = valid.materials.at(1u);
    const CompiledWorld cooked = compile(valid);
    require(cooked.dispatch.cohesiveFaceCount == 1u,
        "one authored paired interface did not cook to one face");
    require((cooked.fem.cohesiveFaces[0].adjacency.w &
            NM_TOPOLOGY_PAIRED_COHESIVE) != 0u,
        "authored pair lost its paired-face topology role");
    require(cooked.fem.cohesiveFaces[0].geometry.w > 0.0f,
        "paired reference area is nonpositive");
    const double initialDamage = cohesive.internalState.at(0u).initialValue;
    const double seedOpening = sourceSeedOpening(cohesive);
    const double seedTraction = parameter(cohesive, "normal_strength") *
        (1.0 - initialDamage);
    const NMCohesiveFaceGPU& face = cooked.fem.cohesiveFaces[0];
    require(std::abs(double(face.state.x) - initialDamage) < 2.0e-7 &&
        std::abs(double(face.state.y) - seedOpening) < 2.0e-6 * seedOpening &&
        std::abs(double(face.state.z) - seedTraction) < 2.0e-6 * seedTraction &&
        face.state.w == 0.0f,
        "compiled face state does not preserve the source initialization");

    auto reject = [](const std::string& label, WorldSource invalid) {
        CompileOptions options;
        options.maximumRateExponent = 0;
        options.emitSpecializedMetal = false;
        const CompileResult result = compileWorld(invalid, options);
        require(!result.succeeded(), "compiler admitted invalid cohesive case: " +
            label);
    };
    auto invalid = valid;
    invalid.objects[0].deformableSelfContact = true;
    reject("self-contact closure conflict", invalid);
    invalid = valid;
    invalid.objects[0].femCohesiveInterfaces[0].sideBNodes[1] = 7u;
    reject("mispaired face nodes", invalid);
    invalid = valid;
    invalid.objects[0].femCohesiveInterfaces[0].materialIndex = 0u;
    reject("bulk material used as interface law", invalid);
    invalid = valid;
    invalid.objects[0].femNodes[4][2] += 1.0e-5;
    reject("non-source initial opening", invalid);
}
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        try {
            require(argc == 3 || (argc == 4 &&
                std::string(argv[3]) == "--cpu-only"),
                "usage: cardboard-cohesive-face-check MEDIUM.nmatter "
                "MODE_I_COUPON.nmatter [--cpu-only]");
            const MaterialProgram bulk = readMaterial(argv[1]);
            const MaterialProgram cohesive = readMaterial(argv[2]);
            const WorldSource authored = worldSource(bulk, cohesive, true);
            cpuAdmissionChecks(authored);
            cpuMeshChecks(bulk, cohesive);
            if (argc == 4) {
                std::cout << "PASS authored paired cohesive compiler checks="
                    << checks << " native_steps=0 physical_board_qualified=false\n";
                return 0;
            }
            Run coupled(compile(authored));
            const RuntimeStateSnapshot initial = coupled.state();
            const RuntimeStateSnapshot loaded = coupled.step();
            require(loaded.cohesiveFaces.size() == 1u,
                "native accepted state lost the cohesive face");
            const NMCohesiveFaceGPU& face = loaded.cohesiveFaces[0];
            require(face.state.x > initial.cohesiveFaces[0].state.x,
                "opening did not advance irreversible damage through native FEM");
            require(face.state.y > initial.cohesiveFaces[0].state.y,
                "accepted max opening did not advance");
            require(face.state.z > 0.0f && face.state.w > 0.0f,
                "native traction/work state is not tensile and work-positive");
            const double expectedWorkJ = face.geometry.w *
                0.5 * (initial.cohesiveFaces[0].state.z + face.state.z) *
                (pairedOpening(loaded) - pairedOpening(initial));
            require(std::abs(face.state.w - expectedWorkJ) <=
                    1.0e-12 + 1.0e-4 * std::abs(expectedWorkJ),
                "native signed interface work lacks the reference area or has wrong units");
            require(pairedOpening(loaded) > pairedOpening(initial),
                "manufactured tensile loading did not open the native pair");

            Run control(compile(worldSource(bulk, cohesive, false)));
            const RuntimeStateSnapshot controlLoaded = control.step();
            require(pairedOpening(loaded) < pairedOpening(controlLoaded),
                "paired-face traction did not reduce opening relative to no-interface control");

            Run closing(compile(authored), -10.0);
            const RuntimeStateSnapshot beforeRejected = closing.state();
            const RuntimeStateSnapshot rejected = closing.step(false);
            require(std::memcmp(rejected.femNodes.data(), beforeRejected.femNodes.data(),
                    beforeRejected.femNodes.size() * sizeof(NMFEMNodeStateGPU)) == 0,
                "nonpositive opening rejection did not roll back FEM nodes");
            require(std::memcmp(rejected.cohesiveFaces.data(),
                    beforeRejected.cohesiveFaces.data(),
                    beforeRejected.cohesiveFaces.size() * sizeof(NMCohesiveFaceGPU)) == 0,
                "nonpositive opening rejection committed cohesive history");

            // A reset restores authored history in one environment while the
            // unreset sibling follows the identical previously accepted state.
            WorldSource resetSource = authored;
            resetSource.environmentCount = 2u;
            const CompiledWorld resetWorld = compile(resetSource);
            Run resetRun(resetWorld);
            const RuntimeStateSnapshot damaged = resetRun.step();
            require(damaged.cohesiveFaces[0].state.x >
                    resetWorld.fem.cohesiveFaces[0].state.x,
                "reset regression did not first create native interface damage");
            Run unresetRun(resetWorld);
            unresetRun.restore(damaged);
            const RuntimeStateSnapshot unresetNext = unresetRun.step();
            Run freshRun(resetWorld, 0.0);
            const RuntimeStateSnapshot freshNext = freshRun.step();
            const RuntimeStateSnapshot resetNext = resetRun.step(true, {1u, 0u});
            require(std::memcmp(&resetNext.cohesiveFaces[0],
                    &freshNext.cohesiveFaces[0], sizeof(NMCohesiveFaceGPU)) == 0,
                "episode reset did not reproduce fresh authored interface history");
            require(std::memcmp(&resetNext.cohesiveFaces[1],
                    &unresetNext.cohesiveFaces[1], sizeof(NMCohesiveFaceGPU)) == 0,
                "episode reset changed the unreset sibling interface history");
            require(std::memcmp(resetNext.femNodes.data(), freshNext.femNodes.data(),
                    resetWorld.dispatch.femNodeCount * sizeof(NMFEMNodeStateGPU)) == 0,
                "reset interface history and nodes did not follow the same fresh episode");
            require(resetNext.statuses[1].code == NM_STATUS_SUCCESS,
                "unreset sibling failed during reset regression");

            // Rejection in the reset transaction must roll back to the authored
            // reset checkpoint, rather than the previous episode's damage.
            Run resetRejectRun(resetWorld);
            (void)resetRejectRun.step();
            std::vector<nm_float4> invalidTargets(
                std::size_t(resetWorld.dispatch.environmentCount) *
                resetWorld.dispatch.femNodeCount);
            for (std::uint32_t node = 0u; node < resetWorld.dispatch.femNodeCount; ++node) {
                const auto& ref = resetWorld.fem.nodes[node].restAndFixed;
                invalidTargets[node] = {ref.x, ref.y,
                    ref.z + (node < 4u ? 1.0e-6f : -1.0e-6f), 3.0f};
            }
            const RuntimeStateSnapshot resetRejected =
                resetRejectRun.step(false, {1u, 0u}, invalidTargets);
            require(std::memcmp(&resetRejected.cohesiveFaces[0],
                    &resetWorld.fem.cohesiveFaces[0], sizeof(NMCohesiveFaceGPU)) == 0,
                "reset-transaction rejection restored stale cohesive checkpoint history");
            require(resetRejected.statuses[1].code == NM_STATUS_SUCCESS,
                "reset-transaction rejection contaminated the unreset sibling");

            const CohesiveMeshResult coarseHistory =
                runCohesiveHistory(bulk, cohesive, false);
            const CohesiveMeshResult refinedHistory =
                runCohesiveHistory(bulk, cohesive, true);
            require(std::abs(coarseHistory.area - refinedHistory.area) <
                    2.0e-6 * coarseHistory.area,
                "coarse and subdivided native histories do not have matched area");
            require(std::abs(coarseHistory.finalForceN - refinedHistory.finalForceN) <
                    0.005 * std::abs(coarseHistory.finalForceN) + 2.0e-4,
                "native integrated reaction changed materially under facet subdivision");
            require(std::abs(coarseHistory.finalTractionPa -
                             refinedHistory.finalTractionPa) <
                    0.003 * std::abs(coarseHistory.finalTractionPa) + 5.0,
                "native area-weighted traction changed materially under facet subdivision");
            require(std::abs(coarseHistory.finalWorkJ - refinedHistory.finalWorkJ) <
                    0.005 * std::abs(coarseHistory.finalWorkJ) + 2.0e-7,
                "native integrated work changed materially under facet subdivision");
            std::cout << "PASS native paired cohesive FEM checks=" << checks
                << " initial_opening_m=" << pairedOpening(initial)
                << " accepted_opening_m=" << pairedOpening(loaded)
                << " no_interface_opening_m=" << pairedOpening(controlLoaded)
                << " accepted_damage=" << face.state.x
                << " accepted_traction_Pa=" << face.state.z
                << " signed_work_J=" << face.state.w
                << " source_mode_i_coarse_peak_force_error_N="
                << coarseHistory.maximumForceError
                << " source_mode_i_refined_peak_force_error_N="
                << refinedHistory.maximumForceError
                << " source_mode_i_coarse_unload_tangent_relative_error="
                << coarseHistory.unloadingTangentRelativeError
                << " source_mode_i_refined_unload_tangent_relative_error="
                << refinedHistory.unloadingTangentRelativeError
                << " source_mode_i_coarse_final_work_error_J="
                << coarseHistory.finalWorkErrorJ
                << " source_mode_i_refined_final_work_error_J="
                << refinedHistory.finalWorkErrorJ
                << " matrix_free_operator_directly_exposed=false"
                << " tested_operator_proxy=finite_difference_of_native_kinematic_reactions"
                << " calibration_grade=single_ply_bleached_clay_coated_not_corrugated_starch"
                << " closure_rejection_rollback=verified"
                << " episode_reset_history_and_isolation=verified"
                << " reset_rejection_checkpoint=verified"
                << " physical_board_qualified=false"
                << " mixed_mode_or_closure_supported=false\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "FAIL native paired cohesive FEM: " << error.what() << '\n';
            return 1;
        }
    }
}
