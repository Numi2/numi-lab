#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/engine_types.h"

#include "cardboard_paper_reference.hpp"
#include "numi/matter/compiler.hpp"
#include "numi/matter/language.hpp"
#include "numi/matter/matter.hpp"
#include "../src/regional_graph_schwarz.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifndef NUMI_MATTER_METALLIB
#error "CMake must define NUMI_MATTER_METALLIB"
#endif

namespace {
using namespace numi::matter;
namespace paper = numi_cardboard_paper;
using Matrix = std::array<double, 9>;
using Point = std::array<double, 3>;

constexpr Matrix baselineDeformation{
    1.000020, 0.000014, -0.000006,
    -0.000008, 0.999985, 0.000011,
    0.000005, -0.000009, 1.000012,
};

constexpr Matrix yieldedDeformation{
    0.99, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
};

void need(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

double determinant(const Matrix& a) {
    return a[0] * (a[4] * a[8] - a[5] * a[7]) -
        a[1] * (a[3] * a[8] - a[5] * a[6]) +
        a[2] * (a[3] * a[7] - a[4] * a[6]);
}

Matrix inverse(const Matrix& a) {
    const double det = determinant(a);
    need(std::isfinite(det) && std::abs(det) > 1.0e-30,
         "reference tetrahedron is singular");
    return {
        (a[4] * a[8] - a[5] * a[7]) / det,
        (a[2] * a[7] - a[1] * a[8]) / det,
        (a[1] * a[5] - a[2] * a[4]) / det,
        (a[5] * a[6] - a[3] * a[8]) / det,
        (a[0] * a[8] - a[2] * a[6]) / det,
        (a[2] * a[3] - a[0] * a[5]) / det,
        (a[3] * a[7] - a[4] * a[6]) / det,
        (a[1] * a[6] - a[0] * a[7]) / det,
        (a[0] * a[4] - a[1] * a[3]) / det,
    };
}

Matrix multiply(const Matrix& a, const Matrix& b) {
    Matrix result{};
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j)
            for (unsigned k = 0; k < 3; ++k)
                result[3 * i + j] += a[3 * i + k] * b[3 * k + j];
    return result;
}

Matrix transpose(const Matrix& a) {
    Matrix result{};
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j)
            result[3 * i + j] = a[3 * j + i];
    return result;
}

Matrix scaled(const Matrix& a, const double scale) {
    Matrix result = a;
    for (double& value : result) value *= scale;
    return result;
}

Matrix add(const Matrix& a, const Matrix& b) {
    Matrix result{};
    for (unsigned i = 0; i < 9; ++i) result[i] = a[i] + b[i];
    return result;
}

Matrix subtract(const Matrix& a, const Matrix& b) {
    Matrix result{};
    for (unsigned i = 0; i < 9; ++i) result[i] = a[i] - b[i];
    return result;
}

Point multiply(const Matrix& a, const Point& x) {
    return {
        a[0] * x[0] + a[1] * x[1] + a[2] * x[2],
        a[3] * x[0] + a[4] * x[1] + a[5] * x[2],
        a[6] * x[0] + a[7] * x[1] + a[8] * x[2],
    };
}

Matrix displacementGradient(
    const Matrix& inverseRest,
    const unsigned localNode,
    const unsigned axis
) {
    Point p[4]{};
    p[localNode][axis] = 1.0;
    const Point p0 = p[0];
    Matrix dp{};
    for (unsigned row = 0; row < 3; ++row) {
        dp[3 * row + 0] = p[1][row] - p0[row];
        dp[3 * row + 1] = p[2][row] - p0[row];
        dp[3 * row + 2] = p[3][row] - p0[row];
    }
    return multiply(dp, inverseRest);
}

Matrix piolaGlobal(
    const paper::MaterialKind kind,
    const Matrix& fGlobal,
    const Matrix& materialFrame,
    const paper::State& acceptedState
) {
    const Matrix fLocal = multiply(fGlobal, materialFrame);
    const auto projected = paper::project(kind, fLocal, acceptedState);
    return scaled(multiply(projected.firstPiola,
                           transpose(materialFrame)), 1.0e6);
}

Point internalForceAtNode(
    const paper::MaterialKind kind,
    const Matrix& fGlobal,
    const Matrix& materialFrame,
    const Matrix& inverseRest,
    const double volume,
    const unsigned localNode,
    const paper::State& acceptedState
) {
    const Matrix piola = piolaGlobal(
        kind, fGlobal, materialFrame, acceptedState);
    const Matrix h = scaled(
        multiply(piola, transpose(inverseRest)), -volume);
    Point force{};
    if (localNode == 0u) {
        for (unsigned row = 0; row < 3; ++row)
            force[row] = -(h[3 * row] + h[3 * row + 1] + h[3 * row + 2]);
    } else {
        const unsigned column = localNode - 1u;
        for (unsigned row = 0; row < 3; ++row)
            force[row] = h[3 * row + column];
    }
    return force;
}

template <typename T>
id<MTLBuffer> makeBuffer(id<MTLDevice> device, const std::vector<T>& values) {
    const T zero{};
    const void* bytes = values.empty() ? static_cast<const void*>(&zero)
                                       : static_cast<const void*>(values.data());
    const NSUInteger length = sizeof(T) * std::max<std::size_t>(1u, values.size());
    return [device newBufferWithBytes:bytes length:length
                              options:MTLResourceStorageModeShared];
}

struct Fixture {
    CompiledWorld world;
    paper::MaterialKind paperKind = paper::MaterialKind::liner;
    paper::State acceptedState{};
    Matrix referenceInverse{};
    Matrix deformation{};
    Matrix materialFrame{};
    double referenceVolume = 0.0;
    double timestep = 0.0;
};

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic<std::uint64_t> serial{0u};
        const auto root = std::filesystem::temp_directory_path();
        for (unsigned attempt = 0u; attempt < 128u; ++attempt) {
            const auto stamp = std::chrono::steady_clock::now()
                .time_since_epoch().count();
            path_ = root / ("numi-cardboard-package-roundtrip-" +
                std::to_string(stamp) + "-" +
                std::to_string(serial.fetch_add(1u)));
            std::error_code error;
            if (std::filesystem::create_directory(path_, error)) {
                created_ = true;
                return;
            }
            if (error && error != std::errc::file_exists)
                throw std::runtime_error("cannot create package-test directory: " +
                                         error.message());
        }
        throw std::runtime_error("cannot allocate a unique package-test directory");
    }

    ~TemporaryDirectory() {
        if (!created_) return;
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
    bool created_ = false;
};

void checkNondefaultPackageRoundTrip(
    const CompileResult& compiled,
    const FEMPreconditionerMode expectedMode
) {
    need(compiled.world.mixedSolver.executionBudgets.z ==
             static_cast<std::uint32_t>(expectedMode),
         "package fixture did not retain its requested FEM preconditioner mode");
    need(!compiled.world.materials.empty() &&
             compiled.world.materials[0].localNewtonIterations == 16u,
         "package fixture did not select the nondefault local Newton budget");

    TemporaryDirectory temporary;
    const auto package = temporary.path() / "regional-tangent.nmpkg";
    std::string error;
    need(writeMatterPackage(compiled, package, &error),
         "cannot write nondefault preconditioner package: " + error);
    CompiledWorld roundTrip;
    error.clear();
    need(readMatterPackage(package, roundTrip, nullptr, &error),
         "cannot read nondefault preconditioner package: " + error);
    error.clear();
    need(validateCompiledWorldLayout(roundTrip, &error),
         "round-tripped preconditioner package is invalid: " + error);
    need(roundTrip.fingerprint == compiled.world.fingerprint &&
             roundTrip.physicsFingerprint == compiled.world.physicsFingerprint,
         "package round-trip changed compiled fingerprints");
    need(roundTrip.mixedSolver.executionBudgets.z ==
             static_cast<std::uint32_t>(expectedMode) &&
             roundTrip.materials.at(0u).localNewtonIterations == 16u,
         "package round-trip lost nondefault preconditioner/material controls");
    std::cout << "package_roundtrip=pass mode="
              << static_cast<std::uint32_t>(expectedMode)
              << " local_material_newton=16\n";
}

Fixture cook(
    const std::filesystem::path& materialPath,
    const paper::State& acceptedState = paper::State{},
    const Matrix& deformation = baselineDeformation,
    const FEMPreconditionerMode preconditioner =
        FEMPreconditionerMode::regionalTangentDiagonal,
    const std::vector<std::uint32_t>& fixedNodes = {}
) {
    auto parsed = parseMatterFile(materialPath);
    need(parsed.succeeded(), "cannot parse material: " + materialPath.string());
    MaterialProgram material = parsed.material;
    need(material.internalState.size() == acceptedState.size(),
         "paper material does not expose the complete 13-value accepted state");
    for (std::size_t state = 0u; state < acceptedState.size(); ++state)
        material.internalState[state].initialValue = acceptedState[state];
    const bool liner = material.name.find("liner") != std::string::npos;
    const paper::MaterialKind paperKind = liner
        ? paper::MaterialKind::liner : paper::MaterialKind::medium;

    const std::array<Point, 4> referenceNodes{
        Point{0.031, -0.018, 0.027},
        Point{0.0333, -0.01783, 0.027004},
        Point{0.03131, -0.01630, 0.027008},
        Point{0.03106, -0.01773, 0.027035},
    };
    constexpr double frameAngle = 0.37;
    const Matrix materialFrame{
        std::cos(frameAngle), -std::sin(frameAngle), 0.0,
        std::sin(frameAngle),  std::cos(frameAngle), 0.0,
        0.0,                   0.0,                  1.0,
    };
    const std::array<double, 4> quaternion{
        0.0, 0.0, std::sin(frameAngle * 0.5), std::cos(frameAngle * 0.5)};

    WorldSource source;
    source.frameTimestep = 2.5e-5;
    source.gravity = {0.0, 0.0, 0.0};
    source.environmentCount = 1u;
    source.mixedSolver.femPreconditioner = preconditioner;
    source.materials = {material};

    ObjectSource object;
    object.name = "tangent_preconditioner_thin_skew_tet";
    object.representation = Representation::fem;
    object.materialIndex = 0u;
    object.mixedFEM = false;
    object.twoWayCoupling = false;
    object.adaptive = false;
    object.identifiable = false;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.femReferenceDisplacementGradient = true;
    object.femMaterialIndices = {0u};
    object.femMaterialSourceIdentity = {1u, 2u, 3u, 4u};
    object.femMaterialFrameRotations = {quaternion};
    object.femMaterialFrameSourceIdentity = {5u, 6u, 7u, 8u};
    object.femReferenceSourceIdentity = {9u, 10u, 11u, 12u};
    object.femReferenceNodes.assign(referenceNodes.begin(), referenceNodes.end());
    object.femNodes.reserve(referenceNodes.size());
    for (const Point& point : referenceNodes) {
        const Point current = multiply(deformation, point);
        object.femNodes.push_back(current);
    }
    object.femFixedNodes = fixedNodes;
    object.tetrahedra = {{{0u, 1u, 2u, 3u}}};
    if (preconditioner == FEMPreconditionerMode::regionalGalerkin ||
        preconditioner ==
            FEMPreconditionerMode::regionalGalerkinBlockJacobi ||
        preconditioner == FEMPreconditionerMode::regionalGalerkinContact ||
        preconditioner ==
            FEMPreconditionerMode::regionalGalerkinGraphSchwarz ||
        preconditioner ==
            FEMPreconditionerMode::regionalGalerkinHybridGraphSchwarz) {
        // Regional Galerkin modes require one authored compact region label
        // per tetrahedron. This one-tet fixture therefore has the exact
        // single-region map {0}; it is part of the preconditioner input.
        object.femPreconditionerRegionIds = {0u};
    }
    object.characteristicLength = 0.001;
    source.objects = {object};

    CompileOptions options;
    options.maximumRateExponent = 0u;
    options.emitSpecializedMetal = false;
    options.localMaterialNewtonIterations = 16u;
    const auto compiled = compileWorld(source, options);
    if (!compiled.succeeded()) {
        std::string diagnostics;
        for (const Diagnostic& item : compiled.diagnostics)
            diagnostics += item.message + "; ";
        throw std::runtime_error("cannot compile tangent fixture: " + diagnostics);
    }
    need(compiled.world.mixedSolver.executionBudgets.z ==
             static_cast<std::uint32_t>(preconditioner),
         "compiled package lost the preconditioner selector");
    need(compiled.world.fem.nodes.size() == 4u &&
             compiled.world.fem.tetrahedra.size() == 1u,
         "compiled fixture is not one explicit four-node tetrahedron");
    checkNondefaultPackageRoundTrip(compiled, preconditioner);

    WorldSource legacy = source;
    legacy.mixedSolver.femPreconditioner =
        FEMPreconditionerMode::scalarDiagonal;
    legacy.objects[0].femPreconditionerRegionIds.clear();
    const auto legacyCompiled = compileWorld(legacy, options);
    need(legacyCompiled.succeeded(), "legacy scalar comparison world did not compile");
    need(legacyCompiled.world.fingerprint != compiled.world.fingerprint,
         "preconditioner selection is absent from the compiled fingerprint");

    WorldSource adaptive = source;
    adaptive.objects[0].adaptive = true;
    need(!compileWorld(adaptive, options).succeeded(),
         "regional tangent selector accepted an adaptive object");
    WorldSource mixed = source;
    mixed.objects[0].mixedFEM = true;
    need(!compileWorld(mixed, options).succeeded(),
         "regional tangent selector accepted a mixed FEM object");
    WorldSource capacity = source;
    capacity.objects[0].femCapacity.tetrahedra = 2u;
    need(!compileWorld(capacity, options).succeeded(),
         "regional tangent selector accepted reserved topology capacity");

    const Point d1{
        referenceNodes[1][0] - referenceNodes[0][0],
        referenceNodes[1][1] - referenceNodes[0][1],
        referenceNodes[1][2] - referenceNodes[0][2]};
    const Point d2{
        referenceNodes[2][0] - referenceNodes[0][0],
        referenceNodes[2][1] - referenceNodes[0][1],
        referenceNodes[2][2] - referenceNodes[0][2]};
    const Point d3{
        referenceNodes[3][0] - referenceNodes[0][0],
        referenceNodes[3][1] - referenceNodes[0][1],
        referenceNodes[3][2] - referenceNodes[0][2]};
    const Matrix edges{
        d1[0], d2[0], d3[0],
        d1[1], d2[1], d3[1],
        d1[2], d2[2], d3[2],
    };
    const double signedVolume = determinant(edges) / 6.0;
    need(std::abs(signedVolume) > 0.0, "thin/skew tet has zero volume");
    return Fixture{
        compiled.world,
        paperKind,
        acceptedState,
        inverse(edges),
        deformation,
        materialFrame,
        std::abs(signedVolume),
        source.frameTimestep,
    };
}

class MetalCheck {
public:
    MetalCheck() {
        device_ = MTLCreateSystemDefaultDevice();
        need(device_ != nil, "Metal device is unavailable");
        const std::string name = [[device_ name] UTF8String];
        need(name.find("Apple") != std::string::npos &&
                 name.find("Paravirtual") == std::string::npos,
             "a physical Apple GPU is required");
        queue_ = [device_ newCommandQueue];
        NSError* error = nil;
        id<MTLLibrary> library = [device_ newLibraryWithURL:
            [NSURL fileURLWithPath:@NUMI_MATTER_METALLIB] error:&error];
        need(library != nil, "cannot load Matter metallib");
        id<MTLFunction> function = [library newFunctionWithName:
            @"numi_matter_metal::nm_fgmres_build_tangent_fem_preconditioner"];
        need(function != nil, "tangent preconditioner kernel is missing from metallib");
        pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
        need(pipeline_ != nil, "cannot create tangent preconditioner pipeline");
        id<MTLFunction> contactLayout = [library newFunctionWithName:
            @"numi_matter_metal::nm_primal_contact_argument_layout"];
        contactEncoder_ = contactLayout == nil
            ? nil : [contactLayout newArgumentEncoderWithBufferIndex:0u];
        need(contactEncoder_ != nil,
             "cannot construct primal contact argument encoder");
        std::cout << "device=" << name
                  << " kernel=nm_fgmres_build_tangent_fem_preconditioner"
                  << " physical_steps=0\n";
    }

    std::vector<nm_float4> evaluate(const Fixture& fixture) {
        const CompiledWorld& world = fixture.world;
        std::vector<NMFEMNodeStateGPU> candidate = world.fem.nodes;
        std::vector<float> environmentParameters;
        environmentParameters.reserve(world.parameters.size());
        for (const NMParameterRangeGPU& parameter : world.parameters)
            environmentParameters.push_back(parameter.valueAndBounds.x);
        std::vector<float> acceptedMaterialState(
            world.dispatch.materialStateStride, 0.0f);
        const NMMaterialGPU material = world.materials.at(0u);
        for (std::uint32_t state = 0u; state < material.stateCount; ++state)
            acceptedMaterialState[state] = world.stateInitials.at(
                material.stateInitialOffset + state);
        std::vector<NMIncidenceRangeGPU> contactNodeRanges(
            world.dispatch.gridNodeCount + world.dispatch.femNodeCount);
        std::vector<NMIncidenceRangeGPU> contactDeformableNodeRanges(
            world.dispatch.environmentCount *
            (world.dispatch.gridNodeCount + world.dispatch.femNodeCount));
        std::vector<NMIncidenceRangeGPU> emptyRange(1u);
        std::vector<NMSchedulerStateGPU> schedulers = world.schedulers;
        if (schedulers.empty()) schedulers.resize(world.dispatch.objectCount);

        const auto dispatchBuffer = makeBuffer(device_,
            std::vector<NMMatterDispatchGPU>{world.dispatch});
        const auto solverBuffer = makeBuffer(device_,
            std::vector<NMMixedSolverGPU>{world.mixedSolver});
        const auto objectBuffer = makeBuffer(device_, world.objects);
        const auto candidateBuffer = makeBuffer(device_, candidate);
        const auto tetrahedronBuffer = makeBuffer(device_, world.fem.tetrahedra);
        const auto incidenceBuffer = makeBuffer(device_, world.fem.nodeIncidence);
        const auto rangeBuffer = makeBuffer(device_, world.fem.nodeRanges);
        const auto schedulerBuffer = makeBuffer(device_, schedulers);
        const auto materialBuffer = makeBuffer(device_, world.materials);
        const auto programBuffer = makeBuffer(device_, world.scalarPrograms);
        const auto instructionBuffer = makeBuffer(device_, world.instructions);
        const auto parameterBuffer = makeBuffer(device_, environmentParameters);
        const auto stateBuffer = makeBuffer(device_, acceptedMaterialState);
        std::vector<nm_float4> output(world.dispatch.femNodeCount, nm_float4{});
        const auto outputBuffer = makeBuffer(device_, output);
        const auto contactRangeBuffer = makeBuffer(device_, contactNodeRanges);
        const auto dummyBuffer = makeBuffer(device_, emptyRange);
        std::vector<NMMatterStatusGPU> status(1u);
        const auto statusBuffer = makeBuffer(device_, status);
        const auto argumentBuffer = [device_ newBufferWithLength:
            contactEncoder_.encodedLength options:MTLResourceStorageModeShared];
        need(argumentBuffer != nil, "cannot allocate contact argument buffer");
        [contactEncoder_ setArgumentBuffer:argumentBuffer offset:0u];
        for (NSUInteger index = 0u; index < 15u; ++index)
        [contactEncoder_ setBuffer:dummyBuffer offset:0u atIndex:index];
        [contactEncoder_ setBuffer:contactRangeBuffer offset:0u atIndex:3u];
        const auto deformableRangeBuffer = makeBuffer(
            device_, contactDeformableNodeRanges);
        [contactEncoder_ setBuffer:deformableRangeBuffer offset:0u atIndex:9u];

        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline_];
        [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
        [encoder setBuffer:solverBuffer offset:0u atIndex:1u];
        [encoder setBuffer:objectBuffer offset:0u atIndex:2u];
        [encoder setBuffer:candidateBuffer offset:0u atIndex:3u];
        [encoder setBuffer:tetrahedronBuffer offset:0u atIndex:4u];
        [encoder setBuffer:incidenceBuffer offset:0u atIndex:5u];
        [encoder setBuffer:rangeBuffer offset:0u atIndex:6u];
        [encoder setBuffer:schedulerBuffer offset:0u atIndex:7u];
        [encoder setBuffer:materialBuffer offset:0u atIndex:8u];
        [encoder setBuffer:programBuffer offset:0u atIndex:9u];
        [encoder setBuffer:instructionBuffer offset:0u atIndex:10u];
        [encoder setBuffer:parameterBuffer offset:0u atIndex:11u];
        [encoder setBuffer:stateBuffer offset:0u atIndex:12u];
        [encoder setBuffer:outputBuffer offset:0u atIndex:13u];
        [encoder setBuffer:argumentBuffer offset:0u atIndex:14u];
        [encoder setBuffer:statusBuffer offset:0u atIndex:15u];
        // This direct-kernel fixture tests the uncached branch. The production
        // cache is opt-in and bound only by Runtime when it has been built for
        // the current Newton candidate/history.
        [encoder setBuffer:dummyBuffer offset:0u atIndex:16u];
        [encoder setBuffer:dummyBuffer offset:0u atIndex:17u];
        const std::uint32_t useTangentCache = 0u;
        [encoder setBytes:&useTangentCache length:sizeof(useTangentCache)
                  atIndex:18u];
        [encoder useResource:dummyBuffer usage:MTLResourceUsageRead];
        [encoder useResource:contactRangeBuffer usage:MTLResourceUsageRead];
        [encoder dispatchThreads:MTLSizeMake(world.dispatch.femNodeCount, 1u, 1u)
           threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        need(command.status == MTLCommandBufferStatusCompleted,
             command.error == nil ? "Metal preconditioner dispatch failed"
                 : std::string([[command.error.localizedDescription description] UTF8String]));
        const auto* values = static_cast<const nm_float4*>(outputBuffer.contents);
        const auto* finalStatus =
            static_cast<const NMMatterStatusGPU*>(statusBuffer.contents);
        need(finalStatus[0].code == 0u,
             "Metal tangent preconditioner reported status " +
                 std::to_string(finalStatus[0].code));
        return std::vector<nm_float4>(values,
                                      values + world.dispatch.femNodeCount);
    }

private:
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLComputePipelineState> pipeline_ = nil;
    id<MTLArgumentEncoder> contactEncoder_ = nil;
};

void checkMaterial(MetalCheck& gpu, const std::filesystem::path& path) {
    const Fixture fixture = cook(path);
    const auto output = gpu.evaluate(fixture);
    const NMContinuumObjectGPU& object = fixture.world.objects.front();
    (void)object;
    const double dt = fixture.timestep;
    constexpr double displacementEpsilon = 1.0e-9;
    double maxRelativeError = 0.0;
    double maxAbsoluteError = 0.0;
    std::uint32_t compared = 0u;
    for (unsigned node = 0u; node < 4u; ++node) {
        const double mass = fixture.world.fem.nodes[node].positionAndMass.w;
        const std::array<double, 3> actualComponents{
            output[node].x, output[node].y, output[node].z};
        for (unsigned axis = 0u; axis < 3u; ++axis) {
            const Matrix direction = displacementGradient(
                fixture.referenceInverse, node, axis);
            const Matrix plus = add(fixture.deformation,
                scaled(direction, displacementEpsilon));
            const Matrix minus = subtract(fixture.deformation,
                scaled(direction, displacementEpsilon));
            const Point forcePlus = internalForceAtNode(
                fixture.paperKind, plus, fixture.materialFrame,
                fixture.referenceInverse, fixture.referenceVolume, node,
                fixture.acceptedState);
            const Point forceMinus = internalForceAtNode(
                fixture.paperKind, minus, fixture.materialFrame,
                fixture.referenceInverse, fixture.referenceVolume, node,
                fixture.acceptedState);
            const double derivative =
                (forcePlus[axis] - forceMinus[axis]) /
                (2.0 * displacementEpsilon);
            const double diagonal = std::max(
                mass - dt * dt * derivative,
                static_cast<double>(fixture.world.mixedSolver.regularization.x));
            const double expected = 1.0 / diagonal;
            const double actual = actualComponents[axis];
            const double relativeError = std::abs(actual - expected) /
                std::max({std::abs(actual), std::abs(expected), 1.0e-30});
            maxRelativeError = std::max(maxRelativeError, relativeError);
            maxAbsoluteError = std::max(maxAbsoluteError,
                                         std::abs(actual - expected));
            ++compared;
        }
    }
    std::cout << path.filename().string()
              << " direct_kernel_diagonal_entries=" << compared
              << " accepted_lambda=" << fixture.acceptedState[12]
              << " fd_relative_max=" << maxRelativeError
              << " fd_absolute_max=" << maxAbsoluteError << '\n';
    need(maxRelativeError < 0.025,
         "GPU tangent FEM preconditioner diagonal disagrees with independent "
         "centered force finite differences");
}

void checkYieldedMaterial(MetalCheck& gpu, const std::filesystem::path& path) {
    const bool liner = path.filename().string().find("liner") != std::string::npos;
    const auto kind = liner ? paper::MaterialKind::liner
                            : paper::MaterialKind::medium;
    const paper::Result preload = paper::project(
        kind, yieldedDeformation, paper::State{});
    need(preload.state[12] > 1.0e-4,
         "yielded-state fixture did not accumulate plastic multiplier");
    need(preload.equivalentStress > 0.999 &&
             preload.equivalentStress < 1.001 &&
             preload.plasticWorkMPa > 0.0,
         "yielded-state preload failed the independent Hill consistency/work check");

    const Fixture fixture = cook(path, preload.state, yieldedDeformation);
    const auto output = gpu.evaluate(fixture);
    const double dt = fixture.timestep;
    constexpr double displacementEpsilon = 1.0e-9;
    double maxRelativeError = 0.0;
    double maxAbsoluteError = 0.0;
    std::uint32_t compared = 0u;
    for (unsigned node = 0; node < 4u; ++node) {
        const double mass = fixture.world.fem.nodes[node].positionAndMass.w;
        const std::array<double, 3> actualComponents{
            output[node].x, output[node].y, output[node].z};
        for (unsigned axis = 0; axis < 3u; ++axis) {
            const Matrix direction = displacementGradient(
                fixture.referenceInverse, node, axis);
            const Matrix plus = add(yieldedDeformation,
                scaled(direction, displacementEpsilon));
            const Matrix minus = subtract(yieldedDeformation,
                scaled(direction, displacementEpsilon));
            const Point forcePlus = internalForceAtNode(
                kind, plus, fixture.materialFrame, fixture.referenceInverse,
                fixture.referenceVolume, node, preload.state);
            const Point forceMinus = internalForceAtNode(
                kind, minus, fixture.materialFrame, fixture.referenceInverse,
                fixture.referenceVolume, node, preload.state);
            const double derivative =
                (forcePlus[axis] - forceMinus[axis]) /
                (2.0 * displacementEpsilon);
            const double diagonal = std::max(
                mass - dt * dt * derivative,
                static_cast<double>(fixture.world.mixedSolver.regularization.x));
            const double expected = 1.0 / diagonal;
            const double actual = actualComponents[axis];
            const double relativeError = std::abs(actual - expected) /
                std::max({std::abs(actual), std::abs(expected), 1.0e-30});
            maxRelativeError = std::max(maxRelativeError, relativeError);
            maxAbsoluteError = std::max(maxAbsoluteError,
                                         std::abs(actual - expected));
            ++compared;
        }
    }
    std::cout << path.filename().string()
              << " yielded_state_fd_entries=" << compared
              << " lambda=" << preload.state[12]
              << " q=" << preload.equivalentStress
              << " work_MPa=" << preload.plasticWorkMPa
              << " fd_relative_max=" << maxRelativeError
              << " fd_absolute_max=" << maxAbsoluteError << '\n';
    need(maxRelativeError < 0.025,
         "GPU tangent FEM preconditioner diagonal disagrees with independent "
         "yielded-state centered force finite differences");
}

class Selector6NativeProbe {
public:
    Selector6NativeProbe() {
        device_ = MTLCreateSystemDefaultDevice();
        need(device_ != nil, "selector6 probe requires a Metal device");
        const std::string name = [[device_ name] UTF8String];
        need(name.find("Apple") != std::string::npos &&
                 name.find("Paravirtual") == std::string::npos,
             "selector6 probe requires a physical Apple GPU");
        queue_ = [device_ newCommandQueue];
        NSError* error = nil;
        library_ = [device_ newLibraryWithURL:
            [NSURL fileURLWithPath:@NUMI_MATTER_METALLIB] error:&error];
        need(library_ != nil, "cannot load selector6 Matter metallib");
        hybridApply_ = makePipeline(
            @"numi_matter_metal::nm_fgmres_graph_apply_hybrid_patches");
        hybridMask_ = makePipeline(
            @"numi_matter_metal::nm_fgmres_graph_validate_hybrid_mask");
        requireRap_ = makePipeline(
            @"numi_matter_metal::nm_fgmres_graph_require_rap_valid");
        factorRap_ = makePipeline(
            @"numi_matter_metal::nm_fgmres_regional_factor_rap");
        factorPatches_ = makePipeline(
            @"numi_matter_metal::nm_fgmres_graph_factor_patches");
        solvePatches_ = makePipeline(
            @"numi_matter_metal::nm_fgmres_graph_solve_patches");
        std::cout << "selector6_native_probe_device=" << name
                  << " max_buffer_length=" << device_.maxBufferLength
                  << " peak_resident_memory=not_measured\n";
    }

    void run(const std::filesystem::path& materialPath) {
        const Fixture fixture = cook(
            materialPath, paper::State{}, baselineDeformation,
            FEMPreconditionerMode::regionalGalerkinHybridGraphSchwarz,
            {2u, 3u});
        const CompiledWorld& world = fixture.world;
        need(world.dispatch.environmentCount == 1u &&
                 world.dispatch.femNodeCount == 4u &&
                 world.dispatch.tetrahedronCount == 1u &&
                 world.dispatch.objectCount == 1u,
             "selector6 native fixture is not the exact compiled 4-node/1-tet topology");

        std::array<std::array<std::uint32_t, 4>, 1> topology{{
            {world.fem.tetrahedra[0].nodes.x,
             world.fem.tetrahedra[0].nodes.y,
             world.fem.tetrahedra[0].nodes.z,
             world.fem.tetrahedra[0].nodes.w}}};
        std::vector<std::uint8_t> authoredFree(world.dispatch.femNodeCount, 0u);
        for (std::uint32_t node = 0u; node < world.dispatch.femNodeCount; ++node)
            authoredFree[node] = world.fem.nodes[node].restAndFixed.w == 0.0f;
        const auto graph = detail::buildHybridRegionalGraphSchwarzLayout(
            world.dispatch.femNodeCount, topology, authoredFree, 1u);
        need(graph.error.empty(), "compiled selector6 topology rejected by hybrid layout: " + graph.error);
        need(graph.patches.size() == 2u && graph.patchNodes.size() == 4u &&
                 graph.nodeCoverage[0] != 0u && graph.nodeCoverage[1] != 0u &&
                 graph.nodeCoverage[2] == 0u && graph.nodeCoverage[3] == 0u,
             "source-derived graph did not preserve the expected covered/free/fixed split");

        std::vector<NMFEMNodeStateGPU> candidate = world.fem.nodes;
        need(std::isfinite(candidate[2].positionAndMass.w) &&
                 candidate[2].positionAndMass.w > 0.0f,
             "selector6 released node has no positive compiled mass");
        candidate[2].restAndFixed.w = 0.0f; // exercise a released source-fixed node
        candidate[2].velocityAndInverseMass.w =
            1.0f / candidate[2].positionAndMass.w;
        need(candidate[3].restAndFixed.w == 1.0f &&
                 candidate[3].velocityAndInverseMass.w == 0.0f,
             "selector6 fixture lost its remaining fixed node");

        std::vector<nm_float4> inverseBlocks(
            std::size_t(world.dispatch.femNodeCount) * 3u, nm_float4{});
        // A symmetric positive-definite inverse block with nonzero
        // off-diagonals makes scalar/componentwise fallback observably wrong
        // while staying inside the selector's certified block-Jacobi domain.
        inverseBlocks[2u * 3u] = {2.0f, 0.3f, -0.2f, 1.0f};
        inverseBlocks[2u * 3u + 1u] = {0.3f, 1.5f, 0.25f, 0.0f};
        inverseBlocks[2u * 3u + 2u] = {-0.2f, 0.25f, 1.25f, 0.0f};
        checkHybridMask(world.dispatch, candidate, graph.nodeCoverage,
                        inverseBlocks, NM_STATUS_SUCCESS);
        auto badMarker = inverseBlocks;
        badMarker[2u * 3u].w = 0.0f;
        checkHybridMask(world.dispatch, candidate, graph.nodeCoverage,
                        badMarker, NM_STATUS_UNSUPPORTED);
        auto nonfiniteBlock = inverseBlocks;
        nonfiniteBlock[2u * 3u + 1u].y = NAN;
        checkHybridMask(world.dispatch, candidate, graph.nodeCoverage,
                        nonfiniteBlock, NM_STATUS_NONFINITE_RESULT);

        checkHybridApply(world.dispatch, candidate, graph, inverseBlocks);
        checkRAPGateFailure(world.dispatch);
        checkPatchLUFailure(world.dispatch, candidate, graph);

        std::uint64_t blockBytes = 0u;
        need(detail::regionalHybridBlockInverseBytes(
                 world.dispatch.femNodeCount, world.dispatch.environmentCount,
                 blockBytes),
             "actual selector6 fixture block arena sizing failed");
        std::cout << "selector6_native_fixture topology=compiled_exact"
                  << " nodes=" << world.dispatch.femNodeCount
                  << " tets=" << world.dispatch.tetrahedronCount
                  << " patches=" << graph.patches.size()
                  << " patch_nodes=" << graph.patchNodes.size()
                  << " graph_arena_bytes=" << graph.deviceBytes
                  << " block_arena_bytes=" << blockBytes
                  << " exact_shipping_topology=false"
                  << " peak_resident_memory=not_measured\n";
    }

private:
    id<MTLComputePipelineState> makePipeline(NSString* functionName) {
        NSError* error = nil;
        id<MTLFunction> function = [library_ newFunctionWithName:functionName];
        need(function != nil, "missing selector6 native kernel " +
             std::string([functionName UTF8String]));
        id<MTLComputePipelineState> pipeline =
            [device_ newComputePipelineStateWithFunction:function error:&error];
        need(pipeline != nil, "cannot create selector6 pipeline " +
             std::string([functionName UTF8String]));
        return pipeline;
    }

    void finish(id<MTLCommandBuffer> command, const char* action) {
        need(command != nil, std::string("cannot allocate command for ") + action);
        [command commit];
        [command waitUntilCompleted];
        need(command.status == MTLCommandBufferStatusCompleted,
             command.error == nil ? std::string(action) + " failed" :
             std::string([[command.error.localizedDescription description] UTF8String]));
    }

    void checkHybridMask(
        const NMMatterDispatchGPU& dispatch,
        const std::vector<NMFEMNodeStateGPU>& candidate,
        const std::vector<std::uint32_t>& coverage,
        const std::vector<nm_float4>& inverseBlocks,
        const std::uint32_t expectedStatus
    ) {
        const auto dispatchBuffer = makeBuffer(device_,
            std::vector<NMMatterDispatchGPU>{dispatch});
        const auto candidateBuffer = makeBuffer(device_, candidate);
        const auto coverageBuffer = makeBuffer(device_, coverage);
        const auto blockBuffer = makeBuffer(device_, inverseBlocks);
        std::vector<NMMatterStatusGPU> statuses(1u);
        const auto statusBuffer = makeBuffer(device_, statuses);
        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:hybridMask_];
        [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
        [encoder setBuffer:candidateBuffer offset:0u atIndex:1u];
        [encoder setBuffer:coverageBuffer offset:0u atIndex:2u];
        [encoder setBuffer:blockBuffer offset:0u atIndex:3u];
        [encoder setBuffer:statusBuffer offset:0u atIndex:4u];
        [encoder dispatchThreads:MTLSizeMake(dispatch.femNodeCount, 1u, 1u)
           threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
        [encoder endEncoding];
        finish(command, "selector6 hybrid-mask validation");
        const auto* result = static_cast<const NMMatterStatusGPU*>(statusBuffer.contents);
        need(result[0].code == expectedStatus,
             "hybrid mask returned status " + std::to_string(result[0].code) +
                 " instead of " + std::to_string(expectedStatus));
    }

    void checkHybridApply(
        const NMMatterDispatchGPU& dispatch,
        const std::vector<NMFEMNodeStateGPU>& candidate,
        const detail::RegionalGraphSchwarzLayout& graph,
        const std::vector<nm_float4>& inverseBlocks
    ) {
        constexpr std::uint32_t modes = 1u;
        constexpr std::uint32_t modeStride = 1u;
        const std::vector<nm_float4> regionalBasis{
            {1.0f, 0.5f, -0.25f, 0.0f},
            {-0.5f, 1.0f, 0.25f, 0.0f},
            {0.25f, -0.5f, 1.0f, 0.0f},
            {1.0f, 1.0f, 1.0f, 0.0f},
        };
        const std::vector<nm_float4> regionalAP{
            {2.0f, -0.5f, 0.25f, 0.0f},
            {-0.5f, 3.0f, 0.2f, 0.0f},
            {0.25f, 0.2f, 1.5f, 0.0f},
            {0.0f, 0.0f, 0.0f, 0.0f},
        };
        const std::vector<float> coefficients{0.4f};
        const std::vector<nm_float4> input{
            {0.25f, -1.0f, 2.0f, 0.0f},
            {-0.7f, 1.2f, -0.3f, 0.0f},
            {3.0f, -2.0f, 0.5f, 0.0f},
            {99.0f, 99.0f, 99.0f, 0.0f},
        };
        const std::vector<nm_float4> patchCorrections{
            {0.3f, -0.2f, 0.1f, 0.0f},
            {-0.5f, 0.25f, 0.75f, 0.0f},
            {-0.1f, 0.3f, 0.2f, 0.0f},
            {0.4f, -0.1f, 0.5f, 0.0f},
        };
        std::vector<nm_float4> output(dispatch.femNodeCount, nm_float4{});
        std::vector<NMFGMRESStateGPU> states(dispatch.environmentCount);
        std::vector<NMMatterStatusGPU> statuses(dispatch.environmentCount);
        const auto dispatchBuffer = makeBuffer(device_,
            std::vector<NMMatterDispatchGPU>{dispatch});
        const auto patchNodesBuffer = makeBuffer(device_, graph.patchNodes);
        const auto offsetsBuffer = makeBuffer(device_, graph.nodePatchOffsets);
        const auto valuesBuffer = makeBuffer(device_, graph.nodePatchValues);
        const auto coverageBuffer = makeBuffer(device_, graph.nodeCoverage);
        const auto basisBuffer = makeBuffer(device_, regionalBasis);
        const auto apBuffer = makeBuffer(device_, regionalAP);
        const auto coefficientBuffer = makeBuffer(device_, coefficients);
        const auto correctionBuffer = makeBuffer(device_, patchCorrections);
        const auto blockBuffer = makeBuffer(device_, inverseBlocks);
        const auto outputBuffer = makeBuffer(device_, output);
        const auto stateBuffer = makeBuffer(device_, states);
        const auto candidateBuffer = makeBuffer(device_, candidate);
        const auto inputBuffer = makeBuffer(device_, input);
        const auto statusBuffer = makeBuffer(device_, statuses);
        const std::vector<NMFEMNodeStateGPU> candidateBefore = candidate;
        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:hybridApply_];
        [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
        [encoder setBuffer:patchNodesBuffer offset:0u atIndex:1u];
        [encoder setBuffer:offsetsBuffer offset:0u atIndex:2u];
        [encoder setBuffer:valuesBuffer offset:0u atIndex:3u];
        [encoder setBuffer:coverageBuffer offset:0u atIndex:4u];
        [encoder setBuffer:basisBuffer offset:0u atIndex:5u];
        [encoder setBuffer:apBuffer offset:0u atIndex:6u];
        [encoder setBuffer:coefficientBuffer offset:0u atIndex:7u];
        [encoder setBuffer:correctionBuffer offset:0u atIndex:8u];
        [encoder setBuffer:blockBuffer offset:0u atIndex:9u];
        [encoder setBuffer:outputBuffer offset:0u atIndex:10u];
        [encoder setBuffer:stateBuffer offset:0u atIndex:11u];
        [encoder setBuffer:candidateBuffer offset:0u atIndex:12u];
        [encoder setBuffer:statusBuffer offset:0u atIndex:13u];
        const std::uint32_t patchNodeCount =
            static_cast<std::uint32_t>(graph.patchNodes.size());
        [encoder setBytes:&patchNodeCount length:sizeof(patchNodeCount) atIndex:14u];
        [encoder setBytes:&modes length:sizeof(modes) atIndex:15u];
        [encoder setBytes:&modeStride length:sizeof(modeStride) atIndex:16u];
        [encoder setBuffer:inputBuffer offset:0u atIndex:17u];
        [encoder dispatchThreads:MTLSizeMake(dispatch.femNodeCount, 1u, 1u)
           threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
        [encoder endEncoding];
        finish(command, "selector6 hybrid apply");
        const auto* actual = static_cast<const nm_float4*>(outputBuffer.contents);
        const auto* status = static_cast<const NMMatterStatusGPU*>(statusBuffer.contents);
        need(status[0].code == NM_STATUS_SUCCESS,
             "valid selector6 hybrid application reported a failure");
        need(std::memcmp(candidateBefore.data(), candidateBuffer.contents,
                         candidateBefore.size() * sizeof(NMFEMNodeStateGPU)) == 0,
             "conditioning-only selector6 application modified candidate state");
        const std::array<Point, 3> p{
            Point{regionalBasis[0].x, regionalBasis[0].y, regionalBasis[0].z},
            Point{regionalBasis[1].x, regionalBasis[1].y, regionalBasis[1].z},
            Point{regionalBasis[2].x, regionalBasis[2].y, regionalBasis[2].z}};
        const std::array<Point, 3> ap{
            Point{regionalAP[0].x, regionalAP[0].y, regionalAP[0].z},
            Point{regionalAP[1].x, regionalAP[1].y, regionalAP[1].z},
            Point{regionalAP[2].x, regionalAP[2].y, regionalAP[2].z}};
        const std::array<Point, 3> r{
            Point{input[0].x, input[0].y, input[0].z},
            Point{input[1].x, input[1].y, input[1].z},
            Point{input[2].x, input[2].y, input[2].z}};
        const auto close3 = [](const nm_float4 actualValue,
                               const Point& expected) {
            constexpr double tolerance = 2.0e-6;
            return std::abs(actualValue.x - expected[0]) <= tolerance &&
                std::abs(actualValue.y - expected[1]) <= tolerance &&
                std::abs(actualValue.z - expected[2]) <= tolerance;
        };
        for (std::size_t node = 0u; node < 2u; ++node) {
            Point expected{};
            const std::uint32_t begin = graph.nodePatchOffsets[node];
            const std::uint32_t end = graph.nodePatchOffsets[node + 1u];
            need(end > begin && end - begin == graph.nodeCoverage[node],
                 "selector6 native CSR disagrees with covered node count");
            for (unsigned axis = 0u; axis < 3u; ++axis)
                expected[axis] = 0.4 * p[node][axis];
            for (std::uint32_t item = begin; item < end; ++item) {
                const std::uint32_t correctionIndex = graph.nodePatchValues[item];
                need(correctionIndex < patchCorrections.size() &&
                         graph.patchNodes[correctionIndex] == node,
                     "selector6 native CSR correction index is invalid");
                expected[0] += patchCorrections[correctionIndex].x /
                    float(end - begin);
                expected[1] += patchCorrections[correctionIndex].y /
                    float(end - begin);
                expected[2] += patchCorrections[correctionIndex].z /
                    float(end - begin);
            }
            need(close3(actual[node], expected),
                 "covered selector6 row differs from Pc plus graph correction");
        }
        Point uncoveredResidual{};
        for (unsigned axis = 0u; axis < 3u; ++axis)
            uncoveredResidual[axis] = r[2][axis] - 0.4 * ap[2][axis];
        const Matrix inverseBlock{
            2.0, 0.3, -0.2,
            0.3, 1.5, 0.25,
            -0.2, 0.25, 1.25};
        const Point blockAction = multiply(inverseBlock, uncoveredResidual);
        Point uncoveredExpected{};
        for (unsigned axis = 0u; axis < 3u; ++axis)
            uncoveredExpected[axis] = 0.4 * p[2][axis] + blockAction[axis];
        need(close3(actual[2], uncoveredExpected),
             "uncovered selector6 row does not apply the full off-diagonal block to r-APc");
        Point firstBasis{};
        for (unsigned axis = 0u; axis < 3u; ++axis) firstBasis[axis] = p[0][axis];
        need(std::abs(r[2][0] - firstBasis[0]) > 1.0 &&
                 std::abs(r[2][1] - firstBasis[1]) > 1.0,
             "selector6 RHS control accidentally matches regional mode zero");
        need(actual[3].x == 0.0f && actual[3].y == 0.0f &&
                 actual[3].z == 0.0f,
             "selector6 hybrid application changed a fixed-node correction");
        std::cout << "selector6_hybrid_apply=pass rhs_distinct_from_P0=1"
                  << " offdiagonal_block=1 fixed_row_zero=1 candidate_unchanged=1\n";
    }

    void checkRAPGateFailure(const NMMatterDispatchGPU& dispatch) {
        const auto dispatchBuffer = makeBuffer(device_,
            std::vector<NMMatterDispatchGPU>{dispatch});
        std::vector<float> rap{0.0f};
        std::vector<std::uint32_t> pivots{0u};
        std::vector<std::uint32_t> valid{1u};
        const auto rapBuffer = makeBuffer(device_, rap);
        const auto pivotBuffer = makeBuffer(device_, pivots);
        const auto validBuffer = makeBuffer(device_, valid);
        const std::uint32_t modeCount = 1u;
        id<MTLCommandBuffer> factorCommand = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> factorEncoder = [factorCommand computeCommandEncoder];
        [factorEncoder setComputePipelineState:factorRap_];
        [factorEncoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
        [factorEncoder setBuffer:rapBuffer offset:0u atIndex:1u];
        [factorEncoder setBuffer:pivotBuffer offset:0u atIndex:2u];
        [factorEncoder setBuffer:validBuffer offset:0u atIndex:3u];
        [factorEncoder setBytes:&modeCount length:sizeof(modeCount) atIndex:4u];
        [factorEncoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        [factorEncoder endEncoding];
        finish(factorCommand, "selector6 singular RAP factor");
        need(static_cast<const std::uint32_t*>(validBuffer.contents)[0] == 0u,
             "singular RAP factor did not clear its validity marker");
        std::vector<NMMatterStatusGPU> statuses(1u);
        const auto statusBuffer = makeBuffer(device_, statuses);
        id<MTLCommandBuffer> gateCommand = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> gateEncoder = [gateCommand computeCommandEncoder];
        [gateEncoder setComputePipelineState:requireRap_];
        [gateEncoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
        [gateEncoder setBuffer:validBuffer offset:0u atIndex:1u];
        [gateEncoder setBuffer:statusBuffer offset:0u atIndex:2u];
        [gateEncoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
           threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
        [gateEncoder endEncoding];
        finish(gateCommand, "selector6 RAP fail-closed gate");
        need(static_cast<const NMMatterStatusGPU*>(statusBuffer.contents)[0].code ==
                 NM_STATUS_LINEAR_SOLVER_FAILURE,
             "selector6 accepted an invalid RAP factor");
        std::cout << "selector6_invalid_RAP=fail_closed status="
                  << NM_STATUS_LINEAR_SOLVER_FAILURE << '\n';
    }

    void checkPatchLUFailure(
        const NMMatterDispatchGPU& dispatch,
        const std::vector<NMFEMNodeStateGPU>& candidate,
        const detail::RegionalGraphSchwarzLayout& graph
    ) {
        need(!graph.patches.empty(), "selector6 LU fixture has no graph patches");
        std::vector<nm_uint4> patches;
        patches.reserve(graph.patches.size());
        for (const auto& descriptor : graph.patches)
            patches.push_back({descriptor.nodeOffset, descriptor.nodeCount,
                               descriptor.matrixFloatOffset,
                               descriptor.pivotOffset});
        std::vector<float> matrices(graph.matrixFloatCount, 0.0f);
        std::vector<std::uint32_t> pivots(graph.pivotCount, 0u);
        std::vector<std::uint32_t> factorValid(graph.patches.size(), 1u);
        std::vector<NMMatterStatusGPU> statuses(1u);
        std::vector<float> original(graph.matrixFloatCount, 0.0f);
        std::vector<NMFGMRESStateGPU> states(1u);
        std::vector<std::uint32_t> priorValid(graph.patches.size(), 0u);
        const auto dispatchBuffer = makeBuffer(device_,
            std::vector<NMMatterDispatchGPU>{dispatch});
        const auto patchBuffer = makeBuffer(device_, patches);
        const auto matrixBuffer = makeBuffer(device_, matrices);
        const auto pivotBuffer = makeBuffer(device_, pivots);
        const auto validBuffer = makeBuffer(device_, factorValid);
        const auto statusBuffer = makeBuffer(device_, statuses);
        const auto originalBuffer = makeBuffer(device_, original);
        const auto stateBuffer = makeBuffer(device_, states);
        const auto priorBuffer = makeBuffer(device_, priorValid);
        const std::uint32_t patchCount =
            static_cast<std::uint32_t>(graph.patches.size());
        const std::uint32_t matrixCount = graph.matrixFloatCount;
        const std::uint32_t pivotCount = graph.pivotCount;
        const std::uint32_t reuse = 0u;
        const std::uint32_t parallel = 0u;
        id<MTLCommandBuffer> factorCommand = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> factorEncoder = [factorCommand computeCommandEncoder];
        [factorEncoder setComputePipelineState:factorPatches_];
        [factorEncoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
        [factorEncoder setBuffer:patchBuffer offset:0u atIndex:1u];
        [factorEncoder setBuffer:matrixBuffer offset:0u atIndex:2u];
        [factorEncoder setBuffer:pivotBuffer offset:0u atIndex:3u];
        [factorEncoder setBuffer:validBuffer offset:0u atIndex:4u];
        [factorEncoder setBuffer:statusBuffer offset:0u atIndex:5u];
        [factorEncoder setBytes:&patchCount length:sizeof(patchCount) atIndex:6u];
        [factorEncoder setBytes:&matrixCount length:sizeof(matrixCount) atIndex:7u];
        [factorEncoder setBytes:&pivotCount length:sizeof(pivotCount) atIndex:8u];
        [factorEncoder setBuffer:originalBuffer offset:0u atIndex:9u];
        [factorEncoder setBuffer:stateBuffer offset:0u atIndex:10u];
        [factorEncoder setBytes:&reuse length:sizeof(reuse) atIndex:11u];
        [factorEncoder setBuffer:priorBuffer offset:0u atIndex:12u];
        [factorEncoder setBytes:&parallel length:sizeof(parallel) atIndex:13u];
        [factorEncoder dispatchThreadgroups:MTLSizeMake(patchCount, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
        [factorEncoder endEncoding];
        finish(factorCommand, "selector6 singular patch LU factor");
        const auto* factorResults =
            static_cast<const std::uint32_t*>(validBuffer.contents);
        need(std::all_of(factorResults, factorResults + graph.patches.size(),
                     [](const std::uint32_t value) { return value == 0u; }) &&
                 static_cast<const NMMatterStatusGPU*>(statusBuffer.contents)[0].code ==
                     NM_STATUS_LINEAR_SOLVER_FAILURE,
             "selector6 singular graph patch did not fail closed");

        std::vector<nm_float4> regionalBasis(dispatch.femNodeCount,
                                             nm_float4{});
        std::vector<nm_float4> regionalAP(dispatch.femNodeCount, nm_float4{});
        std::vector<float> coefficients(1u, 0.0f);
        std::vector<nm_float4> rhs(dispatch.femNodeCount,
                                   nm_float4{1.0f, -0.5f, 0.25f, 0.0f});
        std::vector<nm_float4> corrections(graph.patchNodes.size(),
                                           nm_float4{9.0f, 9.0f, 9.0f, 0.0f});
        statuses.assign(1u, NMMatterStatusGPU{});
        const auto rhsBuffer = makeBuffer(device_, rhs);
        const auto basisBuffer = makeBuffer(device_, regionalBasis);
        const auto apBuffer = makeBuffer(device_, regionalAP);
        const auto coeffBuffer = makeBuffer(device_, coefficients);
        const auto correctionBuffer = makeBuffer(device_, corrections);
        const auto cleanStatusBuffer = makeBuffer(device_, statuses);
        const auto candidateBuffer = makeBuffer(device_, candidate);
        const auto patchNodesBuffer = makeBuffer(device_, graph.patchNodes);
        const std::uint32_t modeCount = 1u;
        const std::uint32_t modeStride = 1u;
        const std::uint32_t patchNodeCount =
            static_cast<std::uint32_t>(graph.patchNodes.size());
        const std::uint32_t refinement = 1u;
        id<MTLCommandBuffer> solveCommand = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> solveEncoder = [solveCommand computeCommandEncoder];
        [solveEncoder setComputePipelineState:solvePatches_];
        [solveEncoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
        [solveEncoder setBuffer:patchBuffer offset:0u atIndex:1u];
        [solveEncoder setBuffer:patchNodesBuffer offset:0u atIndex:2u];
        [solveEncoder setBuffer:rhsBuffer offset:0u atIndex:3u];
        [solveEncoder setBuffer:basisBuffer offset:0u atIndex:4u];
        [solveEncoder setBuffer:apBuffer offset:0u atIndex:5u];
        [solveEncoder setBuffer:coeffBuffer offset:0u atIndex:6u];
        [solveEncoder setBuffer:matrixBuffer offset:0u atIndex:7u];
        [solveEncoder setBuffer:pivotBuffer offset:0u atIndex:8u];
        [solveEncoder setBuffer:validBuffer offset:0u atIndex:9u];
        [solveEncoder setBuffer:correctionBuffer offset:0u atIndex:10u];
        [solveEncoder setBuffer:cleanStatusBuffer offset:0u atIndex:11u];
        [solveEncoder setBytes:&patchCount length:sizeof(patchCount) atIndex:12u];
        [solveEncoder setBytes:&patchNodeCount length:sizeof(patchNodeCount) atIndex:13u];
        [solveEncoder setBytes:&matrixCount length:sizeof(matrixCount) atIndex:14u];
        [solveEncoder setBytes:&pivotCount length:sizeof(pivotCount) atIndex:15u];
        [solveEncoder setBytes:&modeCount length:sizeof(modeCount) atIndex:16u];
        [solveEncoder setBuffer:stateBuffer offset:0u atIndex:17u];
        [solveEncoder setBuffer:originalBuffer offset:0u atIndex:18u];
        [solveEncoder setBytes:&refinement length:sizeof(refinement) atIndex:19u];
        [solveEncoder setBuffer:candidateBuffer offset:0u atIndex:20u];
        [solveEncoder setBytes:&modeStride length:sizeof(modeStride) atIndex:21u];
        [solveEncoder dispatchThreadgroups:MTLSizeMake(patchCount, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
        [solveEncoder endEncoding];
        finish(solveCommand, "selector6 singular patch solve");
        const auto* finalStatus =
            static_cast<const NMMatterStatusGPU*>(cleanStatusBuffer.contents);
        const auto* finalCorrections =
            static_cast<const nm_float4*>(correctionBuffer.contents);
        need(finalStatus[0].code == NM_STATUS_LINEAR_SOLVER_FAILURE &&
                 std::all_of(finalCorrections,
                     finalCorrections + graph.patchNodes.size(),
                     [](const nm_float4 value) {
                         return value.x == 0.0f && value.y == 0.0f &&
                             value.z == 0.0f && value.w == 0.0f;
                     }),
             "selector6 patch solve did not preserve fail-closed zero correction");
        std::cout << "selector6_invalid_patch_LU=fail_closed factor_and_solve=1\n";
    }

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLLibrary> library_ = nil;
    id<MTLComputePipelineState> hybridApply_ = nil;
    id<MTLComputePipelineState> hybridMask_ = nil;
    id<MTLComputePipelineState> requireRap_ = nil;
    id<MTLComputePipelineState> factorRap_ = nil;
    id<MTLComputePipelineState> factorPatches_ = nil;
    id<MTLComputePipelineState> solvePatches_ = nil;
};

template <typename T>
bool sameVectorBytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
        (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

void requireSamePhysicalInputs(
    const CompiledWorld& five,
    const CompiledWorld& six
) {
    need(std::memcmp(&five.dispatch, &six.dispatch,
                     sizeof(NMMatterDispatchGPU)) == 0 &&
             sameVectorBytes(five.objects, six.objects) &&
             sameVectorBytes(five.materials, six.materials) &&
             sameVectorBytes(five.parameters, six.parameters) &&
             sameVectorBytes(five.stateInitials, six.stateInitials) &&
             sameVectorBytes(five.scalarPrograms, six.scalarPrograms) &&
             sameVectorBytes(five.instructions, six.instructions) &&
             sameVectorBytes(five.fem.nodes, six.fem.nodes) &&
             sameVectorBytes(five.fem.tetrahedra, six.fem.tetrahedra) &&
             sameVectorBytes(five.fem.nodeIncidence, six.fem.nodeIncidence) &&
             sameVectorBytes(five.fem.nodeRanges, six.fem.nodeRanges) &&
             sameVectorBytes(five.schedulers, six.schedulers) &&
             sameVectorBytes(five.adaptive, six.adaptive),
         "selector5/6 compile changed a non-preconditioner physical input");
    NMMixedSolverGPU normalizedFive = five.mixedSolver;
    NMMixedSolverGPU normalizedSix = six.mixedSolver;
    normalizedFive.executionBudgets.z = 0u;
    normalizedSix.executionBudgets.z = 0u;
    need(std::memcmp(&normalizedFive, &normalizedSix,
                     sizeof(NMMixedSolverGPU)) == 0,
         "selector5/6 compile changed a solver control besides the selector");
}

template <typename T>
std::string rawVectorHex(const std::vector<T>& values) {
    static constexpr char digits[] = "0123456789abcdef";
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(values.data());
    const std::size_t byteCount = values.size() * sizeof(T);
    std::string result;
    result.resize(byteCount * 2u);
    for (std::size_t i = 0u; i < byteCount; ++i) {
        result[2u * i] = digits[bytes[i] >> 4u];
        result[2u * i + 1u] = digits[bytes[i] & 0x0fu];
    }
    return result;
}

std::string rawStringHex(const std::string& value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.resize(value.size() * 2u);
    for (std::size_t i = 0u; i < value.size(); ++i) {
        const auto byte = static_cast<std::uint8_t>(value[i]);
        result[2u * i] = digits[byte >> 4u];
        result[2u * i + 1u] = digits[byte & 0x0fu];
    }
    return result;
}

void printSelectorRuntimeOutcome(
    const char* label,
    const RuntimeStateSnapshot& snapshot,
    const std::string& error,
    const std::size_t residentBytes
) {
    std::cout << "selector_runtime_outcome label=" << label
              << " snapshot_available=" << snapshot.available
              << " source_fingerprint=" << snapshot.sourcePhysicsFingerprint
              << " device_program_fingerprint=" << snapshot.deviceProgramFingerprint
              << " control_step=" << snapshot.controlStep
              << " physics_substep=" << snapshot.physicsSubstep
              << " status_count=" << snapshot.statuses.size()
              << " certificate_count=" << snapshot.solverCertificates.size()
              << " fem_node_count=" << snapshot.femNodes.size()
              << " resident_bytes_reported=" << residentBytes
              << " message_hex=" << rawStringHex(snapshot.message)
              << " error_hex=" << rawStringHex(error)
              << " statuses_raw_hex=" << rawVectorHex(snapshot.statuses)
              << " certificates_raw_hex=" << rawVectorHex(snapshot.solverCertificates)
              << " fem_nodes_raw_hex=" << rawVectorHex(snapshot.femNodes) << '\n';
    std::cout << "selector_runtime_snapshot_arrays label=" << label
              << " particles=" << rawVectorHex(snapshot.particles)
              << " fem_fields=" << rawVectorHex(snapshot.femFields)
              << " vascular_state=" << rawVectorHex(snapshot.vascularState)
              << " vascular_clock=" << rawVectorHex(snapshot.vascularClock)
              << " fem_topology_nodes=" << rawVectorHex(snapshot.femTopologyNodes)
              << " fem_topology_tetrahedra=" << rawVectorHex(snapshot.femTopologyTetrahedra)
              << " cohesive_faces=" << rawVectorHex(snapshot.cohesiveFaces)
              << " puncture_channels=" << rawVectorHex(snapshot.punctureChannels)
              << " topology_states=" << rawVectorHex(snapshot.topologyStates)
              << " mpm_active_node_indices=" << rawVectorHex(snapshot.mpmActiveNodeIndices)
              << " mpm_node_to_active=" << rawVectorHex(snapshot.mpmNodeToActive)
              << " mpm_active_node_counts=" << rawVectorHex(snapshot.mpmActiveNodeCounts)
              << " rigid_generalized_candidate=" << rawVectorHex(snapshot.rigidGeneralizedCandidate)
              << " learned_weights=" << rawVectorHex(snapshot.learnedWeights)
              << " adaptive=" << rawVectorHex(snapshot.adaptive)
              << " schedulers=" << rawVectorHex(snapshot.schedulers)
              << " reactions=" << rawVectorHex(snapshot.reactions)
              << " rigid_states=" << rawVectorHex(snapshot.rigidStates)
              << " contact_samples=" << rawVectorHex(snapshot.contactSamples)
              << " deformable_contact_failures=" << rawVectorHex(snapshot.deformableContactFailures)
              << " contact_histories=" << rawVectorHex(snapshot.contactHistories)
              << " human_support_histories=" << rawVectorHex(snapshot.humanSupportHistories)
              << " human_support_consequences=" << rawVectorHex(snapshot.humanSupportConsequences)
              << " deformable_contact_histories=" << rawVectorHex(snapshot.deformableContactHistories)
              << " particle_material_state=" << rawVectorHex(snapshot.particleMaterialState)
              << " fem_material_state=" << rawVectorHex(snapshot.femMaterialState)
              << " identification=" << rawVectorHex(snapshot.identification)
              << " environment_parameters=" << rawVectorHex(snapshot.environmentParameters)
              << " fem_regional_linearization_nodes=" << rawVectorHex(snapshot.femRegionalGalerkinLinearizationNodes)
              << " fem_regional_linearization_material_state=" << rawVectorHex(snapshot.femRegionalGalerkinLinearizationMaterialState)
              << " fem_regional_linearization_scheduler=" << rawVectorHex(snapshot.femRegionalGalerkinLinearizationSchedulerExponent)
              << " fem_regional_F_high=" << rawVectorHex(snapshot.femRegionalGalerkinLinearizationLocalFHigh)
              << " fem_regional_F_low=" << rawVectorHex(snapshot.femRegionalGalerkinLinearizationLocalFLow)
              << " fem_regional_basis=" << rawVectorHex(snapshot.femRegionalGalerkinBasisP)
              << " fem_regional_effective_basis=" << rawVectorHex(snapshot.femRegionalGalerkinEffectiveBasisP)
              << " fem_regional_AP=" << rawVectorHex(snapshot.femRegionalGalerkinOperatorAP)
              << " fem_regional_RAP=" << rawVectorHex(snapshot.femRegionalGalerkinPivotedRAP)
              << " fem_regional_pivots=" << rawVectorHex(snapshot.femRegionalGalerkinPivots)
              << " fem_regional_factor_valid=" << rawVectorHex(snapshot.femRegionalGalerkinFactorValid)
              << '\n';
    std::cout << "selector_runtime_snapshot_metadata label=" << label
              << " identification_generation=" << snapshot.identificationGeneration
              << " identification_checkpoint=" << snapshot.identificationCheckpoint
              << " identification_advanced=" << snapshot.identificationAdvanced
              << " allocation_generation=" << snapshot.allocationGeneration
              << " material_state_stride=" << snapshot.materialStateStride
              << " learned_weight_revision=" << snapshot.learnedWeightRevision
              << " regional_diagnostics_available=" << snapshot.femRegionalGalerkinDiagnosticsAvailable
              << " regional_mode_count=" << snapshot.femRegionalGalerkinModeCount
              << " regional_readback_bytes=" << snapshot.femRegionalGalerkinDiagnosticReadbackBytes
              << " regional_message_hex=" << rawStringHex(snapshot.femRegionalGalerkinDiagnosticsMessage)
              << " restart_diagnostics_available=" << snapshot.femRegionalRestartAugmentationDiagnosticsAvailable
              << " restart_base_modes=" << snapshot.femRegionalRestartBaseModeCount
              << " restart_augmented_modes=" << snapshot.femRegionalRestartAugmentedModeCount
              << " restart_control_step=" << snapshot.femRegionalRestartControlStep
              << " restart_substep=" << snapshot.femRegionalRestartPhysicsSubstep
              << " restart_newton_iteration=" << snapshot.femRegionalRestartNewtonIteration
              << " restart_cycle=" << snapshot.femRegionalRestartCycle
              << " restart_attempt_count=" << snapshot.femRegionalRestartAttemptCount
              << " restart_mode_valid=" << snapshot.femRegionalRestartModeValid
              << " restart_augmented_factor_valid=" << snapshot.femRegionalRestartAugmentedFactorValid
              << " restart_readback_bytes=" << snapshot.femRegionalRestartDiagnosticReadbackBytes
              << '\n';
    std::cout << std::setprecision(9);
    for (std::size_t i = 0u; i < snapshot.statuses.size(); ++i) {
        const auto& status = snapshot.statuses[i];
        std::cout << "selector_runtime_status label=" << label << " index=" << i
                  << " code=" << status.code << " environment=" << status.environment
                  << " object=" << status.objectIndex << " failing_index=" << status.failingIndex
                  << " completed_microsteps=" << status.completedMicrosteps
                  << " fgmres_iterations=" << status.fgmresIterations
                  << " contact_count=" << status.contactCount << " event_count=" << status.eventCount
                  << " diagnostics=" << status.diagnostics.x << ',' << status.diagnostics.y << ','
                  << status.diagnostics.z << ',' << status.diagnostics.w << '\n';
    }
    for (std::size_t i = 0u; i < snapshot.solverCertificates.size(); ++i) {
        const auto& certificate = snapshot.solverCertificates[i];
        std::cout << "selector_runtime_certificate label=" << label << " index=" << i
                  << " nonlinear=" << certificate.nonlinear.x << ',' << certificate.nonlinear.y << ','
                  << certificate.nonlinear.z << ',' << certificate.nonlinear.w
                  << " contact=" << certificate.contact.x << ',' << certificate.contact.y << ','
                  << certificate.contact.z << ',' << certificate.contact.w
                  << " transport=" << certificate.transport.x << ',' << certificate.transport.y << ','
                  << certificate.transport.z << ',' << certificate.transport.w
                  << " validity=" << certificate.validity.x << ',' << certificate.validity.y << ','
                  << certificate.validity.z << ',' << certificate.validity.w << '\n';
    }
}

RuntimeStateSnapshot runOneAcceptedRelease(
    const CompiledWorld& world,
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    const float appliedForce,
    std::size_t& residentBytes
) {
    RuntimeConfiguration configuration;
    configuration.metallib = std::filesystem::path(NUMI_MATTER_METALLIB);
    configuration.environmentCount = world.dispatch.environmentCount;
    configuration.captureEvents = false;
    configuration.captureDiagnostics = true;
    configuration.adaptiveTransfer = false;
    Runtime runtime;
    const RuntimeDiagnostics initialized = runtime.initialize(world, configuration);
    residentBytes = initialized.residentBytes;
    std::cout << "selector_runtime_initialization mode="
              << world.mixedSolver.executionBudgets.z
              << " encoded=" << initialized.encoded
              << " resident_bytes_reported=" << initialized.residentBytes
              << " device_hex=" << rawStringHex(initialized.device)
              << " message_hex=" << rawStringHex(initialized.message) << '\n';
    need(initialized.encoded,
         "selector runtime initialization failed: " + initialized.message);

    std::vector<nm_float4> forces(world.dispatch.femNodeCount, nm_float4{});
    forces[2] = {appliedForce, 0.0f, 0.0f, 0.0f};
    std::vector<nm_float4> targets(world.dispatch.femNodeCount, nm_float4{});
    targets[2] = {0.0f, 0.0f, 0.0f, 2.0f};
    const auto forceBuffer = makeBuffer(device, forces);
    const auto targetBuffer = makeBuffer(device, targets);
    std::vector<MRMetalWorldStatusGPU> worldStatuses(world.dispatch.environmentCount);
    for (std::uint32_t environment = 0; environment < world.dispatch.environmentCount;
         ++environment)
        worldStatuses[environment].environment = environment;
    const auto statusBuffer = makeBuffer(device, worldStatuses);
    id<MTLCommandBuffer> command = [queue commandBuffer];
    need(command != nil, "cannot create selector parity command buffer");
    EncodeRequest request;
    request.commandBuffer = (__bridge void*)command;
    request.environmentStatuses = (__bridge void*)statusBuffer;
    request.controlStep = 0u;
    request.physicsSubstep = 0u;
    request.physicsSubsteps = 1u;
    request.timestepSeconds = runtime.timestepSeconds();
    request.runAdaptiveTransfer = false;
    request.femExternalForces = (__bridge void*)forceBuffer;
    request.femExternalForceCount = static_cast<std::uint32_t>(forces.size());
    request.femKinematicTargets = (__bridge void*)targetBuffer;
    request.femKinematicTargetCount = static_cast<std::uint32_t>(targets.size());
    request.phase = EncodePhase::preDynamics;
    RuntimeDiagnostics encoded = runtime.encode(request);
    std::cout << "selector_runtime_encode mode=" << world.mixedSolver.executionBudgets.z
              << " phase=preDynamics encoded=" << encoded.encoded
              << " message_hex=" << rawStringHex(encoded.message) << '\n';
    need(encoded.encoded, "selector preDynamics failed: " + encoded.message);
    request.phase = EncodePhase::postCommit;
    encoded = runtime.encode(request);
    std::cout << "selector_runtime_encode mode=" << world.mixedSolver.executionBudgets.z
              << " phase=postCommit encoded=" << encoded.encoded
              << " message_hex=" << rawStringHex(encoded.message) << '\n';
    need(encoded.encoded, "selector postCommit failed: " + encoded.message);
    [command commit];
    [command waitUntilCompleted];
    std::cout << "selector_runtime_command mode=" << world.mixedSolver.executionBudgets.z
              << " command_status=" << static_cast<unsigned>(command.status)
              << " error_hex=" << rawStringHex(command.error == nil ? std::string{} :
                    std::string([[command.error.localizedDescription description] UTF8String]))
              << '\n';
    need(command.status == MTLCommandBufferStatusCompleted,
         command.error == nil ? "selector parity command failed" :
         std::string([[command.error.localizedDescription description] UTF8String]));
    return runtime.snapshot();
}

void checkSelector5And6AcceptanceParity(const std::filesystem::path& materialPath) {
    const Fixture five = cook(materialPath, paper::State{}, baselineDeformation,
        FEMPreconditionerMode::regionalGalerkinGraphSchwarz, {2u, 3u});
    const Fixture six = cook(materialPath, paper::State{}, baselineDeformation,
        FEMPreconditionerMode::regionalGalerkinHybridGraphSchwarz, {2u, 3u});
    requireSamePhysicalInputs(five.world, six.world);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    need(device != nil, "selector parity requires a Metal device");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    RuntimeStateSnapshot fiveState{};
    RuntimeStateSnapshot sixState{};
    std::string fiveError, sixError;
    std::size_t fiveResidentBytes = 0u, sixResidentBytes = 0u;
    try {
        fiveState = runOneAcceptedRelease(five.world, device, queue, 1.0e-2f,
                                          fiveResidentBytes);
    } catch (const std::exception& error) {
        fiveError = error.what();
    }
    try {
        sixState = runOneAcceptedRelease(six.world, device, queue, 1.0e-2f,
                                         sixResidentBytes);
    } catch (const std::exception& error) {
        sixError = error.what();
    }
    printSelectorRuntimeOutcome("selector5", fiveState, fiveError, fiveResidentBytes);
    printSelectorRuntimeOutcome("selector6", sixState, sixError, sixResidentBytes);
    const auto acceptedOutcome = [](const RuntimeStateSnapshot& state,
                                    const std::string& error,
                                    const std::uint32_t environmentCount) {
        return error.empty() && state.available &&
            state.statuses.size() == environmentCount &&
            state.statuses[0].code == NM_STATUS_SUCCESS &&
            state.solverCertificates.size() == environmentCount &&
            state.solverCertificates[0].validity.w > 0.5f;
    };
    need(acceptedOutcome(fiveState, fiveError, five.world.dispatch.environmentCount) &&
             acceptedOutcome(sixState, sixError, six.world.dispatch.environmentCount),
         "selector accepted-status/certificate gate failed; both complete outcomes were emitted above");
    for (const auto* state : {&fiveState, &sixState}) {
        const float relativeCorrection = state->solverCertificates[0].nonlinear.y;
        need(std::isfinite(relativeCorrection) && relativeCorrection > 0.0f,
             "selector runtime fixture accepted without a nonzero mechanical correction; "
             "the initial nonlinear latch may have bypassed Krylov work");
        need(state->femNodes.size() == five.world.dispatch.femNodeCount &&
                 state->femNodes[2].restAndFixed.w == 0.0f &&
                 state->femNodes[3].restAndFixed.w == 1.0f,
             "selector release/remaining-fixed postcondition failed");
    }
    const auto requireLoadedMotion = [](const RuntimeStateSnapshot& state,
                                        const CompiledWorld& world) {
        const auto& initialLoadedNode = world.fem.nodes[2].positionAndMass;
        const auto& finalLoadedNode = state.femNodes[2].positionAndMass;
        const double loadedNodeMotion = std::sqrt(
            std::pow(double(finalLoadedNode.x - initialLoadedNode.x), 2.0) +
            std::pow(double(finalLoadedNode.y - initialLoadedNode.y), 2.0) +
            std::pow(double(finalLoadedNode.z - initialLoadedNode.z), 2.0));
        need(std::isfinite(loadedNodeMotion) && loadedNodeMotion > 1.0e-8,
             "selector runtime fixture produced no representable loaded-node motion");
        return loadedNodeMotion;
    };
    const double fiveLoadedMotion = requireLoadedMotion(fiveState, five.world);
    const double sixLoadedMotion = requireLoadedMotion(sixState, six.world);
    std::cout << "selector_runtime_loaded_motion mode=5 metres=" << fiveLoadedMotion << '\n'
              << "selector_runtime_loaded_motion mode=6 metres=" << sixLoadedMotion << '\n';
    double maxPositionDifference = 0.0;
    double maxVelocityDifference = 0.0;
    for (std::size_t node = 0u; node < fiveState.femNodes.size(); ++node) {
        const auto& a = fiveState.femNodes[node];
        const auto& b = sixState.femNodes[node];
        for (const auto pair : {
                 std::pair{a.positionAndMass.x, b.positionAndMass.x},
                 std::pair{a.positionAndMass.y, b.positionAndMass.y},
                 std::pair{a.positionAndMass.z, b.positionAndMass.z}})
            maxPositionDifference = std::max(maxPositionDifference,
                                              std::abs(double(pair.first - pair.second)));
        for (const auto pair : {
                 std::pair{a.velocityAndInverseMass.x, b.velocityAndInverseMass.x},
                 std::pair{a.velocityAndInverseMass.y, b.velocityAndInverseMass.y},
                 std::pair{a.velocityAndInverseMass.z, b.velocityAndInverseMass.z}})
            maxVelocityDifference = std::max(maxVelocityDifference,
                                              std::abs(double(pair.first - pair.second)));
    }
    std::cout << "selector5_6_runtime_acceptance=pass"
              << " status5=" << fiveState.statuses[0].code
              << " status6=" << sixState.statuses[0].code
              << " cert5=" << fiveState.solverCertificates[0].validity.w
              << " cert6=" << sixState.solverCertificates[0].validity.w
              << " max_position_difference_m=" << maxPositionDifference
              << " max_velocity_difference_m_per_s=" << maxVelocityDifference
              << " physical_validation=false\n";
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc == 3 && std::string_view(argv[1]) ==
                                 "--selector6-compile-only") {
                const Fixture fixture = cook(
                    argv[2], paper::State{}, baselineDeformation,
                    FEMPreconditionerMode::regionalGalerkinHybridGraphSchwarz,
                    {2u, 3u});
                need(fixture.world.mixedSolver.executionBudgets.z == 6u &&
                         fixture.world.dispatch.environmentCount == 1u &&
                         fixture.world.dispatch.femNodeCount == 4u &&
                         fixture.world.dispatch.tetrahedronCount == 1u &&
                         fixture.world.objects.size() == 1u &&
                         fixture.world.objects[0].representation ==
                             NM_REPRESENTATION_FEM &&
                         fixture.world.objects[0].flags &
                             NM_OBJECT_DISABLE_DEFORMABLE_CONTACT &&
                         fixture.world.objects[0].flags &
                             NM_OBJECT_DISABLE_SELF_CONTACT,
                     "selector6 compile-only fixture lost its authored single-region contract");
                std::cout << "selector6_compile_only=pass region_id=0 "
                             "topology=4nodes/1tet physical_validation=false\n";
                return 0;
            }
            need(argc == 3, "expected liner and medium material paths");
            MetalCheck gpu;
            checkMaterial(gpu, argv[1]);
            checkMaterial(gpu, argv[2]);
            checkYieldedMaterial(gpu, argv[1]);
            checkYieldedMaterial(gpu, argv[2]);
            Selector6NativeProbe selector6;
            selector6.run(argv[1]);
            checkSelector5And6AcceptanceParity(argv[1]);
            std::cout << "PASS direct Metal tangent-diagonal FD; "
                         "selector6_hybrid_kernels=checked "
                         "physical_validation=false\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << error.what() << '\n';
            return 1;
        }
    }
}
