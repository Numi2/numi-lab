#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "cardboard_paper_reference.hpp"
#include "numi/matter/compiler.hpp"
#include "numi/matter/language.hpp"
#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
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

void checkNondefaultPackageRoundTrip(const CompileResult& compiled) {
    need(compiled.world.mixedSolver.executionBudgets.z == 1u,
         "package fixture did not select the nondefault regional tangent mode");
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
    need(roundTrip.mixedSolver.executionBudgets.z == 1u &&
             roundTrip.materials.at(0u).localNewtonIterations == 16u,
         "package round-trip lost nondefault preconditioner/material controls");
    std::cout << "package_roundtrip=pass mode=regional-tangent local_material_newton=16\n";
}

Fixture cook(
    const std::filesystem::path& materialPath,
    const paper::State& acceptedState = paper::State{},
    const Matrix& deformation = baselineDeformation
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
    source.mixedSolver.femPreconditioner =
        FEMPreconditionerMode::regionalTangentDiagonal;
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
    object.tetrahedra = {{{0u, 1u, 2u, 3u}}};
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
    need(compiled.world.mixedSolver.executionBudgets.z == 1u,
         "compiled package lost the preconditioner selector");
    need(compiled.world.fem.nodes.size() == 4u &&
             compiled.world.fem.tetrahedra.size() == 1u,
         "compiled fixture is not one explicit four-node tetrahedron");
    checkNondefaultPackageRoundTrip(compiled);

    WorldSource legacy = source;
    legacy.mixedSolver.femPreconditioner =
        FEMPreconditionerMode::scalarDiagonal;
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
} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            need(argc == 3, "expected liner and medium material paths");
            MetalCheck gpu;
            checkMaterial(gpu, argv[1]);
            checkMaterial(gpu, argv[2]);
            checkYieldedMaterial(gpu, argv[1]);
            checkYieldedMaterial(gpu, argv[2]);
            std::cout << "PASS direct Metal tangent-diagonal FD; "
                         "physical_validation=false\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << error.what() << '\n';
            return 1;
        }
    }
}
