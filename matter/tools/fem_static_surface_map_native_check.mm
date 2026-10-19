#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifndef NUMI_MATTER_METALLIB
#error "CMake must define NUMI_MATTER_METALLIB"
#endif

namespace {
using namespace numi::matter;

constexpr std::uint32_t kEnvironments = 2u;
constexpr float kDt = 0.002f;
unsigned checks = 0u;

void require(const bool value, const std::string& message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}

WorldSource fixture(const MaterialProgram& material) {
    WorldSource source;
    source.environmentCount = kEnvironments;
    source.gravity = {0.0, 0.0, 0.0};
    source.frameTimestep = kDt;
    source.materials = {material};

    ObjectSource adjacent;
    adjacent.name = "shared-interior-face-object";
    adjacent.representation = Representation::fem;
    adjacent.materialIndex = 0u;
    adjacent.mixedFEM = false;
    adjacent.twoWayCoupling = false;
    adjacent.adaptive = false;
    adjacent.deformableContact = false;
    adjacent.deformableSelfContact = false;
    adjacent.femNodes = {
        {0.0, 0.0, 0.0}, {0.01, 0.0, 0.0}, {0.0, 0.01, 0.0},
        {0.0, 0.0, 0.01}, {0.0, 0.0, -0.01},
    };
    // The (0,1,2) face is shared and must be omitted from the surface map.
    adjacent.tetrahedra = {{{0u, 1u, 2u, 3u}}, {{0u, 2u, 1u, 4u}}};

    ObjectSource duplicated;
    duplicated.name = "coincident-duplicated-interface-object";
    duplicated.representation = Representation::fem;
    duplicated.materialIndex = 0u;
    duplicated.mixedFEM = false;
    duplicated.twoWayCoupling = false;
    duplicated.adaptive = false;
    duplicated.deformableContact = false;
    duplicated.deformableSelfContact = false;
    // Its first three nodes coincide with object 0's exterior (0,1,3) face,
    // but have distinct FEM node IDs. The kernel must retain both sides.
    duplicated.femNodes = {
        {0.0, 0.0, 0.0}, {0.01, 0.0, 0.0},
        {0.0, 0.0, 0.01}, {0.0, -0.01, 0.0},
    };
    duplicated.tetrahedra = {{{0u, 1u, 2u, 3u}}};

    source.objects = {std::move(adjacent), std::move(duplicated)};
    return source;
}

CompiledWorld compile(const WorldSource& source) {
    const CompileResult result = compileWorld(source,
        {.maximumRateExponent = 0u, .emitSpecializedMetal = false});
    std::string details;
    for (const Diagnostic& diagnostic : result.diagnostics)
        details += diagnostic.message + "; ";
    require(result.succeeded(), "fixture compile failed: " + details);
    return result.world;
}

template<class T>
id<MTLBuffer> sharedBuffer(
    id<MTLDevice> device,
    const std::vector<T>& values,
    const T& emptyValue = T{}
) {
    const T* data = values.empty() ? &emptyValue : values.data();
    const std::size_t count = values.empty() ? 1u : values.size();
    id<MTLBuffer> result = [device newBufferWithBytes:data
        length:count * sizeof(T) options:MTLResourceStorageModeShared];
    require(result != nil, "Metal input buffer allocation failed");
    return result;
}

template<class T>
std::vector<T> replicated(const std::vector<T>& source) {
    std::vector<T> result;
    result.reserve(source.size() * kEnvironments);
    for (std::uint32_t environment = 0u; environment < kEnvironments;
         ++environment)
        result.insert(result.end(), source.begin(), source.end());
    return result;
}

struct PrimitiveRecord {
    std::uint32_t environment = 0u;
    std::uint32_t sourceFace = 0u;
    std::uint32_t kind = 0u;
    std::uint32_t generation = 0u;
    std::uint32_t flags = 0u;
    std::array<std::uint32_t, 4> nodesAndObject{};
    std::array<float, 4> boundsMinimum{};
    std::array<float, 4> boundsMaximum{};
};

auto key(const PrimitiveRecord& record) {
    return std::tuple(record.environment, record.sourceFace,
        record.nodesAndObject[3],
        record.nodesAndObject[0], record.nodesAndObject[1],
        record.nodesAndObject[2]);
}

bool close(const float a, const float b) {
    return std::isfinite(a) && std::isfinite(b) &&
        std::abs(a - b) <= 2.0e-7f * std::max({1.0f, std::abs(a), std::abs(b)});
}

std::vector<NMFEMNodeStateGPU> deformedNodes(const CompiledWorld& world,
                                             const std::uint32_t environment) {
    std::vector<NMFEMNodeStateGPU> nodes = world.fem.nodes;
    for (NMFEMNodeStateGPU& node : nodes) {
        const float x = node.restAndFixed.x;
        const float y = node.restAndFixed.y;
        const float z = node.restAndFixed.z;
        // Same smooth affine deformation is applied to coincident duplicate
        // nodes. Environment offsets make the two output batches distinguishable.
        const float e = static_cast<float>(environment);
        node.positionAndMass.x = x + 0.013f * y + 0.0003f * e;
        node.positionAndMass.y = y - 0.009f * x + 0.0002f * e;
        node.positionAndMass.z = z + 0.006f * x - 0.0001f * e;
        node.velocityAndInverseMass.x = 0.07f + 0.3f * y + 0.01f * e;
        node.velocityAndInverseMass.y = -0.03f - 0.2f * x + 0.015f * e;
        node.velocityAndInverseMass.z = 0.04f + 0.1f * z - 0.02f * e;
    }
    return nodes;
}

std::vector<PrimitiveRecord> dispatchBuilder(
    id<MTLDevice> device,
    id<MTLComputePipelineState> pipeline,
    const CompiledWorld& world
) {
    NMMatterDispatchGPU dispatch = world.dispatch;
    dispatch.environmentCount = kEnvironments;
    dispatch.gridNodeCount = 0u;
    dispatch.mpmGridCount = 0u;
    dispatch.numericalLimits.x = 2.0e-5f;
    dispatch.gravityAndTimestep.w = kDt;
    const bool compact =
        (dispatch.flags & NM_MATTER_STATIC_FEM_SURFACE_MAP) != 0u;
    const std::uint32_t perEnvironmentCount = compact
        ? dispatch.surfaceFaceCount : 4u * dispatch.tetrahedronCount;
    require(perEnvironmentCount > 0u,
        "surface builder fixture has no primitives");

    NMMicrostepGPU microstep{};
    microstep.microtickCount = 1u;
    microstep.time = {kDt, 1.0f / kDt, 0.0f, kDt};

    std::vector<NMFEMNodeStateGPU> nodes;
    nodes.reserve(world.fem.nodes.size() * kEnvironments);
    for (std::uint32_t environment = 0u; environment < kEnvironments;
         ++environment) {
        auto batch = deformedNodes(world, environment);
        nodes.insert(nodes.end(), batch.begin(), batch.end());
    }
    auto tetrahedra = replicated(world.fem.tetrahedra);
    auto objects = world.objects;
    auto schedulers = replicated(world.schedulers);
    auto adaptive = replicated(world.adaptive);
    require(objects.size() == dispatch.objectCount &&
            schedulers.size() == kEnvironments * dispatch.objectCount &&
            adaptive.size() == kEnvironments * dispatch.objectCount,
        "compiled object/scheduler layout is not environment-major compatible");
    for (std::uint32_t environment = 0u; environment < kEnvironments;
         ++environment) {
        for (std::uint32_t object = 0u; object < dispatch.objectCount; ++object) {
            const std::size_t index =
                std::size_t(environment) * dispatch.objectCount + object;
            schedulers[index].activeExponent = 0u;
            schedulers[index].numerical.w = 1.0f;
            adaptive[index].activeRepresentation = NM_REPRESENTATION_FEM;
        }
    }
    auto nodeIncidence = replicated(world.fem.nodeIncidence);
    auto nodeRanges = replicated(world.fem.nodeRanges);
    for (std::uint32_t environment = 0u; environment < kEnvironments;
         ++environment) {
        const std::uint32_t incidenceOffset =
            environment * static_cast<std::uint32_t>(world.fem.nodeIncidence.size());
        for (std::uint32_t localNode = 0u;
             localNode < world.dispatch.femNodeCount; ++localNode) {
            nodeRanges[environment * world.dispatch.femNodeCount + localNode].first +=
                incidenceOffset;
        }
    }

    const std::vector<NMGridNodeStateGPU> gridNodes(1u);
    const std::vector<NMMPMGridGPU> grids(1u);
    const std::vector<std::uint32_t> gridGenerations(1u, 0u);
    const std::vector<NMFEMSurfaceFaceGPU> faces = compact
        ? world.fem.surfaceFaces : std::vector<NMFEMSurfaceFaceGPU>{};
    const std::size_t total =
        std::size_t(kEnvironments) * perEnvironmentCount;
    std::vector<NMContinuumSurfacePrimitiveGPU> outputTemplate(total);

    id<MTLBuffer> dispatchBuffer = sharedBuffer(device,
        std::vector<NMMatterDispatchGPU>{dispatch});
    id<MTLBuffer> microstepBuffer = sharedBuffer(device,
        std::vector<NMMicrostepGPU>{microstep});
    id<MTLBuffer> objectBuffer = sharedBuffer(device, objects);
    id<MTLBuffer> tetBuffer = sharedBuffer(device, tetrahedra);
    id<MTLBuffer> nodeBuffer = sharedBuffer(device, nodes);
    id<MTLBuffer> schedulerBuffer = sharedBuffer(device, schedulers);
    id<MTLBuffer> gridNodeBuffer = sharedBuffer(device, gridNodes);
    id<MTLBuffer> gridBuffer = sharedBuffer(device, grids);
    id<MTLBuffer> generationBuffer = sharedBuffer(device, gridGenerations);
    id<MTLBuffer> adaptiveBuffer = sharedBuffer(device, adaptive);
    id<MTLBuffer> incidenceBuffer = sharedBuffer(device, nodeIncidence);
    id<MTLBuffer> rangeBuffer = sharedBuffer(device, nodeRanges);
    id<MTLBuffer> outputBuffer = [device newBufferWithBytes:outputTemplate.data()
        length:outputTemplate.size() * sizeof(NMContinuumSurfacePrimitiveGPU)
        options:MTLResourceStorageModeShared];
    const NMFEMSurfaceFaceGPU sentinel{};
    id<MTLBuffer> faceBuffer = sharedBuffer(device, faces, sentinel);
    const std::vector<std::uint32_t> cacheFlagSentinel(kEnvironments, 0u);
    id<MTLBuffer> cacheFlagBuffer = sharedBuffer(device, cacheFlagSentinel);
    require(outputBuffer != nil && cacheFlagBuffer != nil,
        "surface output/cache flag buffer allocation failed");

    id<MTLCommandQueue> queue = [device newCommandQueue];
    require(queue != nil, "Metal command queue allocation failed");
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(command != nil && encoder != nil,
        "surface builder command encoding failed");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
    [encoder setBuffer:microstepBuffer offset:0u atIndex:1u];
    [encoder setBuffer:objectBuffer offset:0u atIndex:2u];
    [encoder setBuffer:tetBuffer offset:0u atIndex:3u];
    [encoder setBuffer:nodeBuffer offset:0u atIndex:4u];
    [encoder setBuffer:schedulerBuffer offset:0u atIndex:5u];
    [encoder setBuffer:gridNodeBuffer offset:0u atIndex:6u];
    [encoder setBuffer:gridBuffer offset:0u atIndex:7u];
    [encoder setBuffer:generationBuffer offset:0u atIndex:8u];
    [encoder setBuffer:adaptiveBuffer offset:0u atIndex:9u];
    [encoder setBuffer:incidenceBuffer offset:0u atIndex:10u];
    [encoder setBuffer:rangeBuffer offset:0u atIndex:11u];
    [encoder setBuffer:outputBuffer offset:0u atIndex:12u];
    // Always bind a valid buffer. The legacy kernel branch must not index it,
    // but Metal still validates declared resources when encoding.
    [encoder setBuffer:faceBuffer offset:0u atIndex:13u];
    [encoder setBuffer:nodeBuffer offset:0u atIndex:14u];
    const std::uint32_t includePrivateBase = 0u;
    [encoder setBytes:&includePrivateBase length:sizeof(includePrivateBase)
              atIndex:15u];
    [encoder setBuffer:cacheFlagBuffer offset:0u atIndex:16u];
    const std::uint32_t useArmijoSurfaceCache = 0u;
    [encoder setBytes:&useArmijoSurfaceCache
              length:sizeof(useArmijoSurfaceCache) atIndex:17u];
    const NSUInteger width = std::min<NSUInteger>(
        pipeline.maxTotalThreadsPerThreadgroup, 64u);
    const NSUInteger groups = (total + width - 1u) / width;
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
        command.error != nil
            ? std::string("surface build kernel failed: ") +
                [[command.error localizedDescription] UTF8String]
            : "surface build kernel did not complete");

    const auto* output = static_cast<const NMContinuumSurfacePrimitiveGPU*>(
        outputBuffer.contents);
    require(output != nullptr, "surface output readback unavailable");
    std::vector<PrimitiveRecord> active;
    for (std::size_t index = 0u; index < total; ++index) {
        const auto& primitive = output[index];
        if ((primitive.identity.w & NM_TOPOLOGY_ACTIVE) == 0u) continue;
        require(primitive.identity.y == NM_CONTINUUM_SURFACE_TRIANGLE,
            "unexpected active non-triangle primitive");
        PrimitiveRecord record;
        record.sourceFace = primitive.identity.x;
        record.environment = static_cast<std::uint32_t>(index / perEnvironmentCount);
        record.kind = primitive.identity.y;
        record.generation = primitive.identity.z;
        record.flags = primitive.identity.w;
        record.nodesAndObject = {
            primitive.nodesAndObject.x, primitive.nodesAndObject.y,
            primitive.nodesAndObject.z, primitive.nodesAndObject.w};
        record.boundsMinimum = {primitive.boundsMinimum.x,
            primitive.boundsMinimum.y, primitive.boundsMinimum.z,
            primitive.boundsMinimum.w};
        record.boundsMaximum = {primitive.boundsMaximum.x,
            primitive.boundsMaximum.y, primitive.boundsMaximum.z,
            primitive.boundsMaximum.w};
        active.push_back(record);
    }
    std::sort(active.begin(), active.end(),
        [](const PrimitiveRecord& a, const PrimitiveRecord& b) {
            return key(a) < key(b);
        });
    return active;
}

std::array<std::array<float, 3>, 3> faceCoordinates(
    const PrimitiveRecord& record,
    const std::vector<NMFEMNodeStateGPU>& nodes,
    const std::uint32_t environment,
    const std::uint32_t nodeCount
) {
    std::array<std::array<float, 3>, 3> coordinates{};
    require(record.environment == environment,
        "surface coordinate fixture selected the wrong environment batch");
    for (std::uint32_t corner = 0u; corner < 3u; ++corner) {
        const std::uint32_t unifiedNode = record.nodesAndObject[corner];
        const std::uint32_t local = unifiedNode; // FEM-only fixture: grid offset is zero.
        require(local < nodeCount, "surface node index outside compiled FEM arena");
        // deformedNodes supplies one environment's local node batch, already
        // shifted for this environment; it is not an environment-major array.
        const std::size_t index = local;
        require(index < nodes.size(), "surface coordinate fixture batch is truncated");
        const auto& position = nodes[index].positionAndMass;
        coordinates[corner] = {position.x, position.y, position.z};
    }
    std::sort(coordinates.begin(), coordinates.end());
    return coordinates;
}

void compareEquivalent(const CompiledWorld& legacy,
                       const CompiledWorld& compact,
                       const std::vector<PrimitiveRecord>& a,
                       const std::vector<PrimitiveRecord>& b) {
    require(a.size() == b.size() && a.size() == 10u * kEnvironments,
        "legacy/compact active surface counts differ or shared face survived");
    std::array<std::size_t, kEnvironments> perEnvironment{};
    bool sawMotion = false;
    for (std::size_t i = 0u; i < a.size(); ++i) {
        require(key(a[i]) == key(b[i]),
            "active source face IDs or node ownership differ");
        require(a[i].kind == b[i].kind &&
                a[i].generation == b[i].generation &&
                a[i].flags == b[i].flags,
            "active surface metadata differs");
        for (std::uint32_t axis = 0u; axis < 4u; ++axis) {
            require(close(a[i].boundsMinimum[axis], b[i].boundsMinimum[axis]) &&
                    close(a[i].boundsMaximum[axis], b[i].boundsMaximum[axis]),
                "active swept bounds/measure differ");
        }
        sawMotion |= a[i].boundsMaximum[3] > 0.0f;
        require(a[i].environment < kEnvironments,
            "surface builder emitted an invalid environment");
        ++perEnvironment[a[i].environment];
        require(a[i].sourceFace != 3u && a[i].sourceFace != 7u,
            "shared interior face was emitted");
    }
    require(sawMotion, "moving-geometry swept travel was not represented");
    require(perEnvironment[0] == 10u && perEnvironment[1] == 10u,
        "surface builder did not emit ten active rows in each environment");

    std::map<std::pair<std::uint32_t, std::uint32_t>, const PrimitiveRecord*>
        duplicateSides;
    for (const PrimitiveRecord& record : a)
        if (record.sourceFace == 2u || record.sourceFace == 11u)
            duplicateSides.emplace(
                std::pair(record.environment, record.sourceFace), &record);
    require(duplicateSides.size() == 2u * kEnvironments,
        "both duplicate-interface sides were not retained");

    for (std::uint32_t environment = 0u; environment < kEnvironments;
         ++environment) {
        const auto* firstSide = duplicateSides.at({environment, 2u});
        const auto* secondSide = duplicateSides.at({environment, 11u});
        require(firstSide->nodesAndObject[3] == 0u &&
                secondSide->nodesAndObject[3] == 1u,
            "duplicated interface sides lost their distinct owning objects");
        const auto legacyNodes = deformedNodes(legacy, environment);
        const auto compactNodes = deformedNodes(compact, environment);
        const auto left = faceCoordinates(*firstSide, legacyNodes,
            environment, legacy.dispatch.femNodeCount);
        const auto right = faceCoordinates(*secondSide, legacyNodes,
            environment, legacy.dispatch.femNodeCount);
        require(left == right,
            "duplicated source faces no longer coincide after deformation");
        const auto compactLeft = faceCoordinates(*firstSide, compactNodes,
            environment, compact.dispatch.femNodeCount);
        const auto compactRight = faceCoordinates(*secondSide, compactNodes,
            environment, compact.dispatch.femNodeCount);
        require(compactLeft == compactRight,
            "compact map changed duplicated-side geometry");
    }
}

void comparePhysicalInputs(const CompiledWorld& legacy,
                           const CompiledWorld& compact) {
    require(legacy.dispatch.objectCount >= 2u && compact.dispatch.objectCount >= 2u,
        "surface-map fixture does not contain two objects");
    require(legacy.dispatch.environmentCount == kEnvironments &&
            compact.dispatch.environmentCount == kEnvironments,
        "surface-map fixture lost environment count");
    require(legacy.dispatch.femNodeCount == compact.dispatch.femNodeCount &&
            legacy.dispatch.tetrahedronCount == compact.dispatch.tetrahedronCount,
        "surface-map option changed FEM dimensions");
    require(legacy.fem.tetrahedra.size() == compact.fem.tetrahedra.size() &&
            std::memcmp(legacy.fem.tetrahedra.data(), compact.fem.tetrahedra.data(),
                legacy.fem.tetrahedra.size() * sizeof(NMTetrahedronGPU)) == 0,
        "surface-map option changed compiled tetrahedra");
    require(legacy.fem.nodes.size() == compact.fem.nodes.size(),
        "surface-map option changed compiled FEM nodes");
    double legacyMass = 0.0;
    double compactMass = 0.0;
    for (std::size_t i = 0u; i < legacy.fem.nodes.size(); ++i) {
        const float left = legacy.fem.nodes[i].positionAndMass.w;
        const float right = compact.fem.nodes[i].positionAndMass.w;
        require(std::bit_cast<std::uint32_t>(left) ==
                std::bit_cast<std::uint32_t>(right),
            "surface-map option changed authored nodal mass");
        legacyMass += left;
        compactMass += right;
    }
    require(legacyMass == compactMass && legacyMass > 0.0,
        "surface-map option changed total physical FEM mass");
}
}

int main(const int argc, char** argv) {
    try {
        const auto materialPath = std::filesystem::path(__FILE__).parent_path()
            .parent_path() / "materials/starch2007_finite_glue_elastic.nmatter";
        const auto parsed = parseMatterFile(materialPath);
        require(parsed.succeeded(), "native surface-map material parse failed");
        WorldSource legacySource = fixture(parsed.material);
        WorldSource compactSource = legacySource;
        compactSource.compactImmutableFEMSurfaces = true;
        const CompiledWorld legacy = compile(legacySource);
        const CompiledWorld compact = compile(compactSource);
        comparePhysicalInputs(legacy, compact);
        require(legacy.fem.surfaceFaces.size() == 11u,
            "legacy all-face source graph has unexpected count");
        require(compact.fem.surfaceFaces.size() == 10u &&
                compact.dispatch.surfaceFaceCount == 10u,
            "compact map exterior count mismatch (faces=" +
                std::to_string(compact.fem.surfaceFaces.size()) +
                ", dispatch=" +
                std::to_string(compact.dispatch.surfaceFaceCount) + ")");
        bool sourceIdDiffersFromCompactSlot = false;
        for (std::uint32_t slot = 0u;
             slot < compact.fem.surfaceFaces.size(); ++slot) {
            const auto& face = compact.fem.surfaceFaces[slot];
            const std::uint32_t sourceFaceId =
                4u * face.adjacency.x + face.sides.x;
            sourceIdDiffersFromCompactSlot |= sourceFaceId != slot;
        }
        require(sourceIdDiffersFromCompactSlot,
            "fixture did not distinguish source face IDs from compact slots");

        if (argc == 2 && std::string(argv[1]) == "--cpu-only") {
            std::cout << "PASS static FEM surface-map CPU source equivalence "
                      << checks << " checks; objects=2; environments="
                      << kEnvironments << "; shared_face_excluded=true"
                      << "; duplicated_interface_sides=2; physical_steps=0\n";
            return 0;
        }
        require(argc == 1, "usage: fem_static_surface_map_native_check [--cpu-only]");

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "Metal device unavailable");
        NSError* error = nil;
        NSString* path = [NSString stringWithUTF8String:NUMI_MATTER_METALLIB];
        NSURL* libraryURL = [NSURL fileURLWithPath:path];
        id<MTLLibrary> library = [device newLibraryWithURL:libraryURL error:&error];
        require(library != nil, error != nil
            ? std::string("cannot load Matter metallib: ") +
                [[error localizedDescription] UTF8String]
            : "cannot load Matter metallib");
        id<MTLFunction> function =
            [library newFunctionWithName:@"numi_matter_metal::nm_contact_build_surface_primitives"];
        require(function != nil, "surface builder kernel is absent from metallib");
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        require(pipeline != nil, error != nil
            ? std::string("cannot create surface builder pipeline: ") +
                [[error localizedDescription] UTF8String]
            : "cannot create surface builder pipeline");

        const auto legacyActive = dispatchBuilder(device, pipeline, legacy);
        const auto compactActive = dispatchBuilder(device, pipeline, compact);
        compareEquivalent(legacy, compact, legacyActive, compactActive);
        std::cout << "PASS native immutable FEM surface map equivalence "
                  << checks << " checks; environments=" << kEnvironments
                  << "; active_surface_faces=" << compactActive.size()
                  << "; physical_steps=0\n";
    } catch (const std::exception& exception) {
        std::cerr << "FAIL native immutable FEM surface map equivalence: "
                  << exception.what() << '\n';
        return 1;
    }
}
